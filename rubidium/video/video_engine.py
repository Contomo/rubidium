# rubidium/video/video_engine.py
from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Dict


@dataclass
class CmdStartRecording:
    session_id: str
    output_path: Path
    cmd_args: List[str]
    log_path: Path
    json_path: Path
    request_mono: float = 0.0


@dataclass
class CmdStopRecording:
    finalize: bool


@dataclass
class CmdLogMark:
    mark_data: Dict[str, Any]


@dataclass
class CmdQueueCut:
    input_path: Path
    output_path: Path
    start_s: float
    duration_s: float
    cmd_args_in: str
    cmd_args_out: str
    clip_metadata: Dict[str, Any]


class VideoEngine(threading.Thread):
    def __init__(self):
        super().__init__(name="RubidiumVideoEngine", daemon=True)
        self._queue = queue.Queue()
        self._proc: Optional[subprocess.Popen] = None
        
        self.session_data: Dict[str, Any] = {}
        self.marks_list: List[Dict[str, Any]] = []
        self.clips_list: List[Dict[str, Any]] = []
        self.pending_cuts: List[CmdQueueCut] = []

        self._startup_lag: float = 0.0
        self.json_path: Optional[Path] = None
        self.ffmpeg_path: str = "ffmpeg"
        self.nice_level: int = 0

    def configure(self, ffmpeg_bin: str, nice_level: int):
        self.ffmpeg_path = ffmpeg_bin
        self.nice_level = nice_level

    def submit(self, cmd: Any):
        if isinstance(cmd, CmdStartRecording):
            cmd.request_mono = time.monotonic()
        self._queue.put(cmd)

    def run(self):
        while True:
            try:
                cmd = self._queue.get()
                self._handle_command(cmd)
                self._queue.task_done()
            except Exception as e:
                logging.exception("rubidium_video_engine: unhandled exception in loop")

    def _handle_command(self, cmd: Any):
        if isinstance(cmd, CmdStartRecording):
            self._do_start(cmd)
        elif isinstance(cmd, CmdStopRecording):
            self._do_stop(cmd)
        elif isinstance(cmd, CmdLogMark):
            self._do_mark(cmd)
        elif isinstance(cmd, CmdQueueCut):
            self.pending_cuts.append(cmd)
        
        self._flush_json()

    def _do_start(self, cmd: CmdStartRecording):
        if self._proc is not None:
            logging.warning("rubidium_video: received start while already running")
            return

        self.json_path = cmd.json_path
        self.session_data = {
            "id": cmd.session_id,
            "file": str(cmd.output_path),
            "start_time": time.time(),
            "active": True,
            "startup_lag": 0.0
        }
        self.marks_list = []
        self.clips_list = []
        self.pending_cuts = []

        try:
            try:
                log_f = open(cmd.log_path, "w", encoding="utf-8")
            except Exception:
                log_f = open(os.devnull, "w")

            spawn_start_mono = time.monotonic()
            
            self._proc = subprocess.Popen(
                cmd.cmd_args,
                stdout=log_f,
                stderr=log_f,
                text=True,
                close_fds=True
            )
            self.session_data["pid"] = self._proc.pid

        except Exception as e:
            logging.error(f"rubidium_video: failed to start ffmpeg: {e}")
            self.session_data["error"] = str(e)
            return

        logging.info("rubidium_video: waiting for video file to appear...")
        
        real_start_mono = None

        for _ in range(200):
            try:
                if cmd.output_path.exists() and cmd.output_path.stat().st_size > 1024:
                    # File exists and has some data (header prob) TODO: check → 1kb may be too low if it writes header before frames arrive?
                    real_start_mono = time.monotonic()
                    break
            except Exception:
                pass
            time.sleep(0.025) # TODO: wait in frame intervals perhaps?
            
            if self._proc.poll() is not None:
                break
        
        if real_start_mono is None:
            logging.warning("rubidium_video: timed out waiting for file creation, sync might be off")
            real_start_mono = spawn_start_mono

        self._startup_lag = real_start_mono - cmd.request_mono
        self.session_data["startup_lag"] = self._startup_lag
        
        logging.info(f"rubidium_video: sync established. startup_lag={self._startup_lag:.4f}s")

    def _do_mark(self, cmd: CmdLogMark):
        self.marks_list.append(cmd.mark_data)

    def _do_stop(self, cmd: CmdStopRecording):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            
            self.session_data["active"] = False
            self.session_data["returncode"] = self._proc.returncode
            self._proc = None

        if cmd.finalize:
            self._run_post_processing()

    def _flush_json(self):
        if not self.json_path:
            return
        
        data = {
            "session": self.session_data,
            "marks": self.marks_list,
            "clips": self.clips_list
        }
        
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass 

    def _run_post_processing(self):
        if not self.pending_cuts:
            return

        try: # nice
            os.nice(self.nice_level)
        except Exception:
            pass

        for cut_job in self.pending_cuts:
            success = self._exec_ffmpeg_cut(cut_job)
            
            result_record = cut_job.clip_metadata.copy()
            result_record["ok"] = success
            result_record["file"] = str(cut_job.output_path)
            self.clips_list.append(result_record)
            
            self._flush_json()

    def _exec_ffmpeg_cut(self, job: CmdQueueCut) -> bool:
        """Directly runs ffmpeg to cut the clip"""

        adjusted_start = max(0.0, job.start_s - self._startup_lag)
        
        cmd = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
        
        if job.cmd_args_in:
            cmd += shlex.split(job.cmd_args_in)
            
        cmd += ["-ss", f"{adjusted_start:.4f}"]
        cmd += ["-i", str(job.input_path)]
        cmd += ["-t", f"{job.duration_s:.4f}"]
        
        if job.cmd_args_out:
            cmd += shlex.split(job.cmd_args_out)
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

        cmd.append(str(job.output_path))

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logging.warning(f"rubidium_video: cut failed: {res.stderr}")
                return False
            return True
        except Exception as e:
            logging.error(f"rubidium_video: cut exception: {e}")
            return False
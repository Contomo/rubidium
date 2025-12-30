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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict


@dataclass
class CmdStartRecording:
    session_id: str
    output_path: Path
    cmd_args: List[str]
    log_path: Path
    json_path: Path
    session_meta: Dict[str, Any] = field(default_factory=dict)
    request_mono: float = 0.0


@dataclass
class StopCompletion:
    event: threading.Event = field(default_factory=threading.Event)
    result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CmdStopRecording:
    finalize: bool
    completion: Optional[StopCompletion] = None


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

        self._lock = threading.Lock()

        # clip cutting (post-processing) progress
        self.cut_active: bool = False
        self.cut_total: int = 0
        self.cut_done: int = 0
        self.cut_ok: int = 0
        self.cut_current: Optional[str] = None
        self.cut_last_error: Optional[str] = None
        self.last_finalize: Optional[Dict[str, Any]] = None

    def configure(self, ffmpeg_bin: str, nice_level: int):
        self.ffmpeg_path = ffmpeg_bin
        self.nice_level = nice_level


    def _rel_path(self, p: Path) -> str:
        if self.json_path is not None:
            base = self.json_path.parent
            try:
                return str(p.relative_to(base))
            except ValueError:
                pass
        return str(p)

    def submit(self, cmd: Any):
        if isinstance(cmd, CmdStartRecording):
            cmd.request_mono = time.monotonic()
        self._queue.put(cmd)


    def get_status(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of the engine state."""
        with self._lock:
            cuts_total = int(self.cut_total)
            cuts_done = int(self.cut_done)
            cuts_ok = int(self.cut_ok)
            progress = (cuts_done / cuts_total) if cuts_total > 0 else None
            return {
                "session": dict(self.session_data),
                "cuts": {
                    "active": bool(self.cut_active),
                    "total": cuts_total,
                    "done": cuts_done,
                    "ok": cuts_ok,
                    "current": self.cut_current,
                    "progress": progress,
                    "last_error": self.cut_last_error,
                },
                "counts": {
                    "marks": len(self.marks_list),
                    "clips": len(self.clips_list),
                    "pending_cuts": len(self.pending_cuts),
                },
                "last_finalize": (None if self.last_finalize is None else dict(self.last_finalize)),
            }
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
            with self._lock:
                self.pending_cuts.append(cmd)
                self.cut_total += 1
        

    def _do_start(self, cmd: CmdStartRecording):
        if self._proc is not None:
            logging.warning("rubidium_video: received start while already running")
            return

        with self._lock:
            self.json_path = cmd.json_path
            self.session_data = {
                "id": cmd.session_id,
                "file": self._rel_path(cmd.output_path),
                "start_time": time.time(),
                "active": True,
                "startup_lag": 0.0,
                "meta": dict(cmd.session_meta or {}),
            }
            self.marks_list = []
            self.clips_list = []
            self.pending_cuts = []

            self.cut_active = False
            self.cut_total = 0
            self.cut_done = 0
            self.cut_ok = 0
            self.cut_current = None
            self.cut_last_error = None
            self.last_finalize = None

        log_f = None
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
            with self._lock:
                self.session_data["pid"] = self._proc.pid

        except Exception as e:
            logging.error(f"rubidium_video: failed to start ffmpeg: {e}")
            self.session_data["error"] = str(e)
            return
        finally:
            if log_f is not None:
                log_f.close()

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
        with self._lock:
            self.session_data["startup_lag"] = self._startup_lag
        
        logging.info(f"rubidium_video: sync established. startup_lag={self._startup_lag:.4f}s")

    def _do_mark(self, cmd: CmdLogMark):
        with self._lock:
            self.marks_list.append(cmd.mark_data)

    def _do_stop(self, cmd: CmdStopRecording):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logging.warning("rubidium_video: ffmpeg did not exit after kill()")

            with self._lock:
                self.session_data["active"] = False
                self.session_data["returncode"] = self._proc.returncode
            self._proc = None

        err: Optional[str] = None
        try:
            if cmd.finalize:
                self._run_post_processing()
        except Exception as e:
            logging.exception("rubidium_video: finalize/post-processing failed")
            err = str(e)
        finally:
            result = {
                "ok": err is None,
                "error": err,
                "json": str(self.json_path) if self.json_path else None,
                "session_id": self.session_data.get("id"),
                "clips": len(self.clips_list),
                "marks": len(self.marks_list),
            }
            with self._lock:
                self.last_finalize = dict(result)
            if cmd.completion is not None:
                cmd.completion.result = dict(result)
                cmd.completion.event.set()

    def _flush_json(self):
        json_path = None
        with self._lock:
            if self.json_path:
                json_path = self.json_path
            session = dict(self.session_data)
            marks = list(self.marks_list)
            clips = list(self.clips_list)
        if not json_path:
            return

        data = {
            "session": session,
            "marks": marks,
            "clips": clips,
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
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
        with self._lock:
            self.cut_active = True
            # cut_total is incremented when CmdQueueCut is received; this is a safety sync.
            self.cut_total = max(self.cut_total, len(self.pending_cuts))
            self.cut_done = 0
            self.cut_ok = 0
            self.cut_current = None
            self.cut_last_error = None

        for cut_job in list(self.pending_cuts):
            with self._lock:
                self.cut_current = str(cut_job.clip_metadata.get("key") or self._rel_path(cut_job.output_path))
            success = self._exec_ffmpeg_cut(cut_job)
            
            result_record = cut_job.clip_metadata.copy()
            result_record["ok"] = success
            result_record["file"] = self._rel_path(cut_job.output_path)
            with self._lock:
                self.clips_list.append(result_record)
                self.cut_done += 1
                if success:
                    self.cut_ok += 1
                else:
                    self.cut_last_error = "cut_failed"
            
            self._flush_json()

        with self._lock:
            self.pending_cuts.clear()
            self.cut_active = False
            self.cut_current = None

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

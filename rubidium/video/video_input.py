# rubidium/video/video_input.py
from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import threading
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from ..core.configview import ConfigView


class PlannerCallback:
    """Schedule reactor callbacks at motion-planner time.

    fire_at_rt = now_rt + (target_print_time + kin_flush_delay - mcu_now_print_time)

    target_print_time  - provided by toolhead lookahead callback
    kin_flush_delay    - corrects for kinematics flush buffering
    mcu_now_pt         - toolhead.mcu.estimated_print_time(now_rt)
    """

    def __init__(self, printer) -> None:
        self.printer = printer

    def get_kin_flush_delay(self) -> float:
        th = self.printer.lookup_object("toolhead", None)
        mq = self.printer.lookup_object("motion_queuing", None)

        # newer klipper moved it into motion_queuing
        if mq is not None and hasattr(mq, "get_kin_flush_delay"):
            return float(mq.get_kin_flush_delay())

        # older klipper/kalico still has it in toolhead
        if th is not None and hasattr(th, "kin_flush_delay"):
            return float(getattr(th, "kin_flush_delay"))

        logging.warning(
            "PlannerCallback (rubidium): unable to determine kin flush delay, using 250ms"
        )
        return 0.250

    def schedule_cb(
        self,
        cb: Callable[[float, float, Any], None],
        payload: Any = None,
    ) -> None:
        r = self.printer.get_reactor()
        th = self.printer.lookup_object("toolhead")

        def lookahead_cb(print_time: float):
            now_rt = float(r.monotonic())
            mcu_now_pt = float(th.mcu.estimated_print_time(now_rt))
            fire_at = now_rt + max(
                0.0, float(print_time) + self.get_kin_flush_delay() - mcu_now_pt
            )

            def timer_handler(eventtime: float):
                cb(float(eventtime), float(print_time), payload)
                return r.NEVER

            r.register_timer(timer_handler, fire_at)

        th.register_lookahead_callback(lookahead_cb)

def _is_url(text: str) -> bool:
    if text.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        return True
    if any(tok in text for tok in ("127.0.0.1", "localhost")):
        return True
    return False


def _json_sanitize(v: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _json_sanitize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_sanitize(x) for x in v]
    return str(v)


def _norm_scale_filter(res: str) -> Optional[str]:
    """Return `WxH` for ffmpeg's scale filter input (`scale=W:H`)."""
    s = (res or "").strip()
    if not s:
        return None
    if "x" in s:
        a, b = s.split("x", 1)
    elif ":" in s:
        a, b = s.split(":", 1)
    else:
        return None
    try:
        w = int(a)
        h = int(b)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{w}:{h}"


def _norm_video_size(res: str) -> Optional[str]:
    """Return `WxH` for ffmpeg's `-video_size` arg."""
    s = (res or "").strip()
    if not s:
        return None
    if "x" in s:
        a, b = s.split("x", 1)
    elif ":" in s:
        a, b = s.split(":", 1)
    else:
        return None
    try:
        w = int(a)
        h = int(b)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{w}x{h}"

def _sh_quote(s: str) -> str:
    """Return a POSIX-shell-safe single-quoted string."""
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"



def _sh_join(cmd: list[str]) -> str:
    return " ".join(_sh_quote(a) for a in cmd)


@dataclass(slots=True)
class VideoMark:
    key: str
    kind: str  # "start" | "end"
    idx: int
    t_s: float
    meta: dict[str, Any]
    eventtime: float
    print_time: float

class VideoInput:
    """Manage a video capture session and motion-timed markers"""

    def __init__(self, printer, *, base_section: Any, section: Any) -> None:
        self.printer = printer
        self.reactor = printer.get_reactor()
        self.gcode = printer.lookup_object("gcode")

        self.cv = cfg = ConfigView(base=base_section, override=section)
        self._planner_cb = PlannerCallback(printer)

        self.source = cfg.require_str("video_source")

        self.ffmpeg = cfg.get_str("ffmpeg", "ffmpeg").strip() or "ffmpeg"
        self.input_kind = cfg.get_str("video_input_kind", "auto").lower().strip()

        # Allow negative offsets to handle cases where marks effectively lag the visible motion.
        self.video_latency_ms = cfg.get_float("video_latency_ms", 0.0)

        self.video_extra_args: Tuple[str, str] = (
            cfg.get_str("video_extra_input_args", "").strip(),
            cfg.get_str("video_extra_output_args", "").strip(),
        )

        if not _is_url(self.source):
            in_args = shlex.split(self.video_extra_args[0])
            if any(opt in in_args for opt in ("-video_size", "-s", "-framerate", "-r")):
                raise section.error(
                    'Please supply video size and/or frame rate via "video_resolution" and "video_framerate" '
                    "and not via extra args!"
                )
            self.scale = cfg.require_str("video_resolution")
            self.framerate = cfg.require_int("video_framerate", minval=1)
        else:
            self.scale = cfg.get_str_opt("video_resolution")
            self.framerate = cfg.get_int_opt("video_framerate", minval=1)

        self.video_session_filename = cfg.get_str(
            "video_session_filename", "rubidium_scan_session"
        )
        self.video_cut_filename = cfg.get_str("video_cut_filename", "rubidium_cut")
        self.video_cut_extra_args: Tuple[str, str] = (
            cfg.get_str("video_cut_extra_input_args", "").strip(),
            cfg.get_str("video_cut_extra_output_args", "").strip(),
        )

        # postprocess (conversion/cutting) runs in a low-priority child process
        self.video_dump_container = cfg.get_str("video_dump_container", "mkv").strip().lower() or "mkv"
        self.video_postprocess_nice = cfg.get_int("video_postprocess_nice", 15, minval=0, maxval=19)
        self.video_convert_extra_args: Tuple[str, str] = (
            cfg.get_str("video_convert_extra_input_args", "").strip(),
            cfg.get_str("video_convert_extra_output_args", "").strip(),
        )

        # session state
        self._stopping: bool = False
        self._proc: Optional[subprocess.Popen] = None
        self._outdir: Optional[Path] = None
        self._session_started_rt: Optional[float] = None
        self._session_id: Optional[str] = None
        self._session_paths: Optional[dict[str, Path]] = None
        self._log_path: Optional[Path] = None
        self._json_path: Optional[Path] = None
        self._ffmpeg_cmd: Optional[list[str]] = None
        self._marks: list[VideoMark] = []
        self._clips: list[dict[str, Any]] = []

        self._video_dump_path: Optional[Path] = None
        self._video_mp4_path: Optional[Path] = None
        self._postproc: Optional[subprocess.Popen] = None
        self._postproc_log_path: Optional[Path] = None
        self._postproc_script_path: Optional[Path] = None
        self._postproc_cmd: Optional[list[str]] = None
        self._logged_mark_debug: bool = False

        # planner-time anchors (THIS is what fixes your timing)
        self._session_start_pt: Optional[float] = None
        self._session_end_pt: Optional[float] = None

    # --------------------------- status/reporting
    def _respond_error(self, msg: str) -> None:
        logging.warning(msg)
        lines = msg.strip().split("\n")
        if len(lines) > 1:
            self.gcode.respond_info("\n".join(lines), log=False)
        self.gcode.respond_raw("!! rubidium_video: %s" % (lines[0].strip(),))

    def _respond_info(self, msg: str, log: bool = False) -> None:
        self.gcode.respond_info("rubidium_video: %s" % (msg,), log=log)

    # --------------------------- ffmpeg command building
    def _pick_input_kind(self) -> str:
        kind = (self.input_kind or "auto").strip().lower()
        if kind == "auto":
            return "url" if _is_url(self.source) else "v4l2"
        return kind

    def _build_record_cmd(self, out_video: Path) -> list[str]:
        src = self.source
        kind = self._pick_input_kind()

        cmd: list[str] = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        cmd += shlex.split(self.video_extra_args[0])

        if kind == "url":
            cmd += ["-i", src]
        elif kind == "v4l2":
            cmd += ["-f", "v4l2"]
            if self.framerate is not None:
                cmd += ["-framerate", str(int(self.framerate))]
            vs = _norm_video_size(self.scale or "")
            if vs is not None:
                cmd += ["-video_size", vs]
            cmd += ["-i", src]
        else:
            raise RuntimeError(f"rubidium_video: invalid video_input_kind '{kind}'")

        vf_parts: list[str] = []
        if kind == "url":
            sf = _norm_scale_filter(self.scale or "")
            if sf is not None:
                vf_parts.append(f"scale={sf}")
            if self.framerate is not None:
                vf_parts.append(f"fps={int(self.framerate)}")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        out_args = self.video_extra_args[1]
        if not out_args:
            if kind == "url":
                out_args = f"-an -c copy"
            else:
                out_args = f"-an -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        cmd += shlex.split(out_args)
        cmd += [str(out_video)]
        return cmd
    

    def _stop_proc_blocking(self, p: Optional[subprocess.Popen]) -> None:
        if p is None:
            return
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=10.0)
                except Exception:
                    p.terminate()
                    try:
                        p.wait(timeout=5.0)
                    except Exception:
                        p.kill()
        except Exception:
            pass

    # --------------------------- session lifecycle
    def start_session(self, outdir: Path) -> None:
        """Start recording a single "session" video into outdir."""
        if self._proc is not None and self._proc.poll() is None:
            self._respond_error("start_session called while ffmpeg is still running")
            return

        ts = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        sid = f"{ts}_{os.getpid()}"

        session_dir = Path(outdir) / f"recording_{sid}"
        session_dir.mkdir(parents=True, exist_ok=True)

        dump_ext = self.video_dump_container
        video_dump_path = session_dir / f"{self.video_session_filename}.dump.{dump_ext}"
        video_mp4_path  = session_dir / f"{self.video_session_filename}.mp4"
        log_path        = session_dir / f"{self.video_session_filename}.record.ffmpeg.log"
        json_path       = session_dir / f"{self.video_session_filename}.json"
        post_log_path   = session_dir / f"{self.video_session_filename}.postprocess.log"
        post_sh_path    = session_dir / f"{self.video_session_filename}.postprocess.sh"

        cmd = self._build_record_cmd(video_dump_path)

        self._marks = []
        self._clips = []
        self._outdir = session_dir
        self._session_started_rt = float(self.reactor.monotonic())
        self._session_id = sid
        self._video_dump_path = video_dump_path
        self._video_mp4_path = video_mp4_path
        self._log_path = log_path
        self._json_path = json_path
        self._postproc_log_path = post_log_path
        self._postproc_script_path = post_sh_path
        self._postproc = None
        self._postproc_cmd = None
        self._ffmpeg_cmd = cmd
        self._session_start_pt = None
        self._session_end_pt = None
        self._logged_mark_debug = False
        self._proc = None

        self._write_state_json()

        def _start_cb(eventtime: float, print_time: float, payload: Any) -> None:
            kin = float(self._planner_cb.get_kin_flush_delay())
            self._session_start_pt = float(print_time) + kin
            self._respond_info(
                "session_start_pt=%.6f print_time=%.6f kin_flush=%.6f eventtime=%.6f"
                % (self._session_start_pt, float(print_time), kin, float(eventtime))
            )

            try:
                try:
                    log_f = open(log_path, "w", encoding="utf-8")
                except Exception:
                    log_f = open(os.devnull, "w")

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=log_f,
                    text=True,
                )
            except Exception as e:
                self._proc = None
                self._respond_error(f"failed to start ffmpeg: {e}")
                self._write_state_json()
                return

            self._write_state_json()
            self._respond_info(f"recording started -> {video_dump_path}")

        self._planner_cb.schedule_cb(_start_cb, payload=None)


    def stop_session(self, *, finalize: bool) -> None:
        """Stop ffmpeg (planner-timed) and optionally cut clips."""
        if self._proc is None:
            return
        if self._stopping:
            return

        self._stopping = True
        p = self._proc

        def _end_cb(eventtime: float, print_time: float, payload: Any) -> None:
            kin = float(self._planner_cb.get_kin_flush_delay())
            self._session_end_pt = float(print_time) + kin
            self._write_state_json()

            def worker() -> None:
                time.sleep(0.05)
                self._stop_proc_blocking(p)
                if self._proc is p:
                    self._proc = None

                if finalize:
                    try:
                        self._spawn_postprocess()
                    except Exception:
                        self._respond_error("rubidium_video: failed spawning postprocess")

                self._stopping = False
                self._write_state_json()

            threading.Thread(target=worker, daemon=True).start()

        self._planner_cb.schedule_cb(_end_cb, payload=None)
        self._respond_info("recording stop scheduled")

    # --------------------------- marking
    def mark(self, *, kind: str, idx: int, meta: Optional[dict[str, Any]] = None, key: Optional[str] = None) -> None:
        """Schedule a motion-timed mark"""
        if self._json_path is None:
            return

        kind_l = (kind or "").strip().lower()
        if kind_l not in ("start", "end"):
            raise ValueError("rubidium_video.mark: kind must be 'start' or 'end'")

        idx_i = int(idx)
        key_s = str(key) if key is not None else f"line_{idx_i:03d}"
        meta_d = dict(meta or {})

        latency_s = float(self.video_latency_ms) / 1000.0

        payload = {
            "key": key_s,
            "kind": kind_l,
            "idx": idx_i,
            "meta": _json_sanitize(meta_d),
            "latency_s": latency_s,
        }

        def _mark_cb(eventtime: float, print_time: float, pl: dict[str, Any]) -> None:
            kin = float(self._planner_cb.get_kin_flush_delay())
            mark_pt = float(print_time) + kin

            if self._session_start_pt is None:
                self._session_start_pt = mark_pt

            t_s = (mark_pt - float(self._session_start_pt)) + float(pl["latency_s"])
            if (not self._logged_mark_debug) and str(pl.get("kind")) == "start":
                self._logged_mark_debug = True
                self._respond_info(
                    "mark_debug key=%s mark_pt=%.6f session_start_pt=%.6f latency=%.6f t_s=%.6f eventtime=%.6f"
                    % (
                        str(pl.get("key")),
                        float(mark_pt),
                        float(self._session_start_pt),
                        float(pl.get("latency_s", 0.0)),
                        float(t_s),
                        float(eventtime),
                    )
                )

            m = VideoMark(
                key=str(pl["key"]),
                kind=str(pl["kind"]),
                idx=int(pl["idx"]),
                t_s=float(t_s),
                meta=dict(pl.get("meta") or {}),
                eventtime=float(eventtime),
                print_time=float(mark_pt),
            )
            self._marks.append(m)
            self._write_state_json()

        self._planner_cb.schedule_cb(_mark_cb, payload=payload)

    # --------------------------- persistence
    def _write_state_json(self) -> None:
        p = self._json_path
        if p is None:
            return

        session = {
            "id": self._session_id,
            "source": self.source,
            "video_dump": str(self._video_dump_path) if self._video_dump_path is not None else None,
            "video_mp4": str(self._video_mp4_path) if self._video_mp4_path is not None else None,
            "video": str(self._video_mp4_path) if self._video_mp4_path is not None else (str(self._video_dump_path) if self._video_dump_path is not None else None),
            "postprocess_nice": int(self.video_postprocess_nice),
            "postprocess_script": str(self._postproc_script_path) if self._postproc_script_path is not None else None,
            "postprocess_log": str(self._postproc_log_path) if self._postproc_log_path is not None else None,
            "postprocess_cmd": list(self._postproc_cmd or []),
            "ffmpeg": self.ffmpeg,
            "ffmpeg_cmd": list(self._ffmpeg_cmd or []),
            "ffmpeg_log": str(self._log_path) if self._log_path is not None else None,
            "started_rt": self._session_started_rt,
            "started_pt": self._session_start_pt,
            "ended_pt": self._session_end_pt,
            "latency_ms": float(self.video_latency_ms),
        }
        if self._proc is not None:
            try:
                session["ffmpeg_pid"] = int(self._proc.pid)
                session["ffmpeg_returncode"] = self._proc.poll()
            except Exception:
                pass

        if self._postproc is not None:
            try:
                session["postprocess_pid"] = int(self._postproc.pid)
                session["postprocess_returncode"] = self._postproc.poll()
            except Exception:
                pass

        marks = [
            {
                "key": m.key,
                "kind": m.kind,
                "idx": m.idx,
                "t_s": m.t_s,
                "eventtime": m.eventtime,
                "print_time": m.print_time,
                "meta": _json_sanitize(m.meta),
            }
            for m in self._marks
        ]

        data = {
            "session": _json_sanitize(session),
            "marks": marks,
            "clips": _json_sanitize(self._clips),
        }

        try:
            p.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
        except Exception:
            logging.exception("rubidium_video: failed writing json state")

    
    # --------------------------- postprocess (convert + cut)
    def _build_convert_cmd(self, in_path: Path, out_path: Path) -> list[str]:
        cmd: list[str] = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        cmd += shlex.split(self.video_convert_extra_args[0].strip())
        cmd += ["-i", str(in_path)]

        out_args = self.video_convert_extra_args[1].strip()
        if not out_args:
            # Default to a cheap remux if possible. If the input isn't mp4-compatible,
            # users can override this via video_convert_extra_output_args.
            out_args = "-an -c copy"
        cmd += shlex.split(out_args)
        cmd += [str(out_path)]
        return cmd

    def _build_cut_cmd(
        self,
        in_path: Path,
        out_path: Path,
        *,
        start_s: float,
        dur_s: float,
    ) -> list[str]:
        cmd: list[str] = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        cmd += shlex.split(self.video_cut_extra_args[0].strip())
        cmd += ["-ss", f"{float(start_s):.6f}", "-i", str(in_path), "-t", f"{float(dur_s):.6f}"]

        out_args = self.video_cut_extra_args[1].strip()
        if not out_args:
            # Default to a sane h264 encode for accurate cuts.
            out_args = "-an -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p"
        cmd += shlex.split(out_args)
        cmd += [str(out_path)]
        return cmd

    def _spawn_postprocess(self) -> None:
        if self._outdir is None or self._video_dump_path is None or self._video_mp4_path is None:
            return
        if self._postproc is not None and self._postproc.poll() is None:
            self._respond_error("postprocess already running, skipping spawn")
            return

        if not self._marks:
            self._respond_error("no marks recorded, skipping postprocess")
            return

        pending: dict[str, VideoMark] = {}
        segments: list[tuple[VideoMark, VideoMark]] = []
        for m in self._marks:
            if m.kind == "start":
                pending[m.key] = m
            elif m.kind == "end":
                sm = pending.pop(m.key, None)
                if sm is not None:
                    segments.append((sm, m))

        if not segments:
            self._respond_error("no complete start/end pairs, skipping postprocess")
            return

        script_path = self._postproc_script_path or (self._outdir / f"{self.video_session_filename}.postprocess.sh")
        log_path = self._postproc_log_path or (self._outdir / f"{self.video_session_filename}.postprocess.log")

        convert_cmd = self._build_convert_cmd(self._video_dump_path, self._video_mp4_path)

        self._clips = []
        cut_cmds: list[tuple[int, str, list[str]]] = []
        for sm, em in segments:
            start_s = max(0.0, float(sm.t_s))
            end_s = max(start_s, float(em.t_s))
            dur_s = max(0.0, end_s - start_s)

            out_name = f"{self.video_cut_filename}_{int(sm.idx):03d}.mp4"
            out_path = self._outdir / out_name

            cut_cmd = self._build_cut_cmd(self._video_mp4_path, out_path, start_s=start_s, dur_s=dur_s)
            cut_cmds.append((int(sm.idx), str(sm.key), cut_cmd))

            self._clips.append(
                {
                    "key": sm.key,
                    "idx": sm.idx,
                    "start": start_s,
                    "end": end_s,
                    "file": str(out_path),
                    "ok": None,
                }
            )

        sh_lines: list[str] = []
        sh_lines.append("#!/bin/sh")
        sh_lines.append("set -u")
        sh_lines.append("")
        sh_lines.append(f"echo {_sh_quote('[rubidium_video] postprocess starting')}")
        sh_lines.append(f"echo {_sh_quote(f'[rubidium_video] nice={int(self.video_postprocess_nice)}')}")
        sh_lines.append("")

        sh_lines.append(f"echo {_sh_quote('[rubidium_video] convert session -> mp4')}")
        sh_lines.append(f"if ! {_sh_join(convert_cmd)}; then")
        sh_lines.append(f"  echo {_sh_quote('[rubidium_video] convert FAILED')}")
        sh_lines.append("  exit 1")
        sh_lines.append("fi")
        sh_lines.append("")

        for idx_i, key_s, cmd in cut_cmds:
            sh_lines.append(f"echo {_sh_quote(f'[rubidium_video] cut idx={idx_i:03d} key={key_s}')}")
            sh_lines.append(f"{_sh_join(cmd)} || echo {_sh_quote(f'[rubidium_video] cut FAILED idx={idx_i:03d} key={key_s}')}")
            sh_lines.append("")

        sh_lines.append(f"echo {_sh_quote('[rubidium_video] postprocess done')}")

        try:
            script_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
            try:
                os.chmod(str(script_path), 0o755)
            except Exception:
                pass
        except Exception as e:
            self._respond_error(f"failed writing postprocess script: {e}")
            return

        # Spawn a detached, low-priority child process.
        try:
            try:
                log_f = open(log_path, "w", encoding="utf-8")
            except Exception:
                log_f = open(os.devnull, "w")

            def _preexec() -> None:
                try:
                    os.setsid()
                except Exception:
                    pass
                try:
                    os.nice(int(self.video_postprocess_nice))
                except Exception:
                    pass

            self._postproc_cmd = [str(script_path)]
            self._postproc = subprocess.Popen(
                [str(script_path)],
                cwd=str(self._outdir),
                stdout=log_f,
                stderr=log_f,
                text=True,
                preexec_fn=_preexec,
                close_fds=True,
            )
        except Exception as e:
            self._postproc = None
            self._respond_error(f"failed to spawn postprocess: {e}")
            self._write_state_json()
            return

        self._respond_info(f"postprocess spawned (nice={int(self.video_postprocess_nice)}) -> {script_path}")
        self._write_state_json()

# --------------------------- cutting
    def _run_ffmpeg(self, cmd: list[str]) -> tuple[int, str]:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return int(p.returncode), out.strip()

    def _cut_clips(self) -> None:
        if self._video_mp4_path is None or self._outdir is None:
            return
        if not self._marks:
            self._respond_error("no marks recorded, skipping clip cut")
            return

        pending: dict[str, VideoMark] = {}
        segments: list[tuple[VideoMark, VideoMark]] = []
        for m in self._marks:
            if m.kind == "start":
                pending[m.key] = m
            elif m.kind == "end":
                sm = pending.pop(m.key, None)
                if sm is not None:
                    segments.append((sm, m))

        if not segments:
            self._respond_error("no complete start/end pairs, skipping clip cut")
            return

        for sm, em in segments:
            start_s = max(0.0, float(sm.t_s))
            end_s = max(start_s, float(em.t_s))

            out_name = f"{self.video_cut_filename}_{int(sm.idx):03d}.mp4"
            out_path = self._outdir / out_name

            cmd: list[str] = [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
            ]
            cmd += shlex.split(self.video_cut_extra_args[0].strip())
            cmd += ["-ss", f"{start_s:.6f}", "-i", str(self._video_mp4_path), "-t", f"{max(0.0, end_s - start_s):.6f}"]

            cmd += shlex.split(self.video_cut_extra_args[1].strip())
            cmd += [str(out_path)]

            rc, out = self._run_ffmpeg(cmd)
            if rc != 0:
                self._respond_error(
                    f"ffmpeg cut failed for idx={sm.idx} key={sm.key} (rc={rc})\n{out}"
                )
                self._clips.append(
                    {
                        "key": sm.key,
                        "idx": sm.idx,
                        "start": start_s,
                        "end": end_s,
                        "file": str(out_path),
                        "ok": False,
                        "returncode": rc,
                    }
                )
                continue

            self._clips.append(
                {
                    "key": sm.key,
                    "idx": sm.idx,
                    "start": start_s,
                    "end": end_s,
                    "file": str(out_path),
                    "ok": True,
                }
            )

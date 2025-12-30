# rubidium/video/video_input.py
from __future__ import annotations


import logging
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Dict


from ..core.configview import ConfigView
from .video_engine import VideoEngine, CmdStartRecording, CmdStopRecording, CmdLogMark, CmdQueueCut, StopCompletion


def _is_url(text: str) -> bool:
    if text.startswith(("http://", "https://", "rtsp://", "rtsps://")):
        return True
    if any(tok in text for tok in ("127.0.0.1", "localhost")):
        return True
    return False


def _json_sanitize(v: Any) -> Any:
    """Best-effort conversion to JSON-serializable types"""
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
    """Return `WxH` for ffmpeg's scale filter input"""
    s = (res or "").strip()
    if not s: return None
    if "x" in s: a, b = s.split("x", 1)
    elif ":" in s: a, b = s.split(":", 1)
    else: return None
    try:
        w, h = int(a), int(b)
        if w <= 0 or h <= 0: return None
        return f"{w}:{h}"
    except Exception:
        return None


def _norm_video_size(res: str) -> Optional[str]:
    """Return `WxH` for ffmpeg's `-video_size` arg"""
    s = (res or "").strip()
    if not s: return None
    if "x" in s: a, b = s.split("x", 1)
    elif ":" in s: a, b = s.split(":", 1)
    else: return None
    try:
        w, h = int(a), int(b)
        if w <= 0 or h <= 0: return None
        return f"{w}x{h}"
    except Exception:
        return None



def _sanitize_path_segment(name: Optional[str]) -> str:
    s = (name or "").strip()
    if not s:
        return "scan"
    s = re.sub(r"\s+", "_", s)
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s or "scan"


class PlannerCallback:
    """Schedule reactor callbacks at motion-planner time"""
    def __init__(self, printer) -> None:
        self.printer = printer

    def get_kin_flush_delay(self) -> float:
        th = self.printer.lookup_object("toolhead", None)
        mq = self.printer.lookup_object("motion_queuing", None)
        # newer klipper
        if mq is not None and hasattr(mq, "get_kin_flush_delay"):
            return float(mq.get_kin_flush_delay())
        # older klipper
        if th is not None and hasattr(th, "kin_flush_delay"):
            return float(getattr(th, "kin_flush_delay"))
        return 0.250

    def schedule_cb(self, cb, payload=None):
        r = self.printer.get_reactor()
        th = self.printer.lookup_object("toolhead")

        def lookahead_cb(print_time: float):
            target_pt = float(print_time)
            now_rt = float(r.monotonic())
            mcu_now_pt = float(th.mcu.estimated_print_time(now_rt))

            fire_at = now_rt + max(0.0, target_pt - mcu_now_pt)

            def timer_handler(eventtime: float):
                now_pt = float(th.mcu.estimated_print_time(eventtime))
                if now_pt < target_pt:
                    return eventtime + 0.001
                cb(float(eventtime), target_pt, payload)
                return r.NEVER

            r.register_timer(timer_handler, fire_at)

        th.register_lookahead_callback(lookahead_cb)


class VideoInput:
    """Manage video capture by delegating heavy tasks to VideoEngine"""

    def __init__(self, printer, *, base_section: Any, section: Any) -> None:
        self.printer = printer
        self.reactor = printer.get_reactor()
        self.gcode = printer.lookup_object("gcode")
        self.cv = cfg = ConfigView(base=base_section, override=section)
        self._planner_cb = PlannerCallback(printer)

        self.source = cfg.require_str("video_source")
        self.ffmpeg_bin = cfg.get_str("ffmpeg", "ffmpeg").strip()
        self.input_kind = cfg.get_str("video_input_kind", "auto").lower().strip() # TODO: getchoice
        
        self.latency_s = cfg.get_float("video_latency_ms", 0.0) / 1000.0

        self.video_extra_args: Tuple[str, str] = (
            cfg.get_str("video_extra_input_args", "").strip(),
            cfg.get_str("video_extra_output_args", "").strip(),
        )
        
        if not _is_url(self.source):
            in_args = shlex.split(self.video_extra_args[0])
            if any(opt in in_args for opt in ("-video_size", "-s", "-framerate", "-r")):
                raise section.error(
                    'Please use "video_resolution" and "video_framerate" instead of raw ffmpeg args.'
                )
            self.scale = cfg.require_str("video_resolution")
            self.framerate = cfg.require_int("video_framerate", minval=1)
        else:
            self.scale = cfg.get_str_opt("video_resolution")
            self.framerate = cfg.get_int_opt("video_framerate", minval=1)

        self.video_session_filename = cfg.get_str("video_session_filename", "rubidium_scan_session")
        self.video_cut_filename = cfg.get_str("video_cut_filename", "rubidium_cut")
        self.video_dump_container = cfg.get_str("video_dump_container", "mkv").strip().lower()

        self.video_cut_extra_args: Tuple[str, str] = (
            cfg.get_str("video_cut_extra_input_args", "").strip(),
            cfg.get_str("video_cut_extra_output_args", "").strip(),
        )

        self.engine = VideoEngine()
        self.engine.configure(
            ffmpeg_bin=self.ffmpeg_bin,
            nice_level=cfg.get_int("video_postprocess_nice", 15, minval=0, maxval=20)
        )
        self.engine.start()

        self._pending_completion: Optional[StopCompletion] = None
        self._last_finalize: Optional[Dict[str, Any]] = None

        self._session_start_pt: Optional[float] = None
        self._active_marks: Dict[str, Tuple[float, Dict[str, Any]]] = {} 
        self._old_latency_s: Optional[float] = None
        self._current_outdir: Optional[Path] = None
        self._clips_dir: Optional[Path] = None

    def _pick_input_kind(self) -> str:
        kind = self.input_kind
        if kind == "auto":
            return "url" if _is_url(self.source) else "v4l2"
        return kind

    def _build_base_ffmpeg_cmd(self) -> list[str]:
        kind = self._pick_input_kind()
        cmd = [self.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        cmd += shlex.split(self.video_extra_args[0])

        if kind == "url":
            cmd += ["-i", self.source]
        elif kind == "v4l2":
            cmd += ["-f", "v4l2"]
            if self.framerate:
                cmd += ["-framerate", str(int(self.framerate))]
            vs = _norm_video_size(self.scale or "")
            if vs:
                cmd += ["-video_size", vs]
            cmd += ["-i", self.source]
        else:
            raise RuntimeError(f"rubidium_video: invalid video_input_kind '{kind}'")

        vf_parts = []
        if kind == "url":
            sf = _norm_scale_filter(self.scale or "")
            if sf: vf_parts.append(f"scale={sf}")
            if self.framerate: vf_parts.append(f"fps={int(self.framerate)}")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        out_args = self.video_extra_args[1]
        if not out_args:
            if kind == "url":
                out_args = "-an -c copy"
            else:
                out_args = "-an -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        
        cmd += shlex.split(out_args)
        return cmd

    def set_latency_offset(self, offset: Optional[float]):
        if offset is None: return
        if self._old_latency_s is None:
            self._old_latency_s = self.latency_s
        # G-code override uses ms like config; store seconds internally.
        self.latency_s = float(offset) / 1000.0

    def reset_latency_offset(self):
        if self._old_latency_s is None: return
        self.latency_s = self._old_latency_s

    def start_session(self, outdir: Path, *, meta: Optional[Dict[str, Any]] = None, dirname: Optional[str] = None) -> None:
        """Prepare session paths and schedule recording start"""
        
        session_name = f"{_sanitize_path_segment(dirname)}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
        session_dir = Path(outdir) / session_name
        if session_dir.exists():
            session_dir = Path(outdir) / f"{session_name}_{os.getpid()}"
        session_dir.mkdir(parents=True, exist_ok=True)

        clips_dir = session_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        self._current_outdir = session_dir
        self._clips_dir = clips_dir
        video_path = session_dir / f"{self.video_session_filename}.dump.{self.video_dump_container}"
        log_path = session_dir / f"{self.video_session_filename}.record.log"
        json_path = session_dir / f"{self.video_session_filename}.json"

        meta_s = _json_sanitize(meta or {})

        cmd_args = self._build_base_ffmpeg_cmd()
        cmd_args.append(str(video_path))

        def _start_cb(eventtime: float, print_time: float, _: Any) -> None:
            self._session_start_pt = print_time
            self._active_marks.clear()
            
            self.gcode.respond_info(f"rubidium_video: starting recording -> {video_path.name}")
            
            self.engine.submit(CmdStartRecording(
                session_id=session_dir.name,
                output_path=video_path,
                cmd_args=cmd_args,
                log_path=log_path,
                json_path=json_path,
                session_meta=meta_s
            ))

        self._planner_cb.schedule_cb(_start_cb, payload=None)

    def stop_session(self, *, finalize: bool, notify_done: bool = False) -> None:
        completion = StopCompletion() if notify_done else None
        if completion is not None:
            self._pending_completion = completion
        def _end_cb(eventtime: float, print_time: float, _: Any) -> None:
            self.gcode.respond_info("rubidium_video: stopping recording")
            self.engine.submit(CmdStopRecording(finalize=finalize, completion=completion))
            self._session_start_pt = None
            self._active_marks.clear()
            self.reset_latency_offset()
            self._clips_dir = None
            self._current_outdir = None

        self._planner_cb.schedule_cb(_end_cb, payload=None)

        if completion is not None:
            def _poll_done(eventtime: float):
                c = self._pending_completion
                if c is None:
                    return self.reactor.NEVER
                if not c.event.is_set():
                    return eventtime + 0.20
                self._pending_completion = None
                self._last_finalize = dict(c.result)
                msg = "rubidium_video: post-processing complete"
                if self._last_finalize.get("ok") is False:
                    msg += f" (failed: {self._last_finalize.get('error')})"
                self.gcode.respond_info(msg)
                return self.reactor.NEVER

            self.reactor.register_timer(_poll_done, self.reactor.monotonic() + 0.20)

    def mark(self, *, kind: str, idx: int, meta: Optional[dict[str, Any]] = None, key: Optional[str] = None) -> None:
        kind_l = (kind or "").strip().lower()
        idx_i = int(idx)
        key_s = str(key) if key is not None else f"line_{idx_i:03d}"
        meta_d = _json_sanitize(meta or {})

        payload = {
            "key": key_s, "kind": kind_l, "idx": idx_i, "meta": meta_d
        }

        def _mark_cb(eventtime: float, print_time: float, pl: dict[str, Any]) -> None:
            if self._session_start_pt is None: return

            # Latency shifts marks earlier in time (compensation).
            t_s = (print_time - self._session_start_pt) - self.latency_s
            t_s = max(0.0, t_s) 

            mark_data = {
                "key": pl["key"], "kind": pl["kind"], "idx": pl["idx"],
                "t_s": t_s, "meta": pl["meta"],
                "eventtime": eventtime, "print_time": print_time
            }
            self.engine.submit(CmdLogMark(mark_data))

            if pl["kind"] == "start":
                self._active_marks[pl["key"]] = (t_s, dict(pl.get("meta", {}) or {}))
                
            elif pl["kind"] == "end":
                mark = self._active_marks.pop(pl["key"], None)
                if mark is not None and self._current_outdir and self._clips_dir:
                    start_ts, start_meta = mark
                    duration = t_s - start_ts
                    
                    out_name = f"{self.video_cut_filename}_{pl['idx']:03d}.mp4"
                    out_path = self._clips_dir / out_name
                    dump_path = self._current_outdir / f"{self.video_session_filename}.dump.{self.video_dump_container}"

                    clip_meta = {
                        "key": pl["key"],
                        "idx": pl["idx"],
                        "start": start_ts,
                        "end": t_s
                    }
                    if start_meta:
                        clip_meta.update(start_meta)
                    if pl["meta"]:
                        clip_meta.update(pl["meta"])

                    self.engine.submit(CmdQueueCut(
                        input_path=dump_path,
                        output_path=out_path,
                        start_s=start_ts,
                        duration_s=duration,
                        cmd_args_in=self.video_cut_extra_args[0],
                        cmd_args_out=self.video_cut_extra_args[1],
                        clip_metadata=clip_meta
                    ))

        self._planner_cb.schedule_cb(_mark_cb, payload=payload)

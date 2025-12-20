# rubidium/video/video_input.py
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.configview import ConfigView


@dataclass(slots=True)
class VideoMark:
    key: str
    kind: str
    t_s: float
    meta: dict[str, Any]


class VideoInput:
    """Continuous recorder with motion-synchronized marks"""

    def __init__(self, printer, *, base_section, section) -> None:
        self.printer = printer
        self.reactor = printer.get_reactor()
        self.gcode = printer.lookup_object("gcode")
        self.toolhead = None

        self.cv = ConfigView(base=base_section, override=section)
        where = getattr(section, "get_name", lambda: "[rubidium scan]")()

        self.enabled = self.cv.get_bool("video_enable", True)

        self.ffmpeg = self.cv.get_str("ffmpeg", "ffmpeg")

        self.source = self.cv.get_str_opt("video_source") or ""
        self.resolution = self.cv.get_str_opt("video_resolution") or ""
        self.framerate = self.cv.get_int("video_framerate", 25, minval=1)

        self.input_kind = self.cv.get_str("video_input_kind", "auto").lower().strip()
        self.video_args = self.cv.get_str("video_args", "").strip()

        # NOTE: do not sleep inside gcode handlers; these delays are meant to be
        # implemented as G4 dwells in the caller (scan/print templates).
        self.start_delay_s = self.cv.get_float("ffmpeg_start_delay", 0.0, minval=0.0)
        self.stop_delay_s = self.cv.get_float("ffmpeg_stop_delay", 0.0, minval=0.0)

        self.ffmpeg_common_args = self.cv.get_str(
            "ffmpeg_common_args", "-hide_banner -loglevel error"
        )
        self.ffmpeg_input_args_url = self.cv.get_str(
            "ffmpeg_input_args_url", "-an -i {src}"
        )
        self.ffmpeg_input_args_v4l2 = self.cv.get_str(
            "ffmpeg_input_args_v4l2",
            "-f v4l2 -framerate {fps} -video_size {resx} -i {src}",
        )

        self.ffmpeg_session_output_args = self.cv.get_str(
            "ffmpeg_session_output_args",
            "-vf scale={res} -r {fps} -pix_fmt yuv420p {out}",
        )
        self.ffmpeg_cut_args = self.cv.get_str(
            "ffmpeg_cut_args",
            "-ss {ss} -to {to} -i {session} -vf scale={res} -r {fps} -pix_fmt yuv420p {out}",
        )

        self.session_filename = self.cv.get_str(
            "video_session_filename", "rubidium_scan_session.mp4"
        )

        if self.enabled and not self.source:
            raise base_section.error(
                f"rubidium: video_enable True but missing video_source in {where} or [rubidium]"
            )

        # video_resolution is only required when the configured ffmpeg templates
        # actually reference it. (URLs generally don't need it; v4l2 often does.)
        if self.enabled and not self.resolution:
            needs_res = (
                "{res}" in self.ffmpeg_session_output_args
                or "{res}" in self.ffmpeg_cut_args
                or "{resx}" in self.ffmpeg_input_args_v4l2
            )
            if needs_res and self._resolve_input_kind() == "v4l2":
                raise base_section.error(
                    "rubidium: video_resolution is required for v4l2 with the default ffmpeg_* args "
                    f"(set video_resolution: 1280:720 or override ffmpeg_input_args_v4l2/ffmpeg_*_args). "
                    f"Location: {where}"
                )

        self._session_proc: Optional[subprocess.Popen] = None
        self._session_path: Optional[Path] = None
        self._session_t0: float = 0.0
        self._marks: list[VideoMark] = []
        self._mark_timers: list[Any] = []

        self._finalizer_thread: Optional[threading.Thread] = None
        self._startup_check_timer = None
        self._last_error: Optional[str] = None

        self.printer.register_event_handler("klippy:connect", self._on_connect)
        self.printer.register_event_handler("klippy:ready", self._on_ready)

        self.gcode.register_command(
            "RUBIDIUM_VIDEO_MARK",
            self.cmd_RUBIDIUM_VIDEO_MARK,
            desc="Internal: add a motion-synchronized video mark",
        )
        self.gcode.register_command(
            "RUBIDIUM_VIDEO_STATUS",
            self.cmd_RUBIDIUM_VIDEO_STATUS,
            desc="Show rubidium video capture status",
        )
        self.gcode.register_command(
            "RUBIDIUM_VIDEO_TEST",
            self.cmd_RUBIDIUM_VIDEO_TEST,
            desc="Test whether the configured video source can be opened",
        )

    # -----------------------------------------------------------------
    def _on_connect(self) -> None:
        self.toolhead = self.printer.lookup_object("toolhead")

    def _on_ready(self) -> None:
        if not self.enabled:
            return
        if self._startup_check_timer is not None:
            try:
                self.reactor.unregister_timer(self._startup_check_timer)
            except Exception:
                pass
            self._startup_check_timer = None

        def _timer(eventtime):
            self._startup_check_timer = None
            self._start_async_test(reason="startup")
            return self.reactor.NEVER

        self._startup_check_timer = self.reactor.register_timer(_timer)
        self.reactor.update_timer(
            self._startup_check_timer, self.reactor.monotonic() + 5.0
        )

    # -----------------------------------------------------------------
    def _is_url(self) -> bool:
        return self.source.startswith(("http://", "https://"))

    def _resolve_input_kind(self) -> str:
        if self.input_kind in ("url", "v4l2"):
            return self.input_kind
        return "url" if self._is_url() else "v4l2"

    def _expand(self, s: str, *, out: str, session: str = "") -> list[str]:
        repl = {
            "{src}": self.source,
            "{out}": out,
            "{fps}": str(self.framerate),
            "{res}": self.resolution,
            "{resx}": self.resolution.replace(":", "x"),
            "{session}": session,
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return shlex.split(s) if s else []

    def _build_session_cmd(self, session_path: Path) -> list[str]:
        kind = self._resolve_input_kind()
        in_args = self.ffmpeg_input_args_url if kind == "url" else self.ffmpeg_input_args_v4l2

        cmd = [self.ffmpeg, "-y"]
        cmd += self._expand(self.ffmpeg_common_args, out=str(session_path))
        cmd += self._expand(in_args, out=str(session_path))
        if self.video_args:
            cmd += shlex.split(self.video_args)
        cmd += self._expand(self.ffmpeg_session_output_args, out=str(session_path))
        if cmd[-1] != str(session_path):
            cmd.append(str(session_path))
        return cmd

    # -----------------------------------------------------------------

    def start_session(self, outdir: Path) -> None:
        if not self.enabled:
            return
        if self._session_proc is not None:
            return

        outdir.mkdir(parents=True, exist_ok=True)
        session_path = outdir / self.session_filename

        self._session_path = session_path
        self._marks.clear()
        self._session_t0 = self.reactor.monotonic()

        cmd = self._build_session_cmd(session_path)
        logging.info("rubidium: ffmpeg cmd: %s", " ".join(cmd))
        try:
            # Do not inherit stdin (ffmpeg can block reading from it).
            self._session_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as e:
            self._last_error = f"ffmpeg not found: {e}"
            raise self.gcode.error(self._last_error)
        except Exception as e:
            self._last_error = f"failed to start ffmpeg: {e}"
            raise self.gcode.error(self._last_error)

    def stop_session(self, *, finalize: bool = True) -> None:
        if not self.enabled:
            return

        proc = self._session_proc
        session_path = self._session_path
        self._session_proc = None
        self._session_path = None

        # Never block the reactor here. Finalization runs in a worker thread.
        if proc is None and session_path is None:
            return

        self._cancel_mark_timers()

        if self._finalizer_thread is not None and self._finalizer_thread.is_alive():
            logging.warning("rubidium: previous video finalizer still running")
            return

        self._finalizer_thread = threading.Thread(
            target=self._finalize_worker,
            name="rubidium_video_finalize",
            args=(proc, session_path, bool(finalize)),
            daemon=True,
        )
        self._finalizer_thread.start()

    # -----------------------------------------------------------------

    def _write_marks(self, outdir: Path) -> None:
        try:
            payload = {
                "t0_monotonic": float(self._session_t0),
                "marks": [
                    {"key": m.key, "kind": m.kind, "t_s": float(m.t_s), "meta": dict(m.meta)}
                    for m in self._marks
                ],
            }
            (outdir / "rubidium_video_marks.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True)
            )
        except Exception:
            logging.exception("rubidium: failed to write rubidium_video_marks.json")

    def _record_mark_now(self, *, key: str, kind: str, meta: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._session_proc is None:
            return
        t = self.reactor.monotonic() - self._session_t0
        self._marks.append(VideoMark(key=key, kind=kind, t_s=float(t), meta=meta))

    # -----------------------------------------------------------------
    # gcode hook

    def cmd_RUBIDIUM_VIDEO_MARK(self, gcmd) -> None:
        if not self.enabled:
            return
        if self._session_proc is None:
            return

        key = gcmd.get("KEY", "")
        kind = gcmd.get("KIND", "")
        if not key or not kind:
            raise gcmd.error("RUBIDIUM_VIDEO_MARK requires KEY= and KIND=")
        if kind not in ("start", "end"):
            raise gcmd.error("RUBIDIUM_VIDEO_MARK KIND must be start or end")

        meta: dict[str, Any] = {
            "pa": gcmd.get_float("PA", 0.0),
            "idx": gcmd.get_int("IDX", -1),
        }

        toolhead, reactor = self.toolhead, self.reactor

        def _kin_flush_delay() -> float:
            mq = self.printer.lookup_object("motion_queuing", None)
            if mq and hasattr(mq, "get_kin_flush_delay"):
                return float(mq.get_kin_flush_delay())
            return float(getattr(toolhead, "kin_flush_delay", 0.0))

        def _timer_handler(eventtime):
            try:
                self._record_mark_now(key=key, kind=kind, meta=meta)
            except Exception:
                logging.exception("rubidium: mark callback failed")
            return reactor.NEVER

        def _lookahead_cb(print_time):
            now = reactor.monotonic()
            mcu_now = float(toolhead.mcu.estimated_print_time(now))  # type: ignore
            fire_at = now + max(0.0, float(print_time) + _kin_flush_delay() - mcu_now)
            tmr = reactor.register_timer(_timer_handler)
            self._mark_timers.append(tmr)
            reactor.update_timer(tmr, fire_at)

        toolhead.register_lookahead_callback(_lookahead_cb)  # type: ignore

    # -----------------------------------------------------------------
    # status / test

    def cmd_RUBIDIUM_VIDEO_STATUS(self, gcmd) -> None:
        if not self.enabled:
            gcmd.respond_info("rubidium video: disabled")
            return
        kind = self._resolve_input_kind()
        running = self._session_proc is not None
        msg = (
            f"rubidium video: enabled kind={kind} running={int(running)} "
            f"fps={self.framerate} res='{self.resolution}' src='{self.source}'"
        )
        if self._last_error:
            msg += f" last_error='{self._last_error}'"
        gcmd.respond_info(msg)

    def cmd_RUBIDIUM_VIDEO_TEST(self, gcmd) -> None:
        if not self.enabled:
            gcmd.respond_info("rubidium video: disabled")
            return
        self._start_async_test(reason="manual")
        gcmd.respond_info("rubidium video: test started (check console/log for result)")

    # -----------------------------------------------------------------
    # internals

    def _cancel_mark_timers(self) -> None:
        if not self._mark_timers:
            return
        for tmr in self._mark_timers:
            try:
                self.reactor.unregister_timer(tmr)
            except Exception:
                pass
        self._mark_timers.clear()

    def _finalize_worker(
        self,
        proc: Optional[subprocess.Popen],
        session_path: Optional[Path],
        finalize: bool,
    ) -> None:
        err_tail: Optional[str] = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

            if self.stop_delay_s:
                time.sleep(self.stop_delay_s)

            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass

            try:
                if proc.stderr is not None:
                    tail = proc.stderr.read()[-2000:]
                    err_tail = tail.strip() if tail else None
            except Exception:
                pass

        if session_path is not None:
            self._write_marks(session_path.parent)

        if finalize and session_path is not None:
            try:
                self._cut_clips(session_path)
            except Exception:
                logging.exception("rubidium: failed to cut clips")

        def _notify(_eventtime):
            parts = ["rubidium video: session stopped"]
            if session_path is not None:
                parts.append(f"session='{session_path.name}'")
                parts.append(f"marks={len(self._marks)}")
            if err_tail:
                parts.append("ffmpeg_stderr_tail=... (see klippy.log)")
                logging.info("rubidium: ffmpeg stderr tail: %s", err_tail)
            self.gcode.respond_info(" ".join(parts))
            return self.reactor.NEVER

        try:
            self.reactor.register_timer(_notify, self.reactor.NOW)
        except Exception:
            pass

    def _start_async_test(self, *, reason: str) -> None:
        def _worker():
            ok, msg = self._do_test_ffmpeg()
            self._last_error = None if ok else msg

            def _notify(_eventtime):
                tag = "OK" if ok else "FAIL"
                self.gcode.respond_info(f"rubidium video test ({reason}): {tag}: {msg}")
                return self.reactor.NEVER

            try:
                self.reactor.register_timer(_notify, self.reactor.NOW)
            except Exception:
                pass

        t = threading.Thread(target=_worker, name="rubidium_video_test", daemon=True)
        t.start()

    def _do_test_ffmpeg(self) -> tuple[bool, str]:
        kind = self._resolve_input_kind()
        if not self.source:
            return (False, "missing video_source")

        if kind == "url":
            cmd = [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-an",
                "-i",
                self.source,
                "-t",
                "0.2",
                "-f",
                "null",
                "-",
            ]
        else:
            if not self.resolution:
                return (False, "missing video_resolution for v4l2")
            cmd = [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "v4l2",
                "-framerate",
                str(self.framerate),
                "-video_size",
                self.resolution.replace(":", "x"),
                "-i",
                self.source,
                "-t",
                "0.2",
                "-f",
                "null",
                "-",
            ]

        try:
            cp = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except FileNotFoundError as e:
            return (False, f"ffmpeg not found: {e}")
        except subprocess.TimeoutExpired:
            return (False, "timeout while probing source")
        except Exception as e:
            return (False, f"exception while probing source: {e}")

        if cp.returncode == 0:
            return (True, "source opened successfully")

        stderr = (cp.stderr or "").strip()
        if stderr:
            stderr = stderr[-400:]
        return (False, f"ffmpeg exit={cp.returncode} stderr_tail='{stderr}'")

    # -----------------------------------------------------------------
    # clip cutting

    def _cut_clips(self, session_path: Path) -> None:
        if not self._marks:
            return

        by_key: dict[str, dict[str, VideoMark]] = {}
        for m in self._marks:
            d = by_key.setdefault(m.key, {})
            d[m.kind] = m

        outdir = session_path.parent
        for key, md in sorted(by_key.items()):
            ms = md.get("start")
            me = md.get("end")
            if ms is None or me is None:
                continue

            ss = float(ms.t_s)
            to = float(me.t_s)
            if to <= ss:
                continue

            idx = ms.meta.get("idx")
            pa = ms.meta.get("pa")
            if idx is not None and pa is not None and int(idx) >= 0:
                out = outdir / f"rubedo_line_{int(idx):03d}_pa_{float(pa):.5f}.mp4"
            else:
                out = outdir / f"{key}.mp4"

            cmd = [self.ffmpeg, "-y"]
            cmd += self._expand(self.ffmpeg_common_args, out=str(out), session=str(session_path))
            cmd += self._expand(self.ffmpeg_cut_args, out=str(out), session=str(session_path))
            cmd = [str(x).replace("{ss}", f"{ss:.3f}").replace("{to}", f"{to:.3f}") for x in cmd]

            logging.info("rubidium: cutting %s -> %s (%.3f..%.3f)", key, out.name, ss, to)
            subprocess.check_call(cmd)

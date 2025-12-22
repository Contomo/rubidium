# rubidium/modes/latency_mode.py
from __future__ import annotations

import logging
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..core.configview import ConfigView


@dataclass(slots=True)
class LatencyResult:
    anchor_print_time: float
    detected_print_time: float
    delay_ms_mcu: float

    arm_eventtime: float
    detected_eventtime: float
    delay_ms_host: float

    metric: float
    threshold: float
    baseline_mean: float
    baseline_std: float


def _pick_base_section(config: Any, base_name: str) -> Any:
    for section in config.get_prefix_sections(base_name):
        suffix = section.get_name()[len(base_name) :].strip()
        if suffix == "":
            return section
    raise config.error("a base [rubidium] section is required")


def _is_url(src: str) -> bool:
    s = (src or "").strip().lower()
    return "://" in s or s.startswith("http") or s.startswith("rtsp") or s.startswith("rtsps")


def _mean_std(vals: list[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    v = sum((x - m) * (x - m) for x in vals) / len(vals)
    return m, v**0.5


class _MotionDetector:
    """
    ffmpeg -> gray raw frames -> mean abs diff metric.
    Collect baseline first, then arm, then trigger on metric >= threshold.
    """

    def __init__(
        self,
        *,
        reactor,
        ffmpeg_cmd: list[str],
        w: int,
        h: int,
        warmup_frames: int,
        baseline_metrics: int,
        threshold_sigma: float,
        threshold_floor: float,
        consecutive: int,
    ) -> None:
        self.reactor = reactor
        self.ffmpeg_cmd = ffmpeg_cmd
        self.w = int(w)
        self.h = int(h)

        self.warmup_frames = max(0, int(warmup_frames))
        self.baseline_metrics = max(8, int(baseline_metrics))
        self.threshold_sigma = float(threshold_sigma)
        self.threshold_floor = float(threshold_floor)
        self.consecutive = max(1, int(consecutive))

        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        self._ready_frames = threading.Event()
        self._ready_baseline = threading.Event()

        self._armed_eventtime: Optional[float] = None
        self._completion = reactor.completion()

        self._baseline: list[float] = []
        self._baseline_mean = 0.0
        self._baseline_std = 0.0
        self._threshold = float(self.threshold_floor)

        self._last_metric = 0.0

        self._done = False  # prevent double-complete

    @property
    def baseline_mean(self) -> float:
        return self._baseline_mean

    @property
    def baseline_std(self) -> float:
        return self._baseline_std

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def last_metric(self) -> float:
        return self._last_metric

    def start(self) -> None:
        if self._thread is not None:
            return
        self._proc = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        t = threading.Thread(target=self._run, name="rubidium_latency_det", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()
        p = self._proc
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass

    def arm(self, eventtime: float) -> None:
        self._armed_eventtime = float(eventtime)

    def wait_frames_ready(self, timeout_s: float) -> bool:
        return bool(self._ready_frames.wait(timeout=max(0.0, float(timeout_s))))

    def wait_baseline_ready(self, timeout_s: float) -> bool:
        return bool(self._ready_baseline.wait(timeout=max(0.0, float(timeout_s))))

    def wait_trigger(self, timeout_s: float) -> Optional[Tuple[float, float]]:
        return self._completion.wait(
            self.reactor.monotonic() + max(0.0, float(timeout_s)),
            waketime_result=None,
        )

    def _complete(self, value) -> None:
        if self._done:
            return
        self._done = True
        self.reactor.async_complete(self._completion, value)

    def _read_exact(self, n: int) -> Optional[bytes]:
        p = self._proc
        if p is None or p.stdout is None:
            return None
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = p.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _metric(self, a: memoryview, b: memoryview) -> float:
        s = 0
        for x, y in zip(a, b):
            d = x - y
            s += -d if d < 0 else d
        return float(s) / float(len(a) or 1)

    def _compute_threshold(self) -> None:
        m, sd = _mean_std(self._baseline)
        self._baseline_mean = m
        self._baseline_std = sd
        self._threshold = max(self.threshold_floor, m + self.threshold_sigma * sd)

    def _run(self) -> None:
        try:
            frame_n = self.w * self.h
            prev: Optional[memoryview] = None
            warm = self.warmup_frames
            above = 0

            while not self._stop.is_set():
                raw = self._read_exact(frame_n)
                if raw is None:
                    break

                if not self._ready_frames.is_set():
                    self._ready_frames.set()

                cur = memoryview(raw)
                if prev is None:
                    prev = cur
                    continue

                if warm > 0:
                    warm -= 1
                    prev = cur
                    continue

                metric = self._metric(prev, cur)
                self._last_metric = float(metric)
                prev = cur

                # baseline phase
                if not self._ready_baseline.is_set():
                    self._baseline.append(metric)
                    if len(self._baseline) >= self.baseline_metrics:
                        self._compute_threshold()
                        self._ready_baseline.set()
                    continue

                # armed phase
                arm_t = self._armed_eventtime
                if arm_t is None:
                    continue

                now = float(self.reactor.monotonic())
                if now < arm_t:
                    continue

                if metric >= self._threshold:
                    above += 1
                else:
                    above = 0

                if above >= self.consecutive:
                    self._complete((now, float(metric)))
                    return

        except Exception:
            logging.exception("rubidium latency: detector crashed")
        finally:
            try:
                p = self._proc
                if p is not None:
                    p.kill()
            except Exception:
                pass
            self._complete(None)


class RubidiumLatency:
    cmd_help = "Measure camera-to-motion latency (lookahead-timed arm + frame diff trigger)"

    def __init__(self, config: Any) -> None:
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.gcode = self.printer.lookup_object("gcode")

        base_name = config.get_name().split(None, 1)[0]
        base = _pick_base_section(config, base_name)
        self.cv = ConfigView(base, config)

        self.ffmpeg = self.cv.get_str("ffmpeg", "ffmpeg")
        self.video_source = self.cv.get_str_opt("video_source") or ""
        if not self.video_source:
            raise config.error("[rubidium latency] requires video_source")

        self.video_input_kind = (self.cv.get_str_opt("video_input_kind") or "auto").strip().lower()

        self.video_resolution = (self.cv.get_str_opt("video_resolution") or "").strip()
        self.video_framerate = self.cv.get_int("video_framerate", 25, minval=1)

        self.ffmpeg_common_args = self.cv.get_str("ffmpeg_common_args", "-hide_banner -loglevel error")
        self.ffmpeg_input_args_url = self.cv.get_str("ffmpeg_input_args_url", "-an -i {src}")
        self.ffmpeg_input_args_v4l2 = self.cv.get_str(
            "ffmpeg_input_args_v4l2",
            "-f v4l2 -framerate {fps} -video_size {resx} -i {src}",
        )

        self.detect_w = self.cv.get_int("latency_detect_width", 160, minval=32)
        self.detect_h = self.cv.get_int("latency_detect_height", 120, minval=32)

        self.warmup_frames = self.cv.get_int("latency_warmup_frames", 4, minval=0)
        self.baseline_metrics = self.cv.get_int("latency_baseline_metrics", 30, minval=8)

        self.threshold_sigma = self.cv.get_float("latency_threshold_sigma", 10.0, minval=1.0)
        self.threshold_floor = self.cv.get_float("latency_threshold_floor", 2.0, minval=0.0)
        self.consecutive = self.cv.get_int("latency_consecutive", 2, minval=1)

        self.arm_lead_s = self.cv.get_float("latency_arm_lead_s", 0.02, minval=0.0)
        self.timeout_s = self.cv.get_float("latency_timeout", 5.0, minval=0.5)

        self.last: Optional[LatencyResult] = None

        self.gcode.register_command("RUBIDIUM_LATENCY", self.cmd_RUBIDIUM_LATENCY, desc=self.cmd_help)

    def _kin_flush_delay(self, toolhead) -> float:
        mq = self.printer.lookup_object("motion_queuing", None)
        if mq and hasattr(mq, "get_kin_flush_delay"):
            return float(mq.get_kin_flush_delay())
        return float(getattr(toolhead, "kin_flush_delay", 0.0))

    def _build_ffmpeg_cmd(self) -> list[str]:
        src = self.video_source
        kind = self.video_input_kind
        if kind == "auto":
            kind = "url" if _is_url(src) else "v4l2"

        cmd = [self.ffmpeg]
        cmd += shlex.split(self.ffmpeg_common_args)

        if kind == "url":
            cmd += shlex.split(self.ffmpeg_input_args_url.replace("{src}", src))
        elif kind == "v4l2":
            args = self.ffmpeg_input_args_v4l2
            args = args.replace("{src}", src)
            args = args.replace("{fps}", str(int(self.video_framerate)))
            args = args.replace("{res}", self.video_resolution)
            args = args.replace("{resx}", self.video_resolution.replace(":", "x"))
            cmd += shlex.split(args)
        else:
            raise RuntimeError(f"rubidium latency: invalid video_input_kind '{kind}'")

        # no fps= filter here; it can duplicate frames and create false "no motion" plateaus
        vf = f"scale={self.detect_w}:{self.detect_h},format=gray"
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        return cmd

    def cmd_RUBIDIUM_LATENCY(self, gcmd) -> None:
        toolhead = self.printer.lookup_object("toolhead")
        reactor = self.reactor

        axis = (gcmd.get("AXIS", "X") or "X").strip().upper()
        if axis not in ("X", "Y"):
            raise gcmd.error("RUBIDIUM_LATENCY AXIS must be X or Y")

        dist = gcmd.get_float("DIST", 30.0)
        speed = gcmd.get_float("SPEED", 100.0, above=0.0)
        timeout_s = gcmd.get_float("TIMEOUT", self.timeout_s, above=0.0)
        do_return = bool(gcmd.get_int("RETURN", 1))

        toolhead.wait_moves()

        det = _MotionDetector(
            reactor=reactor,
            ffmpeg_cmd=self._build_ffmpeg_cmd(),
            w=self.detect_w,
            h=self.detect_h,
            warmup_frames=self.warmup_frames,
            baseline_metrics=self.baseline_metrics,
            threshold_sigma=self.threshold_sigma,
            threshold_floor=self.threshold_floor,
            consecutive=self.consecutive,
        )
        det.start()

        if not det.wait_frames_ready(1.0):
            det.stop()
            raise gcmd.error("rubidium latency: ffmpeg produced no frames")

        if not det.wait_baseline_ready(3.0):
            det.stop()
            raise gcmd.error("rubidium latency: baseline never stabilized")

        # bounds + target
        start_pos = list(toolhead.get_position())
        idx = 0 if axis == "X" else 1

        kin = toolhead.get_kinematics()
        ks = kin.get_status(eventtime=None)
        mins = ks.get("axis_minimum")
        maxs = ks.get("axis_maximum")
        if isinstance(mins, dict):
            lo = float(mins[axis.lower()])
            hi = float(maxs[axis.lower()])
        else:
            lo = float(mins[idx])
            hi = float(maxs[idx])

        target = start_pos[:]
        proposed = float(target[idx]) + float(dist)
        if proposed < lo or proposed > hi:
            dist = -float(dist)
            proposed = float(target[idx]) + float(dist)
        if proposed < lo or proposed > hi:
            det.stop()
            raise gcmd.error(f"rubidium latency: move would exceed {axis} limits")

        target[idx] = proposed

        # Arm timing using lookahead callback (same scheme as VideoInput markers)
        arm_comp = reactor.completion()
        flush = self._kin_flush_delay(toolhead)

        def _lookahead_cb(print_time):
            now = float(reactor.monotonic())
            mcu_now = float(toolhead.mcu.estimated_print_time(now))  # type: ignore

            anchor_print_time = float(print_time) + flush
            arm_eventtime = now + max(0.0, anchor_print_time - mcu_now) - self.arm_lead_s

            reactor.async_complete(arm_comp, (anchor_print_time, arm_eventtime))

        toolhead.register_lookahead_callback(_lookahead_cb)  # type: ignore

        # enqueue move
        toolhead.manual_move([target[0], target[1], target[2]], speed)

        arm_info = arm_comp.wait(reactor.monotonic() + 1.0, waketime_result=None)
        if arm_info is None:
            det.stop()
            toolhead.wait_moves()
            raise gcmd.error("rubidium latency: failed to obtain lookahead timing")

        anchor_print_time, arm_eventtime = arm_info

        # if TIMEOUT is too small to even reach the arming point, fail immediately (this matches your logs)
        now2 = float(reactor.monotonic())
        if arm_eventtime > now2 + timeout_s:
            det.stop()
            toolhead.wait_moves()
            raise gcmd.error(
                "rubidium latency: TIMEOUT too small "
                f"(arm_in={(arm_eventtime - now2)*1000.0:.1f}ms, timeout={timeout_s*1000.0:.1f}ms). "
                "Increase TIMEOUT."
            )

        det.arm(arm_eventtime)

        hit = det.wait_trigger(timeout_s)
        det.stop()

        toolhead.wait_moves()
        if do_return:
            toolhead.manual_move([start_pos[0], start_pos[1], start_pos[2]], speed)
            toolhead.wait_moves()

        if hit is None:
            raise gcmd.error(
                "rubidium latency: timeout waiting for motion "
                f"(last_metric={det.last_metric:.2f}, thr={det.threshold:.2f}, "
                f"baseline={det.baseline_mean:.2f}±{det.baseline_std:.2f})"
            )

        detected_eventtime, metric = hit
        detected_print_time = float(toolhead.mcu.estimated_print_time(detected_eventtime))  # type: ignore

        delay_ms_mcu = (detected_print_time - anchor_print_time) * 1000.0
        delay_ms_host = (float(detected_eventtime) - arm_eventtime) * 1000.0

        self.last = LatencyResult(
            anchor_print_time=float(anchor_print_time),
            detected_print_time=float(detected_print_time),
            delay_ms_mcu=float(delay_ms_mcu),
            arm_eventtime=float(arm_eventtime),
            detected_eventtime=float(detected_eventtime),
            delay_ms_host=float(delay_ms_host),
            metric=float(metric),
            threshold=float(det.threshold),
            baseline_mean=float(det.baseline_mean),
            baseline_std=float(det.baseline_std),
        )

        gcmd.respond_info(
            "rubidium latency: "
            f"{delay_ms_mcu:.1f}ms (mcu) / {delay_ms_host:.1f}ms (host), "
            f"metric={metric:.2f} thr={det.threshold:.2f} base={det.baseline_mean:.2f}±{det.baseline_std:.2f}"
        )

    def get_status(self, eventtime=None):
        if self.last is None:
            return {"last": None}
        r = self.last
        return {
            "last": {
                "delay_ms_mcu": r.delay_ms_mcu,
                "delay_ms_host": r.delay_ms_host,
                "metric": r.metric,
                "threshold": r.threshold,
                "baseline_mean": r.baseline_mean,
                "baseline_std": r.baseline_std,
            }
        }

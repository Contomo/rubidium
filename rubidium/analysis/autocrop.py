# rubidium/analysis/autocrop.py
from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .clips import ClipInfo
from .image_processing import CropConfig, crop_frame, detect_laser_joint
from .types import AnalysisConfig


# ----------------------------------------------------------------------------
# Policy knobs
# ----------------------------------------------------------------------------

_SAMPLE_CAP = 80

_TRACK_ALPHA = 0.15
_GATE_FRAC = 0.10
_MAX_JUMP_FRAC = 0.50

# How many full-frame measurements we require before we "lock" and
# switch to crop-window tracking.
_LOCK_MIN_SAMPLES = 6

# True: moves original files to outdir/backup/.
# False: deletes originals.
_BACKUP_ORIGINALS = True


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoCropSample:
    clip_idx: int
    frame_idx: int
    global_frame: int
    x_raw: float
    y_raw: float
    x_center_raw: float
    y_center_raw: float
    score: float


@dataclass(frozen=True, slots=True)
class AutoCropResult:
    status: str
    applied: bool
    center_xy: Tuple[float, float]
    frame_wh: Tuple[int, int]
    center_space: str
    n_samples: int
    n_valid: int
    rejects: Dict[str, int]


@dataclass(frozen=True, slots=True)
class _ClipMeta:
    clip: ClipInfo
    frame_wh: Tuple[int, int]
    frame_count: int
    fps: float
    global_start: int


@dataclass(slots=True)
class _CenterTracker:
    """EMA center tracker in *raw* (unmirrored) coordinates."""

    alpha: float
    gate_px: float
    max_jump_px: float
    center_xy: Tuple[float, float]
    locked: bool = False

    def update(self, meas_xy: Tuple[float, float], *, force: bool = False) -> bool:
        mx, my = meas_xy
        cx, cy = self.center_xy
        if not (math.isfinite(mx) and math.isfinite(my)):
            return False

        # Acquisition: first accepted measurement locks the tracker.
        # (We normally seed from a robust median; see _collect_samples.)
        if not self.locked:
            self.center_xy = (float(mx), float(my))
            self.locked = True
            return True

        dx0 = mx - cx
        dy0 = my - cy
        if not force and (abs(dx0) > self.gate_px or abs(dy0) > self.gate_px):
            return False

        dx, dy = _clamp_delta(dx0, dy0, self.max_jump_px)
        a = self.alpha
        self.center_xy = (
            (1.0 - a) * cx + a * (cx + dx),
            (1.0 - a) * cy + a * (cy + dy),
        )
        return True


@dataclass(frozen=True, slots=True)
class _CenterCurve:
    """A continuous (piecewise-linear) center curve in raw coordinates."""

    t: np.ndarray  # int frame index (global)
    x: np.ndarray
    y: np.ndarray

    def predict(self, t: int) -> Tuple[float, float]:
        if self.t.size == 0:
            return 0.0, 0.0
        ti = float(t)
        x = float(np.interp(ti, self.t, self.x))
        y = float(np.interp(ti, self.t, self.y))
        return x, y

    def evaluate_dense(self, *, step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.t.size == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        t0 = int(self.t[0])
        t1 = int(self.t[-1])
        step = max(1, int(step))
        td = np.arange(t0, t1 + 1, step, dtype=np.int32)
        xd = np.interp(td.astype(np.float64), self.t, self.x).astype(np.float32)
        yd = np.interp(td.astype(np.float64), self.t, self.y).astype(np.float32)
        return td, xd, yd


# ----------------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------------


class AutoCropper:
    def __init__(self, cfg: AnalysisConfig, *, outdir: Path) -> None:
        self.cfg = cfg
        self.outdir = outdir
        self.backup_dir = outdir / "backup"

    def run(self, clips: Iterable[ClipInfo]) -> Optional[AutoCropResult]:
        if not self.cfg.autocrop.enable:
            return None

        clips_list = sorted(list(clips), key=lambda c: c.idx)
        if not clips_list:
            return self._write_result("no_clips", applied=False, rejects={})

        out_w, out_h = (int(self.cfg.crop.wh[0]), int(self.cfg.crop.wh[1]))
        if out_w <= 0 or out_h <= 0:
            return self._write_result(
                "crop_wh_required",
                applied=False,
                rejects={"crop_wh_required": 1},
                frame_wh=(0, 0),
                center_xy=(0.0, 0.0),
            )

        rejects: Dict[str, int] = {}
        metas = self._build_timeline(clips_list, rejects)
        if not metas:
            return self._write_result("open_failed", applied=False, rejects=rejects)

        src_wh = metas[0].frame_wh
        # Acquisition always starts on the full frame. We do not bias towards any
        # previously configured crop center.
        src_w, src_h = int(src_wh[0]), int(src_wh[1])
        base_center = (src_w / 2.0, src_h / 2.0)

        base_px = float(min(out_w, out_h))
        tracker = _CenterTracker(
            alpha=_TRACK_ALPHA,
            gate_px=_GATE_FRAC * base_px,
            max_jump_px=_MAX_JUMP_FRAC * base_px,
            center_xy=base_center,
            locked=False,
        )

        samples = self._collect_samples(metas, tracker, (out_w, out_h), rejects)
        if not samples:
            return self._write_result("no_samples", applied=False, rejects=rejects, frame_wh=(out_w, out_h))

        curve = _build_center_curve(samples)
        if curve is None:
            return self._write_result("no_valid_curve", applied=False, rejects=rejects, frame_wh=(out_w, out_h))

        self._write_points(samples)
        self._generate_debug_plot(samples, curve, metas)

        if _BACKUP_ORIGINALS:
            self.backup_dir.mkdir(exist_ok=True, parents=True)

        rendered = 0
        for i, meta in enumerate(metas):
            if self._render_stabilized_video(meta, curve, (out_w, out_h), i + 1, len(metas)):
                self._handle_file_swap(meta.clip.path)
                rendered += 1
            else:
                rejects["render_fail"] = rejects.get("render_fail", 0) + 1

        if rendered == 0:
            return self._write_result("render_failed", applied=False, rejects=rejects, frame_wh=(out_w, out_h))

        # After render: the clips are already cropped to out_wh.
        return self._write_result(
            "ok",
            applied=True,
            center_xy=(out_w / 2.0, out_h / 2.0),
            frame_wh=(out_w, out_h),
            center_space="output",
            n_samples=len(samples),
            n_valid=len(samples),
            rejects=rejects,
        )

    # ------------------------------------------------------------------------
    # Timeline / metadata
    # ------------------------------------------------------------------------

    def _build_timeline(self, clips: List[ClipInfo], rejects: Dict[str, int]) -> List[_ClipMeta]:
        metas: List[_ClipMeta] = []
        global_start = 0
        frame_wh: Optional[Tuple[int, int]] = None

        for clip in clips:
            cap = cv2.VideoCapture(str(clip.path))
            try:
                if not cap.isOpened():
                    rejects["open_failed"] = rejects.get("open_failed", 0) + 1
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            finally:
                cap.release()

            if w <= 0 or h <= 0 or n <= 0:
                rejects["bad_meta"] = rejects.get("bad_meta", 0) + 1
                continue

            if frame_wh is None:
                frame_wh = (w, h)
            elif frame_wh != (w, h):
                rejects["mixed_frame_wh"] = rejects.get("mixed_frame_wh", 0) + 1
                continue

            metas.append(
                _ClipMeta(
                    clip=clip,
                    frame_wh=(w, h),
                    frame_count=n,
                    fps=fps,
                    global_start=global_start,
                )
            )
            global_start += n

        return metas

    # ------------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------------

    def _collect_samples(
        self,
        metas: List[_ClipMeta],
        tracker: _CenterTracker,
        out_wh: Tuple[int, int],
        rejects: Dict[str, int],
    ) -> List[AutoCropSample]:
        out: List[AutoCropSample] = []
        lock_buf: List[AutoCropSample] = []
        out_w, out_h = out_wh
        lc = self.cfg.laser

        for meta in metas:
            cap = cv2.VideoCapture(str(meta.clip.path))
            try:
                if not cap.isOpened():
                    rejects["open_failed"] = rejects.get("open_failed", 0) + 1
                    continue

                for fi in _sample_frame_indices(meta.frame_count, cap=_SAMPLE_CAP):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        rejects["read_failed"] = rejects.get("read_failed", 0) + 1
                        continue

                    mirror_x = bool(meta.clip.mirror_x)
                    frame_proc = frame_bgr[:, ::-1] if mirror_x else frame_bgr

                    # Phase 1: full-frame acquisition. We collect a few robust
                    # measurements before switching to crop-window tracking.
                    if not tracker.locked:
                        joint_xy, score, _mask = detect_laser_joint(frame_proc, lc)
                        if joint_xy is None:
                            rejects["no_joint"] = rejects.get("no_joint", 0) + 1
                            continue

                        jx, jy = float(joint_xy[0]), float(joint_xy[1])
                        if not (math.isfinite(jx) and math.isfinite(jy) and math.isfinite(float(score))):
                            rejects["nan"] = rejects.get("nan", 0) + 1
                            continue

                        mx_proc, my_proc = jx, jy
                        mx_raw = _mirror_x_coord(mx_proc, meta.frame_wh[0]) if mirror_x else mx_proc
                        my_raw = my_proc

                        lock_buf.append(
                            AutoCropSample(
                                clip_idx=meta.clip.idx,
                                frame_idx=fi,
                                global_frame=meta.global_start + fi,
                                x_raw=mx_raw,
                                y_raw=my_raw,
                                x_center_raw=mx_raw,
                                y_center_raw=my_raw,
                                score=float(score),
                            )
                        )

                        if len(lock_buf) >= _LOCK_MIN_SAMPLES:
                            tracker.center_xy = _robust_median_xy(lock_buf)
                            tracker.locked = True
                            for s in lock_buf:
                                if tracker.update((s.x_raw, s.y_raw), force=True):
                                    cx, cy = tracker.center_xy
                                    out.append(
                                        AutoCropSample(
                                            clip_idx=s.clip_idx,
                                            frame_idx=s.frame_idx,
                                            global_frame=s.global_frame,
                                            x_raw=s.x_raw,
                                            y_raw=s.y_raw,
                                            x_center_raw=float(cx),
                                            y_center_raw=float(cy),
                                            score=s.score,
                                        )
                                    )
                            lock_buf.clear()

                        continue

                    # Phase 2: crop-window tracking, centered on the current center.
                    cx_raw, cy_raw = tracker.center_xy
                    cx_raw, cy_raw = _clamp_center_to_crop_bounds((cx_raw, cy_raw), meta.frame_wh, out_wh)
                    cx_proc = _mirror_x_coord(cx_raw, meta.frame_wh[0]) if mirror_x else cx_raw
                    search_crop, (offx, offy) = crop_frame(
                        frame_proc,
                        CropConfig(center_xy=(cx_proc, cy_raw), wh=(out_w, out_h), ref_wh=None),
                    )

                    joint_xy, score, _mask = detect_laser_joint(search_crop, lc)
                    if joint_xy is None:
                        rejects["no_joint"] = rejects.get("no_joint", 0) + 1
                        continue

                    jx, jy = float(joint_xy[0]), float(joint_xy[1])
                    if not (math.isfinite(jx) and math.isfinite(jy) and math.isfinite(float(score))):
                        rejects["nan"] = rejects.get("nan", 0) + 1
                        continue

                    mx_proc = float(offx) + jx
                    my_proc = float(offy) + jy

                    mx_raw = _mirror_x_coord(mx_proc, meta.frame_wh[0]) if mirror_x else mx_proc
                    my_raw = my_proc

                    if not tracker.update((mx_raw, my_raw)):
                        rejects["tracker_gate"] = rejects.get("tracker_gate", 0) + 1
                        continue

                    cx, cy = tracker.center_xy

                    out.append(
                        AutoCropSample(
                            clip_idx=meta.clip.idx,
                            frame_idx=fi,
                            global_frame=meta.global_start + fi,
                            x_raw=mx_raw,
                            y_raw=my_raw,
                            x_center_raw=float(cx),
                            y_center_raw=float(cy),
                            score=float(score),
                        )
                    )
            finally:
                cap.release()

        # If we never reached the lock threshold, lock with what we have.
        if lock_buf and not tracker.locked:
            tracker.center_xy = _robust_median_xy(lock_buf)
            tracker.locked = True
            for s in lock_buf:
                if tracker.update((s.x_raw, s.y_raw), force=True):
                    cx, cy = tracker.center_xy
                    out.append(
                        AutoCropSample(
                            clip_idx=s.clip_idx,
                            frame_idx=s.frame_idx,
                            global_frame=s.global_frame,
                            x_raw=s.x_raw,
                            y_raw=s.y_raw,
                            x_center_raw=float(cx),
                            y_center_raw=float(cy),
                            score=s.score,
                        )
                    )
            lock_buf.clear()

        out.sort(key=lambda s: s.global_frame)
        return out

    # ------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------

    def _render_stabilized_video(
        self,
        meta: _ClipMeta,
        curve: _CenterCurve,
        out_wh: Tuple[int, int],
        idx: int,
        total: int,
    ) -> bool:
        out_w, out_h = out_wh

        cap = cv2.VideoCapture(str(meta.clip.path))
        if not cap.isOpened():
            return False

        temp_path = meta.clip.path.with_suffix(".temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_path), fourcc, float(meta.fps or 30.0), (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return False

        mirror_x = bool(meta.clip.mirror_x)
        w_src, h_src = meta.frame_wh

        try:
            fi = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if fi % 60 == 0:
                    print(f"rubidium: autocrop render {idx}/{total} frame={fi}", end="\r")

                frame_proc = frame_bgr[:, ::-1] if mirror_x else frame_bgr
                t_global = meta.global_start + fi
                cx_raw, cy_raw = curve.predict(int(t_global))
                cx_raw, cy_raw = _clamp_center_to_crop_bounds((cx_raw, cy_raw), (w_src, h_src), out_wh)
                cx_proc = _mirror_x_coord(cx_raw, w_src) if mirror_x else cx_raw

                crop, _ = crop_frame(
                    frame_proc,
                    CropConfig(center_xy=(cx_proc, cy_raw), wh=(out_w, out_h), ref_wh=None),
                )
                if crop.shape[0] != out_h or crop.shape[1] != out_w:
                    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

                writer.write(crop)
                fi += 1
        except Exception:
            writer.release()
            cap.release()
            if temp_path.exists():
                temp_path.unlink()
            return False

        writer.release()
        cap.release()
        return True

    def _handle_file_swap(self, original: Path) -> None:
        temp = original.with_suffix(".temp.mp4")
        if not temp.exists():
            return

        if _BACKUP_ORIGINALS:
            backup = self.backup_dir / original.name
            shutil.move(str(original), str(backup))
        else:
            original.unlink(missing_ok=True)

        shutil.move(str(temp), str(original))

    # ------------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------------

    def _write_points(self, samples: List[AutoCropSample]) -> None:
        p = self.outdir / "autocrop_points.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "clip_idx",
                    "frame_idx",
                    "global_frame",
                    "x_raw",
                    "y_raw",
                    "x_center_raw",
                    "y_center_raw",
                    "score",
                ]
            )
            for s in samples:
                w.writerow(
                    [
                        s.clip_idx,
                        s.frame_idx,
                        s.global_frame,
                        s.x_raw,
                        s.y_raw,
                        s.x_center_raw,
                        s.y_center_raw,
                        s.score,
                    ]
                )

    def _generate_debug_plot(self, samples: List[AutoCropSample], curve: _CenterCurve, metas: List[_ClipMeta]) -> None:
        if not samples:
            return

        t = np.asarray([s.global_frame for s in samples], dtype=np.int32)
        x = np.asarray([s.x_raw for s in samples], dtype=np.float32)
        y = np.asarray([s.y_raw for s in samples], dtype=np.float32)

        td, xd, yd = curve.evaluate_dense(step=max(1, int((t[-1] - t[0]) // 2500) or 1))

        fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax_x.scatter(t, x, s=3, alpha=0.5, label="meas")
        ax_y.scatter(t, y, s=3, alpha=0.5, label="meas")
        if td.size:
            ax_x.plot(td, xd, lw=2, label="center")
            ax_y.plot(td, yd, lw=2, label="center")

        # Clip boundaries
        for m in metas[1:]:
            ax_x.axvline(m.global_start, lw=0.8, alpha=0.25)
            ax_y.axvline(m.global_start, lw=0.8, alpha=0.25)

        ax_x.set_ylabel("X (raw px)")
        ax_y.set_ylabel("Y (raw px)")
        ax_y.set_xlabel("global frame")
        ax_x.grid(True)
        ax_y.grid(True)
        ax_x.legend(loc="upper right")
        ax_y.legend(loc="upper right")
        ax_x.set_title("Autocrop center trajectory (continuous across clips)")

        fig.tight_layout()
        fig.savefig(str(self.outdir / "autocrop_trace.png"))
        plt.close(fig)

    def _write_result(
        self,
        status: str,
        *,
        applied: bool,
        rejects: Dict[str, int],
        center_xy: Tuple[float, float] = (0.0, 0.0),
        frame_wh: Tuple[int, int] = (0, 0),
        center_space: str = "output",
        n_samples: int = 0,
        n_valid: int = 0,
    ) -> AutoCropResult:
        res = AutoCropResult(
            status=str(status),
            applied=bool(applied),
            center_xy=(float(center_xy[0]), float(center_xy[1])),
            frame_wh=(int(frame_wh[0]), int(frame_wh[1])),
            center_space=str(center_space),
            n_samples=int(n_samples),
            n_valid=int(n_valid),
            rejects=dict(rejects),
        )
        (self.outdir / "autocrop.json").write_text(
            json.dumps(asdict(res), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return res


# ----------------------------------------------------------------------------
# Curve building (pure)
# ----------------------------------------------------------------------------


def _build_center_curve(samples: List[AutoCropSample]) -> Optional[_CenterCurve]:
    if len(samples) < 8:
        return None

    t = np.asarray([s.global_frame for s in samples], dtype=np.int32)
    # Use the tracker center (already gated + smoothed) for the stabilization
    # curve. Raw measurements are still kept for debug plots.
    x = np.asarray([s.x_center_raw for s in samples], dtype=np.float64)
    y = np.asarray([s.y_center_raw for s in samples], dtype=np.float64)

    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 8:
        return None
    t = t[ok]
    x = x[ok]
    y = y[ok]

    # Robust outlier rejection using MAD in each axis.
    x, keep_x = _mad_filter_1d(x)
    y, keep_y = _mad_filter_1d(y)
    keep = keep_x & keep_y
    if np.count_nonzero(keep) < 8:
        return None
    t = t[keep]
    x = x[keep]
    y = y[keep]

    # Sort by time (global frame)
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    # Time-binned robust smoothing: collapse high-frequency jitter into
    # per-window medians, then apply a gentle EMA.
    t_b, x_b, y_b = _time_bin_median(t, x, y, target_knots=250)
    x_s = _ema(x_b, alpha=0.12)
    y_s = _ema(y_b, alpha=0.12)

    # Collapse duplicate timestamps (rare, but possible if sample indices repeat)
    t_u, x_u, y_u = _collapse_duplicates(t_b, x_s, y_s)
    if t_u.size < 2:
        return None

    return _CenterCurve(t=t_u.astype(np.float64), x=x_u.astype(np.float64), y=y_u.astype(np.float64))


def _time_bin_median(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_knots: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce jitter by taking median points in time bins.

    The bin size is chosen to produce approximately `target_knots` outputs.
    """
    if t.size == 0:
        return t, x, y

    t = np.asarray(t, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    t0 = int(t[0])
    t1 = int(t[-1])
    span = max(1, t1 - t0)
    k = max(2, int(target_knots))
    bin_size = max(1, int(round(span / k)))

    bins = (t - t0) // bin_size
    uniq = np.unique(bins)
    if uniq.size <= 1:
        return t.astype(np.float64), x, y

    t_out = np.empty((uniq.size,), dtype=np.int64)
    x_out = np.empty((uniq.size,), dtype=np.float64)
    y_out = np.empty((uniq.size,), dtype=np.float64)
    for i, b in enumerate(uniq):
        m = bins == b
        t_out[i] = int(round(float(np.median(t[m]))))
        x_out[i] = float(np.median(x[m]))
        y_out[i] = float(np.median(y[m]))

    return t_out.astype(np.float64), x_out, y_out


def _mad_filter_1d(v: np.ndarray, *, z: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    vv = np.asarray(v, dtype=np.float64)
    med = float(np.median(vv))
    mad = float(np.median(np.abs(vv - med)))
    if mad < 1e-9:
        keep = np.isfinite(vv)
        return vv, keep
    sigma = 1.4826 * mad
    keep = np.abs(vv - med) <= (z * sigma)
    return vv, keep


def _ema(v: np.ndarray, *, alpha: float) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64)
    if vv.size == 0:
        return vv
    a = float(alpha)
    a = min(max(a, 0.0), 1.0)
    out = np.empty_like(vv)
    out[0] = vv[0]
    for i in range(1, vv.size):
        out[i] = (1.0 - a) * out[i - 1] + a * vv[i]
    return out


def _collapse_duplicates(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t.size == 0:
        return t, x, y
    t = np.asarray(t)
    x = np.asarray(x)
    y = np.asarray(y)

    uniq_t = [int(t[0])]
    acc_x = [float(x[0])]
    acc_y = [float(y[0])]
    cnt = [1]

    for i in range(1, t.size):
        ti = int(t[i])
        if ti == uniq_t[-1]:
            acc_x[-1] += float(x[i])
            acc_y[-1] += float(y[i])
            cnt[-1] += 1
        else:
            uniq_t.append(ti)
            acc_x.append(float(x[i]))
            acc_y.append(float(y[i]))
            cnt.append(1)

    tx = np.asarray(uniq_t, dtype=np.int32)
    xx = np.asarray([acc_x[i] / cnt[i] for i in range(len(uniq_t))], dtype=np.float64)
    yy = np.asarray([acc_y[i] / cnt[i] for i in range(len(uniq_t))], dtype=np.float64)
    return tx, xx, yy


def _robust_median_xy(samples: List[AutoCropSample]) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    x = np.asarray([s.x_raw for s in samples], dtype=np.float64)
    y = np.asarray([s.y_raw for s in samples], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return 0.0, 0.0
    x = x[ok]
    y = y[ok]
    return float(np.median(x)), float(np.median(y))


# ----------------------------------------------------------------------------
# Misc helpers (pure)
# ----------------------------------------------------------------------------


def _clamp_delta(dx: float, dy: float, max_jump_px: float) -> Tuple[float, float]:
    mj = float(max(0.0, max_jump_px))
    if mj <= 0.0:
        return 0.0, 0.0
    n = math.hypot(dx, dy)
    if n <= mj or n <= 1e-9:
        return float(dx), float(dy)
    s = mj / n
    return float(dx * s), float(dy * s)


def _mirror_x_coord(x: float, w: int) -> float:
    return float(w - 1) - float(x)


def _sample_frame_indices(total_frames: int, *, cap: int) -> List[int]:
    if total_frames <= 1:
        return [0]
    n = min(int(cap), int(total_frames))
    # Avoid the very first and very last frames; they are often garbage.
    return [int(round((total_frames - 1) * ((i + 1) / (n + 1)))) for i in range(n)]


def _clamp_center_to_crop_bounds(
    center_xy: Tuple[float, float],
    frame_wh: Tuple[int, int],
    crop_wh: Tuple[int, int],
) -> Tuple[float, float]:
    w, h = int(frame_wh[0]), int(frame_wh[1])
    cw, ch = int(crop_wh[0]), int(crop_wh[1])
    if w <= 0 or h <= 0 or cw <= 0 or ch <= 0:
        return float(center_xy[0]), float(center_xy[1])

    cw = min(cw, w)
    ch = min(ch, h)
    lo_x = cw / 2.0
    hi_x = (w - cw) + cw / 2.0
    lo_y = ch / 2.0
    hi_y = (h - ch) + ch / 2.0

    cx = float(center_xy[0])
    cy = float(center_xy[1])
    cx = min(max(cx, lo_x), hi_x)
    cy = min(max(cy, lo_y), hi_y)
    return cx, cy


__all__ = [
    "AutoCropResult",
    "AutoCropSample",
    "AutoCropper",
]

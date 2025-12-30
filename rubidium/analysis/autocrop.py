# rubidium/analysis/autocrop.py
from __future__ import annotations

import csv
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Generator

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .clips import ClipInfo
from .image_processing import (
    CropConfig, 
    clamp_center_to_crop_bounds, 
    crop_frame, 
    detect_laser_joint,
    LaserExtractConfig
)
from .types import AnalysisConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_SAMPLE_CAP = 80
_LOCK_MIN_SAMPLES = 8

# Tracker / Collection Settings
_TRACK_ALPHA = 0.50
_GATE_FRAC = 0.25
_MAX_JUMP_FRAC = 0.50

# Curve / Smoothing Settings
# Lower = Smoother. 
# 0.015 is very stiff, ignoring almost all vibration/hysteresis.
_CURVE_ALPHA = 0.015
AUTOCROP_CURVE_MODE: str = 'smoothing' # 'linear' | 'smoothing'

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AutoCropSample:
    clip_idx: int
    frame_idx: int
    global_frame: int
    x_raw: float
    y_raw: float
    score: float

@dataclass(frozen=True, slots=True)
class AutoCropResult:
    status: str
    applied: bool
    frame_wh: Tuple[int, int]
    crop_wh: Tuple[int, int]
    curve_kind: str
    curve_params: Dict[str, float]
    n_samples: int
    n_valid: int
    rejects: Dict[str, int]
    timing_s: Dict[str, float]

@dataclass(frozen=True, slots=True)
class _ClipMeta:
    clip: ClipInfo
    frame_wh: Tuple[int, int]
    frame_count: int
    fps: float
    global_start: int

# -----------------------------------------------------------------------------
# Trajectory Abstraction
# -----------------------------------------------------------------------------

class Trajectory(ABC):
    def __init__(self, t0: int, t1: int):
        self.t0 = t0
        self.t1 = t1

    @abstractmethod
    def predict(self, t: Union[int, float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        pass
    
    def evaluate_dense(self, step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.t1 < self.t0: return np.array([]), np.array([]), np.array([])
        t = np.arange(self.t0, self.t1 + 1, step, dtype=np.float64)
        x, y = self.predict(t)
        return t, x, y

class LinearTrajectory(Trajectory):
    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        t0, t1 = int(np.min(t)), int(np.max(t))
        super().__init__(t0, t1)
        self.mx, self.bx = self._robust_fit(t, x)
        self.my, self.by = self._robust_fit(t, y)

    def predict(self, t: Union[int, float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        t_c = np.clip(t, self.t0, self.t1)
        return (self.mx * t_c + self.bx, self.my * t_c + self.by)

    def get_params(self) -> Dict[str, float]:
        return {"mx": self.mx, "bx": self.bx, "my": self.my, "by": self.by, "t0": self.t0, "t1": self.t1}

    @staticmethod
    def _robust_fit(t: np.ndarray, v: np.ndarray, max_iter: int = 4, z: float = 6.0) -> Tuple[float, float]:
        mask = np.isfinite(t) & np.isfinite(v)
        if np.count_nonzero(mask) < 2: return 0.0, float(np.nanmedian(v)) if np.any(np.isfinite(v)) else 0.0
        tt, vv = t[mask], v[mask]
        for _ in range(max_iter):
            if tt.size < 2: break
            m, b = np.polyfit(tt, vv, 1)
            resid = vv - (m * tt + b)
            mad = np.median(np.abs(resid - np.median(resid)))
            if mad < 1e-9: break
            keep = np.abs(resid - np.median(resid)) <= (z * 1.4826 * mad)
            if np.count_nonzero(keep) == tt.size: break
            tt, vv = tt[keep], vv[keep]
        m, b = np.polyfit(tt, vv, 1)
        return float(m), float(b)

class SmoothEMATrajectory(Trajectory):
    """
    Zero-Phase Forward-Backward EMA with Edge Padding.
    Uses 'edge' padding to settle the filter on the start/end values without slope artifacts.
    """
    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float):
        t0, t1 = int(np.min(t)), int(np.max(t))
        super().__init__(t0, t1)
        self.alpha = alpha
        self._dense_t = np.arange(t0, t1 + 1, dtype=np.int32)
        
        u_t, u_idx = np.unique(t, return_index=True)
        x_int = np.interp(self._dense_t, u_t, x[u_idx])
        y_int = np.interp(self._dense_t, u_t, y[u_idx])
        
        self._dense_x = self._zero_phase_ema(x_int, alpha)
        self._dense_y = self._zero_phase_ema(y_int, alpha)

    def predict(self, t: Union[int, float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        x = np.interp(t, self._dense_t, self._dense_x)
        y = np.interp(t, self._dense_t, self._dense_y)
        if np.isscalar(t): return float(x), float(y)
        return x, y

    def get_params(self) -> Dict[str, float]:
        return {"t0": self.t0, "t1": self.t1, "alpha": self.alpha}

    @staticmethod
    def _ema(d: np.ndarray, alpha: float) -> np.ndarray:
        out = np.empty_like(d)
        curr = d[0]
        for i in range(len(d)):
            curr = curr * (1.0 - alpha) + d[i] * alpha
            out[i] = curr
        return out

    @classmethod
    def _zero_phase_ema(cls, data: np.ndarray, alpha: float) -> np.ndarray:
        # Pad with edge (repeat first/last value) to prevent trend skew at boundaries
        pad_len = int(3.0 / alpha)
        if len(data) < pad_len: pad_len = len(data) - 1
        
        if pad_len > 0:
            d_pad = np.pad(data, pad_len, mode='edge')
        else:
            d_pad = data

        fwd = cls._ema(d_pad, alpha)
        rev = cls._ema(d_pad[::-1], alpha)[::-1]
        out = (fwd + rev) / 2.0
        
        if pad_len > 0:
            return out[pad_len:-pad_len]
        return out

# -----------------------------------------------------------------------------
# Tracker Logic
# -----------------------------------------------------------------------------

class _CropTracker:
    def __init__(self, initial_xy: Tuple[float, float], out_wh: Tuple[int, int], lc: LaserExtractConfig):
        self.xy = initial_xy
        self.locked = False
        self.lock_buffer: List[Tuple[float, float]] = []
        self.gate_px = _GATE_FRAC * float(min(out_wh))
        self.max_jump = _MAX_JUMP_FRAC * float(min(out_wh))
        self.strong_signal_thresh = float(lc.min_profile_energy) * 3.0

    def update(self, detected_xy: Tuple[float, float], score: float) -> Tuple[bool, str]:
        dx, dy = detected_xy[0] - self.xy[0], detected_xy[1] - self.xy[1]
        
        if not self.locked:
            self.lock_buffer.append(detected_xy)
            if len(self.lock_buffer) >= _LOCK_MIN_SAMPLES:
                arr = np.array(self.lock_buffer)
                self.xy = (float(np.median(arr[:,0])), float(np.median(arr[:,1])))
                self.locked = True
            return True, "locking"

        dist = math.hypot(dx, dy)
        is_strong = score > self.strong_signal_thresh
        
        if dist < self.gate_px or is_strong:
            dx_c, dy_c = _clamp_delta(dx, dy, self.max_jump)
            self.xy = (
                self.xy[0] * (1 - _TRACK_ALPHA) + (self.xy[0] + dx_c) * _TRACK_ALPHA,
                self.xy[1] * (1 - _TRACK_ALPHA) + (self.xy[1] + dy_c) * _TRACK_ALPHA
            )
            return True, "tracked"
        
        return False, "gate_rejected"

# -----------------------------------------------------------------------------
# Main Class
# -----------------------------------------------------------------------------

class AutoCropper:
    def __init__(self, cfg: AnalysisConfig, *, outdir: Path) -> None:
        self.cfg = cfg
        self.outdir = outdir

    def run(self, clips: Iterable[ClipInfo]) -> Tuple[Optional[AutoCropResult], Optional[Trajectory], Dict[int, int]]:
        if not self.cfg.autocrop.enable: return None, None, {}

        t_all: Dict[str, float] = {}
        clips_list = sorted(list(clips), key=lambda c: c.idx)
        if not clips_list: return self._fail("no_clips", t_all), None, {}
        
        out_w, out_h = (int(self.cfg.crop.wh[0]), int(self.cfg.crop.wh[1]))
        if out_w <= 0 or out_h <= 0: return self._fail("crop_wh_required", t_all, crop_wh=(out_w, out_h)), None, {}

        # 1. Timeline
        t0 = time.perf_counter()
        rejects: Dict[str, int] = {}
        metas = self._build_timeline(clips_list, rejects)
        t_all["timeline"] = time.perf_counter() - t0
        if not metas: return self._fail("open_failed", t_all, rejects=rejects), None, {}

        # 2. Collection
        t0 = time.perf_counter()
        samples = self._collect_samples(metas, (out_w, out_h), rejects, t_all)
        t_all["samples"] = time.perf_counter() - t0
        if len(samples) < _LOCK_MIN_SAMPLES: 
            return self._fail("insufficient_samples", t_all, rejects=rejects, frame_wh=metas[0].frame_wh, crop_wh=(out_w, out_h)), None, {}

        # 3. Curve Fitting
        t0 = time.perf_counter()
        traj = self._build_trajectory(samples)
        t_all["curve"] = time.perf_counter() - t0
        if not traj: 
            return self._fail("curve_fit_failed", t_all, rejects=rejects, frame_wh=metas[0].frame_wh, crop_wh=(out_w, out_h)), None, {}

        # 4. Output
        t0 = time.perf_counter()
        self._write_csv(samples)
        self._generate_debug_plot(samples, traj, metas)
        t_all["debug"] = time.perf_counter() - t0
        
        res = self._success(
            "ok",
            trajectory=traj,
            n_samples=len(samples),
            rejects=rejects,
            timing_s=t_all,
            frame_wh=metas[0].frame_wh,
            crop_wh=(out_w, out_h)
        )
        return res, traj, {int(m.clip.idx): int(m.global_start) for m in metas}

    def _build_timeline(self, clips: List[ClipInfo], rejects: Dict[str, int]) -> List[_ClipMeta]:
        metas = []
        g_start = 0
        ref_wh = None
        for c in clips:
            cap = cv2.VideoCapture(str(c.path))
            if not cap.isOpened():
                rejects["open_failed"] = rejects.get("open_failed", 0) + 1; continue
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if w <= 0: rejects["bad_meta"] = rejects.get("bad_meta", 0) + 1; continue
            if ref_wh is None: ref_wh = (w, h)
            elif ref_wh != (w, h): rejects["mixed_frame_wh"] = rejects.get("mixed_frame_wh", 0) + 1; continue
            
            metas.append(_ClipMeta(c, (w, h), n, fps, g_start))
            g_start += n
        return metas

    def _collect_samples(self, metas: List[_ClipMeta], out_wh: Tuple[int, int], rejects: Dict[str, int], timing: Dict[str, float]) -> List[AutoCropSample]:
        samples = []
        lc = self.cfg.laser
        
        center_init = (metas[0].frame_wh[0] / 2.0, metas[0].frame_wh[1] / 2.0)
        tracker = _CropTracker(center_init, out_wh, lc)
        
        t_read, t_proc = 0.0, 0.0

        for meta in metas:
            indices = _sample_frame_indices(meta.frame_count, _SAMPLE_CAP)
            
            for frame_bgr, frame_idx in _yield_frames(meta.clip.path, indices, timing):
                ts = time.perf_counter()
                
                mirror = bool(meta.clip.mirror_x)
                f_proc = frame_bgr[:, ::-1] if mirror else frame_bgr
                
                if tracker.locked:
                    cx, cy = clamp_center_to_crop_bounds(tracker.xy, meta.frame_wh, out_wh)
                    cx_proc = _mirror_x_coord(cx, meta.frame_wh[0]) if mirror else cx
                    search_img, off = crop_frame(f_proc, CropConfig(center_xy=(cx_proc, cy), wh=out_wh))
                else:
                    search_img, off = f_proc, (0, 0)

                joint_xy, score, _ = detect_laser_joint(search_img, lc)
                
                if joint_xy:
                    det_x = float(off[0]) + float(joint_xy[0])
                    det_y = float(off[1]) + float(joint_xy[1])
                    raw_x = _mirror_x_coord(det_x, meta.frame_wh[0]) if mirror else det_x
                    raw_y = det_y
                    
                    accepted, reason = tracker.update((raw_x, raw_y), float(score))
                    if accepted:
                        samples.append(AutoCropSample(meta.clip.idx, frame_idx, meta.global_start + frame_idx, raw_x, raw_y, float(score)))
                    else:
                        rejects[reason] = rejects.get(reason, 0) + 1
                else:
                    rejects["no_joint"] = rejects.get("no_joint", 0) + 1
                
                t_proc += time.perf_counter() - ts

        timing['samples_proc'] = float(timing.get('samples_proc', 0)) + t_proc
        samples.sort(key=lambda s: s.global_frame)
        return samples

    def _build_trajectory(self, samples: List[AutoCropSample]) -> Optional[Trajectory]:
        if not samples: return None
        scores = np.array([s.score for s in samples])
        if len(scores) > 20:
            samples = [s for s in samples if s.score >= np.percentile(scores, 10)]
            if not samples: return None
            
        t = np.array([s.global_frame for s in samples], dtype=np.float64)
        x = np.array([s.x_raw for s in samples], dtype=np.float64)
        y = np.array([s.y_raw for s in samples], dtype=np.float64)
        
        return SmoothEMATrajectory(t, x, y, _CURVE_ALPHA) if AUTOCROP_CURVE_MODE == 'smoothing' else LinearTrajectory(t, x, y)

    def _generate_debug_plot(self, samples: List[AutoCropSample], traj: Trajectory, metas: List[_ClipMeta]) -> None:
        if not samples: return
        t = np.array([s.global_frame for s in samples])
        x = np.array([s.x_raw for s in samples])
        y = np.array([s.y_raw for s in samples])
        td, xd, yd = traj.evaluate_dense(step=max(1, (int(t[-1]) - int(t[0])) // 1000))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        dims = [(x, xd, "X"), (y, yd, "Y")]
        
        for ax, (meas, fit, label) in zip(axes, dims):
            ax.scatter(t, meas, s=3, alpha=0.4, label="Meas", c='tab:blue')
            if len(td): ax.plot(td, fit, 'tab:red', lw=2, label="Fit")
            for m in metas[1:]: ax.axvline(m.global_start, c='k', lw=0.5, alpha=0.3)
            ax.set_ylabel(f"{label} (Raw Px)")
            ax.grid(True, alpha=0.5)
        
        axes[0].legend(); axes[0].set_title(f"Autocrop: {len(samples)} pts")
        axes[1].set_xlabel("Global Frame")
        fig.tight_layout()
        fig.savefig(str(self.outdir / "autocrop_trace.png")); plt.close(fig)

    def _success(self, status: str, trajectory: Trajectory, n_samples: int, rejects: Dict, timing_s: Dict, frame_wh: Tuple[int, int], crop_wh: Tuple[int, int]) -> AutoCropResult:
        res = AutoCropResult(
            status=status,
            applied=True,
            frame_wh=frame_wh,
            crop_wh=crop_wh,
            curve_kind=AUTOCROP_CURVE_MODE,
            curve_params=trajectory.get_params(),
            n_samples=n_samples,
            n_valid=n_samples,
            rejects=rejects,
            timing_s=timing_s
        )
        self._write_json(res)
        return res

    def _fail(self, status: str, timing_s: Dict, *, rejects=None, frame_wh=(0,0), crop_wh=(0,0)) -> AutoCropResult:
        res = AutoCropResult(
            status=status,
            applied=False,
            frame_wh=frame_wh,
            crop_wh=crop_wh,
            curve_kind="none",
            curve_params={},
            n_samples=0,
            n_valid=0,
            rejects=rejects or {},
            timing_s=timing_s
        )
        self._write_json(res)
        return res

    def _write_csv(self, samples: List[AutoCropSample]) -> None:
        with (self.outdir / "autocrop_points.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_idx", "frame_idx", "global_frame", "x_raw", "y_raw", "score"])
            for s in samples: w.writerow([s.clip_idx, s.frame_idx, s.global_frame, s.x_raw, s.y_raw, s.score])

    def _write_json(self, res: AutoCropResult) -> None:
        (self.outdir / "autocrop.json").write_text(json.dumps(asdict(res), indent=2, sort_keys=True), encoding="utf-8")

def _yield_frames(path: Path, indices: List[int], timing: Dict) -> Generator[Tuple[np.ndarray, int], None, None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return
    
    frame_pos = 0
    t_acc = 0.0
    
    for target in indices:
        while frame_pos < target:
            if not cap.grab(): cap.release(); return
            frame_pos += 1
            
        if not cap.grab(): cap.release(); return
        ts = time.perf_counter()
        ok, frame = cap.retrieve()
        t_acc += time.perf_counter() - ts
        if not ok: cap.release(); return
        
        yield frame, target
        frame_pos += 1
        
    timing['samples_read'] = float(timing.get('samples_read', 0)) + t_acc
    cap.release()

def _clamp_delta(dx: float, dy: float, limit: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    if n <= limit or limit <= 0: return dx, dy
    s = limit / n
    return dx * s, dy * s

def _mirror_x_coord(x: float, w: int) -> float:
    return float(w - 1) - float(x)

def _sample_frame_indices(total: int, cap: int) -> List[int]:
    if total <= 0: return []
    n = min(cap, total)
    return sorted([int((total - 1) * (i + 1) / (n + 2)) for i in range(n)])

__all__ = ["AutoCropResult", "AutoCropSample", "AutoCropper"]
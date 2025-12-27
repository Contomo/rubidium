# rubidium/analysis/image_processing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol
import numpy as np
import cv2

JOINT_PERP_MODE: str = "centroid"
JOINT_MIN_STRIPE_PX: int = 3 

# ----------------------------
# Configs
# ----------------------------

@dataclass(slots=True)
class CropConfig:
    center_xy: Tuple[float, float] = (980, 520)
    wh: Tuple[int, int] = (0, 0)
    ref_wh: Optional[Tuple[int, int]] = None

@dataclass(slots=True)
class LaserExtractConfig:
    hsv_lower: Optional[Tuple[int, int, int]] = None
    hsv_upper: Optional[Tuple[int, int, int]] = None
    bright_percentile: float = -1.0
    weight_power: float = 4.0
    min_row_energy: float = 1.0
    median_ksize: int = 5
    morph_ksize: int = 3
    use_clahe: bool = True
    blur_ksize: int = 5
    clahe_clip: float = 3.0
    clahe_grid: Tuple[int, int] = (8, 8)
    joint_samples: int = 120
    joint_perp_half_len_px: int = 16
    joint_perp_half_wid_px: int = 2
    joint_min_mask_px: int = 50
    min_profile_energy: float = 2.0

# ----------------------------
# Pipeline Core
# ----------------------------

class Step(Protocol):
    name: str
    def __call__(self, ctx: "PipelineCtx") -> "PipelineCtx": ...

@dataclass
class PipelineCtx:
    frame_bgr: np.ndarray
    cfg_crop: Optional[CropConfig] = None
    cfg_laser: Optional[LaserExtractConfig] = None
    offset_xy: Tuple[int, int] = (0, 0)
    cropped_bgr: Optional[np.ndarray] = None
    src_u8: Optional[np.ndarray] = None
    mask_u8: Optional[np.ndarray] = None
    gray_f32: Optional[np.ndarray] = None
    centroid_x: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

def run_pipeline(frame_bgr: np.ndarray, steps: List[Step], *,
                 cfg_crop: Optional[CropConfig] = None,
                 cfg_laser: Optional[LaserExtractConfig] = None,
                 keep_debug: bool = True) -> PipelineCtx:
    ctx = PipelineCtx(frame_bgr=frame_bgr, cfg_crop=cfg_crop, cfg_laser=cfg_laser)
    for st in steps:
        ctx = st(ctx)
        if not keep_debug:
            ctx.debug.clear()
    return ctx

# ----------------------------
# Laser Joint Detection
# ----------------------------

def detect_laser_joint(frame_bgr: np.ndarray, cfg: Optional[LaserExtractConfig] = None) -> Tuple[Optional[Tuple[float, float]], float, np.ndarray]:
    """
    Finds the center of the laser "nub".
    Uses Sub-pixel Weighted Averaging to handle saturation/plateaus.
    """
    lc = cfg or LaserExtractConfig()
    src_u8, mask_u8 = build_laser_mask(frame_bgr, lc)
    
    geom = _compute_stripe_geometry(mask_u8, lc)
    if geom is None: return None, 0.0, mask_u8
    
    profiles, _, _ = _sample_stripe_grids(src_u8, geom, lc)
    if profiles is None: return None, 0.0, mask_u8

    scores, offsets = _compute_profile_scores(profiles, mask_u8, lc)
    if scores is None: return None, 0.0, mask_u8

    t_vals = geom['ti']
    cx_candidates = geom['x0'] + t_vals * geom['vx'] + offsets * geom['px']
    cy_candidates = geom['y0'] + t_vals * geom['vy'] + offsets * geom['py']

    result = _refine_peak_position(cx_candidates, cy_candidates, scores)
    
    if result is None: return None, 0.0, mask_u8
        
    return result, float(np.max(scores)), mask_u8

def _compute_stripe_geometry(mask_u8: np.ndarray, lc: LaserExtractConfig) -> Optional[Dict]:
    ys, xs = np.nonzero(mask_u8)
    if xs.size < int(lc.joint_min_mask_px): return None

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    vx, vy, x0, y0 = (np.asarray(a).reshape(-1)[0].item() for a in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01))
    
    nrm = float((vx**2 + vy**2)**0.5) or 1.0
    vx, vy = vx/nrm, vy/nrm
    
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    tmin, tmax = float(np.percentile(t, 5.0)), float(np.percentile(t, 95.0))
    
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin: return None

    n_samples = int(max(9, lc.joint_samples))
    ti = np.linspace(tmin, tmax, num=n_samples, dtype=np.float32)

    return {'x0': x0, 'y0': y0, 'vx': vx, 'vy': vy, 'px': -vy, 'py': vx, 'ti': ti}

def _sample_stripe_grids(src_u8: np.ndarray, geom: Dict, lc: LaserExtractConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = src_u8.shape
    half_len = int(max(3, lc.joint_perp_half_len_px))
    half_wid = int(max(0, lc.joint_perp_half_wid_px))
    s_vals = np.arange(-half_len, half_len + 1, dtype=np.float32)
    u_vals = np.arange(-half_wid, half_wid + 1, dtype=np.float32)

    cx = geom['x0'] + geom['ti'][:, None, None] * geom['vx']
    cy = geom['y0'] + geom['ti'][:, None, None] * geom['vy']
    
    xs = cx + s_vals[None, :, None] * geom['px'] + u_vals[None, None, :] * geom['vx']
    ys = cy + s_vals[None, :, None] * geom['py'] + u_vals[None, None, :] * geom['vy']

    ix = np.rint(xs).astype(np.int32)
    iy = np.rint(ys).astype(np.int32)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)

    src_f = src_u8.astype(np.float32)
    samp = src_f[iy, ix]
    samp[~valid] = np.nan
    
    prof = np.nanmean(samp, axis=2)
    prof = np.where(np.isfinite(prof), prof, 0.0).astype(np.float32)
    
    return prof, ix, iy

def _compute_profile_scores(profiles: np.ndarray, mask_u8: np.ndarray, lc: LaserExtractConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y0 = profiles[:, 0]
    y1 = profiles[:, -1]
    L = profiles.shape[1]
    frac = np.linspace(0, 1, L, dtype=np.float32)
    baseline = y0[:, None] + (y1 - y0)[:, None] * frac[None, :]
    
    resid = np.maximum(profiles - baseline, 0.0)
    denom = resid.sum(axis=1)
    
    ok = denom > 1e-6
    scores = denom / float(L)
    ok &= scores >= float(getattr(lc, "min_profile_energy", 0.0))
    
    if not np.any(ok): return None, None

    half_len = int(max(3, lc.joint_perp_half_len_px))
    s_vals = np.arange(-half_len, half_len + 1, dtype=np.float32)
    s_bar = (resid * s_vals[None, :]).sum(axis=1) / denom

    scores[~ok] = 0.0
    return scores, s_bar

def _refine_peak_position(cx: np.ndarray, cy: np.ndarray, scores: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Robust Peak Finding:
    Instead of argmax (which snaps to grid and jitters on plateaus),
    we take the weighted average of the top 5% of scores.
    This provides sub-pixel accuracy along the line vector.
    """
    if not np.any(scores > 0): return None
    
    # 1. Isolate the "Nub" (Top 5% of energy)
    thresh = 0.95 * np.max(scores)
    peak_mask = scores >= thresh
    
    p_scores = scores[peak_mask]
    p_cx = cx[peak_mask]
    p_cy = cy[peak_mask]
    
    w_sum = np.sum(p_scores)
    if w_sum <= 0: return None
    
    # 2. Weighted Average
    best_x = np.sum(p_cx * p_scores) / w_sum
    best_y = np.sum(p_cy * p_scores) / w_sum
    
    return float(best_x), float(best_y)

# ----------------------------
# Utilities
# ----------------------------

def build_laser_mask(frame_bgr: np.ndarray, cfg: Optional[LaserExtractConfig]) -> Tuple[np.ndarray, np.ndarray]:
    lc = cfg or LaserExtractConfig()
    src_u8 = _extract_v_u8(frame_bgr)
    
    if lc.use_clahe:
        src_u8 = _apply_clahe_u8(src_u8, lc.clahe_clip, lc.clahe_grid)
    
    k = int(lc.blur_ksize)
    if k >= 3 and k % 2 == 1:
        src_u8 = cv2.GaussianBlur(src_u8, (k, k), 0)

    if lc.bright_percentile < 0:
        if int(np.max(src_u8)) < 30:
            mask_u8 = np.zeros_like(src_u8)
        else:
            _, mask_u8 = cv2.threshold(src_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        v = src_u8.astype(np.float32)
        p = float(np.clip(lc.bright_percentile, 0.0, 100.0))
        thr = float(np.percentile(v, p))
        mask_u8 = (v >= thr).astype(np.uint8) * 255

    if lc.hsv_lower is not None and lc.hsv_upper is not None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gate = cv2.inRange(hsv, np.array(lc.hsv_lower, dtype=np.uint8), np.array(lc.hsv_upper, dtype=np.uint8))
        mask_u8 = cv2.bitwise_and(mask_u8, gate)

    mk = int(lc.morph_ksize)
    if mk > 1:
        kernel = np.ones((mk, mk), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return src_u8, mask_u8

def crop_frame(frame: np.ndarray, cfg: CropConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = frame.shape[:2]
    crop_w, crop_h = int(round(cfg.wh[0])), int(round(cfg.wh[1]))
    if crop_w <= 0 or crop_h <= 0: return frame, (0, 0)

    cx, cy = cfg.center_xy
    if cfg.ref_wh and cfg.ref_wh[0] > 0:
        sx, sy = w / float(cfg.ref_wh[0]), h / float(cfg.ref_wh[1])
        cx, cy = cx * sx, cy * sy

    crop_w, crop_h = min(max(1, crop_w), w), min(max(1, crop_h), h)
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))
    
    return frame[y0:y0+crop_h, x0:x0+crop_w], (x0, y0)

def clamp_center_to_crop_bounds(center_xy: Tuple[float, float], frame_wh: Tuple[int, int], crop_wh: Tuple[int, int]) -> Tuple[float, float]:
    w, h = frame_wh
    cw, ch = crop_wh
    if min(w, h, cw, ch) <= 0: return center_xy
    
    cw, ch = min(cw, w), min(ch, h)
    half_w, half_h = cw / 2.0, ch / 2.0
    
    return (
        min(max(center_xy[0], half_w), w - half_w),
        min(max(center_xy[1], half_h), h - half_h)
    )

def _extract_v_u8(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

def _apply_clahe_u8(img_u8: np.ndarray, clip: float, grid: Tuple[int, int]) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=grid).apply(img_u8)

# ----------------------------
# Standard Steps
# ----------------------------

@dataclass(slots=True)
class StepCrop:
    name: str = "crop"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        if ctx.cfg_crop:
            ctx.cropped_bgr, ctx.offset_xy = crop_frame(ctx.frame_bgr, ctx.cfg_crop)
        else:
            ctx.cropped_bgr, ctx.offset_xy = ctx.frame_bgr, (0, 0)
        ctx.debug[self.name] = ctx.cropped_bgr
        return ctx

@dataclass(slots=True)
class StepExtractBrightness:
    name: str = "brightness"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        ctx.src_u8 = _extract_v_u8(ctx.cropped_bgr) # type: ignore
        ctx.debug[self.name] = ctx.src_u8
        return ctx

@dataclass(slots=True)
class StepCLAHE:
    name: str = "clahe"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        if cfg.use_clahe and ctx.src_u8 is not None:
            ctx.src_u8 = _apply_clahe_u8(ctx.src_u8, cfg.clahe_clip, cfg.clahe_grid)
        ctx.debug[self.name] = ctx.src_u8
        return ctx

@dataclass(slots=True)
class StepBlur:
    name: str = "blur"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        k = int(cfg.blur_ksize)
        if k >= 3 and k % 2 == 1 and ctx.src_u8 is not None:
            ctx.src_u8 = cv2.GaussianBlur(ctx.src_u8, (k, k), 0)
        ctx.debug[self.name] = ctx.src_u8
        return ctx

@dataclass(slots=True)
class StepThreshold:
    name: str = "threshold"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        if cfg.bright_percentile < 0 and ctx.src_u8 is not None:
             _, ctx.mask_u8 = cv2.threshold(ctx.src_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            v = _extract_v_u8(ctx.cropped_bgr).astype(np.float32) # type: ignore
            p = float(np.clip(cfg.bright_percentile, 0.0, 100.0))
            thr = float(np.percentile(v, p))
            ctx.mask_u8 = (v >= thr).astype(np.uint8) * 255
        ctx.debug[self.name] = ctx.mask_u8
        return ctx

@dataclass(slots=True)
class StepHSVGate:
    name: str = "hsv_gate"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        if cfg.hsv_lower and cfg.hsv_upper and ctx.mask_u8 is not None:
            hsv = cv2.cvtColor(ctx.cropped_bgr, cv2.COLOR_BGR2HSV) # type: ignore
            gate = cv2.inRange(hsv, np.array(cfg.hsv_lower, dtype=np.uint8), np.array(cfg.hsv_upper, dtype=np.uint8))
            ctx.mask_u8 = cv2.bitwise_and(ctx.mask_u8, gate)
        ctx.debug[self.name] = ctx.mask_u8
        return ctx

@dataclass(slots=True)
class StepMorph:
    name: str = "morph"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        k = int(cfg.morph_ksize)
        if k > 1 and ctx.mask_u8 is not None:
            kernel = np.ones((k, k), dtype=np.uint8)
            ctx.mask_u8 = cv2.morphologyEx(ctx.mask_u8, cv2.MORPH_OPEN, kernel)
            ctx.mask_u8 = cv2.morphologyEx(ctx.mask_u8, cv2.MORPH_CLOSE, kernel)
        ctx.debug[self.name] = ctx.mask_u8
        return ctx

@dataclass(slots=True)
class StepMaskedGray:
    name: str = "masked_gray"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        gray = cv2.cvtColor(ctx.cropped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) # type: ignore
        m = ctx.mask_u8.astype(np.float32) / 255.0 # type: ignore
        ctx.gray_f32 = gray * m
        ctx.debug[self.name] = ctx.gray_f32
        return ctx

@dataclass(slots=True)
class StepStripeCentroidPerRow:
    name: str = "centroid"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        h, w = ctx.gray_f32.shape[:2] # type: ignore
        x = np.arange(w, dtype=np.float32)
        weights = np.power(np.clip(ctx.gray_f32, 0.0, None), max(1.0, cfg.weight_power))
        s = np.sum(weights, axis=1)
        good = s > float(cfg.min_row_energy)
        
        out = np.full((h,), np.nan, dtype=np.float32)
        if np.any(good):
            out[good] = (weights[good] @ x) / s[good]

        k = int(cfg.median_ksize)
        if k >= 3 and k % 2 == 1:
            tmp = out.copy()
            isn = np.isnan(tmp)
            if np.any(~isn):
                tmp[isn] = float(np.nanmedian(tmp))
                tmp = cv2.medianBlur(tmp.reshape(-1, 1), k).reshape(-1).astype(np.float32)
                out = np.where(isn, np.nan, tmp)
        
        ctx.centroid_x = out
        ctx.debug[self.name] = out
        return ctx

_STEP_FACTORIES = {
    "crop": StepCrop, "brightness": StepExtractBrightness, "clahe": StepCLAHE, "blur": StepBlur,
    "threshold": StepThreshold, "hsv_gate": StepHSVGate, "morph": StepMorph, "masked_gray": StepMaskedGray,
    "centroid": StepStripeCentroidPerRow,
}

def build_laser_pipeline(step_names: Optional[List[str]] = None) -> List[Step]:
    if not step_names: return build_default_laser_pipeline()
    return [_STEP_FACTORIES[str(n).strip()]() for n in step_names if str(n).strip() in _STEP_FACTORIES]

def build_default_laser_pipeline() -> List[Step]:
    return build_laser_pipeline(["crop", "brightness", "clahe", "blur", "threshold", "hsv_gate", "morph", "masked_gray", "centroid"])
# rubidium/analysis/image_processing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol
import numpy as np
import cv2


# ----------------------------
# configs

@dataclass(slots=True)
class CropConfig:
    center_xy: Tuple[float, float] = (980, 520)
    wh: Tuple[int, int] = (0, 0)
    ref_wh: Optional[Tuple[int, int]] = None


@dataclass(slots=True)
class LaserExtractConfig:
    hsv_lower: Optional[Tuple[int, int, int]] = None
    hsv_upper: Optional[Tuple[int, int, int]] = None

    bright_percentile: float = -1.0  # < 0 means "use Otsu path"
    weight_power: float = 4.0
    min_row_energy: float = 1.0

    median_ksize: int = 5
    morph_ksize: int = 3

    use_clahe: bool = True
    blur_ksize: int = 5
    clahe_clip: float = 3.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # Joint (laser-line intersection) detector knobs (used for auto-crop prepass)
    joint_samples: int = 120
    joint_perp_half_len_px: int = 16
    joint_perp_half_wid_px: int = 2
    joint_min_mask_px: int = 50


# ----------------------------
# pipeline core

class Step(Protocol):
    name: str
    def __call__(self, ctx: "PipelineCtx") -> "PipelineCtx": ...


@dataclass
class PipelineCtx:
    frame_bgr: np.ndarray
    cfg_crop: Optional[CropConfig] = None
    cfg_laser: Optional[LaserExtractConfig] = None

    # mutable working set
    offset_xy: Tuple[int, int] = (0, 0)
    cropped_bgr: Optional[np.ndarray] = None
    src_u8: Optional[np.ndarray] = None      # usually V channel or other 8-bit source
    mask_u8: Optional[np.ndarray] = None
    gray_f32: Optional[np.ndarray] = None
    centroid_x: Optional[np.ndarray] = None  # per-row centroid (float32, NaNs for bad rows)

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
# utilities

def crop_frame(frame: np.ndarray, cfg: CropConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = frame.shape[:2]

    crop_w = int(round(cfg.wh[0]))
    crop_h = int(round(cfg.wh[1]))
    if crop_w <= 0 or crop_h <= 0:
        return frame, (0, 0)

    # Resolve center in current-frame pixels (optionally scaled from a reference resolution).
    if cfg.ref_wh is not None and cfg.ref_wh[0] > 0 and cfg.ref_wh[1] > 0:
        ref_w, ref_h = cfg.ref_wh
        sx = w / float(ref_w)
        sy = h / float(ref_h)
        cx = float(cfg.center_xy[0]) * sx
        cy = float(cfg.center_xy[1]) * sy
    else:
        cx = float(cfg.center_xy[0])
        cy = float(cfg.center_xy[1])

    crop_w = int(max(1, crop_w))
    crop_h = int(max(1, crop_h))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))

    # Clamp so the crop keeps its requested size whenever possible.
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    return frame[y0:y1, x0:x1], (x0, y0)


def build_laser_mask(frame_bgr: np.ndarray, cfg: Optional[LaserExtractConfig]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (src_u8, mask_u8) for the laser stripe."""
    lc = cfg or LaserExtractConfig()

    src_u8 = _extract_v_u8(frame_bgr)
    if lc.use_clahe:
        src_u8 = _apply_clahe_u8(src_u8, lc.clahe_clip, lc.clahe_grid)
    k = int(lc.blur_ksize)
    if k >= 3 and (k % 2) == 1:
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
        gate = cv2.inRange(
            hsv,
            np.array(lc.hsv_lower, dtype=np.uint8),
            np.array(lc.hsv_upper, dtype=np.uint8),
        )
        mask_u8 = cv2.bitwise_and(mask_u8, gate)

    mk = int(lc.morph_ksize)
    if mk > 1:
        kernel = np.ones((mk, mk), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return src_u8, mask_u8


def detect_laser_joint(frame_bgr: np.ndarray, cfg: Optional[LaserExtractConfig]) -> Tuple[Optional[Tuple[float, float]], float, np.ndarray]:
    """Detect a stable "joint" point along a line laser.

    Returns:
      - center_xy (float px, local to frame_bgr) or None
      - score (higher is better)
      - mask_u8 (for debugging)
    """
    lc = cfg or LaserExtractConfig()
    h, w = frame_bgr.shape[:2]

    src_u8, mask_u8 = build_laser_mask(frame_bgr, lc)
    ys, xs = np.nonzero(mask_u8)
    if xs.size < int(lc.joint_min_mask_px):
        return None, 0.0, mask_u8

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

    # Fit laser line
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
    nrm = float((vx * vx + vy * vy) ** 0.5) or 1.0
    vx, vy = vx / nrm, vy / nrm
    px, py = -vy, vx  # perpendicular

    # Candidate range along the fitted line (robust to outliers)
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    tmin = float(np.percentile(t, 5.0))
    tmax = float(np.percentile(t, 95.0))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return None, 0.0, mask_u8

    # Gradient magnitude on the source image
    src_f = src_u8.astype(np.float32)
    gx = cv2.Sobel(src_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src_f, cv2.CV_32F, 0, 1, ksize=3)

    half_len = int(max(1, lc.joint_perp_half_len_px))
    half_wid = int(max(0, lc.joint_perp_half_wid_px))
    n_samples = int(max(5, lc.joint_samples))

    def _edge_energy(cx: float, cy: float) -> float:
        acc = 0.0
        cnt = 0
        for s in range(-half_len, half_len + 1):
            for u in range(-half_wid, half_wid + 1):
                x = cx + s * px + u * vx
                y = cy + s * py + u * vy
                ix = int(round(x))
                iy = int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    gxx = float(gx[iy, ix])
                    gyy = float(gy[iy, ix])
                    acc += float((gxx * gxx + gyy * gyy) ** 0.5)
                    cnt += 1
        return acc / float(max(1, cnt))

    best_score = -1.0
    best_xy: Optional[Tuple[float, float]] = None

    for i in range(n_samples):
        ti = tmin + (tmax - tmin) * (i / float(max(1, n_samples - 1)))
        cx = x0 + ti * vx
        cy = y0 + ti * vy
        if cx < 0.0 or cx > (w - 1) or cy < 0.0 or cy > (h - 1):
            continue
        sc = _edge_energy(cx, cy)
        if sc > best_score:
            best_score = sc
            best_xy = (cx, cy)

    if best_xy is None or not np.isfinite(best_score):
        return None, 0.0, mask_u8

    return best_xy, float(best_score), mask_u8


def _extract_v_u8(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


def _apply_clahe_u8(img_u8: np.ndarray, clip: float, grid: Tuple[int, int]) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(grid))
    return clahe.apply(img_u8)


# ----------------------------
# steps (small + reorderable)

@dataclass(slots=True)
class StepCrop:
    name: str = "crop"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        if ctx.cfg_crop is None:
            ctx.cropped_bgr = ctx.frame_bgr
            ctx.offset_xy = (0, 0)
        else:
            cropped, off = crop_frame(ctx.frame_bgr, ctx.cfg_crop)
            ctx.cropped_bgr = cropped
            ctx.offset_xy = off
        ctx.debug[self.name] = ctx.cropped_bgr
        return ctx


@dataclass(slots=True)
class StepExtractBrightness:
    name: str = "brightness"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        assert ctx.cropped_bgr is not None
        ctx.src_u8 = _extract_v_u8(ctx.cropped_bgr)
        ctx.debug[self.name] = ctx.src_u8
        return ctx


@dataclass(slots=True)
class StepCLAHE:
    name: str = "clahe"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        assert ctx.src_u8 is not None
        if cfg.use_clahe:
            ctx.src_u8 = _apply_clahe_u8(ctx.src_u8, cfg.clahe_clip, cfg.clahe_grid)
        ctx.debug[self.name] = ctx.src_u8
        return ctx


@dataclass(slots=True)
class StepBlur:
    name: str = "blur"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        assert ctx.src_u8 is not None
        k = int(cfg.blur_ksize)
        if k >= 3 and (k % 2) == 1:
            ctx.src_u8 = cv2.GaussianBlur(ctx.src_u8, (k, k), 0)
        ctx.debug[self.name] = ctx.src_u8
        return ctx


@dataclass(slots=True)
class StepThreshold:
    """
    If cfg.bright_percentile < 0 -> Otsu.
    Else -> percentile threshold on V (float32).
    """
    name: str = "threshold"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        assert ctx.cropped_bgr is not None

        if cfg.bright_percentile < 0:
            assert ctx.src_u8 is not None
            if int(np.max(ctx.src_u8)) < 30:
                ctx.mask_u8 = np.zeros_like(ctx.src_u8)
            else:
                _, m = cv2.threshold(ctx.src_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ctx.mask_u8 = m
        else:
            hsv = cv2.cvtColor(ctx.cropped_bgr, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(np.float32)
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
        assert ctx.cropped_bgr is not None
        assert ctx.mask_u8 is not None

        if cfg.hsv_lower is not None and cfg.hsv_upper is not None:
            hsv = cv2.cvtColor(ctx.cropped_bgr, cv2.COLOR_BGR2HSV)
            color = cv2.inRange(
                hsv,
                np.array(cfg.hsv_lower, dtype=np.uint8),
                np.array(cfg.hsv_upper, dtype=np.uint8),
            )
            ctx.mask_u8 = cv2.bitwise_and(ctx.mask_u8, color)

        ctx.debug[self.name] = ctx.mask_u8
        return ctx


@dataclass(slots=True)
class StepMorph:
    name: str = "morph"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        assert ctx.mask_u8 is not None
        k = int(cfg.morph_ksize)
        if k > 1:
            kernel = np.ones((k, k), dtype=np.uint8)
            ctx.mask_u8 = cv2.morphologyEx(ctx.mask_u8, cv2.MORPH_OPEN, kernel)
            ctx.mask_u8 = cv2.morphologyEx(ctx.mask_u8, cv2.MORPH_CLOSE, kernel)
        ctx.debug[self.name] = ctx.mask_u8
        return ctx


@dataclass(slots=True)
class StepMaskedGray:
    name: str = "masked_gray"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        assert ctx.cropped_bgr is not None
        assert ctx.mask_u8 is not None
        gray = cv2.cvtColor(ctx.cropped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = ctx.mask_u8.astype(np.float32) / 255.0
        ctx.gray_f32 = gray * m
        ctx.debug[self.name] = ctx.gray_f32
        return ctx


@dataclass(slots=True)
class StepStripeCentroidPerRow:
    name: str = "centroid"
    def __call__(self, ctx: PipelineCtx) -> PipelineCtx:
        cfg = ctx.cfg_laser or LaserExtractConfig()
        assert ctx.gray_f32 is not None

        h, w = ctx.gray_f32.shape[:2]
        x = np.arange(w, dtype=np.float32)

        pw = float(max(1.0, cfg.weight_power))
        weights = np.power(np.clip(ctx.gray_f32, 0.0, None), pw)

        s = np.sum(weights, axis=1)
        good = s > float(cfg.min_row_energy)

        out = np.full((h,), np.nan, dtype=np.float32)
        if np.any(good):
            out[good] = (weights[good] @ x) / s[good]

        k = int(cfg.median_ksize)
        if k >= 3 and (k % 2) == 1:
            tmp = out.copy()
            isn = np.isnan(tmp)
            if np.any(~isn):
                fill = float(np.nanmedian(tmp))
                tmp[isn] = fill
                tmp = cv2.medianBlur(tmp.reshape(-1, 1), k).reshape(-1).astype(np.float32)
                out = np.where(isn, np.nan, tmp)

        ctx.centroid_x = out
        ctx.debug[self.name] = out
        return ctx


def build_default_laser_pipeline() -> List[Step]:
    return [
        StepCrop(),
        StepExtractBrightness(),
        StepCLAHE(),
        StepBlur(),
        StepThreshold(),
        StepHSVGate(),
        StepMorph(),
        StepMaskedGray(),
        StepStripeCentroidPerRow(),
    ]

_STEP_FACTORIES = {
    "crop": StepCrop,
    "brightness": StepExtractBrightness,
    "clahe": StepCLAHE,
    "blur": StepBlur,
    "threshold": StepThreshold,
    "hsv_gate": StepHSVGate,
    "morph": StepMorph,
    "masked_gray": StepMaskedGray,
    "centroid": StepStripeCentroidPerRow,
}


def build_laser_pipeline(step_names: Optional[List[str]] = None) -> List[Step]:
    """
    Build a laser extraction pipeline.
    - step_names=None -> default pipeline
    - step_names=[...] -> explicit ordering by step name
    """
    if not step_names:
        return build_default_laser_pipeline()

    out: List[Step] = []
    for raw in step_names:
        name = str(raw).strip()
        if not name:
            continue
        cls = _STEP_FACTORIES.get(name)
        if cls is None:
            raise ValueError(f"Unknown pipeline step: {name!r}. Known: {sorted(_STEP_FACTORIES.keys())}")
        out.append(cls())
    return out

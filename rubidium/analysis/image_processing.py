# rubidium/analysis/image_processing.py
"""Image processing helpers for Rubidium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


@dataclass(slots=True)
class CropConfig:
    """ROI definition in reference coordinates."""
    center_xy: tuple[float, float] = (640.0, 360.0)
    wh: tuple[int, int] = (900, 260)
    ref_wh: tuple[int, int] = (1280, 720)


@dataclass(slots=True)
class LaserExtractConfig:
    """Laser stripe extraction settings."""
    hsv_lower: Optional[tuple[int, int, int]] = None
    hsv_upper: Optional[tuple[int, int, int]] = None

    bright_percentile: float = 99.5

    weight_power: float = 4.0

    min_row_energy: float = 1.0

    median_ksize: int = 5
    morph_ksize: int = 3


def crop_frame(frame: np.ndarray, cfg: CropConfig) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a frame around the given centre with the given size"""
    h, w = frame.shape[:2]
    ref_w, ref_h = cfg.ref_wh

    sx = w / float(ref_w)
    sy = h / float(ref_h)

    cx = int(round(cfg.center_xy[0] * sx))
    cy = int(round(cfg.center_xy[1] * sy))

    half_w = int(round(cfg.wh[0] / 2.0))
    half_h = int(round(cfg.wh[1] / 2.0))

    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)

    return frame[y0:y1, x0:x1], (x0, y0)


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("rubidium: cv2 not available in this python env")


def _hsv_mask(frame_bgr: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    _ensure_cv2()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))


def _bright_mask(frame_bgr: np.ndarray, percentile: float) -> np.ndarray:
    _ensure_cv2()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    p = float(np.clip(percentile, 50.0, 100.0))
    thr = float(np.percentile(v, p))
    mask = (v >= thr).astype(np.uint8) * 255
    return mask


def build_laser_mask(frame_bgr: np.ndarray, cfg: LaserExtractConfig) -> np.ndarray:
    """Build a binary mask of where the laser stripe probably is.""" # (probably)
    _ensure_cv2()

    mask = _bright_mask(frame_bgr, cfg.bright_percentile)

    if cfg.hsv_lower is not None and cfg.hsv_upper is not None:
        mask = cv2.bitwise_and(mask, _hsv_mask(frame_bgr, cfg.hsv_lower, cfg.hsv_upper))

    k = int(cfg.morph_ksize)
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def masked_gray(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply a binary mask."""
    _ensure_cv2()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = (mask.astype(np.float32) / 255.0)
    return gray * m


def stripe_centroid_per_row(gray_masked: np.ndarray, cfg: LaserExtractConfig) -> np.ndarray:
    """Compute a sub-pixel x centroid for each image row.

    Returns a float32 array of length H; rows with no stripe are NaN.
    """
    h, w = gray_masked.shape[:2]
    x = np.arange(w, dtype=np.float32)

    # Emphasize the laser core without going full supernova.
    pw = float(max(1.0, cfg.weight_power))
    weights = np.power(np.clip(gray_masked, 0.0, None), pw)

    s = np.sum(weights, axis=1)
    good = s > float(cfg.min_row_energy)

    out = np.full((h,), np.nan, dtype=np.float32)
    if np.any(good):
        out[good] = (weights[good] @ x) / s[good]

    # Mild median filter to reduce salt-and-pepper centroid jumps.
    k = int(cfg.median_ksize)
    if cv2 is not None and k >= 3 and (k % 2) == 1:
        tmp = out.copy()
        # OpenCV medianBlur wants finite values; fill NaNs with nearest.
        isn = np.isnan(tmp)
        if np.any(~isn):
            fill = float(np.nanmedian(tmp))
            tmp[isn] = fill
            tmp = cv2.medianBlur(tmp.reshape(-1, 1), k).reshape(-1).astype(np.float32)
            out = np.where(isn, np.nan, tmp)

    return out


def pick_stripe_points(
    frame_bgr: np.ndarray,
    crop_cfg: CropConfig,
    laser_cfg: LaserExtractConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract stripe points from a frame.

    Returns (u_px, v_px, intensity) in ORIGINAL frame pixel coords.
    """
    _ensure_cv2()
    cropped, (x0, y0) = crop_frame(frame_bgr, crop_cfg)
    mask = build_laser_mask(cropped, laser_cfg)
    gray = masked_gray(cropped, mask)

    cx = stripe_centroid_per_row(gray, laser_cfg)
    h = gray.shape[0]
    ys = np.arange(h, dtype=np.float32)

    good = np.isfinite(cx)
    if not np.any(good):
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Use masked gray values near the centroid as a crude intensity estimate.
    ix = np.clip(np.round(cx[good]).astype(np.int32), 0, gray.shape[1] - 1)
    inten = gray[good, :][np.arange(ix.size), ix]

    u = cx[good] + float(x0)
    v = ys[good] + float(y0)

    return u.astype(np.float32), v.astype(np.float32), inten.astype(np.float32)

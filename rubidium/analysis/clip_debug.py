# rubidium/analysis/clip_debug.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .autocrop import _CropTracker, _mirror_x_coord
from .image_processing import CropConfig, crop_frame, clamp_center_to_crop_bounds, detect_laser_joint
from .types import AnalysisConfig


@dataclass(slots=True)
class ClipDebugOptions:
    tile_scale: int = 2
    max_frames: int = 0  # 0 = all


def render_scan_debug_preview_clip(
    *,
    cfg: AnalysisConfig,
    clip_path: Path,
    out_mp4: Path,
    mirror_x: bool,
    clip_idx: int,
    opts: Optional[ClipDebugOptions] = None,
) -> Path:
    """Render a tiled debug video visualizing detector + crop window for one clip."""

    opts = opts or ClipDebugOptions()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"rubidium: failed opening clip: {clip_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    crop_w, crop_h = int(cfg.crop.wh[0]), int(cfg.crop.wh[1])
    crop_wh = (max(1, min(crop_w, w)), max(1, min(crop_h, h)))

    # Keep behavior aligned with the real autocrop path by using the same tracker.
    tracker = _CropTracker((w / 2.0, h / 2.0), crop_wh, cfg.laser)
    scale = max(1, int(opts.tile_scale))

    tile_w, tile_h = w * scale, h * scale
    out_w, out_h = tile_w * 2, tile_h * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, out_h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"rubidium: failed opening VideoWriter for {out_mp4}")

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if opts.max_frames and frame_i > int(opts.max_frames):
            break

        raw = frame
        proc = raw[:, ::-1] if mirror_x else raw

        # choose crop
        crop_rect_proc: Optional[Tuple[int, int, int, int]] = None
        if tracker.locked:
            cx, cy = clamp_center_to_crop_bounds(tracker.xy, (w, h), crop_wh)
            # clamp_center_to_crop_bounds uses raw coords; convert to proc if mirrored
            cx_proc = _mirror_x_coord(cx, w) if mirror_x else float(cx)
            search_img, off = crop_frame(proc, CropConfig(center_xy=(cx_proc, cy), wh=crop_wh))
            crop_rect_proc = (int(off[0]), int(off[1]), int(crop_wh[0]), int(crop_wh[1]))
        else:
            search_img, off = proc, (0, 0)

        joint_xy, score, mask_u8 = detect_laser_joint(search_img, cfg.laser)
        accepted = False
        reason = "no_joint"
        det_raw_xy: Optional[Tuple[float, float]] = None
        det_crop_xy: Optional[Tuple[float, float]] = None

        if joint_xy is not None:
            det_crop_xy = (float(joint_xy[0]), float(joint_xy[1]))
            det_x_proc = float(off[0]) + float(joint_xy[0])
            det_y_proc = float(off[1]) + float(joint_xy[1])
            det_x_raw = _mirror_x_coord(det_x_proc, w) if mirror_x else det_x_proc
            det_raw_xy = (det_x_raw, det_y_proc)

            accepted, reason = tracker.update(det_raw_xy, float(score))

        # panels --------------------------------------------------------
        p_raw = _panel_full_frame(raw, crop_rect_proc, det_raw_xy, tracker.xy, mirror_x)
        p_crop = _panel_crop(search_img, det_crop_xy, crop_wh)
        p_mask = _panel_full_mask(mask_u8, crop_rect_proc, w, h, crop_wh, mirror_x)
        p_txt = _panel_text(
            clip_idx=clip_idx,
            frame_idx=frame_i - 1,
            locked=tracker.locked,
            accepted=accepted,
            reason=reason,
            score=float(score),
            det_raw_xy=det_raw_xy,
            track_xy=tracker.xy,
            crop_wh=crop_wh,
        )

        tiled = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        tiled[0:tile_h, 0:tile_w] = _resize(p_raw, (tile_w, tile_h))
        tiled[0:tile_h, tile_w:out_w] = _resize(p_crop, (tile_w, tile_h))
        tiled[tile_h:out_h, 0:tile_w] = _resize(p_mask, (tile_w, tile_h))
        tiled[tile_h:out_h, tile_w:out_w] = _resize(p_txt, (tile_w, tile_h))
        vw.write(tiled)

    vw.release()
    cap.release()
    return out_mp4


def _resize(img: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, wh, interpolation=cv2.INTER_NEAREST)


def _panel_full_frame(
    raw_bgr: np.ndarray,
    crop_rect_proc: Optional[Tuple[int, int, int, int]],
    det_raw_xy: Optional[Tuple[float, float]],
    track_xy: Tuple[float, float],
    mirror_x: bool,
) -> np.ndarray:
    out = raw_bgr.copy()
    h, w = out.shape[:2]

    if crop_rect_proc is not None:
        x0p, y0p, cw, ch = crop_rect_proc
        # map crop rect from proc coords to raw coords
        if mirror_x:
            x0 = int(w - x0p - cw)
        else:
            x0 = int(x0p)
        y0 = int(y0p)
        cv2.rectangle(out, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (0, 255, 255), 1)

    _draw_cross(out, (int(track_xy[0]), int(track_xy[1])), (0, 255, 0))
    if det_raw_xy is not None:
        _draw_cross(out, (int(det_raw_xy[0]), int(det_raw_xy[1])), (0, 0, 255))
    cv2.putText(out, "RAW", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _panel_crop(
    crop_bgr: np.ndarray,
    det_crop_xy: Optional[Tuple[float, float]],
    crop_wh: Tuple[int, int],
) -> np.ndarray:
    out = crop_bgr.copy()
    cw, ch = int(crop_wh[0]), int(crop_wh[1])
    _draw_cross(out, (cw // 2, ch // 2), (0, 255, 0))
    if det_crop_xy is not None:
        _draw_cross(out, (int(det_crop_xy[0]), int(det_crop_xy[1])), (0, 0, 255))
    cv2.putText(out, "CROP", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _panel_full_mask(
    mask_u8: np.ndarray,
    crop_rect_proc: Optional[Tuple[int, int, int, int]],
    w: int,
    h: int,
    crop_wh: Tuple[int, int],
    mirror_x: bool,
) -> np.ndarray:
    full = np.zeros((h, w), dtype=np.uint8)
    if mask_u8 is not None and mask_u8.size:
        if crop_rect_proc is None:
            # mask already full-frame
            if mask_u8.shape[:2] == (h, w):
                full = mask_u8
            else:
                # fallback: just center it
                mh, mw = mask_u8.shape[:2]
                y0 = max(0, (h - mh) // 2)
                x0 = max(0, (w - mw) // 2)
                full[y0:y0 + mh, x0:x0 + mw] = mask_u8
        else:
            x0p, y0p, cw, ch = crop_rect_proc
            x0 = int(w - x0p - cw) if mirror_x else int(x0p)
            y0 = int(y0p)
            m = mask_u8
            if m.shape[0] != ch or m.shape[1] != cw:
                m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            full[y0:y0 + ch, x0:x0 + cw] = m

    out = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    if crop_rect_proc is not None:
        x0p, y0p, cw, ch = crop_rect_proc
        x0 = int(w - x0p - cw) if mirror_x else int(x0p)
        y0 = int(y0p)
        cv2.rectangle(out, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (0, 255, 255), 1)
    cv2.putText(out, "MASK", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _panel_text(
    *,
    clip_idx: int,
    frame_idx: int,
    locked: bool,
    accepted: bool,
    reason: str,
    score: float,
    det_raw_xy: Optional[Tuple[float, float]],
    track_xy: Tuple[float, float],
    crop_wh: Tuple[int, int],
) -> np.ndarray:
    out = np.zeros((240, 240, 3), dtype=np.uint8)
    lines = [
        f"clip={clip_idx:03d}  frame={frame_idx}",
        f"crop_wh={crop_wh[0]}x{crop_wh[1]}",
        f"locked={int(locked)}  accepted={int(accepted)}",
        f"reason={reason}",
        f"score={score:.3f}",
        f"track=({track_xy[0]:.1f},{track_xy[1]:.1f})",
    ]
    if det_raw_xy is not None:
        lines.append(f"det=({det_raw_xy[0]:.1f},{det_raw_xy[1]:.1f})")
    else:
        lines.append("det=(None)")

    y = 22
    for s in lines:
        cv2.putText(out, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    cv2.putText(out, "INFO", (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def _draw_cross(img: np.ndarray, xy: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = int(xy[0]), int(xy[1])
    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)


__all__ = [
    "ClipDebugOptions",
    "render_scan_debug_preview_clip",
]

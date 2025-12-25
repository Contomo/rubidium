# rubidium/analysis/video_analyzer.py
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .image_processing import CropConfig, build_laser_pipeline, run_pipeline
from .scoring import ScoreBreakdown, score_heightmap
from .types import (
    _BUMP_FAIL_SCORE,
    _BUMP_MIN_PX,
    AnalysisConfig,
    CameraCalibration,
    LineAnalysis,
)
from .visualization import create_debug_thumbnails, render_heightmap_plot


class VideoAnalyzer:
    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.steps = build_laser_pipeline(cfg.pipeline_steps)
        self._calib_cache: Optional[CameraCalibration] = None
        self._calib_path: Optional[str] = None

    def analyze(self, *, path: Path, idx: int, pa: float, mirror_x: bool = False) -> LineAnalysis:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logging.warning(f"rubidium: failed to open video: {path}")
            return LineAnalysis(
                video_path=path,
                idx=idx,
                pa=pa,
                breakdown=ScoreBreakdown(score=float("inf"), roughness=float("inf"), dropouts=1.0),
                height_map_kind="err",
                ok=False,
            )

        total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        debug_frame_idx = max(1, total_frames_est // 2)

        crop_cfg = None
        centers: List[np.ndarray] = []
        kept = 0
        frames_seen = 0

        t_crop = t_mask = t_track = None
        expected_h: Optional[int] = None
        offset_xy: Optional[Tuple[int, int]] = None
        frame_w: Optional[int] = None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames_seen += 1

            if crop_cfg is None:
                crop_cfg = self._build_crop_cfg(first_frame=frame, mirror_x=mirror_x)

            if mirror_x:
                frame = cv2.flip(frame, 1)

            is_process = (self.cfg.frame_step == 1) or ((frames_seen - 1) % self.cfg.frame_step == 0)
            is_debug = frames_seen == debug_frame_idx
            if not is_process and not is_debug:
                continue

            ctx = run_pipeline(
                frame,
                self.steps,
                cfg_crop=crop_cfg,
                cfg_laser=self.cfg.laser,
                keep_debug=False,
            )

            if frame_w is None:
                frame_w = int(frame.shape[1])
            if offset_xy is None:
                offset_xy = ctx.offset_xy

            cx = ctx.centroid_x
            if cx is not None:
                if expected_h is None:
                    expected_h = int(cx.shape[0])
                elif int(cx.shape[0]) != expected_h:
                    # Crop height changing across frames is an invariant break.
                    # Skip this frame; continuing keeps the run usable.
                    continue

            if is_process and cx is not None:
                centers.append(cx)
                kept += 1

            if is_debug and ctx.mask_u8 is not None and cx is not None:
                try:
                    cropped = ctx.cropped_bgr if ctx.cropped_bgr is not None else frame
                    t_crop, t_mask, t_track = create_debug_thumbnails(cropped, ctx.mask_u8, cx)
                except Exception:
                    pass

            if self.cfg.max_frames and kept >= self.cfg.max_frames:
                break

        cap.release()

        if not centers:
            return LineAnalysis(
                video_path=path,
                idx=idx,
                pa=pa,
                breakdown=ScoreBreakdown(score=float("inf"), roughness=float("inf"), dropouts=1.0),
                height_map_kind="empty",
                ok=False,
            )

        center_map = np.stack(centers, axis=0).astype(np.float32)
        height_map, height_map_kind, tri_ok = self._compute_height_map(
            center_map=center_map,
            offset_xy=offset_xy or (0, 0),
            frame_w=int(frame_w or 0),
            mirror_x=mirror_x,
        )

        breakdown = score_heightmap(
            height_map,
            trim_frac=float(self.cfg.score_trim_frac),
        )

        bump_px = float(_profile_bump_px(center_map))
        if not tri_ok and bump_px < float(_BUMP_MIN_PX):
            breakdown = ScoreBreakdown(
                score=max(float(breakdown.score), float(_BUMP_FAIL_SCORE)),
                roughness=breakdown.roughness,
                dropouts=breakdown.dropouts,
            )
            height_map_kind = "pixel_gate"

        if tri_ok:
            height_map_kind = "triangulated_gate"
            breakdown = self._apply_triangulation_gates(height_map, breakdown)

        plot_path = None
        npz_path = None
        outdir = Path(self.cfg.output_dir) if self.cfg.output_dir else (path.parent / "analysis")
        outdir.mkdir(parents=True, exist_ok=True)

        if self.cfg.write_plots:
            plot_path = outdir / f"line_{idx:03d}_pa_{pa:.5f}.png"
            img = render_heightmap_plot(height_map, width_px=800, height_px=300, limit_scale=None)
            ok = cv2.imwrite(str(plot_path), img)
            if not ok:
                plot_path = None

        if self.cfg.write_npz:
            npz_path = outdir / f"line_{idx:03d}_pa_{pa:.5f}.npz"
            step_names = np.array([s.name for s in self.steps], dtype=object)
            np.savez_compressed(
                str(npz_path),
                height_map=height_map,
                center_map=center_map,
                score=float(breakdown.score),
                steps=step_names,
            )

        return LineAnalysis(
            video_path=path,
            idx=idx,
            pa=pa,
            breakdown=breakdown,
            height_map_kind=height_map_kind,
            height_map=height_map,
            thumb_crop=t_crop,
            thumb_mask=t_mask,
            thumb_track=t_track,
            plot_path=plot_path,
            npz_path=npz_path,
            ok=True,
            bump_px=bump_px,
        )

    def _build_crop_cfg(self, *, first_frame: np.ndarray, mirror_x: bool) -> CropConfig:
        """Build the per-video crop config.

        When mirror_x is True, VideoAnalyzer flips frames before running the pipeline.
        This method mirrors center_x as well, so the cropped region stays physically
        aligned to cfg.crop.center_xy.
        """
        if not mirror_x:
            return self.cfg.crop

        cx, cy = self.cfg.crop.center_xy
        if self.cfg.crop.ref_wh is not None and self.cfg.crop.ref_wh[0] > 0:
            ref_w = float(self.cfg.crop.ref_wh[0])
            cx_m = (ref_w - 1.0) - float(cx)
        else:
            fw = float(first_frame.shape[1])
            cx_m = (fw - 1.0) - float(cx)
        return replace(self.cfg.crop, center_xy=(float(cx_m), float(cy)))

    def _compute_height_map(
        self,
        *,
        center_map: np.ndarray,
        offset_xy: Tuple[int, int],
        frame_w: int,
        mirror_x: bool,
    ) -> Tuple[np.ndarray, str, bool]:
        baseline = np.nanmedian(center_map, axis=0)
        height_map = (center_map - baseline[None, :]).astype(np.float32)
        height_map_kind = "pixel"

        if not self.cfg.triangulation.enabled:
            return height_map, height_map_kind, False

        tri = self.cfg.triangulation
        if not (tri.camera_calibration_path and tri.laser_plane_abcd and tri.bed_plane_abcd):
            logging.warning(f"rubidium: triangulation enabled but missing calibration/planes for {frame_w=}")
            return height_map, height_map_kind, False

        if frame_w <= 0:
            logging.warning("rubidium: triangulation skipped (invalid frame width)")
            return height_map, height_map_kind, False

        try:
            calib = self._load_calibration(tri.camera_calibration_path)
            tri_map = _triangulate_height_map(
                center_map,
                offset_xy=offset_xy,
                frame_w=frame_w,
                mirror_x=bool(mirror_x),
                calib=calib,
                laser_plane_abcd=tri.laser_plane_abcd,
                bed_plane_abcd=tri.bed_plane_abcd,
            )
            return tri_map, "triangulated", True
        except Exception as e:
            logging.warning(f"rubidium: triangulation failed: {e}")
            return height_map, height_map_kind, False

    def _apply_triangulation_gates(self, height_map: np.ndarray, breakdown: ScoreBreakdown) -> ScoreBreakdown:
        tri = self.cfg.triangulation
        gate_failed = False
        if float(tri.min_valid_frac) > 0.0:
            valid_frac = float(np.mean(np.isfinite(height_map)))
            if valid_frac < float(tri.min_valid_frac):
                gate_failed = True
        if float(tri.min_height_range) > 0.0:
            ranges = _height_range_per_frame(height_map)
            med_range = float(np.nanmedian(ranges)) if ranges.size > 0 else 0.0
            if med_range < float(tri.min_height_range):
                gate_failed = True
        if not gate_failed:
            return breakdown
        return ScoreBreakdown(
            score=max(float(breakdown.score), float(tri.gate_fail_score)),
            roughness=breakdown.roughness,
            dropouts=breakdown.dropouts,
        )

    def _load_calibration(self, path: str) -> CameraCalibration:
        if self._calib_cache is not None and self._calib_path == path:
            return self._calib_cache
        calib = _load_camera_calibration(Path(path).expanduser())
        self._calib_cache = calib
        self._calib_path = path
        return calib


def _load_camera_calibration(path: Path) -> CameraCalibration:
    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        return CameraCalibration(
            K=np.asarray(data["K"], dtype=np.float32),
            dist=np.asarray(data["dist"], dtype=np.float32),
        )

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"rubidium: failed to read calib: {path}")

    K = fs.getNode("K").mat()
    if K is None or K.size == 0:
        K = fs.getNode("camera_matrix").mat()

    dist = fs.getNode("dist").mat()
    if dist is None or dist.size == 0:
        dist = fs.getNode("distortion_coefficients").mat()

    fs.release()
    return CameraCalibration(K=np.asarray(K, dtype=np.float32), dist=np.asarray(dist, dtype=np.float32))


def _profile_bump_px(center_map: np.ndarray) -> float:
    """Estimate bump amplitude (px) from the median centroid profile."""
    if center_map.ndim != 2:
        return 0.0
    profile = np.nanmedian(center_map, axis=0)
    if profile.size < 8:
        return 0.0
    idx = np.nonzero(np.isfinite(profile))[0].astype(np.float32)
    if idx.size < 8:
        return 0.0
    vals = profile[idx.astype(np.int32)].astype(np.float32)
    coeff = np.polyfit(idx.astype(np.float64), vals.astype(np.float64), 1)
    a = float(coeff[0])
    b = float(coeff[1])
    pred = a * idx + b
    resid = vals - pred
    bump = float(np.percentile(np.abs(resid), 95.0))
    if not np.isfinite(bump):
        return 0.0
    return bump


def _triangulate_height_map(
    center_map_px: np.ndarray,
    *,
    offset_xy: Tuple[int, int],
    frame_w: int,
    mirror_x: bool,
    calib: CameraCalibration,
    laser_plane_abcd: Tuple[float, float, float, float],
    bed_plane_abcd: Tuple[float, float, float, float],
) -> np.ndarray:
    """Triangulate a per-frame height map from centroid pixels."""
    laser = np.asarray(laser_plane_abcd, dtype=np.float32).reshape(4)
    bed = np.asarray(bed_plane_abcd, dtype=np.float32).reshape(4)
    bed_n = bed[:3]
    bed_n_norm = float(np.linalg.norm(bed_n))
    if bed_n_norm <= 0.0:
        raise ValueError("invalid bed plane (zero normal)")

    frames, rows = center_map_px.shape
    out = np.full((frames, rows), np.nan, dtype=np.float32)

    off_x = float(offset_xy[0])
    off_y = float(offset_xy[1])
    k = np.asarray(calib.K, dtype=np.float32)
    dist = np.asarray(calib.dist, dtype=np.float32)

    for i in range(frames):
        cx = center_map_px[i]
        valid = np.isfinite(cx)
        if not np.any(valid):
            continue
        row_idx = np.nonzero(valid)[0].astype(np.float32)
        xs = cx[valid].astype(np.float32) + off_x
        if mirror_x:
            xs = (float(frame_w - 1) - xs)
        ys = row_idx + off_y

        pts = np.column_stack([xs, ys]).astype(np.float32)
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2), k, dist)
        und = und.reshape(-1, 2)

        rays = np.concatenate([und, np.ones((und.shape[0], 1), dtype=np.float32)], axis=1)
        denom = laser[0] * rays[:, 0] + laser[1] * rays[:, 1] + laser[2] * rays[:, 2]
        good = np.isfinite(denom) & (np.abs(denom) > 1e-6)
        if not np.any(good):
            continue

        t = -laser[3] / denom
        good &= np.isfinite(t) & (t > 0.0)
        if not np.any(good):
            continue

        rays_g = rays[good]
        t_g = t[good].reshape(-1, 1)
        pts3 = rays_g * t_g

        height = (bed[0] * pts3[:, 0] + bed[1] * pts3[:, 1] + bed[2] * pts3[:, 2] + bed[3]) / bed_n_norm

        out_idx = row_idx[good].astype(np.int32)
        out[i, out_idx] = height.astype(np.float32)

    return out


def _height_range_per_frame(height_map: np.ndarray) -> np.ndarray:
    """Return robust per-frame height ranges (95-5 percentile)."""
    ranges: List[float] = []
    for i in range(int(height_map.shape[0])):
        vals = height_map[i]
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            continue
        lo = float(np.percentile(vals, 5.0))
        hi = float(np.percentile(vals, 95.0))
        if np.isfinite(lo) and np.isfinite(hi):
            ranges.append(hi - lo)
    return np.asarray(ranges, dtype=np.float32)


__all__ = [
    "VideoAnalyzer",
]

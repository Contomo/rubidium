# rubidium/analysis/analyzer.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2

from .image_processing import (
    CropConfig,
    LaserExtractConfig,
    build_laser_pipeline,
    run_pipeline,
)
from .scoring import ScoreBreakdown, score_heightmap
from .visualization import render_heightmap_plot, create_debug_thumbnails, save_dashboard


@dataclass(slots=True)
class CameraCalibration:
    K: np.ndarray
    dist: np.ndarray


@dataclass(slots=True)
class TriangulationConfig:
    enabled: bool = False
    camera_calibration_path: Optional[str] = None
    laser_plane_abcd: Optional[Tuple[float, float, float, float]] = None
    bed_plane_abcd: Optional[Tuple[float, float, float, float]] = None


@dataclass(slots=True)
class AnalysisConfig:
    crop: CropConfig = field(default_factory=CropConfig)
    laser: LaserExtractConfig = field(default_factory=LaserExtractConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)

    frame_step: int = 1
    max_frames: int = 0
    write_plots: bool = False
    write_npz: bool = True
    output_dir: Optional[str] = None

    # NEW: optional pipeline ordering by step names
    pipeline_steps: Optional[List[str]] = None


@dataclass(slots=True)
class LineAnalysis:
    video_path: Path
    idx: int
    pa: float
    breakdown: ScoreBreakdown
    height_map_kind: str
    height_map: Optional[np.ndarray] = None
    thumb_crop: Optional[np.ndarray] = None
    thumb_mask: Optional[np.ndarray] = None
    thumb_track: Optional[np.ndarray] = None
    plot_path: Optional[Path] = None
    npz_path: Optional[Path] = None
    ok: bool = True


@dataclass(slots=True)
class AnalysisSummary:
    dirpath: Path
    results: List[LineAnalysis]
    best: Optional[LineAnalysis]
    summary_csv: Optional[Path]
    summary_sheet: Optional[Path]


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


def analyze_video(path: Path, idx: int, pa: float, cfg: AnalysisConfig) -> LineAnalysis:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logging.warning(f"rubidium: failed to open video: {path}")
        return LineAnalysis(path, idx, pa, ScoreBreakdown(float("inf"), float("inf"), 0.0, 1.0), "err", ok=False)

    steps = build_laser_pipeline(cfg.pipeline_steps)

    centers: List[np.ndarray] = []
    frames = 0
    kept = 0
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    debug_frame_idx = max(1, total_frames_est // 2)

    t_crop, t_mask, t_track = None, None, None
    expected_h: Optional[int] = None  # centroid vector length (crop height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        is_process = (cfg.frame_step == 1) or ((frames - 1) % cfg.frame_step) == 0
        is_debug = (frames == debug_frame_idx)
        if not is_process and not is_debug:
            continue

        ctx = run_pipeline(
            frame,
            steps,
            cfg_crop=cfg.crop,
            cfg_laser=cfg.laser,
            keep_debug=False,
        )

        cropped = ctx.cropped_bgr if ctx.cropped_bgr is not None else frame
        mask = ctx.mask_u8
        cx = ctx.centroid_x

        if cx is not None:
            if expected_h is None:
                expected_h = int(cx.shape[0])
            elif int(cx.shape[0]) != expected_h:
                # if crop height changes unexpectedly, skip this frame
                continue

        if is_process:
            if cx is not None:
                centers.append(cx)
                kept += 1

        if is_debug:
            if mask is not None and cx is not None:
                try:
                    t_crop, t_mask, t_track = create_debug_thumbnails(cropped, mask, cx)
                except Exception:
                    pass

        if cfg.max_frames and kept >= cfg.max_frames:
            break

    cap.release()

    if not centers:
        return LineAnalysis(path, idx, pa, ScoreBreakdown(float("inf"), float("inf"), 0.0, 1.0), "empty", ok=False)

    center_map = np.stack(centers, axis=0).astype(np.float32)
    baseline = np.nanmedian(center_map, axis=0)
    height_map = center_map - baseline[None, :]

    breakdown = score_heightmap(height_map)

    outdir = Path(cfg.output_dir) if cfg.output_dir else (path.parent / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    plot_path: Optional[Path] = None
    if cfg.write_plots:
        try:
            plot_path = outdir / f"line_{idx:03d}_pa_{pa:.5f}.png"
            img = render_heightmap_plot(height_map, width_px=800, height_px=300, limit_scale=None)
            cv2.imwrite(str(plot_path), img)
        except Exception:
            pass

    npz_path: Optional[Path] = None
    if cfg.write_npz:
        try:
            npz_path = outdir / f"line_{idx:03d}_pa_{pa:.5f}.npz"
            step_names = np.array([s.name for s in steps], dtype=object)
            np.savez_compressed(
                str(npz_path),
                height_map=height_map,
                center_map=center_map,
                score=breakdown.score,
                steps=step_names,
            )
        except Exception:
            pass

    return LineAnalysis(
        video_path=path,
        idx=idx,
        pa=pa,
        breakdown=breakdown,
        height_map_kind="pixel",
        height_map=height_map,
        thumb_crop=t_crop,
        thumb_mask=t_mask,
        thumb_track=t_track,
        plot_path=plot_path,
        npz_path=npz_path,
        ok=True,
    )


def analyze_session_json(json_path: Path, cfg: AnalysisConfig) -> AnalysisSummary:
    if not json_path.exists():
        raise FileNotFoundError(f"Session not found: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed parsing JSON: {e}")

    session_dir = json_path.parent
    clips = data.get("clips", [])

    outdir = Path(cfg.output_dir) if cfg.output_dir else (session_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(outdir)

    results: List[LineAnalysis] = []

    for clip in clips:
        if clip.get("ok") is False:
            continue

        fname = Path(clip.get("file", ""))
        if not fname.name:
            continue
        vid_path = fname if fname.is_absolute() else session_dir / fname.name
        if not vid_path.exists():
            continue

        idx = int(clip.get("idx", -1))
        pa = float(clip.get("parameter_value", 0.0))

        res = analyze_video(vid_path, idx, pa, cfg)
        if res.ok:
            results.append(res)

    results.sort(key=lambda r: r.pa)
    best = min(results, key=lambda r: r.breakdown.score) if results else None

    csv_path: Optional[Path] = outdir / "summary.csv"
    try:
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "pa", "score", "roughness", "transient", "dropouts", "pk_pk_px"])
            for r in results:
                pk_pk = 0.0
                if r.height_map is not None:
                    vals = r.height_map[np.isfinite(r.height_map)]
                    if vals.size > 0:
                        pk_pk = float(np.max(vals) - np.min(vals))
                w.writerow([r.idx, r.pa, r.breakdown.score, r.breakdown.roughness, r.breakdown.transient, r.breakdown.dropouts, pk_pk])
    except Exception:
        csv_path = None

    sheet_path: Optional[Path] = outdir / "analysis_dashboard.jpg"
    try:
        save_dashboard(results, sheet_path)
    except Exception as e:
        logging.error(f"Failed to save dashboard: {e}")
        sheet_path = None
        
    return AnalysisSummary(session_dir, results, best, csv_path, sheet_path)

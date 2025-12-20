# rubidium/analysis/analyzer.py
"""Video analysis routines for Rubidium"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .image_processing import (
    CropConfig,
    LaserExtractConfig,
    crop_frame,
    build_laser_mask,
    masked_gray,
    stripe_centroid_per_row,
)
from .scoring import ScoreBreakdown, score_heightmap


@dataclass(slots=True)
class CameraCalibration:
    K: np.ndarray
    dist: np.ndarray


@dataclass(slots=True)
class TriangulationConfig:
    enabled: bool = False
    camera_calibration_path: Optional[str] = None
    laser_plane_abcd: Optional[tuple[float, float, float, float]] = None
    bed_plane_abcd: Optional[tuple[float, float, float, float]] = None


@dataclass(slots=True)
class AnalysisConfig:
    crop: CropConfig = CropConfig()
    laser: LaserExtractConfig = LaserExtractConfig()

    frame_step: int = 1
    max_frames: int = 0  # 0 => all

    triangulation: TriangulationConfig = TriangulationConfig()

    write_plots: bool = True
    write_npz: bool = True

    output_dir: Optional[str] = None


@dataclass(slots=True)
class LineAnalysis:
    video_path: Path
    idx: int
    pa: float
    breakdown: ScoreBreakdown
    height_map_kind: str  # "pixel" or "triangulate"
    plot_path: Optional[Path]
    npz_path: Optional[Path]


@dataclass(slots=True)
class AnalysisSummary:
    dirpath: Path
    results: list[LineAnalysis]
    best: Optional[LineAnalysis]
    summary_csv: Optional[Path]
    summary_plot: Optional[Path]


_pa_re = re.compile(r"pa_([-+]?[0-9]*\.?[0-9]+)")
_idx_re = re.compile(r"rubedo_line_(\d+)")


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("rubidium: cv2 not available in this python env")


def _parse_plane_abcd(abcd: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
    n = np.asarray([abcd[0], abcd[1], abcd[2]], dtype=np.float32)
    d = float(abcd[3])
    nn = float(np.linalg.norm(n))
    if nn <= 0.0:
        raise ValueError("invalid plane normal")
    n = n / nn
    d = d / nn
    return n, d


def _load_camera_calibration(path: Path) -> CameraCalibration:
    _ensure_cv2()

    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        K = np.asarray(data["K"], dtype=np.float32)
        dist = np.asarray(data["dist"], dtype=np.float32)
        return CameraCalibration(K=K, dist=dist)

    # OpenCV YAML/XML via FileStorage
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"rubidium: failed to read camera calibration file: {path}")

    K = fs.getNode("K").mat()
    if K is None or K.size == 0:
        K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist").mat()
    if dist is None or dist.size == 0:
        dist = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if K is None or dist is None:
        raise RuntimeError(f"rubidium: calibration file missing K/dist: {path}")

    return CameraCalibration(K=np.asarray(K, dtype=np.float32), dist=np.asarray(dist, dtype=np.float32))


def _triangulate_points(
    u_px: np.ndarray,
    v_px: np.ndarray,
    calib: CameraCalibration,
    laser_plane_abcd: tuple[float, float, float, float],
) -> np.ndarray:
    """Intersect camera rays with the laser plane.

    Returns Nx3 points in camera coordinates.
    """
    _ensure_cv2()

    n, d = _parse_plane_abcd(laser_plane_abcd)

    pts = np.stack([u_px, v_px], axis=1).reshape((-1, 1, 2)).astype(np.float32)
    und = cv2.undistortPoints(pts, calib.K, calib.dist)
    xy = und.reshape((-1, 2)).astype(np.float32)

    dirs = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)
    denom = dirs @ n

    # Avoid division by near-zero; those rays are effectively parallel to the plane.
    eps = 1e-9
    good = np.abs(denom) > eps

    out = np.full((dirs.shape[0], 3), np.nan, dtype=np.float32)
    if not np.any(good):
        return out

    t = (-d) / denom[good]
    out[good] = dirs[good] * t[:, None]
    return out


def _height_from_points(
    pts_cam: np.ndarray,
    bed_plane_abcd: Optional[tuple[float, float, float, float]],
) -> np.ndarray:
    """Convert 3D points to a scalar height.

    If bed_plane is available: signed distance to that plane.
    Otherwise: use camera Z as a fallback (still useful for relative scoring).
    """
    if bed_plane_abcd is None:
        return pts_cam[:, 2].astype(np.float32)

    n, d = _parse_plane_abcd(bed_plane_abcd)
    return (pts_cam @ n + d).astype(np.float32)


def _plot_heightmap(height_map: np.ndarray, out_png: Path, *, title: str) -> None:
    # Imported lazily: Klipper envs vary.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm = height_map
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlabel("bin")
    ax.set_ylabel("frame")

    vmin = float(np.nanpercentile(hm, 5))
    vmax = float(np.nanpercentile(hm, 95))
    ax.imshow(hm, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def analyze_video(path: Path, cfg: AnalysisConfig) -> LineAnalysis:
    """Analyse a single video file and return a score + artifacts."""
    _ensure_cv2()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"rubidium: failed to open video: {path}")

    centers: list[np.ndarray] = []
    frames = 0
    kept = 0

    # Store crop origin so we can rebuild absolute pixels for triangulation.
    crop_x0 = 0
    crop_y0 = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if cfg.frame_step > 1 and ((frames - 1) % cfg.frame_step) != 0:
            continue

        cropped, (x0, y0) = crop_frame(frame, cfg.crop)
        crop_x0, crop_y0 = x0, y0

        mask = build_laser_mask(cropped, cfg.laser)
        gray = masked_gray(cropped, mask)
        cx = stripe_centroid_per_row(gray, cfg.laser)
        centers.append(cx)

        kept += 1
        if cfg.max_frames and kept >= cfg.max_frames:
            break

    cap.release()

    if not centers:
        bd = ScoreBreakdown(score=float("inf"), roughness=float("inf"), transient=0.0, dropouts=1.0)
        return LineAnalysis(path, idx=-1, pa=0.0, breakdown=bd, height_map_kind="pixel", plot_path=None, npz_path=None)

    center_map = np.stack(centers, axis=0).astype(np.float32)  # (frames, rows)

    # Baseline remove: subtract per-row median.
    baseline = np.nanmedian(center_map, axis=0)
    delta_px = center_map - baseline[None, :]

    hm_kind = "pixel"
    height_map = delta_px

    # Optional triangulation.
    tri = cfg.triangulation
    if tri.enabled:
        if tri.camera_calibration_path is None or tri.laser_plane_abcd is None:
            raise RuntimeError("rubidium: triangulation enabled but camera_calibration_path / laser_plane_abcd is missing")

        calib = _load_camera_calibration(Path(tri.camera_calibration_path))
        laser_plane = tri.laser_plane_abcd
        bed_plane = tri.bed_plane_abcd

        h = center_map.shape[1]
        ys = np.arange(h, dtype=np.float32)

        heights: list[np.ndarray] = []
        for cx in center_map:
            good = np.isfinite(cx)
            if not np.any(good):
                heights.append(np.full((h,), np.nan, dtype=np.float32))
                continue

            u = (cx[good] + float(crop_x0)).astype(np.float32)
            v = (ys[good] + float(crop_y0)).astype(np.float32)

            pts = _triangulate_points(u, v, calib, laser_plane)
            z = _height_from_points(pts, bed_plane)

            row_h = np.full((h,), np.nan, dtype=np.float32)
            row_h[good] = z
            heights.append(row_h)

        height_map = np.stack(heights, axis=0).astype(np.float32)

        # Baseline remove: subtract per-row median again (in whatever height units).
        b2 = np.nanmedian(height_map, axis=0)
        height_map = height_map - b2[None, :]

        hm_kind = "triangulate"

    breakdown = score_heightmap(height_map)

    # Parse idx + pa from filename.
    idx = -1
    pa = 0.0
    mi = _idx_re.search(path.name)
    if mi:
        try:
            idx = int(mi.group(1))
        except Exception:
            idx = -1
    mp = _pa_re.search(path.name)
    if mp:
        try:
            pa = float(mp.group(1))
        except Exception:
            pa = 0.0

    outdir = Path(cfg.output_dir) if cfg.output_dir else (path.parent / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    plot_path: Optional[Path] = None
    if cfg.write_plots:
        try:
            plot_path = outdir / f"analysis_line_{idx:03d}_pa_{pa:.5f}.png"
            _plot_heightmap(height_map, plot_path, title=f"line {idx:03d}  pa={pa:.5f}  score={breakdown.score:.3f}")
        except Exception:
            plot_path = None

    npz_path: Optional[Path] = None
    if cfg.write_npz:
        try:
            npz_path = outdir / f"analysis_line_{idx:03d}_pa_{pa:.5f}.npz"
            np.savez_compressed(
                str(npz_path),
                height_map=height_map,
                center_map=center_map,
                baseline=baseline,
                score=np.float32(breakdown.score),
                roughness=np.float32(breakdown.roughness),
                transient=np.float32(breakdown.transient),
                dropouts=np.float32(breakdown.dropouts),
                kind=np.array([hm_kind]),
            )
        except Exception:
            npz_path = None

    return LineAnalysis(
        video_path=path,
        idx=idx,
        pa=pa,
        breakdown=breakdown,
        height_map_kind=hm_kind,
        plot_path=plot_path,
        npz_path=npz_path,
    )


def _plot_summary(results: list[LineAnalysis], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.pa for r in results]
    ys = [r.breakdown.score for r in results]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Rubidium analysis: score vs PA")
    ax.set_xlabel("pressure advance")
    ax.set_ylabel("score (lower is better)")
    ax.plot(xs, ys, marker="o")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def analyze_directory(dirpath: Path, cfg: Optional[AnalysisConfig] = None) -> AnalysisSummary:
    """Analyse all matching video files in a directory."""
    cfg = cfg or AnalysisConfig()

    videos = sorted(dirpath.glob("rubedo_line_*_pa_*.mp4"))
    if not videos:
        return AnalysisSummary(dirpath=dirpath, results=[], best=None, summary_csv=None, summary_plot=None)

    outdir = Path(cfg.output_dir) if cfg.output_dir else (dirpath / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[LineAnalysis] = []
    for vp in videos:
        results.append(analyze_video(vp, cfg))

    # Sort by PA for consistent plots.
    results.sort(key=lambda r: r.pa)

    best = min(results, key=lambda r: r.breakdown.score) if results else None

    # Write summary CSV.
    summary_csv = outdir / "analysis_summary.csv"
    try:
        with summary_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "pa", "score", "roughness", "transient", "dropouts", "kind", "video", "plot", "npz"])
            for r in results:
                w.writerow(
                    [
                        r.idx,
                        f"{r.pa:.6f}",
                        f"{r.breakdown.score:.6f}",
                        f"{r.breakdown.roughness:.6f}",
                        f"{r.breakdown.transient:.6f}",
                        f"{r.breakdown.dropouts:.6f}",
                        r.height_map_kind,
                        r.video_path.name,
                        r.plot_path.name if r.plot_path else "",
                        r.npz_path.name if r.npz_path else "",
                    ]
                )
    except Exception:
        summary_csv = None

    summary_plot = outdir / "analysis_summary.png"
    try:
        _plot_summary(results, summary_plot)
    except Exception:
        summary_plot = None

    # Write best PA text.
    if best is not None:
        try:
            (outdir / "best_pa.txt").write_text(f"{best.pa:.6f}\n")
        except Exception:
            pass

    return AnalysisSummary(
        dirpath=dirpath,
        results=results,
        best=best,
        summary_csv=summary_csv,
        summary_plot=summary_plot,
    )

# rubidium/analysis/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .image_processing import CropConfig, LaserExtractConfig
from .scoring import ScoreBreakdown


_BUMP_MIN_PX = 1.0
_BUMP_FAIL_SCORE = 3.0


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
    min_valid_frac: float = 0.0
    min_height_range: float = 0.0
    gate_fail_score: float = 3.0


@dataclass(slots=True)
class AutoCropConfig:
    """Session-wide crop-center prepass."""

    enable: bool = False
    # None means "use full frame".
    search_wh: Optional[Tuple[int, int]] = None
    samples_per_clip: int = 1
    max_samples: int = 0  # 0 -> no explicit cap
    keep_percentile: float = 50.0
    min_kept: int = 8


@dataclass(slots=True)
class AnalysisConfig:
    crop: CropConfig = field(default_factory=CropConfig)
    laser: LaserExtractConfig = field(default_factory=LaserExtractConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    autocrop: AutoCropConfig = field(default_factory=AutoCropConfig)

    frame_step: int = 1
    max_frames: int = 0
    write_plots: bool = False
    write_npz: bool = True
    output_dir: Optional[str] = None
    pipeline_steps: Optional[List[str]] = None
    score_trim_frac: float = 0.10


@dataclass(slots=True)
class LineAnalysis:
    video_path: Path
    idx: int
    pa: float
    breakdown: ScoreBreakdown
    height_map_kind: str

    # grid patterns (optional)
    pa2: Optional[float] = None
    grid_row: Optional[int] = None
    grid_col: Optional[int] = None

    height_map: Optional[np.ndarray] = None
    thumb_crop: Optional[np.ndarray] = None
    thumb_mask: Optional[np.ndarray] = None
    thumb_track: Optional[np.ndarray] = None
    plot_path: Optional[Path] = None
    npz_path: Optional[Path] = None
    ok: bool = True
    bump_px: Optional[float] = None


@dataclass(slots=True)
class AnalysisSummary:
    dirpath: Path
    results: List[LineAnalysis]
    best: Optional[LineAnalysis]
    summary_csv: Optional[Path]
    summary_sheet: Optional[Path]
    warnings: List[str] = field(default_factory=list)


__all__ = [
    "_BUMP_MIN_PX",
    "_BUMP_FAIL_SCORE",
    "CameraCalibration",
    "TriangulationConfig",
    "AutoCropConfig",
    "AnalysisConfig",
    "LineAnalysis",
    "AnalysisSummary",
]

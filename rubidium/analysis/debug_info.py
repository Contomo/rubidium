# rubidium/analysis/debug_info.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(slots=True)
class CropDebug:
    center_xy: Tuple[float, float]
    wh: Tuple[int, int]
    ref_wh: Optional[Tuple[int, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "wh": [int(self.wh[0]), int(self.wh[1])],
            "ref_wh": list(self.ref_wh) if self.ref_wh is not None else None,
        }


@dataclass(slots=True)
class SessionCounts:
    clips_total: int
    clips_flag_ok: int
    clips_missing_name: int
    clips_missing_file: int
    clips_analyzed: int
    results_ok: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clips_total": int(self.clips_total),
            "clips_flag_ok": int(self.clips_flag_ok),
            "clips_missing_name": int(self.clips_missing_name),
            "clips_missing_file": int(self.clips_missing_file),
            "clips_analyzed": int(self.clips_analyzed),
            "results_ok": int(self.results_ok),
        }


@dataclass(slots=True)
class AutoCropDebug:
    enable: bool
    search_wh: Optional[Tuple[int, int]]
    samples_per_clip: int
    max_samples: int
    keep_percentile: float
    min_kept: int
    # extra runtime signals (optional)
    rejects: Optional[Dict[str, int]] = None
    stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "enable": bool(self.enable),
            "search_wh": list(self.search_wh) if self.search_wh is not None else None,
            "samples_per_clip": int(self.samples_per_clip),
            "max_samples": int(self.max_samples),
            "keep_percentile": float(self.keep_percentile),
            "min_kept": int(self.min_kept),
        }
        if self.rejects:
            d["rejects"] = dict(self.rejects)
        if self.stats:
            d["stats"] = dict(self.stats)
        return d


@dataclass(slots=True)
class TriangulationDebug:
    enabled: bool
    camera_calibration_path: Optional[str]
    laser_plane_abcd: Optional[Tuple[float, float, float, float]]
    bed_plane_abcd: Optional[Tuple[float, float, float, float]]
    min_valid_frac: float
    min_height_range: float
    gate_fail_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "camera_calibration_path": self.camera_calibration_path,
            "laser_plane_abcd": list(self.laser_plane_abcd) if self.laser_plane_abcd is not None else None,
            "bed_plane_abcd": list(self.bed_plane_abcd) if self.bed_plane_abcd is not None else None,
            "min_valid_frac": float(self.min_valid_frac),
            "min_height_range": float(self.min_height_range),
            "gate_fail_score": float(self.gate_fail_score),
        }


@dataclass(slots=True)
class SessionDebug:
    session: str
    output_dir: str
    warnings: List[str]
    counts: SessionCounts
    crop_initial: CropDebug
    crop_final: CropDebug
    autocrop: AutoCropDebug
    pipeline_steps: List[str]
    triangulation: TriangulationDebug

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": str(self.session),
            "output_dir": str(self.output_dir),
            "warnings": list(self.warnings),
            "counts": self.counts.to_dict(),
            "crop_initial": self.crop_initial.to_dict(),
            "crop_final": self.crop_final.to_dict(),
            "autocrop": self.autocrop.to_dict(),
            "pipeline_steps": list(self.pipeline_steps),
            "triangulation": self.triangulation.to_dict(),
        }


__all__ = [
    "AutoCropDebug",
    "CropDebug",
    "SessionCounts",
    "SessionDebug",
    "TriangulationDebug",
]

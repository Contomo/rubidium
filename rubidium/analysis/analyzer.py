# rubidium/analysis/analyzer.py
"""Public analysis entrypoints.

This module intentionally stays small:
  * config/data models live in types.py
  * clip parsing lives in clips.py
  * autocrop prepass lives in autocrop.py
  * per-video analysis lives in video_analyzer.py
  * session orchestration lives in session_analyzer.py

Older code imported dataclasses from this module, so we re-export them.
"""

from __future__ import annotations

from pathlib import Path

from .session_analyzer import SessionAnalyzer
from .types import (
    AnalysisConfig,
    AnalysisSummary,
    AutoCropConfig,
    CameraCalibration,
    LineAnalysis,
    TriangulationConfig,
)


def analyze_session_json(json_path: Path, cfg: AnalysisConfig) -> AnalysisSummary:
    return SessionAnalyzer(cfg).analyze_session_json(json_path)


__all__ = [
    "AnalysisConfig",
    "AnalysisSummary",
    "AutoCropConfig",
    "CameraCalibration",
    "LineAnalysis",
    "TriangulationConfig",
    "analyze_session_json",
]

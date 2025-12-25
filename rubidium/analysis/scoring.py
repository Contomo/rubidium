# rubidium/analysis/scoring.py
"""Scoring functions for Rubidium"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ScoreBreakdown:
    score: float
    roughness: float
    transient: float
    dropouts: float


def _nanmean(x: np.ndarray) -> float:
    v = np.nanmean(x)
    return float(v) if np.isfinite(v) else 0.0


def score_heightmap(
    height_map: np.ndarray,
    *,
    edge_frames: int = 10,
) -> ScoreBreakdown:
    """Compute a scalar score from a (frames x bins) height map."""
    if height_map.ndim != 2:
        raise ValueError("height_map must be 2D")

    frames = int(height_map.shape[0])
    if frames < 3:
        return ScoreBreakdown(score=0.0, roughness=0.0, transient=0.0, dropouts=0.0)

    hm = height_map.astype(np.float32)

    drop_frac = float(np.mean(~np.isfinite(hm)))
    dropouts = 10.0 * drop_frac

    a = max(0, int(frames * 0.10))
    b = min(frames, int(frames * 0.90))
    mid = hm[a:b] if (b - a) >= 2 else hm

    roughness = _nanmean(np.nanstd(mid, axis=0))

    ef = int(max(1, min(edge_frames, frames // 3)))
    start = hm[:ef]
    end = hm[-ef:]

    mid_mean = np.nanmean(mid, axis=0)
    start_mean = np.nanmean(start, axis=0)
    end_mean = np.nanmean(end, axis=0)

    transient = _nanmean(np.abs(start_mean - mid_mean)) + _nanmean(np.abs(end_mean - mid_mean))

    score = float(roughness + 0.5 * transient + dropouts)

    return ScoreBreakdown(score=score, roughness=roughness, transient=transient, dropouts=dropouts)

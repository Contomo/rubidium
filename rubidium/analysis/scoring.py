# rubidium/analysis/scoring.py
"""Scoring functions for Rubidium"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ScoreBreakdown:
    score: float
    roughness: float
    dropouts: float


def _nanmean(x: np.ndarray) -> float:
    v = np.nanmean(x)
    return float(v) if np.isfinite(v) else 0.0


def score_heightmap(
    height_map: np.ndarray,
    *,
    trim_frac: float = 0.10,
) -> ScoreBreakdown:
    """Compute a scalar score from a (frames x bins) height map."""
    if height_map.ndim != 2:
        raise ValueError("height_map must be 2D")

    frames = int(height_map.shape[0])
    if frames < 3:
        return ScoreBreakdown(score=0.0, roughness=0.0, dropouts=0.0)

    hm = height_map.astype(np.float32)
    tf = float(np.clip(trim_frac, 0.0, 0.45))
    trim_n = int(round(frames * tf))
    if trim_n * 2 >= (frames - 2):
        trim_n = max(0, (frames - 2) // 2)
    if trim_n > 0:
        hm = hm[trim_n : frames - trim_n]
        frames = int(hm.shape[0])
        if frames < 3:
            return ScoreBreakdown(score=0.0, roughness=0.0, dropouts=0.0)

    drop_frac = float(np.mean(~np.isfinite(hm)))
    dropouts = 10.0 * drop_frac

    roughness = _nanmean(np.nanstd(hm, axis=0))
    score = float(roughness + dropouts)

    return ScoreBreakdown(score=score, roughness=roughness, dropouts=dropouts)

# rubidium/analysis/clips.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def derive_mirror_x(clip: Dict[str, Any]) -> bool:
    """Infer whether to mirror a frame based on motion direction.

    No explicit flag is stored; we infer from raw_start/raw_end coordinates.
    For horizontal line scans this mirrors when travel is in -X.
    """
    rs = clip.get("raw_start") or clip.get("buf_start")
    re_ = clip.get("raw_end") or clip.get("buf_end")
    if not (
        isinstance(rs, (list, tuple))
        and isinstance(re_, (list, tuple))
        and len(rs) >= 2
        and len(re_) >= 2
    ):
        return False
    try:
        return float(re_[0]) < float(rs[0])
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class ClipInfo:
    idx: int
    pa: float
    path: Path
    mirror_x: bool
    pa2: Optional[float] = None
    grid_row: Optional[int] = None
    grid_col: Optional[int] = None


@dataclass(slots=True)
class ClipCounts:
    clips_total: int = 0
    clips_flag_ok: int = 0
    clips_missing_name: int = 0
    clips_missing_file: int = 0


def parse_session_clips(session_dir: Path, clips: Iterable[Dict[str, Any]]) -> Tuple[List[ClipInfo], ClipCounts]:
    """Parse raw session JSON clip entries into normalized ClipInfo objects."""
    out: List[ClipInfo] = []
    counts = ClipCounts()

    for clip in clips:
        counts.clips_total += 1
        if clip.get("ok") is False:
            continue
        counts.clips_flag_ok += 1

        raw = clip.get("file") or clip.get("path")
        if not raw:
            counts.clips_missing_name += 1
            continue

        p = Path(str(raw))
        vid_path = p if p.is_absolute() else (session_dir / p.name)
        if not vid_path.exists():
            counts.clips_missing_file += 1
            continue

        idx = int(clip.get("idx", -1))
        pa = float(clip.get("parameter_value", 0.0))
        pa2 = _opt_float(clip.get("parameter_value2", None))

        grid_row = _opt_int(clip.get("grid_row", None))
        grid_col = _opt_int(clip.get("grid_col", None))
        mirror_x = derive_mirror_x(clip)

        out.append(
            ClipInfo(
                idx=idx,
                pa=pa,
                pa2=pa2,
                grid_row=grid_row,
                grid_col=grid_col,
                path=vid_path,
                mirror_x=mirror_x,
            )
        )

    return out, counts


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


__all__ = [
    "ClipCounts",
    "ClipInfo",
    "derive_mirror_x",
    "parse_session_clips",
]

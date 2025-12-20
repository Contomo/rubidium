"""Simple geometric primitives"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Iterator
import math


@dataclass(frozen=True)
class Pt:
    """A point in three dimensional space"""
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class LineSegment:
    """A contiguous portion of a line at a single speed label"""
    start: Pt
    end: Pt
    speed_label: str # TODO: store actual speed and not label?

    def length_xy(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.hypot(dx, dy)


@dataclass
class Line:
    """A full line consisting of one or more segments."""
    idx: int
    pa_value: float
    start: Pt
    end: Pt
    segments: Tuple[LineSegment, ...]
    
    def length_xy(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.hypot(dx, dy)

@dataclass
class Lines:
    """A container for many lines"""
    lines: Tuple[Line, ...]

    def __iter__(self) -> Iterator[Line]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Line:
        return self.lines[idx]

    def outline_and_center_xy(self) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        """Return a bounding polygon and its centre in the XY plane"""
        if not self.lines:
            return [], (0.0, 0.0)
        xs: List[float] = []
        ys: List[float] = []
        for ln in self.lines:
            xs.extend([ln.start.x, ln.end.x])
            ys.extend([ln.start.y, ln.end.y])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        outline = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        centre = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        return outline, centre

    def points_xy(self) -> List[Tuple[float, float]]:
        """Return all segment endpoints as (x, y) tuples."""
        pts: List[Tuple[float, float]] = []
        for ln in self.lines:
            pts.append((ln.start.x, ln.start.y))
            pts.append((ln.end.x, ln.end.y))
            for seg in ln.segments:
                pts.append((seg.start.x, seg.start.y))
                pts.append((seg.end.x, seg.end.y))
        return pts

    @staticmethod
    def _convex_hull_xy(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Compute a convex hull using the monotonic chain algorithm"""
        pts = sorted(set(points))
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        hull = lower[:-1] + upper[:-1]
        return hull

    def convex_hull_xy(self) -> List[Tuple[float, float]]:
        """Return the convex hull of all points in the XY plane."""
        if not self.lines:
            return []
        return self._convex_hull_xy(self.points_xy())

    def rotated_xy(self, angle_radians: float, *, origin: Tuple[float, float] | None = None) -> "Lines":
        """Return a new Lines rotated in the XY plane by angle_radians.

        Rotation is performed around origin. If origin is None, the current
        outline center is used.
        """
        if not self.lines:
            return self
        if origin is None:
            _, origin = self.outline_and_center_xy()
        ox, oy = origin
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)

        def r_pt(p: Pt) -> Pt:
            x = p.x - ox
            y = p.y - oy
            return Pt(x=ox + x * c - y * s, y=oy + x * s + y * c, z=p.z)

        new_lines: List[Line] = []
        for ln in self.lines:
            new_segments: List[LineSegment] = []
            for seg in ln.segments:
                new_segments.append(
                    LineSegment(
                        speed_label=seg.speed_label,
                        start=r_pt(seg.start),
                        end=r_pt(seg.end),
                    )
                )
            new_lines.append(
                Line(
                    idx=ln.idx,
                    pa_value=ln.pa_value,
                    start=r_pt(ln.start),
                    end=r_pt(ln.end),
                    segments=tuple(new_segments),
                )
            )
        return Lines(lines=tuple(new_lines))

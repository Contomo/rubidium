"""Scanning mode for Rubidium"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from ..core.base import RubidiumBase
from ..core.lines import Lines, Line, Pt  # type: ignore
from ..analysis.graphing import RubidiumPaths
from ..video.video_input import VideoInput


class RubidiumScan(RubidiumBase):
    """Handler class for the RUBIDIUM_SCAN command"""

    cmd_help = "Stream scan moves and optionally record per-line videos via ffmpeg"

    def __init__(self, config) -> None:
        super().__init__(
            config,
            register_name="SCAN",
            provider_name="Rubidium scan",
        )
        self.scan_speed  = self._get("scan_speed", self.v_travel, method="getfloat", above=0.0)
        self.scan_buffer = self._get("scan_buffer", 0.0, method="getfloat", minval=0.0)
        self.graphing    = RubidiumPaths(self.printer, base_section=self.base_section, section=self.section)
        self.video       = VideoInput(self.printer, base_section=self.base_section, section=self.section)
        self._outdir     = Path(self.graphing.dirs.scan)

    def _on_connect(self) -> None:
        super()._on_connect()
        self.graphing.ensure_dirs()

    def _extra_settings_from(self, gcmd) -> Dict[str, Any]:
        return {
            "scan_speed":  gcmd.get_float("SCAN_SPEED",  self.scan_speed, above=0.0),
            "scan_buffer": gcmd.get_float("SCAN_BUFFER", self.scan_buffer, minval=0.0),
        }

    def handle_shutdown(self) -> None:
        try:
            self.video.stop_session(finalize=False)
        except Exception:
            pass

    def _iter_run(self):
        return self._iter_scan()

    def _iter_scan(self):
        """Yield G-Code lines to perform a scan along the lines."""
        r = self
        s = self._run_settings or {}
        lines = self._run_lines or Lines(tuple())
        n = len(lines.lines)

        for ln in r._render_template_lines(
            self.templates.start, 
            r._tmpl_ctx(mode="scan", s=s), 
            gcmd=self._run_gcmd
        ):
            yield ln
        yield "G90"

        self.video.start_session(self._outdir)

        travel_speed = float(s.get("travel_speed", r.v_travel))
        scan_speed = float(s.get("scan_speed", r.v_travel))
        scan_buffer = float(s.get("scan_buffer", 0.0))
        travel_f = travel_speed * 60.0
        scan_f = scan_speed * 60.0
        offx, offy, offz = self.offset_x, self.offset_y, self.offset_z
        # Iterate over lines
        for i, pl in enumerate(lines):
            self.progress = i / max(1, n)
            sx, sy, sz = pl.start.x + offx, pl.start.y + offy, pl.start.z + offz
            ex, ey     = pl.end.x + offx,   pl.end.y + offy
            dx = ex - sx
            dy = ey - sy
            dist = math.hypot(dx, dy)
            if dist <= 1e-9:
                continue
            ux = dx / dist
            uy = dy / dist
            buf = min(scan_buffer, 0.49 * dist)
            bsx = sx + ux * buf
            bsy = sy + uy * buf
            bex = ex - ux * buf
            bey = ey - uy * buf
            scan_ctx = r._tmpl_ctx(
                mode="scan",
                s=s,
                line=pl,
                scan={
                    "buf_start": Pt(bsx, bsy, sz),
                    "buf_end":   Pt(bex, bey, sz),
                    "raw_start": Pt(sx, sy, sz),
                    "raw_end":   Pt(ex, ey, sz),
                },
            )
            for ln in r._render_template_lines(
                self.templates.before_line, 
                scan_ctx, 
                gcmd=self._run_gcmd
            ):
                yield ln

            yield f"G0 X{bsx:.3f} Y{bsy:.3f} Z{sz:.3f} F{travel_f:.1f}"
            yield f"RUBIDIUM_VIDEO_MARK KEY=line_{getattr(pl, 'idx', i):03d} KIND=start IDX={getattr(pl, 'idx', i)} PA={getattr(pl, 'pa_value', 0.0):.9f}"
            yield f"G1 X{bex:.3f} Y{bey:.3f} F{scan_f:.1f}"
            yield f"RUBIDIUM_VIDEO_MARK KEY=line_{getattr(pl, 'idx', i):03d} KIND=end IDX={getattr(pl, 'idx', i)} PA={getattr(pl, 'pa_value', 0.0):.9f}"

            for ln in r._render_template_lines(
                self.templates.after_line, 
                scan_ctx, 
                gcmd=self._run_gcmd
            ):
                yield ln

        self.progress = 1.0

        # Stop recording and cut per-line clips
        self.video.stop_session(finalize=True)

        for ln in r._render_template_lines(
            self.templates.end, 
            r._tmpl_ctx(mode="scan", s=s), 
            gcmd=self._run_gcmd
        ):
            yield ln

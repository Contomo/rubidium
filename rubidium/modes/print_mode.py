"""Printing mode for the Rubidium Klipper extra."""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.base import RubidiumBase
from ..core.lines import Lines, LineSegment, Line, Pt  # type: ignore
from ..patterns.patterns import RubidiumPatternObject

# Constants used when defining exclude objects
EXCLUDE_OBJECT_MARGIN: float = 1.0


def _tokenize(raw: str) -> List[str]:
    return str(raw).translate(str.maketrans("=:,", "   ")).split()


def parse_speeds(raw: str) -> Dict[str, float]:
    tokens = _tokenize(raw)
    if not tokens:
        return {}
    if len(tokens) % 2 != 0:
        raise ValueError(f"Invalid speeds format: '{raw}'")
    out: Dict[str, float] = {}
    it = iter(tokens)
    for label, value in zip(it, it):
        out[label.strip()] = float(value)
    return out


class RubidiumPrint(RubidiumBase):
    """Handler class for the RUBIDIUM_PRINT command."""

    cmd_help = "Stream a Rubidium pattern via virtual_sdcard (printing)"

    def __init__(self, config) -> None:
        super().__init__(
            config,
            register_name="PRINT",
            provider_name="Rubidium print",
        )
        self.flow_default: float = self._get(
            "flow",          1.0,   method="getfloat", above=0.0
        ) # type: ignore
        self.print_speed: float = self._get(
            "print_speed",   25.0,  method="getfloat", above=0.0
        ) # type: ignore
        self.brim_walls: int = self._get(
            "brim_walls",     2,    method="getint",   minval=0
        ) # type: ignore
        self.brim_overlap: float = self._get(
            "brim_overlap",   0.0,  method="getfloat", minval=0.0, maxval=100.0
        ) # type: ignore
        self.line_width: float = self._get(
            "line_width",     0.0,  method="getfloat", minval=0.0
        ) # type: ignore
        self.line_width_pct: float = self._get(
            "line_width_pct", 120,  method="getfloat", above=0.0
        ) # type: ignore
        raw_speeds = self._get("speeds", "", method="get")
        try:
            self.speeds_default = parse_speeds(raw_speeds) # type: ignore
        except Exception as e:
            raise config.error(f"Invalid speeds in [{self.section.get_name()}]: {e}") from e  # type: ignore
        
        if self.registry.lookup_pattern("default", None) is None:
            default_pattern = RubidiumPatternObject.from_config(config)
            self.printer.add_object(
                config.get_name().split(None, 1)[0] + ' pattern',
                default_pattern
            )

    # -- helper utilities ------------------------------------------------
    def _compile_param_command_fmt(self, command: str, parameter: str) -> str:
        """Build a printf format string for tuning commands"""
        if self.gcode.is_traditional_gcode(command):
            return f"{command} {parameter}%0.9f"
        return f"{command} {parameter}=%0.9f"

    def _extra_settings_from(self, gcmd) -> Dict[str, Any]:
        """Add print-only settings to the run settings dict."""
        s: Dict[str, Any] = {}

        s["flow"]           = gcmd.get_float("FLOW",           self.flow_default,   above=0.0)
        s["line_width"]     = gcmd.get_float("LINE_WIDTH",     self.line_width,     minval=0.0)
        s["line_width_pct"] = gcmd.get_float("LINE_WIDTH_PCT", self.line_width_pct, above=0.0)

        speeds_map: Dict[str, float] = dict(self.speeds_default)
        for label, base in list(speeds_map.items()):
            speeds_map[label] = gcmd.get_float('SPEED_' + label.upper(), base, above=0.0)
        s["speeds"] = speeds_map

        pat = self._run_pattern
        if pat is not None:
            cmd = str(pat.settings.get("tuning_command", "SET_PRESSURE_ADVANCE"))
            par = str(pat.settings.get("tuning_parameter", "ADVANCE"))
            s["tuning_fmt"] = self._compile_param_command_fmt(cmd, par)

            cmd2 = str(pat.settings.get("tuning_command2", "")).strip()
            par2 = str(pat.settings.get("tuning_parameter2", "")).strip()
            if cmd2 and par2:
                s["tuning_fmt2"] = self._compile_param_command_fmt(cmd2, par2)

        return s

    def _prepare_lines_for_run(self, lines: Lines, *, s: Dict[str, Any], gcmd, keepout_xy=None) -> Lines:
        """Add a rectangular multi wall brim around the pattern"""
        if not lines.lines:
            return lines
        
        speed_label = gcmd.get_float("PRINT_SPEED",  self.print_speed,  above=0.0)
        walls       = gcmd.get_int("BRIM_WALLS",     self.brim_walls,   minval=0)
        overlap_pct = gcmd.get_float("BRIM_OVERLAP", self.brim_overlap, above=0.0)

        if walls <= 0: # type: ignore
            return lines

        overlap_pct = max(0.0, min(100.0, overlap_pct)) # type: ignore

        line_w = float(self._get_line_width(s=s))
        if line_w <= 0.0:
            return lines

        sep = line_w * max(0.0, 1.0 - overlap_pct / 100.0)

        if keepout_xy is not None:
            outline = keepout_xy
        else:
            outline, _ = lines.outline_and_center_xy()
        if not outline:
            return lines

        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        z = float(lines.lines[0].start.z)

        brim_lines: List[Line] = []
        for i in range(walls): # type: ignore
            off = sep + (line_w * i)
            x0, x1 = minx - off, maxx + off
            y0, y1 = miny - off, maxy + off

            p0, p1 = Pt(x0, y0, z), Pt(x1, y0, z)
            p2, p3 = Pt(x1, y1, z), Pt(x0, y1, z)

            segs = (
                LineSegment(start=p0, end=p1, speed_label=speed_label), # type: ignore
                LineSegment(start=p1, end=p2, speed_label=speed_label), # type: ignore
                LineSegment(start=p2, end=p3, speed_label=speed_label), # type: ignore
                LineSegment(start=p3, end=p0, speed_label=speed_label), # type: ignore
            )
            brim_lines.append(
                Line(
                    idx=-(walls - i), # type: ignore
                    parameter_value=0.0,
                    parameter_value2=0.0,
                    start=p0,
                    end=p2,
                    segments=segs,
                )
            )

        return Lines(tuple(brim_lines) + lines.lines)

    def define_exclude_object(self, lines: Lines) -> str:
        """Lines → Exclude object define (closed poly)"""
        if self.printer.lookup_object("exclude_object", None) is None:
            return ""

        poly_pts, _ = lines.outline_and_center_xy()
        if not poly_pts:
            return ""

        xs = [p[0] for p in poly_pts]
        ys = [p[1] for p in poly_pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        ko = self._keepout_xy
        if ko:
            kxs, kys = [p[0] for p in ko], [p[1] for p in ko]
            minx, maxx = min(minx, min(kxs)), max(maxx, max(kxs))
            miny, maxy = min(miny, min(kys)), max(maxy, max(kys))

        m = EXCLUDE_OBJECT_MARGIN
        minx -= m; maxx += m
        miny -= m; maxy += m

        cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5

        rect = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        poly = "[" + ",".join(f"[{x:.3f},{y:.3f}]" for x, y in rect) + "]"
        return f"EXCLUDE_OBJECT_DEFINE NAME=Pattern CENTER={cx:.3f},{cy:.3f} POLYGON={poly}"


    def _iter_run(self):
        return self._iter_print()

    def _get_line_width(self, *, s: Dict[str, Any]) -> float:
        extruder = self.toolhead.get_extruder()  # type: ignore
        abs_w = float(s.get("line_width", 0.0))
        if abs_w > 0.0:
            return abs_w
        nozzle = float(getattr(extruder, "nozzle_diameter", 0.4))
        pct = s.get("line_width_pct", 0.0)
        if pct > 0.0:
            return nozzle * (pct / 100.0)
        return nozzle * 1.20

    def _iter_print(self):
        """Yield GCode lines to perform the print."""
        settings = self._run_settings or {}
        lines = self._run_lines or Lines(tuple())
        n = len(lines.lines)

        travel_speed = float(settings.get("travel_speed", self.v_travel))
        fast_feed = travel_speed * 60.0

        offx, offy, offz = self.offset_x, self.offset_y, self.offset_z

        speeds_map: Dict[str, float] = settings.get("speeds", {})

        yield self.define_exclude_object(lines)

        for ln in self._render_template_lines(
            self.templates.start,
            self._tmpl_ctx(mode="print", s=settings),
            gcmd=self._run_gcmd
        ):
            yield ln

        yield "G90"
        yield "M83"
        yield "G92 E0"

        extruder = self.toolhead.get_extruder()  # type: ignore
        filament_area = float(getattr(extruder, "filament_area", 2.4052))

        layer_h = float(settings.get("layer_height", 0.2))
        flow = float(settings.get("flow", 1.0))
        line_w = self._get_line_width(s=settings)

        e_per_mm = (line_w * layer_h / filament_area) * flow

        tuning_fmt = str(settings.get("tuning_fmt", "SET_PRESSURE_ADVANCE ADVANCE=%0.9f"))
        tuning_fmt2 = str(settings.get("tuning_fmt2", ""))

        for idx, pl in enumerate(lines):
            self.progress = idx / max(1, n)
            last_line = lines[idx - 1] if idx > 0 else None
            next_line = lines[idx + 1] if idx + 1 < n else None
            ctx = self._tmpl_ctx(
                mode="print",
                s=settings,
                line=pl,
                last_line=last_line,
                next_line=next_line,
            )

            for ln in self._render_template_lines(
                self.templates.before_line,
                ctx,
                gcmd=self._run_gcmd
            ):
                yield ln

            yield f"G0 X{pl.start.x + offx:.3f} Y{pl.start.y + offy:.3f} Z{pl.start.z + offz:.3f} F{fast_feed:.1f}"

            yield (tuning_fmt % (pl.parameter_value,))

            if pl.parameter_value2 is not None and tuning_fmt2:
                yield (tuning_fmt2 % (pl.parameter_value2,))
            
            yield "G90"

            for seg in pl.segments:
                speed_mm_s = self._segment_speed_mm_s(seg, speeds_map)
                yield (
                    f"G1 X{seg.end.x + offx:.3f} "
                    f"Y{seg.end.y + offy:.3f} "
                    f"E{seg.length_xy() * e_per_mm:.6f} F{speed_mm_s * 60.0:.1f}"
                )

            for ln in self._render_template_lines(
                self.templates.after_line,
                ctx,
                gcmd=self._run_gcmd
            ):
                yield ln

        self.progress = 1.0

        for ln in self._render_template_lines(
            self.templates.end,
            self._tmpl_ctx(mode="print", s=settings),
            gcmd=self._run_gcmd
        ):
            yield ln

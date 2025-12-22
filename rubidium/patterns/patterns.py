"""Live pattern objects"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

from ..core.lines import Lines, Line, LineSegment, Pt


# ------------------------------------------------------------
# parsing


def _tokenize(raw: str) -> List[str]:
    return str(raw).translate(str.maketrans("=:,", "   ")).split()


def parse_segments(raw: str) -> List[Tuple[str, float]]:
    """Parse pattern_segments into a normalised list of (tag, fraction).

    Accepts:
        slow: 0.25, fast: 0.5, slow: 0.25
        50.0: 0.1, 75.0: 0.8, 50.0: 0.1

    Fractions are normalised to sum to 1.0.
    """
    tokens = _tokenize(raw)
    if not tokens:
        return []
    if len(tokens) % 2 != 0:
        raise ValueError(f"Invalid pattern_segments format: '{raw}'")

    pairs: List[Tuple[str, float]] = []
    total = 0.0
    for tag, val in zip(tokens[::2], tokens[1::2]):
        fval = float(val)
        pairs.append((str(tag).strip(), fval))
        total += fval

    if total <= 0.0:
        raise ValueError("pattern_segments must sum to a value greater than zero")

    return [(tag, frac / total) for (tag, frac) in pairs]


# ------------------------------------------------------------
# option spec


CtxFn = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class OptSpec:
    key: str
    typ: str  # 'float' | 'int' | 'str'
    gcmd: Optional[str]
    default: Any
    above: Optional[float] = None
    minval: Optional[float] = None
    maxval: Optional[float] = None

    def default_value(self, ctx: Dict[str, Any]) -> Any:
        if callable(self.default):
            return self.default(ctx)
        return self.default


def _read_opt_from_config(section: Any, opt: OptSpec, *, ctx: Dict[str, Any]) -> Any:
    dv = opt.default_value(ctx)
    if opt.typ == "float":
        return section.getfloat(opt.key, dv, above=opt.above, minval=opt.minval, maxval=opt.maxval)
    if opt.typ == "int":
        mn = int(opt.minval) if opt.minval is not None else None
        mx = int(opt.maxval) if opt.maxval is not None else None
        return section.getint(opt.key, dv, minval=mn, maxval=mx)
    if opt.typ == "str":
        return section.get(opt.key, dv)
    raise ValueError(f"Unknown option type: {opt.typ}")


def _read_opt_from_gcmd(gcmd: Any, opt: OptSpec, cur: Any) -> Any:
    if not opt.gcmd:
        return cur
    if opt.typ == "float":
        return gcmd.get_float(opt.gcmd, cur, above=opt.above, minval=opt.minval, maxval=opt.maxval)
    if opt.typ == "int":
        mn = int(opt.minval) if opt.minval is not None else None
        mx = int(opt.maxval) if opt.maxval is not None else None
        return gcmd.get_int(opt.gcmd, cur, minval=mn, maxval=mx)
    if opt.typ == "str":
        return gcmd.get(opt.gcmd, cur)
    raise ValueError(f"Unknown option type: {opt.typ}")


# ------------------------------------------------------------
# pattern types


def _kin_bed_center(printer) -> Dict[str, float]:
    toolhead = printer.lookup_object("toolhead", None)
    if toolhead is None:
        return {"bed_center_x": 0.0, "bed_center_y": 0.0}
    try:
        kin = toolhead.get_kinematics().get_status(eventtime=None)
        mins = kin.get("axis_minimum")
        maxs = kin.get("axis_maximum")

        def axis(ix: int, key: str) -> Tuple[float, float]:
            if isinstance(mins, dict):
                return float(mins[key]), float(maxs[key])
            return float(mins[ix]), float(maxs[ix])

        x0, x1 = axis(0, "x")
        y0, y1 = axis(1, "y")
        return {"bed_center_x": (x0 + x1) * 0.5, "bed_center_y": (y0 + y1) * 0.5}
    except Exception:
        return {"bed_center_x": 0.0, "bed_center_y": 0.0}


class PatternType:
    def __init__(self, name: str, *, options: Tuple[OptSpec, ...]) -> None:
        self.name: str = name
        self.options: Tuple[OptSpec, ...] = options

    def build_keepout(self, *, settings: Dict[str, Any], segments: List[Tuple[str, float]]) -> Optional[Lines]:
        raise NotImplementedError
    
    def build(self, *, settings: Dict[str, Any], segments: List[Tuple[str, float]]) -> Lines:
        raise NotImplementedError


class LinesPatternType(PatternType):
    def build(self, *, settings: Dict[str, Any], segments: List[Tuple[str, float]]) -> Lines:
        param_start  = float(settings["param_start"])
        param_stop   = float(settings["param_stop"])
        param_count  = int(settings["param_count"])
        origin_x     = float(settings["origin_x"])
        origin_y     = float(settings["origin_y"])
        origin_z     = float(settings["origin_z"])
        line_length  = float(settings["line_length"])
        line_spacing = float(settings["line_spacing"])

        count = max(1, param_count)
        lines_list: List[Line] = []
        for idx in range(count):
            if count > 1:
                pa = param_start + (param_stop - param_start) * (idx / (count - 1))
            else:
                pa = param_start

            y = origin_y + line_spacing * idx
            start_pt = Pt(origin_x, y, origin_z)
            end_pt = Pt(origin_x + line_length, y, origin_z)

            segs: List[LineSegment] = []
            x0 = origin_x
            for tag, ratio in segments:
                seg_len = line_length * float(ratio)
                x1 = x0 + seg_len
                seg_start = Pt(x0, y, origin_z)
                seg_end = Pt(x1, y, origin_z)
                segs.append(LineSegment(seg_start, seg_end, str(tag)))
                x0 = x1

            lines_list.append(Line(idx=idx, parameter_value=pa, start=start_pt, end=end_pt, segments=tuple(segs)))
        return Lines(tuple(lines_list))
    
    def build_keepout(self, *, settings: Dict[str, Any], segments: List[Tuple[str, float]]) -> Lines:
        s2 = dict(settings)
        s2["param_count"] = int(settings["param_count"]) + 2
        s2["origin_y"] = float(settings["origin_y"]) - float(settings["line_spacing"])
        return self.build(settings=s2, segments=segments)

def _bed_center_x(ctx: Dict[str, Any]) -> float: # TODO lambda this
    return float(ctx.get("bed_center_x", 0.0))


def _bed_center_y(ctx: Dict[str, Any]) -> float: # TODO lambda this
    return float(ctx.get("bed_center_y", 0.0))


PATTERN_SPECS: Dict[str, PatternType] = {
    "lines": LinesPatternType(
        "lines",
        options=( # TODO : Move this into the actual LinesPatternType class as a function that returns this instead, it is specific to the class and belongs in there
            OptSpec("param_start", "float", "PARAM_START", 0.0),
            OptSpec("param_stop", "float", "PARAM_STOP", 0.1),
            OptSpec("param_count", "int", "PARAM_COUNT", 10, minval=1),
            OptSpec("tuning_command", "str", "TUNING_COMMAND", "SET_PRESSURE_ADVANCE"),
            OptSpec("tuning_parameter", "str", "TUNING_PARAMETER", "ADVANCE"),
            OptSpec("origin_x", "float", "ORIGIN_X", _bed_center_x),
            OptSpec("origin_y", "float", "ORIGIN_Y", _bed_center_y),
            OptSpec("origin_z", "float", "ORIGIN_Z", 0.0),
            OptSpec("layer_height", "float", "LAYER_HEIGHT", 0.2, above=0.0),
            OptSpec("line_length", "float", "LINE_LENGTH", 30.0, above=0.0),
            OptSpec("line_spacing", "float", "LINE_SPACING", 3.0, above=0.0),
        ),
    )
}


def _has_option(section: Any, name: str) -> bool:
    has_opt = getattr(section, "has_option", None)
    if callable(has_opt):
        return bool(has_opt(name))
    getter = getattr(section, "get", None)
    if callable(getter):
        return getter(name, None) is not None
    return False

# ------------------------------ pattern object and pattern config loading -----------------------------------

class RubidiumPatternObject:
    """Live, mutable pattern settings + built geometry."""
    def __init__(
        self,
        printer: Any,
        *,
        name: str,
        ptype: "PatternType",
        settings: Optional[Dict[str, Any]] = None,
        segments: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        self.printer = printer
        self.name = str(name)
        self.ptype = ptype
        self.settings: Dict[str, Any] = dict(settings or {})
        self.segments: List[Tuple[str, float]] = list(segments or [])
        self.lines: Optional["Lines"] = None

        self.config: Optional[Any] = None
        self._did_inherit_default: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "RubidiumPatternObject":
        """Create or update a live pattern from the given config wrapper."""
        printer = config.get_printer()
        sec_name = config.get_name()

        parts = sec_name.split(None, 2)
        pname = parts[2].strip() if len(parts) >= 3 and parts[1] == "pattern" else "default"

        root = printer.lookup_object("rubidium", None)
        existing = root.lookup_pattern(pname, None) if root is not None else None
        default_obj = root.lookup_pattern("default", None) if root is not None else None

        fallback_name = None
        if existing is not None:
            fallback_name = existing.ptype.name
        elif default_obj is not None:
            fallback_name = default_obj.ptype.name # TODO: clean this a tad

        if fallback_name is None:
            ptype_name = config.get("pattern")
        else:  # TODO: clean this a tad
            ptype_name = config.get("pattern", fallback_name)

        ptype = PATTERN_SPECS.get(ptype_name)
        if ptype is None:
            raise config.error(
                f"Unknown pattern type '{ptype_name}'. Known patterns: {', '.join(sorted(PATTERN_SPECS.keys()))}"
            )
        elif existing is not None:
            ptype = existing.ptype
        elif default_obj is not None:  # TODO: clean this a tad
            ptype = default_obj.ptype
        else:
            if pname == "default":
                ptype = PATTERN_SPECS["lines"]
            else:
                raise config.error(f"[{sec_name}] missing required 'pattern' (e.g. pattern: lines)")

        settings: Dict[str, Any] = {}
        ctx0 = {"bed_center_x": 0.0, "bed_center_y": 0.0}
        for opt in ptype.options:
            if _has_option(config, opt.key):
                settings[opt.key] = _read_opt_from_config(config, opt, ctx=ctx0)

        segs: Optional[List[Tuple[str, float]]] = None
        raw_seg = config.get("pattern_segments", "")
        if raw_seg:
            try:
                segs = parse_segments(raw_seg)
            except Exception as e:
                raise config.error(f"Invalid pattern_segments in [{sec_name}]: {e}") from e

        if existing is not None:
            if ptype is not existing.ptype:
                existing.ptype = ptype
            existing.settings.update(settings)   # TODO: clean this a tad
            if segs is not None:
                existing.segments = segs
            existing.config = config
            return existing

        obj = cls(printer, name=pname, ptype=ptype, settings=settings, segments=segs or [])
        obj.config = config
        obj._did_inherit_default = default_obj is not None and pname != "default"

        if default_obj is not None and pname != "default":
            obj.inherit_from(default_obj)

        if root is not None:
            root.register_pattern(pname, obj, config=config)

        printer.register_event_handler("klippy:connect", obj._on_connect)
        return obj

    def _on_connect(self) -> None:
        try:
            if not self._did_inherit_default:
                root = self.printer.lookup_object("rubidium", None)
                default_pattern = root.lookup_pattern("default", None) if root is not None else None
                if default_pattern is not None and default_pattern is not self:
                    self.inherit_from(default_pattern)
                    self._did_inherit_default = True

            self.rebuild()
        except Exception as e:
            if self.config is None:
                raise
            raise self.config.error(
                f"Unable to build pattern '{self.name}': {e}"
            ) from e

    def rebuild(self) -> None:
        ctx = _kin_bed_center(self.printer)
        for opt in self.ptype.options:
            if opt.key not in self.settings:
                self.settings[opt.key] = opt.default_value(ctx)
        if not self.segments:
            self.segments = [("segment", 1.0)]
        self.lines = self.ptype.build(settings=self.settings, segments=self.segments)

    def apply_gcmd(self, gcmd: Any) -> None:
        for opt in self.ptype.options:
            cur = self.settings.get(opt.key)
            if cur is None:
                cur = opt.default_value(_kin_bed_center(self.printer))
            self.settings[opt.key] = _read_opt_from_gcmd(gcmd, opt, cur)

        raw_seg = gcmd.get("PATTERN_SEGMENTS", None)
        if raw_seg is not None:
            self.segments = parse_segments(raw_seg)

        self.rebuild()

    def get_keepout_outline_xy(self) -> Optional[List[Tuple[float, float]]]:
        ko = self.ptype.build_keepout(settings=self.settings, segments=self.segments)
        if ko is None or not ko.lines:
            return None
        poly, _ = ko.outline_and_center_xy()
        if not poly:
            return None
        return poly

    def inherit_from(self, other: "RubidiumPatternObject") -> None:
        """Fill missing settings/segments from another pattern."""
        for k, v in other.settings.items():
            if k not in self.settings:
                self.settings[k] = v
        if not self.segments:
            self.segments = list(other.segments)

    def get_status(self, eventtime=None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "type": self.ptype.name,
            "settings": dict(self.settings),
            "pattern_segments": list(self.segments),
        }
        if self.lines is None:
            out["lines"] = None
            return out
        outline, center = self.lines.outline_and_center_xy()
        out["lines"] = {
            "count": len(self.lines),
            "outline_xy": outline,
            "center_xy": center,
        }
        return out

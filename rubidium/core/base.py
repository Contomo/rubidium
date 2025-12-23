
"""Shared base classes and utilities for Rubidium"""

from __future__ import annotations


from typing import Any, Dict, Optional, Tuple, Literal, List


from .lines import Lines, Line, LineSegment, Pt


class TemplateSet:
    """Container for templated G-Code snippets."""
    __slots__ = ("start", "end", "before_line", "after_line")
    def __init__(self, start, end, before_line, after_line) -> None:
        self.start = start
        self.end = end
        self.before_line = before_line
        self.after_line = after_line


class RubidiumBase:
    """Common functionality shared by printing and scanning modes."""
    cmd_help: str = ""
    def __init__(self, config, *, register_name: str, provider_name: str) -> None:

        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.gcode = self.printer.lookup_object("gcode")
        self.provider_name = provider_name

        base_name = config.get_name().split(None, 1)[0]
        base_section = self._get_section(config, base_name)
        suffix = config.get_name()[len(base_name):].strip()

        override_section = config if suffix else None
        self.base_section = base_section
        self.section = override_section or base_section
        self.registry = self.printer.lookup_object("rubidium")

        self.v_travel: float = self._get("travel_speed", 150.0, method="getfloat", above=0.0) # type: ignore
        self.offset_x: float = self._get("offset_x", 0.0, method="getfloat") # type: ignore
        self.offset_y: float = self._get("offset_y", 0.0, method="getfloat") # type: ignore
        self.offset_z: float = self._get("offset_z", 0.0, method="getfloat") # type: ignore

        load_tmpl = self.printer.load_object(base_section, "gcode_macro").load_template
        def pick_template_section(name: str) -> Any:
            return self.section if self._has_option(self.section, name) else self.base_section
        self.templates = TemplateSet(
            load_tmpl(pick_template_section("start_gcode"),  "start_gcode",  ""),
            load_tmpl(pick_template_section("end_gcode"),    "end_gcode",    ""),
            load_tmpl(pick_template_section("before_line_gcode"), "before_line_gcode", ""),
            load_tmpl(pick_template_section("after_line_gcode"),  "after_line_gcode",  ""),
        )
        if register_name:
            self.gcode.register_command('RUBIDIUM_' + register_name, self.RUBIDIUM, desc=self.cmd_help)
        self.sdcard, self.toolhead = None, None # later

        self.printer.register_event_handler("klippy:connect", self._on_connect)
        self._run_lines: Optional[Lines] = None
        self._run_settings: Optional[Dict[str, Any]] = None
        self._run_gcmd = None
        self._run_pattern = None
        self._keepout_xy: Optional[List[Tuple[float, float]]] = None
        self.progress: float = 0.0

    # -- lifecycle ------------------------------------------------------
    def _on_connect(self) -> None:
        self.toolhead = self.printer.lookup_object("toolhead")
        self.sdcard = self.printer.lookup_object("virtual_sdcard", None)
        if self.sdcard is None:
            raise self.printer.config_error(
                "virtual_sdcard must be enabled to use rubidium"
            )
        
    # -- config helpers -------------------------------------------------- a tad weird but ok
    def _has_option(self, section: Any, name: str) -> bool:
        """Return True if the given config section defines the option."""
        has_opt = getattr(section, "has_option", None)
        if callable(has_opt):
            return bool(has_opt(name))
        getter = getattr(section, "get", None)
        if callable(getter):
            return getter(name, None) is not None
        return False

    def _get(self, name: str, default, *, method: str = "get", **kwargs):
        """Fetch a value from the override or base section with fallback"""
        getter = getattr(self.section, method, None)
        if callable(getter) and self._has_option(self.section, name):
            return getter(name, **kwargs)
        base_get = getattr(self.base_section, method, None)
        if callable(base_get) and self._has_option(self.base_section, name):
            return base_get(name, **kwargs)
        return default

    def _get_section(self, config, name: str) -> Optional[Any]: # utter gibberish TODO: Remove this!
        """Return the first section matching the given prefix."""
        for section in config.get_prefix_sections(name):
            suffix = section.get_name()[len(name):].strip()
            if suffix == "":
                return section
        return None

    # -- gcode entry -----------------------------------------------------
    def RUBIDIUM(self, gcmd) -> None:
        """Entry point for each G‑Code command invocation."""
        pattern_name = self._select_pattern_name(gcmd)
        pattern = self.registry.lookup_pattern(pattern_name)
        pattern.apply_gcmd(gcmd)
        lines = pattern.lines
        if lines is None:
            raise gcmd.error(f"{self.provider_name} pattern '{pattern_name}' has no built lines")

        keepout_xy = pattern.get_keepout_outline_xy()

        self._run_pattern = pattern
        settings = self._build_settings_from(gcmd, pattern)
        lines = self._prepare_lines_for_run(
            lines,
            s=settings,
            gcmd=gcmd,
            keepout_xy=keepout_xy
        )
        self._ensure_within_bounds(lines, gcmd=gcmd)
        self._run_lines = lines
        self._run_settings = settings
        self._keepout_xy = keepout_xy 
        self._run_gcmd = gcmd # wtf
        self.progress = 0.0
        gcmd.respond_info(f"{self.provider_name} running with pattern '{pattern_name}'")
        self.sdcard.print_with_gcode_provider(self)  # type: ignore # perhaps not deligate this to self but a subclass instantiated with all those settings instead

    def _select_pattern_name(self, gcmd) -> str:
        pattern_name = gcmd.get("PATTERN", None)
        if pattern_name is not None:
            if pattern_name not in self.registry.patterns:
                raise gcmd.error(f'Invalid pattern "{pattern_name}" out of avalible [{", ".join(self.registry.patterns.keys())}]')
            return pattern_name
        keys = sorted(self.registry.patterns.keys())
        if not keys:
            raise gcmd.error('No Patterns configured.')
        return keys[0]

    def _build_settings_from(self, gcmd, pattern) -> Dict[str, Any]:
        """Build the per-run settings dict."""
        settings: Dict[str, Any] = {}
        settings["travel_speed"] = gcmd.get_float("TRAVEL_SPEED", self.v_travel, above=0.0)

        settings["pattern"] = getattr(pattern, "name", None)
        settings["pattern_settings"] = dict(getattr(pattern, "settings", {}))
        settings["pattern_segments"] = list(getattr(pattern, "segments", []))
        settings.update(self._extra_settings_from(gcmd))
        return settings

    def _extra_settings_from(self, gcmd) -> Dict[str, Any]:
        """Hook for subclasses to add additional runtime settings."""
        return {}

    def _prepare_lines_for_run(self, lines: Lines, *, s: Dict[str, Any], gcmd, keepout_xy=None) -> Lines:
        """Hook for subclasses to add/modify geometry for this run."""
        return lines

    def _tmpl_ctx(
        self,
        *,
        mode: str,
        s: Optional[Dict[str, Any]] = None,
        line: Optional[Line] = None,
        scan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a context dictionary for template rendering."""
        s = s or {}
        def pick(key: str, fallback: Any):
            return s[key] if key in s else fallback
        speeds = s.get("speeds", {})
        ctx: Dict[str, Any] = {
            "mode": mode,
            "travel_speed": pick("travel_speed", self.v_travel),
            "scan_speed":   pick("scan_speed", self.v_travel),
            "scan_buffer":  pick("scan_buffer", 0.0),
            "line": line,
            "pattern": s.get("pattern"),
            "pattern_settings": s.get("pattern_settings", {}),
            "pattern_segments": s.get("pattern_segments", []),
            "speeds": speeds,
        }
        if scan is not None:
            ctx["scan"] = scan
        return ctx

    # -- geometry --------------------------------------------------------
    def _ensure_within_bounds(self, lines: Lines, gcmd=None) -> None:
        """Check that all lines lie within the printer's kinematic limits."""
        kin_status = self.toolhead.get_kinematics().get_status(eventtime=None) # type: ignore
        mins = kin_status.get("axis_minimum")
        maxs = kin_status.get("axis_maximum")

        def axis_minmax(axis: Literal["x", "y", "z"]) -> Tuple[float, float]:
            if isinstance(mins, dict):
                return float(mins[axis]), float(maxs[axis])
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            return float(mins[idx]), float(maxs[idx])
        
        def check(val: float, val_name: str, axis: Literal["x", "y", "z"]) -> None:
            lo, hi = axis_minmax(axis)
            if not (lo <= val <= hi):
                msg = f"rubidium: {val_name} ({val:.3f}) must be within {lo:.3f} ↔ {hi:.3f}"
                if gcmd is not None:
                    raise gcmd.error(msg)
                raise ValueError(msg)
            
        poly_pts, _ = lines.outline_and_center_xy()
        if poly_pts:
            xs = [p[0] for p in poly_pts]
            ys = [p[1] for p in poly_pts]
        else:
            xs = [ln.start.x for ln in lines.lines] + [ln.end.x for ln in lines.lines]
            ys = [ln.start.y for ln in lines.lines] + [ln.end.y for ln in lines.lines]
        if xs and ys:
            check(min(xs), "pattern min_x", "x")
            check(max(xs), "pattern max_x", "x") # perhaps just make this an actual for loop, this sucks
            check(min(ys), "pattern min_y", "y")
            check(max(ys), "pattern max_y", "y")
        zs = [ln.start.z for ln in lines.lines] + [ln.end.z for ln in lines.lines]
        if zs:
            check(min(zs), "pattern min_z", "z")
            check(max(zs), "pattern max_z", "z")

    def _render_template_lines(self, template, extra_context: Dict[str, Any], gcmd=None) -> list[str]: # gcmd should never be None...
        """Render a template into a list of trimmed GCode lines."""
        ctx = template.create_template_context()
        if gcmd is not None:
            ctx.update({
                "rawparams": gcmd.get_raw_command_parameters(),
                "params": dict(gcmd.get_command_parameters()),
            })
        ctx["rubidium"] = extra_context
        script = template.render(ctx).strip()
        if not script:
            return []
        return [ln.strip() for ln in script.splitlines() if ln.strip()]

    def _setting_value(self, key: str, fallback: Any) -> Any:
        s = self._run_settings or {} # get rid of this prob
        return s.get(key, fallback)

    def _segment_speed_mm_s(self, segment: LineSegment, speeds_map: Dict[str, float]) -> float: # possibly belongs into printing, or into constructing and resolving pattern (store speed as float mm/s)
        """Resolve a speed label to an actual mm/s value."""
        label = segment.speed_label
        try:
            direct_speed = float(label)
            if direct_speed > 0.0:
                return direct_speed
        except Exception:
            pass
        if label not in speeds_map:
            raise ValueError(f"rubidium: missing speed configuration for segment label '{label}'")
        speed = speeds_map[label]
        if speed <= 0:
            raise ValueError(f"rubidium: speed for '{label}' segments must be positive")
        return float(speed)

    # -- provider interface ----------------------------------------------
    def get_gcode(self):
        """Return an iterator over GCode lines for the current run."""
        if self._run_lines is None or self._run_settings is None:
            return iter([])
        return self._iter_run()

    def get_stats(self, eventtime):
        return True, self.provider_name # this is the virtual sd card crap, why is this in the base class.... this should be offloaded into a true generator only class

    def get_status(self, eventtime): 
        st = {
            "file_path": self.get_name(),
            "progress": self.progress,
            "file_position": 0,
            "file_size": 0,
        }
        if self._run_settings is not None:
            st["rubidium"] = {
                "provider": self.provider_name,
                "settings": dict(self._run_settings),
            }
        if self._run_pattern is not None:
            try:
                st.setdefault("rubidium", {})["pattern"] = self._run_pattern.get_status(eventtime)
            except Exception:
                pass
        return st

    def get_name(self) -> str:
        return self.provider_name

    def reset(self) -> None:
        self.progress = 0.0 

    def handle_shutdown(self) -> None:
        pass

    def _iter_run(self):
        raise NotImplementedError("subclasses must implement _iter_run()")
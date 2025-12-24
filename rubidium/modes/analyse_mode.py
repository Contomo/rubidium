# rubidium/modes/analyse_mode.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.configview import ConfigView
from ..analysis.graphing import RubidiumPaths


class RubidiumAnalyse:
    def __init__(self, config) -> None:
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")

        self.base_section = getattr(self, "base_section", None) or config
        self.section = config

        self.cv = ConfigView(base=self.base_section, override=self.section)
        self.paths = RubidiumPaths(self.printer, base_section=self.base_section, section=self.section)

        self.write_plots = self.cv.get_bool("analysis_write_plots", False)
        self.write_npz = self.cv.get_bool("analysis_write_npz", False)

        self.crop_center = self.cv.get_str_opt("crop_center")
        self.crop_search_size = self.cv.get_str_opt("crop_search_size")
        self.crop_size = self.cv.get_str_opt("crop_size")
        self.crop_auto_center = self.cv.get_bool("crop_auto_center", False)
        self.crop_auto_center_samples_per_clip = self.cv.get_int("crop_auto_center_samples_per_clip", 1, minval=1)
        self.crop_auto_center_keep_percentile = self.cv.get_float("crop_auto_center_keep_percentile", 50.0, minval=0.0, maxval=100.0)
        self.crop_auto_center_max_samples = self.cv.get_int("crop_auto_center_max_samples", 0, minval=0)
        self.crop_auto_center_min_kept = self.cv.get_int("crop_auto_center_min_kept", 8, minval=1)
        self.base_resolution = self.cv.get_str_opt("base_resolution")

        self.laser_hsv_lower = self.cv.get_str_opt("laser_hsv_lower")
        self.laser_hsv_upper = self.cv.get_str_opt("laser_hsv_upper")
        self.laser_bright_percentile = self.cv.get_float("laser_bright_percentile", -1.0, minval=-1.0, maxval=100.0)
        self.laser_weight_power = self.cv.get_float("laser_weight_power", 4.0, minval=1.0)
        self.laser_min_row_energy = self.cv.get_float("laser_min_row_energy", 1.0, minval=0.0)
        self.laser_median_ksize = self.cv.get_int("laser_median_ksize", 5, minval=0)
        self.laser_morph_ksize = self.cv.get_int("laser_morph_ksize", 3, minval=0)

        self.laser_use_clahe = self.cv.get_bool("laser_use_clahe", True)
        self.laser_blur_ksize = self.cv.get_int("laser_blur_ksize", 5, minval=0)
        self.laser_clahe_clip = self.cv.get_float("laser_clahe_clip", 3.0, minval=0.0)
        self.laser_clahe_grid = self.cv.get_int_list("laser_clahe_grid", (8, 8), count=2)

        self.analysis_pipeline = self.cv.get_str_opt("analysis_pipeline")

        self.frame_step = self.cv.get_int("analysis_frame_step", 1, minval=1)
        self.max_frames = self.cv.get_int("analysis_max_frames", 0, minval=0)

        self.triangulate_enable = self.cv.get_bool("triangulate_enable", False)
        self.camera_calibration = self.cv.get_str_opt("camera_calibration")
        self.laser_plane = self.cv.get_str_opt("laser_plane")
        self.bed_plane = self.cv.get_str_opt("bed_plane")

        self.gcode.register_command(
            "RUBIDIUM_ANALYSE",
            self.cmd_RUBIDIUM_ANALYSE,
            desc=self.cmd_RUBIDIUM_ANALYSE_help
        )

    @staticmethod
    def _parse_nums(s: Optional[str], n: int, cast=float) -> Optional[tuple]:
        if not s: return None
        try:
            parts = [p.strip() for p in str(s).replace(",", " ").split() if p.strip()]
            if len(parts) < n: return None
            return tuple(cast(parts[i]) for i in range(n))
        except Exception: return None

    def _resolve_scan_dir(self, gcmd) -> Optional[Path]:
        self.paths.ensure_dirs()
        scan_root = self.paths.dirs.scan

        raw_dir = gcmd.get("DIR", None)
        raw_sess = gcmd.get("SESSION", None)

        if raw_dir:
            p = Path(str(raw_dir)).expanduser()
            return p if p.is_absolute() else (scan_root / p)
        if raw_sess:
            return scan_root / str(raw_sess)
        return self.paths.latest_run_dir(scan_root)

    cmd_RUBIDIUM_ANALYSE_help = "Analyse recorded scan clips. Usage: RUBIDIUM_ANALYSE [DIR=...]"

    def cmd_RUBIDIUM_ANALYSE(self, gcmd) -> None:
        scan_dir = self._resolve_scan_dir(gcmd)
        if scan_dir is None or not scan_dir.is_dir():
            self.gcode.respond_info("rubidium: no scan directory found.")
            return

        json_path = scan_dir / "rubidium_scan_session.json"
        if not json_path.exists():
            self.gcode.respond_info(f"rubidium: no session JSON found in {scan_dir.name}")
            return

        try:
            from ..analysis.analyzer import AnalysisConfig, TriangulationConfig, analyze_session_json
            from ..analysis.image_processing import CropConfig, LaserExtractConfig
        except ImportError:
            self.gcode.respond_info("rubidium: missing dependencies (opencv-python-headless, matplotlib, numpy)")
            return

        session_name = scan_dir.name
        output_dir = self.paths.dirs.analysis / session_name

        crop = CropConfig()
        cc = self._parse_nums(self.crop_center, 2, float)
        css = self._parse_nums(self.crop_search_size, 2, int)
        cs = self._parse_nums(self.crop_size, 2, int)
        br = self._parse_nums(self.base_resolution, 2, int)
        
        if cc: crop.center_xy = (float(cc[0]), float(cc[1]))
        if cs: crop.wh = (int(cs[0]), int(cs[1]))
        if br: 
            crop.ref_wh = (int(br[0]), int(br[1]))
        else:
            crop.ref_wh = None

        hsv_lo = self._parse_nums(self.laser_hsv_lower, 3, int)
        hsv_hi = self._parse_nums(self.laser_hsv_upper, 3, int)

        laser = LaserExtractConfig(
            hsv_lower=hsv_lo, hsv_upper=hsv_hi,
            bright_percentile=float(self.laser_bright_percentile),
            weight_power=float(self.laser_weight_power),
            min_row_energy=float(self.laser_min_row_energy),
            median_ksize=int(self.laser_median_ksize),
            morph_ksize=int(self.laser_morph_ksize),
            use_clahe=bool(self.laser_use_clahe),
            blur_ksize=int(self.laser_blur_ksize),
            clahe_clip=float(self.laser_clahe_clip),
            clahe_grid=tuple(int(v) for v in self.laser_clahe_grid),
        )

        tri = TriangulationConfig(
            enabled=bool(self.triangulate_enable),
            camera_calibration_path=self.camera_calibration,
            laser_plane_abcd=self._parse_nums(self.laser_plane, 4, float),
            bed_plane_abcd=self._parse_nums(self.bed_plane, 4, float),
        )

        pipeline_steps = None
        if self.analysis_pipeline is not None:
            pipeline_steps = [p.strip() for p in str(self.analysis_pipeline).replace("\n", " ").replace(",", " ").split() if p.strip()]
        cfg = AnalysisConfig(
             crop=crop, laser=laser, triangulation=tri,
             frame_step=int(self.frame_step), max_frames=int(self.max_frames),
             write_plots=bool(self.write_plots), write_npz=bool(self.write_npz),
            output_dir=str(output_dir),
            pipeline_steps=pipeline_steps,
            autocrop_enable=bool(self.crop_auto_center),
            autocrop_search_wh=(int(css[0]), int(css[1])) if css else None,
            autocrop_samples_per_clip=int(self.crop_auto_center_samples_per_clip),
            autocrop_max_samples=int(self.crop_auto_center_max_samples),
            autocrop_keep_percentile=float(self.crop_auto_center_keep_percentile),
            autocrop_min_kept=int(self.crop_auto_center_min_kept),
         )

        self.gcode.respond_info(f"rubidium: analyzing session {scan_dir.name}...")
        
        try:
            summary = analyze_session_json(json_path, cfg)
        except Exception as e:
            self.gcode.respond_info(f"rubidium analyse failed: {e}")
            return
        
        for warn in getattr(summary, "warnings", []):
            self.gcode.respond_info(warn)

        if self.crop_auto_center:
            try:
                ap = Path(str(output_dir)) / "autocrop.json"
                if ap.exists():
                    import json as _json
                    d = _json.loads(ap.read_text(encoding="utf-8"))
                    cx, cy = d.get("center_xy") or (None, None)
                    space = str(d.get("center_space") or "")
                    if cx is not None and cy is not None:
                        self.gcode.respond_info(f"rubidium autocrop: center={float(cx):.2f},{float(cy):.2f} ({space})")
            except Exception:
                pass

        if not summary.results:
            self.gcode.respond_info("rubidium analyse: no valid clips found to analyze.")
            return

        if summary.best:
            pa1 = float(summary.best.pa)
            pa2 = summary.best.pa2
            score = float(summary.best.breakdown.score)
            if pa2 is not None:
                rr = summary.best.grid_row
                cc = summary.best.grid_col
                if rr is not None and cc is not None:
                    self.gcode.respond_info(
                        f"rubidium analyse: best-value={pa1:.6f},{float(pa2):.6f} (score={score:.3f}) [r{int(rr)},c{int(cc)}]"
                    )
                else:
                    self.gcode.respond_info(
                        f"rubidium analyse: best-value={pa1:.6f},{float(pa2):.6f} (score={score:.3f})"
                    )
            else:
                self.gcode.respond_info(
                    f"rubidium analyse: best-value={pa1:.6f} (score={score:.3f})"
                )
        else:
            self.gcode.respond_info("rubidium analyse complete, no clear winner.")

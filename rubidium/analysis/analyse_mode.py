# rubidium/analysis/analyse_mode.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..core.configview import ConfigView
from .graphing import RubidiumPaths


class RubidiumAnalyse:
    def __init__(self, config, *, register: bool = True) -> None:
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")

        self.base_section = getattr(self, "base_section", None) or config
        self.section = config

        self.cv = ConfigView(base=self.base_section, override=self.section)
        self.paths = RubidiumPaths(self.printer, base_section=self.base_section, section=self.section)

        self.analysis_enable = self.cv.get_bool("analysis_enable", True)

        # Plot/npz output toggles.
        self.write_plots = self.cv.get_bool("analysis_write_plots", True)
        self.write_npz = self.cv.get_bool("analysis_write_npz", True)

        # Crop settings.
        self.crop_center     = self.cv.get_str_opt("crop_center")          # "x y" or "x,y"
        self.crop_size       = self.cv.get_str_opt("crop_size")              # "w h"
        self.base_resolution = self.cv.get_str_opt("base_resolution")  # "w h"

        # Laser extraction tuning.
        self.laser_hsv_lower = self.cv.get_str_opt("laser_hsv_lower")  # "h s v"
        self.laser_hsv_upper = self.cv.get_str_opt("laser_hsv_upper")
        self.laser_bright_percentile = self.cv.get_float("laser_bright_percentile", 99.5, minval=50.0, maxval=100.0)
        self.laser_weight_power = self.cv.get_float("laser_weight_power", 4.0, minval=1.0)
        self.laser_min_row_energy = self.cv.get_float("laser_min_row_energy", 1.0, minval=0.0)
        self.laser_median_ksize = self.cv.get_int("laser_median_ksize", 5, minval=0)
        self.laser_morph_ksize = self.cv.get_int("laser_morph_ksize", 3, minval=0)

        # Frame sampling.
        self.frame_step = self.cv.get_int("analysis_frame_step", 1, minval=1)
        self.max_frames = self.cv.get_int("analysis_max_frames", 0, minval=0)

        # Triangulation.
        self.triangulate_enable = self.cv.get_bool("triangulate_enable", False)
        self.camera_calibration = self.cv.get_str_opt("camera_calibration")  # file path
        self.laser_plane = self.cv.get_str_opt("laser_plane")                # "a b c d"
        self.bed_plane = self.cv.get_str_opt("bed_plane")                    # "a b c d"

        self.gcode.register_command(
            "RUBIDIUM_ANALYSE",
            self.cmd_RUBIDIUM_ANALYSE,
            desc=self.cmd_RUBIDIUM_ANALYSE_help,
        )

    # -----------------------------------------------------------------

    @staticmethod
    def _parse_nums(s: Optional[str], n: int, cast=float) -> Optional[tuple]:
        if not s:
            return None
        try:
            parts = [p.strip() for p in str(s).replace(",", " ").split() if p.strip()]
            if len(parts) < n:
                return None
            return tuple(cast(parts[i]) for i in range(n))
        except Exception:
            return None

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

    # -----------------------------------------------------------------
    
    cmd_RUBIDIUM_ANALYSE_help = "Analyse recorded scan clips and pick the best value"

    def cmd_RUBIDIUM_ANALYSE(self, gcmd) -> None:
        if not self.analysis_enable:
            self.gcode.respond_info("rubidium: analysis is disabled (analysis_enable: False)")
            return

        scan_dir = self._resolve_scan_dir(gcmd)
        if scan_dir is None or not scan_dir.is_dir():
            self.gcode.respond_info("rubidium: no scan directory found (set DIR=... or run a scan first)")
            return

        outdir = scan_dir / "analysis"
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            from .analyzer import AnalysisConfig, TriangulationConfig, analyze_directory
            from .image_processing import CropConfig, LaserExtractConfig
        except Exception as e:
            self.gcode.respond_info(f"rubidium: analysis unavailable ({e})")
            return

        crop = CropConfig()
        cc = self._parse_nums(self.crop_center, 2, float)
        cs = self._parse_nums(self.crop_size, 2, int)
        br = self._parse_nums(self.base_resolution, 2, int)
        if cc:
            crop.center_xy = (float(cc[0]), float(cc[1]))
        if cs:
            crop.wh = (int(cs[0]), int(cs[1]))
        if br:
            crop.ref_wh = (int(br[0]), int(br[1]))

        hsv_lo = self._parse_nums(self.laser_hsv_lower, 3, int)
        hsv_hi = self._parse_nums(self.laser_hsv_upper, 3, int)

        laser = LaserExtractConfig(
            hsv_lower=hsv_lo,
            hsv_upper=hsv_hi,
            bright_percentile=float(self.laser_bright_percentile),
            weight_power=float(self.laser_weight_power),
            min_row_energy=float(self.laser_min_row_energy),
            median_ksize=int(self.laser_median_ksize),
            morph_ksize=int(self.laser_morph_ksize),
        )

        lp = self._parse_nums(self.laser_plane, 4, float)
        bp = self._parse_nums(self.bed_plane, 4, float)

        tri = TriangulationConfig(
            enabled=bool(self.triangulate_enable),
            camera_calibration_path=self.camera_calibration,
            laser_plane_abcd=lp,
            bed_plane_abcd=bp,
        )

        cfg = AnalysisConfig(
            crop=crop,
            laser=laser,
            frame_step=int(self.frame_step),
            max_frames=int(self.max_frames),
            triangulation=tri,
            write_plots=bool(self.write_plots),
            write_npz=bool(self.write_npz),
            output_dir=str(outdir),
        )

        try:
            summary = analyze_directory(scan_dir, cfg)
        except Exception as e:
            self.gcode.respond_info(f"rubidium: analysis failed ({e})")
            return

        if not summary.results or summary.best is None:
            self.gcode.respond_info("rubidium: analysis produced no results")
            return

        self.gcode.respond_info(
            f"rubidium: best_pa={summary.best.pa:.6f} score={summary.best.breakdown.score:.3f} kind={summary.best.height_map_kind}"
        )

# rubidium/analysis/graphing.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.configview import ConfigView, config_dir_from_printer


@dataclass(slots=True)
class RunDirs:
    root: Path
    scan: Path
    analysis: Path


class RubidiumPaths:
    """Pure path policy for Rubidium."""

    def __init__(self, printer, *, base_section: Any, section: Any) -> None:
        self.printer = printer
        self.cv = ConfigView(base=base_section, override=section)

        cfg_dir = config_dir_from_printer(printer)
        output_dir = self.cv.get_str_opt("output_dir")

        if output_dir:
            root = Path(output_dir).expanduser()
        else:
            if cfg_dir is None:
                raise printer.config_error(
                    "rubidium: cannot determine config directory; set output_dir in [rubidium] or [rubidium scan]/[rubidium analyse]"
                )
            root = cfg_dir / "rubidium"

        scan_dir = Path(self.cv.get_str("scan_dir", str(root / "scan"))).expanduser()
        analysis_dir = Path(self.cv.get_str("analysis_dir", str(root / "analysis"))).expanduser()

        self.dirs = RunDirs(root=root, scan=scan_dir, analysis=analysis_dir)
        self.run_dir_format = self.cv.get_str("run_dir_format", "%Y%m%d_%H%M%S")

    def ensure_dirs(self) -> None:
        self.dirs.root.mkdir(parents=True, exist_ok=True)
        self.dirs.scan.mkdir(parents=True, exist_ok=True)
        self.dirs.analysis.mkdir(parents=True, exist_ok=True)

    def new_run_dir(self, parent: Path, *, prefix: str) -> Path:
        import os, time

        parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime(self.run_dir_format, time.localtime())
        name = f"{prefix}_{ts}_{os.getpid()}"
        p = parent / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def latest_run_dir(self, parent: Path) -> Optional[Path]:
        try:
            dirs = [p for p in parent.iterdir() if p.is_dir()]
        except Exception:
            return None
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0]

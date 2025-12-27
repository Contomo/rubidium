# rubidium/analysis/session_analyzer.py
from __future__ import annotations

import csv
import json
from dataclasses import replace, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .autocrop import AutoCropper
from .clips import ClipCounts, ClipInfo, parse_session_clips
from .debug_info import AutoCropDebug, CropDebug, SessionCounts, SessionDebug, TriangulationDebug
from .types import AnalysisConfig, AnalysisSummary, LineAnalysis
from .visualization import save_dashboard
from .video_analyzer import VideoAnalyzer


class SessionAnalyzer:
    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg

    def analyze_session_json(self, json_path: Path) -> AnalysisSummary:
        data = self._load_json(json_path)
        session_dir = json_path.parent
        outdir = Path(self.cfg.output_dir) if self.cfg.output_dir else (session_dir / "analysis")
        outdir.mkdir(parents=True, exist_ok=True)

        clips_raw = data.get("clips", [])
        if not isinstance(clips_raw, list):
            raise ValueError("rubidium: session JSON 'clips' must be a list")

        clips, counts = parse_session_clips(session_dir, clips_raw)

        warnings: List[str] = []
        cfg_run = replace(self.cfg, output_dir=str(outdir))

        crop_initial = CropDebug(
            center_xy=(float(cfg_run.crop.center_xy[0]), float(cfg_run.crop.center_xy[1])),
            wh=(int(cfg_run.crop.wh[0]), int(cfg_run.crop.wh[1])),
            ref_wh=tuple(cfg_run.crop.ref_wh) if cfg_run.crop.ref_wh is not None else None,
        )

        autocrop_state = None
        if cfg_run.autocrop.enable and clips:
            ac = AutoCropper(cfg_run, outdir=outdir)
            autocrop_state = ac.run(clips)
            if autocrop_state is not None:
                warnings.append(f"autocrop: status={autocrop_state.status} (applied={int(autocrop_state.applied)})")
                if autocrop_state.applied:
                    cfg_run = replace(cfg_run, crop=replace(cfg_run.crop, center_xy=autocrop_state.center_xy))

        crop_final = CropDebug(
            center_xy=(float(cfg_run.crop.center_xy[0]), float(cfg_run.crop.center_xy[1])),
            wh=(int(cfg_run.crop.wh[0]), int(cfg_run.crop.wh[1])),
            ref_wh=tuple(cfg_run.crop.ref_wh) if cfg_run.crop.ref_wh is not None else None,
        )

        video = VideoAnalyzer(cfg_run)
        results: List[LineAnalysis] = []
        clips_analyzed = 0
        for clip in clips:
            clips_analyzed += 1
            res = video.analyze(path=clip.path, idx=clip.idx, pa=clip.pa, mirror_x=clip.mirror_x)
            if not res.ok:
                continue
            res.pa2 = clip.pa2
            res.grid_row = clip.grid_row
            res.grid_col = clip.grid_col
            results.append(res)

        # Ordering: primary param, secondary param, then idx.
        results.sort(key=lambda r: (float(r.pa), float(r.pa2) if r.pa2 is not None else float("-inf"), int(r.idx)))
        best = min(results, key=lambda r: float(r.breakdown.score)) if results else None

        csv_path = self._write_summary_csv(outdir, results)

        debug = self._build_debug(
            session=session_dir.name,
            outdir=str(outdir),
            warnings=warnings,
            counts=counts,
            clips_analyzed=clips_analyzed,
            results_ok=len(results),
            crop_initial=crop_initial,
            crop_final=crop_final,
            cfg=cfg_run,
            autocrop_state=autocrop_state,
        )

        sheet_path = outdir / "analysis_dashboard.jpg"
        save_dashboard(results, sheet_path, debug=debug.to_dict())

        return AnalysisSummary(
            dirpath=session_dir,
            results=results,
            best=best,
            summary_csv=csv_path,
            summary_sheet=sheet_path,
            warnings=warnings,
        )

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed parsing JSON: {e}")

    def _write_summary_csv(self, outdir: Path, results: List[LineAnalysis]) -> Optional[Path]:
        csv_path = outdir / "summary.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "pa", "pa2", "grid_row", "grid_col", "score", "roughness", "dropouts", "pk_pk_px", "bump_px"])
            for r in results:
                pk_pk = 0.0
                if r.height_map is not None:
                    vals = r.height_map[np.isfinite(r.height_map)]
                    if vals.size > 0:
                        pk_pk = float(np.max(vals) - np.min(vals))
                w.writerow([
                    int(r.idx),
                    float(r.pa),
                    float(r.pa2) if r.pa2 is not None else "",
                    int(r.grid_row) if r.grid_row is not None else "",
                    int(r.grid_col) if r.grid_col is not None else "",
                    float(r.breakdown.score),
                    float(r.breakdown.roughness),
                    float(r.breakdown.dropouts),
                    float(pk_pk),
                    float(r.bump_px) if r.bump_px is not None else "",
                ])
        return csv_path

    def _build_debug(
        self,
        *,
        session: str,
        outdir: str,
        warnings: List[str],
        counts: ClipCounts,
        clips_analyzed: int,
        results_ok: int,
        crop_initial: CropDebug,
        crop_final: CropDebug,
        cfg: AnalysisConfig,
        autocrop_state,
    ) -> SessionDebug:
        ac_rejects = getattr(autocrop_state, "rejects", None) if autocrop_state is not None else None
        
        # CRASH FIX: Use asdict instead of calling a non-existent method
        ac_stats = None
        if autocrop_state is not None:
            ac_stats = asdict(autocrop_state)
            # Remove redundant keys
            if isinstance(ac_stats, dict):
                ac_stats.pop("rejects", None)

        return SessionDebug(
            session=session,
            output_dir=outdir,
            warnings=warnings,
            counts=SessionCounts(
                clips_total=counts.clips_total,
                clips_flag_ok=counts.clips_flag_ok,
                clips_missing_name=counts.clips_missing_name,
                clips_missing_file=counts.clips_missing_file,
                clips_analyzed=clips_analyzed,
                results_ok=results_ok,
            ),
            crop_initial=crop_initial,
            crop_final=crop_final,
            autocrop=AutoCropDebug(
                enable=cfg.autocrop.enable,
                rejects=ac_rejects,
                stats=ac_stats,
            ),
            pipeline_steps=list(cfg.pipeline_steps) if cfg.pipeline_steps else [],
            triangulation=TriangulationDebug(
                enabled=bool(cfg.triangulation.enabled),
                camera_calibration_path=cfg.triangulation.camera_calibration_path,
                laser_plane_abcd=cfg.triangulation.laser_plane_abcd,
                bed_plane_abcd=cfg.triangulation.bed_plane_abcd,
                min_valid_frac=float(cfg.triangulation.min_valid_frac),
                min_height_range=float(cfg.triangulation.min_height_range),
                gate_fail_score=float(cfg.triangulation.gate_fail_score),
            ),
        )


__all__ = [
    "SessionAnalyzer",
]
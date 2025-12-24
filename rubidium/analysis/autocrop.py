# rubidium/analysis/autocrop.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .clips import ClipInfo
from .image_processing import CropConfig, crop_frame, detect_laser_joint
from .types import AnalysisConfig


@dataclass(frozen=True, slots=True)
class AutoCropSample:
    clip_idx: int
    frame_idx: int
    mirror_x: bool
    cx_full_px: float
    cy_full_px: float
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class AutoCropResult:
    applied: bool
    center_xy: Tuple[float, float]
    center_space: str
    search_wh_used: Tuple[int, int]
    status: str
    samples_total: int
    samples_valid: int
    samples_kept: int
    score_threshold: Optional[float]
    mad_px: Optional[Tuple[float, float]]
    rejects: Dict[str, int]

    def to_json(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "center_space": str(self.center_space),
            "search_wh": [int(self.search_wh_used[0]), int(self.search_wh_used[1])],
            "samples_total": int(self.samples_total),
            "samples_valid": int(self.samples_valid),
            "samples_kept": int(self.samples_kept),
            "score_threshold": float(self.score_threshold) if self.score_threshold is not None else None,
            "mad_px": [float(self.mad_px[0]), float(self.mad_px[1])] if self.mad_px is not None else None,
            "rejects": dict(self.rejects),
        }


class AutoCropper:
    """Session-wide crop-center prepass.

    This class is intentionally narrow:
      * sample a handful of frames across clips
      * detect a joint/feature point
      * compute a stable median center
      * write debug artifacts

    It never swallows programmer errors (bad config types, etc.).
    """

    def __init__(self, cfg: AnalysisConfig, *, outdir: Path) -> None:
        self.cfg = cfg
        self.outdir = outdir

        self.path_json = outdir / "autocrop.json"
        self.path_points = outdir / "autocrop_points.csv"
        self.path_preview = outdir / "autocrop_preview.jpg"
        self.path_final = outdir / "autocrop_final.jpg"
        self.path_mask = outdir / "autocrop_mask.jpg"

    def run(self, clips: List[ClipInfo]) -> Optional[AutoCropResult]:
        if not self.cfg.autocrop.enable:
            return None

        self.outdir.mkdir(parents=True, exist_ok=True)

        samples: List[AutoCropSample] = []
        rejects: Dict[str, int] = {}
        # (frame_proc_bgr, search_crop_bgr, mask_u8, joint_local_xy, search_off_xy, score, frame_w, mirror_x)
        best_debug: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], Tuple[int, int], float, int, bool]
        ] = None
        first_frame_wh: Optional[Tuple[int, int]] = None

        total_samples = 0
        for clip_i, clip in enumerate(clips):
            cap = cv2.VideoCapture(str(clip.path))
            if not cap.isOpened():
                _inc(rejects, "video_open_failed")
                continue

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for frame_idx in _sample_frame_indices(total, int(max(1, self.cfg.autocrop.samples_per_clip))):
                if self.cfg.autocrop.max_samples and total_samples >= self.cfg.autocrop.max_samples:
                    break

                total_samples += 1
                fr = _read_frame_at(cap, frame_idx)
                if fr is None:
                    _inc(rejects, "frame_read_failed")
                    samples.append(AutoCropSample(clip_i, frame_idx, clip.mirror_x, float("nan"), float("nan"), float("nan"), "frame_read_failed"))
                    continue

                frame_w = int(fr.shape[1])
                frame_h = int(fr.shape[0])
                if first_frame_wh is None:
                    first_frame_wh = (frame_w, frame_h)

                fr_proc = cv2.flip(fr, 1) if clip.mirror_x else fr

                search_wh = self._resolve_search_wh(frame_w=frame_w, frame_h=frame_h)
                search_cfg = CropConfig(center_xy=self.cfg.crop.center_xy, wh=search_wh, ref_wh=self.cfg.crop.ref_wh)
                search_crop, search_off = crop_frame(fr_proc, search_cfg)

                joint_xy, score, mask_u8 = detect_laser_joint(search_crop, self.cfg.laser)
                if joint_xy is None:
                    _inc(rejects, "joint_not_found")
                    samples.append(AutoCropSample(clip_i, frame_idx, clip.mirror_x, float("nan"), float("nan"), float(score), "joint_not_found"))
                    continue

                if not np.isfinite(score) or float(score) <= 0.0:
                    _inc(rejects, "score_invalid")
                    samples.append(AutoCropSample(clip_i, frame_idx, clip.mirror_x, float("nan"), float("nan"), float(score), "score_invalid"))
                    continue

                cx_proc = float(search_off[0]) + float(joint_xy[0])
                cy_proc = float(search_off[1]) + float(joint_xy[1])

                # Convert to unmirrored full-frame coordinates so all clips are comparable.
                cx_full = float(frame_w - 1) - cx_proc if clip.mirror_x else cx_proc
                cy_full = cy_proc

                samples.append(AutoCropSample(clip_i, frame_idx, clip.mirror_x, cx_full, cy_full, float(score), "ok"))

                if best_debug is None or float(score) > float(best_debug[5]):
                    best_debug = (
                        fr_proc.copy(),
                        search_crop.copy(),
                        mask_u8.copy() if mask_u8 is not None else np.zeros((1, 1), dtype=np.uint8),
                        (float(joint_xy[0]), float(joint_xy[1])),
                        (int(search_off[0]), int(search_off[1])),
                        float(score),
                        int(frame_w),
                        bool(clip.mirror_x),
                    )

            cap.release()

            if self.cfg.autocrop.max_samples and total_samples >= self.cfg.autocrop.max_samples:
                break

        self._write_points(samples)

        if not samples:
            res = self._result_failure(
                status="no_samples",
                rejects=rejects,
                first_frame_wh=first_frame_wh,
            )
            self._write_json(res)
            return res

        valid = [s for s in samples if s.reason == "ok" and np.isfinite(s.cx_full_px) and np.isfinite(s.cy_full_px) and np.isfinite(s.score) and s.score > 0.0]
        if not valid:
            res = self._result_failure(
                status="no_valid_detections",
                rejects=rejects,
                first_frame_wh=first_frame_wh,
                samples_total=len(samples),
            )
            self._write_json(res)
            return res

        scores = np.asarray([float(s.score) for s in valid], dtype=np.float32)
        thr = float(np.percentile(scores, float(np.clip(self.cfg.autocrop.keep_percentile, 0.0, 100.0))))
        kept = [s for s in valid if float(s.score) >= thr]

        if len(kept) < int(self.cfg.autocrop.min_kept):
            res = self._result_failure(
                status="too_few_kept",
                rejects=rejects,
                first_frame_wh=first_frame_wh,
                samples_total=len(samples),
                samples_valid=len(valid),
                samples_kept=len(kept),
                score_threshold=thr,
            )
            self._write_json(res)
            return res

        center_full = _median_center(kept)
        mad = _median_abs_deviation(kept, center_full)

        if first_frame_wh is not None:
            scatter_limit = _scatter_limit_px(self.cfg, first_frame_wh)
            if mad[0] > scatter_limit or mad[1] > scatter_limit:
                res = self._result_failure(
                    status="unstable",
                    rejects=rejects,
                    first_frame_wh=first_frame_wh,
                    samples_total=len(samples),
                    samples_valid=len(valid),
                    samples_kept=len(kept),
                    score_threshold=thr,
                    mad_px=mad,
                )
                self._write_json(res)
                return res

        center_cfg, space = self._convert_full_px_to_cfg(center_full, first_frame_wh)

        res = AutoCropResult(
            applied=True,
            center_xy=center_cfg,
            center_space=space,
            search_wh_used=self._resolve_search_wh(frame_w=first_frame_wh[0], frame_h=first_frame_wh[1]) if first_frame_wh else (0, 0),
            status="ok",
            samples_total=len(samples),
            samples_valid=len(valid),
            samples_kept=len(kept),
            score_threshold=thr,
            mad_px=mad,
            rejects=rejects,
        )

        self._write_json(res)
        if best_debug is not None and first_frame_wh is not None:
            self._write_images(best_debug, center_full_px=center_full)

        return res

    def _resolve_search_wh(self, *, frame_w: int, frame_h: int) -> Tuple[int, int]:
        cfg_wh = self.cfg.autocrop.search_wh
        if cfg_wh is None:
            return (int(frame_w), int(frame_h))

        sw, sh = int(cfg_wh[0]), int(cfg_wh[1])
        if sw <= 0 or sh <= 0:
            raise ValueError("autocrop: crop_search_size must be > 0")
        if sw > frame_w or sh > frame_h:
            raise ValueError(
                f"autocrop: crop_search_size {sw}x{sh} exceeds frame {frame_w}x{frame_h}"
            )
        return (sw, sh)

    def _convert_full_px_to_cfg(
        self,
        center_full_px: Tuple[float, float],
        first_frame_wh: Optional[Tuple[int, int]],
    ) -> Tuple[Tuple[float, float], str]:
        """Convert full-frame pixels into cfg.crop.center_xy space."""
        cx_px, cy_px = float(center_full_px[0]), float(center_full_px[1])
        ref = self.cfg.crop.ref_wh
        if ref is None or ref[0] <= 0 or ref[1] <= 0 or first_frame_wh is None:
            return (cx_px, cy_px), "px"

        frame_w, frame_h = first_frame_wh
        sx = float(frame_w) / float(ref[0])
        sy = float(frame_h) / float(ref[1])
        if sx <= 0.0 or sy <= 0.0:
            return (cx_px, cy_px), "px"
        return (cx_px / sx, cy_px / sy), "ref"

    def _write_points(self, samples: List[AutoCropSample]) -> None:
        with self.path_points.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_idx", "frame_idx", "mirror_x", "cx_full_px", "cy_full_px", "score", "reason"])
            for s in samples:
                w.writerow([
                    int(s.clip_idx),
                    int(s.frame_idx),
                    int(bool(s.mirror_x)),
                    float(s.cx_full_px),
                    float(s.cy_full_px),
                    float(s.score),
                    str(s.reason),
                ])

    def _write_json(self, res: AutoCropResult) -> None:
        self.path_json.write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")

    def _write_images(
        self,
        best_debug: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], Tuple[int, int], float, int, bool],
        *,
        center_full_px: Tuple[float, float],
    ) -> None:
        fr_proc, search_crop, mask_u8, joint_local, search_off, _score, frame_w, mirror_x = best_debug

        # best_debug images are in processed (potentially mirrored) space. Convert the
        # chosen center back into that same space for visualization.
        cx_full, cy_full = float(center_full_px[0]), float(center_full_px[1])
        cxp = float(frame_w - 1) - cx_full if mirror_x else cx_full
        cyp = cy_full

        # final crop preview from processed full frame using the chosen center in pixel space
        final_cfg_px = CropConfig(center_xy=(cxp, cyp), wh=self.cfg.crop.wh, ref_wh=None)
        final_crop, _ = crop_frame(fr_proc, final_cfg_px)

        prev = search_crop.copy()
        cv2.circle(prev, (int(round(joint_local[0])), int(round(joint_local[1]))), 4, (0, 0, 255), -1)

        med_local_x = cxp - float(search_off[0])
        med_local_y = cyp - float(search_off[1])
        cv2.circle(prev, (int(round(med_local_x)), int(round(med_local_y))), 4, (0, 255, 0), -1)

        fw, fh = int(self.cfg.crop.wh[0]), int(self.cfg.crop.wh[1])
        if fw <= 0 or fh <= 0:
            fh_full, fw_full = fr_proc.shape[:2]
            fw, fh = int(fw_full), int(fh_full)
        x0 = int(round(med_local_x - fw / 2.0))
        y0 = int(round(med_local_y - fh / 2.0))
        cv2.rectangle(prev, (x0, y0), (x0 + fw, y0 + fh), (0, 255, 0), 2)

        cv2.imwrite(str(self.path_preview), prev)
        cv2.imwrite(str(self.path_final), final_crop)
        cv2.imwrite(str(self.path_mask), cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR))

    def _result_failure(
        self,
        *,
        status: str,
        rejects: Dict[str, int],
        first_frame_wh: Optional[Tuple[int, int]],
        samples_total: int = 0,
        samples_valid: int = 0,
        samples_kept: int = 0,
        score_threshold: Optional[float] = None,
        mad_px: Optional[Tuple[float, float]] = None,
    ) -> AutoCropResult:
        if first_frame_wh is None:
            search_wh_used = (0, 0)
        else:
            search_wh_used = self._resolve_search_wh(frame_w=first_frame_wh[0], frame_h=first_frame_wh[1])
        return AutoCropResult(
            applied=False,
            center_xy=self.cfg.crop.center_xy,
            center_space="cfg",
            search_wh_used=search_wh_used,
            status=str(status),
            samples_total=int(samples_total),
            samples_valid=int(samples_valid),
            samples_kept=int(samples_kept),
            score_threshold=score_threshold,
            mad_px=mad_px,
            rejects=dict(rejects),
        )


def _inc(d: Dict[str, int], k: str, n: int = 1) -> None:
    d[k] = int(d.get(k, 0)) + int(n)


def _sample_frame_indices(total: int, n: int) -> List[int]:
    if total <= 0:
        return [0] * max(1, n)
    if n <= 1:
        return [max(0, total // 2)]
    lo = int(round(total * 0.20))
    hi = int(round(total * 0.80))
    hi = max(lo + 1, min(total - 1, hi))
    out: List[int] = []
    for i in range(n):
        t = (i + 1) / float(n + 1)
        out.append(int(round(lo + (hi - lo) * t)))
    return out


def _read_frame_at(cap: cv2.VideoCapture, target_idx: int) -> Optional[np.ndarray]:
    if target_idx < 0:
        target_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
    ok, fr = cap.read()
    if ok and fr is not None:
        return fr

    # fallback: rewind and step
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            return None
        if i >= target_idx:
            return fr
        i += 1


def _median_center(kept: List[AutoCropSample]) -> Tuple[float, float]:
    cxs = np.asarray([float(s.cx_full_px) for s in kept], dtype=np.float32)
    cys = np.asarray([float(s.cy_full_px) for s in kept], dtype=np.float32)
    return (float(np.median(cxs)), float(np.median(cys)))


def _median_abs_deviation(kept: List[AutoCropSample], center: Tuple[float, float]) -> Tuple[float, float]:
    cx, cy = float(center[0]), float(center[1])
    cxs = np.asarray([float(s.cx_full_px) for s in kept], dtype=np.float32)
    cys = np.asarray([float(s.cy_full_px) for s in kept], dtype=np.float32)
    mad_x = float(np.median(np.abs(cxs - cx)))
    mad_y = float(np.median(np.abs(cys - cy)))
    return (mad_x, mad_y)


def _scatter_limit_px(cfg: AnalysisConfig, frame_wh: Tuple[int, int]) -> float:
    final_w, final_h = float(cfg.crop.wh[0]), float(cfg.crop.wh[1])
    if final_w <= 0.0 or final_h <= 0.0:
        final_w, final_h = float(frame_wh[0]), float(frame_wh[1])
    return max(3.0, 0.25 * min(final_w, final_h))


__all__ = [
    "AutoCropResult",
    "AutoCropSample",
    "AutoCropper",
]

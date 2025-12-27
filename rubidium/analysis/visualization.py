# rubidium/analysis/visualization.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


def render_heightmap_plot(
    height_map: np.ndarray,
    width_px: int = 500,
    height_px: int = 120,
    limit_scale: Optional[float] = None,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm_t = height_map.T
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if limit_scale is None:
        vals = hm_t[np.isfinite(hm_t)]
        limit = np.percentile(np.abs(vals), 99) if vals.size > 0 else 1.0
        limit = max(float(limit), 0.1)
    else:
        limit = float(limit_scale)

    ax.imshow(
        hm_t,
        aspect="auto",
        interpolation="nearest",
        vmin=-limit,
        vmax=limit,
        cmap="viridis",
    )

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="png", dpi=dpi)
    plt.close(fig)
    io_buf.seek(0)

    file_bytes = np.asarray(bytearray(io_buf.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    if img.shape[0] != height_px or img.shape[1] != width_px:
        img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)

    return img


def render_score_lineplot(
    x_vals: List[float],
    y_vals: List[float],
    *,
    width_px: int = 900,
    height_px: int = 220,
) -> np.ndarray:
    """Render a simple line plot (BGR image) for score vs parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.06, 0.18, 0.92, 0.75])

    if x_vals and y_vals:
        ax.plot(x_vals, y_vals, marker="o")

    ax.set_xlabel("parameter")
    ax.set_ylabel("score")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="png", dpi=dpi)
    plt.close(fig)
    io_buf.seek(0)

    file_bytes = np.asarray(bytearray(io_buf.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    if img.shape[0] != height_px or img.shape[1] != width_px:
        img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)

    return img


def render_score_surface_plot(
    x_vals: List[float],
    y_vals: List[float],
    z_vals: List[float],
    *,
    width_px: int = 900,
    height_px: int = 900,
    interp: bool = True,
) -> np.ndarray:
    """Render a 3D surface plot (score vs param1/param2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xs = np.asarray(x_vals, dtype=np.float32)
    ys = np.asarray(y_vals, dtype=np.float32)
    zs = np.asarray(z_vals, dtype=np.float32)

    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    xs = xs[ok]
    ys = ys[ok]
    zs = zs[ok]

    if xs.size == 0:
        return np.zeros((height_px, width_px, 3), dtype=np.uint8)

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=35, azim=235)
    ax.set_xlabel("param1")
    ax.set_ylabel("param2")
    ax.set_zlabel("score")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.6))
    except Exception:
        pass

    try:
        if xs.size >= 3:
            tri = mtri.Triangulation(xs, ys)
            if interp and xs.size >= 4:
                xi = np.linspace(float(xs.min()), float(xs.max()), 40)
                yi = np.linspace(float(ys.min()), float(ys.max()), 40)
                xi, yi = np.meshgrid(xi, yi)
                interp_f = mtri.LinearTriInterpolator(tri, zs)
                zi = interp_f(xi, yi)
                zi = np.ma.masked_invalid(zi)
                ax.plot_surface(xi, yi, zi, cmap="viridis", linewidth=0, antialiased=True)
            else:
                ax.plot_trisurf(tri, zs, cmap="viridis", linewidth=0.2, antialiased=True)
        ax.scatter(xs, ys, zs, c="k", s=12, depthshade=True)
    except Exception:
        ax.scatter(xs, ys, zs, c="k", s=12, depthshade=True)

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="png", dpi=dpi)
    plt.close(fig)
    io_buf.seek(0)

    file_bytes = np.asarray(bytearray(io_buf.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    if img.shape[0] != height_px or img.shape[1] != width_px:
        img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)

    return img


def create_debug_thumbnails(
    crop: np.ndarray, mask: np.ndarray, centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tracking = crop.copy()
    h, w = tracking.shape[:2]

    step = max(1, h // 50)
    for y in range(0, h, step):
        cx = centroids[y]
        if np.isfinite(cx):
            x_int = int(round(cx))
            if 0 <= x_int < w:
                cv2.circle(tracking, (x_int, y), 2, (0, 255, 0), -1)

    return crop, mask_bgr, tracking


def _global_limit_from_results(results: List) -> float:
    all_abs_vals = []
    for res in results:
        hm = getattr(res, "height_map", None)
        if hm is None:
            continue
        valid = hm[np.isfinite(hm)]
        if valid.size > 0:
            all_abs_vals.append(np.abs(valid))

    global_limit = 1.0
    if all_abs_vals:
        concat = np.concatenate(all_abs_vals)
        if concat.size > 0:
            global_limit = float(np.percentile(concat, 99))
            if global_limit < 0.1:
                global_limit = 0.1
    return global_limit


def _center_crop_resize(img: np.ndarray, out_w: int, out_h: int, *, bg_color=(30, 30, 30)) -> np.ndarray:
    """Resize with center-crop so that the output is exactly out_w x out_h."""
    if out_w <= 0 or out_h <= 0:
        return np.zeros((max(1, out_h), max(1, out_w), 3), dtype=np.uint8)
    if img is None:
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        out[:] = bg_color
        return out

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        out[:] = bg_color
        return out

    s = max(out_w / float(w), out_h / float(h))
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = max(0, (nw - out_w) // 2)
    y0 = max(0, (nh - out_h) // 2)
    return resized[y0 : y0 + out_h, x0 : x0 + out_w]


def _wrap_text_to_width(text: str, *, font, font_scale: float, thickness: int, max_w: int) -> List[str]:
    """Best-effort word wrap for cv2.putText."""
    s = str(text).replace("\r", "").strip()
    if not s:
        return [""]

    words = s.split()
    lines: List[str] = []
    cur: List[str] = []

    def line_w(t: str) -> int:
        (w, _), _ = cv2.getTextSize(t, font, font_scale, thickness)
        return int(w)

    for w in words:
        test = (" ".join(cur + [w])).strip()
        if cur and line_w(test) > max_w:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)

    if cur:
        lines.append(" ".join(cur))

    # if a single token is wider than max, just hard-cut it.
    out: List[str] = []
    for ln in lines:
        if line_w(ln) <= max_w:
            out.append(ln)
            continue
        # hard slice
        buf = ""
        for ch in ln:
            if line_w(buf + ch) > max_w and buf:
                out.append(buf)
                buf = ch
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def _render_debug_panel(
    *,
    results: List,
    debug: Optional[Dict[str, Any]],
    width_px: int,
    height_px: int = 220,
) -> np.ndarray:
    BORDER = 2
    BG_COLOR = (30, 30, 30)
    BORDER_COLOR = (100, 100, 100)
    TEXT_COLOR = (220, 220, 220)

    panel = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    panel[:] = BG_COLOR

    if not debug:
        cv2.rectangle(panel, (0, 0), (width_px - 1, height_px - 1), BORDER_COLOR, 1)
        return panel

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Left text area
    w_text = max(380, min(720, int(width_px * 0.45)))
    w_text = min(w_text, width_px - 2 * BORDER)

    # Right images area (optional)
    w_imgs = width_px - w_text - 3 * BORDER

    # ---- compose text lines
    lines: List[str] = []

    session = str(debug.get("session") or "")
    if session:
        lines.append(f"session: {session}")

    counts = debug.get("counts") or {}
    if isinstance(counts, dict):
        ct = counts.get("clips_total", None)
        ca = counts.get("clips_analyzed", None)
        cr = counts.get("results_ok", None)
        if ct is not None or ca is not None or cr is not None:
            parts = []
            if ct is not None:
                parts.append(f"clips={int(ct)}")
            if ca is not None:
                parts.append(f"analyzed={int(ca)}")
            if cr is not None:
                parts.append(f"ok={int(cr)}")
            if parts:
                lines.append("counts: " + ", ".join(parts))

    crop_i = debug.get("crop_initial") or {}
    crop_f = debug.get("crop_final") or {}

    def _fmt_crop(prefix: str, c: Dict[str, Any]) -> str:
        cxcy = c.get("center_xy")
        wh = c.get("wh")
        ref = c.get("ref_wh")
        s = prefix
        if isinstance(cxcy, (list, tuple)) and len(cxcy) >= 2:
            s += f" center=({float(cxcy[0]):.2f},{float(cxcy[1]):.2f})"
        if isinstance(wh, (list, tuple)) and len(wh) >= 2:
            s += f" wh={int(wh[0])}x{int(wh[1])}"
        if isinstance(ref, (list, tuple)) and len(ref) >= 2:
            s += f" ref={int(ref[0])}x{int(ref[1])}"
        return s

    if isinstance(crop_i, dict):
        lines.append(_fmt_crop("crop(init):", crop_i))
    if isinstance(crop_f, dict):
        lines.append(_fmt_crop("crop(final):", crop_f))

    ac_cfg = debug.get("autocrop") or {}
    ac_enabled = bool(isinstance(ac_cfg, dict) and ac_cfg.get("enable"))
    if ac_enabled:
        parts = ["autocrop: enabled"]
        stats = ac_cfg.get("stats") if isinstance(ac_cfg.get("stats"), dict) else {}
        if isinstance(stats, dict):
            status = stats.get("status")
            if status:
                parts.append(f"status={status}")
            applied = stats.get("applied")
            if applied is not None:
                parts.append(f"applied={int(bool(applied))}")
            wh = stats.get("frame_wh")
            if isinstance(wh, (list, tuple)) and len(wh) >= 2:
                parts.append(f"out={int(wh[0])}x{int(wh[1])}")
            n_samples = stats.get("n_samples")
            n_valid = stats.get("n_valid")
            if n_samples is not None or n_valid is not None:
                bits = []
                if n_samples is not None:
                    bits.append(f"samples={int(n_samples)}")
                if n_valid is not None:
                    bits.append(f"valid={int(n_valid)}")
                if bits:
                    parts.append(",".join(bits))
            cxy = stats.get("center_xy")
            space = str(stats.get("center_space") or "")
            if isinstance(cxy, (list, tuple)) and len(cxy) >= 2:
                s = f"center=({float(cxy[0]):.2f},{float(cxy[1]):.2f})"
                if space:
                    s += f"[{space}]"
                parts.append(s)
        lines.append(" | ".join(parts))

        rejects = ac_cfg.get("rejects")
        if isinstance(rejects, dict) and rejects:
            pairs = []
            for k, v in rejects.items():
                try:
                    pairs.append((str(k), int(v)))
                except Exception:
                    continue
            pairs.sort(key=lambda kv: (-kv[1], kv[0]))
            if pairs:
                msg = ", ".join([f"{k}={v}" for (k, v) in pairs[:6]])
                lines.append(f"autocrop(rejects): {msg}")
    else:
        lines.append("autocrop: disabled")

    out_dir_s = str(debug.get("output_dir") or "")

    # best line summary
    if results:
        best = min(results, key=lambda r: float(getattr(r.breakdown, "score", 0.0)))
        try:
            score = float(best.breakdown.score)
        except Exception:
            score = float("nan")
        pa1 = float(getattr(best, "pa", 0.0))
        pa2 = getattr(best, "pa2", None)
        rr = getattr(best, "grid_row", None)
        cc = getattr(best, "grid_col", None)
        if pa2 is None:
            s = f"best: p={pa1:.6f} score={score:.3f}"
        else:
            s = f"best: p={pa1:.6f},{float(pa2):.6f} score={score:.3f}"
        if rr is not None and cc is not None:
            s += f" [r{int(rr)},c{int(cc)}]"
        lines.append(s)

    # warnings (autocrop first)
    warnings = debug.get("warnings") or []
    if isinstance(warnings, (list, tuple)) and warnings:
        ac_warn = [str(w) for w in warnings if str(w).lower().startswith("autocrop:")]
        other_warn = [str(w) for w in warnings if not str(w).lower().startswith("autocrop:")]
        show = ac_warn[:6] + other_warn[:6]
        if show:
            lines.append("warnings:")
            for w in show:
                lines.append(f"- {w}")

    # Draw text
    x = BORDER + 8
    y = BORDER + 18
    max_w = w_text - 2 * BORDER - 16
    font_scale = 0.45
    thickness = 1
    line_h = 18

    for ln in lines:
        for seg in _wrap_text_to_width(ln, font=font, font_scale=font_scale, thickness=thickness, max_w=max_w):
            if y + line_h > height_px - BORDER:
                break
            cv2.putText(panel, seg, (x, y), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
            y += line_h
        if y + line_h > height_px - BORDER:
            break

    # Images on the right (autocrop previews)
    if w_imgs >= 260 and out_dir_s:
        img_area_x0 = w_text + 2 * BORDER
        img_area_w = width_px - img_area_x0 - BORDER
        slot_w = max(1, (img_area_w - 2 * BORDER) // 3)
        slot_h = height_px - 2 * BORDER

        paths = [
            ("ac_preview", Path(out_dir_s) / "autocrop_preview.jpg"),
            ("ac_final", Path(out_dir_s) / "autocrop_final.jpg"),
            ("ac_mask", Path(out_dir_s) / "autocrop_mask.jpg"),
        ]

        sx = img_area_x0
        for label, p in paths:
            tile = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
            tile[:] = BG_COLOR
            img = None
            if p.exists():
                try:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                except Exception:
                    img = None
            if img is not None:
                tile = _center_crop_resize(img, slot_w, slot_h, bg_color=BG_COLOR)

            # label strip
            overlay = tile.copy()
            cv2.rectangle(overlay, (0, 0), (slot_w - 1, 18), (0, 0, 0), -1)
            tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
            cv2.putText(tile, label, (6, 14), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            panel[BORDER : BORDER + slot_h, sx : sx + slot_w] = tile
            sx += slot_w + BORDER

    cv2.rectangle(panel, (0, 0), (width_px - 1, height_px - 1), BORDER_COLOR, 1)
    return panel


def _render_dashboard_rows(results: List) -> Optional[np.ndarray]:
    """Original per-line dashboard (RAW/MASK/TRACK + heightmap per row)."""
    if not results:
        return None

    H_ROW = 120
    W_INFO = 260
    W_IMG = 100
    W_PLOT = 500
    BORDER = 2
    BG_COLOR = (30, 30, 30)
    BORDER_COLOR = (100, 100, 100)
    TEXT_COLOR = (220, 220, 220)

    sorted_res = sorted(results, key=lambda r: r.pa)

    global_limit = _global_limit_from_results(results)

    rows = []
    total_w = W_INFO + 3 * W_IMG + W_PLOT + 4 * BORDER
    header = np.zeros((40, total_w, 3), dtype=np.uint8)
    header[:] = BG_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX

    def put_txt(img, text, x):
        cv2.putText(img, text, (x, 25), font, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    put_txt(header, "PA | SCORE", 10)
    off = W_INFO + BORDER
    put_txt(header, "RAW", off + 10)
    off += W_IMG + BORDER
    put_txt(header, "MASK", off + 10)
    off += W_IMG + BORDER
    put_txt(header, "TRACK", off + 10)
    off += W_IMG + BORDER
    put_txt(header, f"DEVIATION +/-{global_limit:.2f}px", off + 10)
    rows.append(header)

    def format_thumb(img: np.ndarray) -> np.ndarray:
        return _center_crop_resize(img, W_IMG, H_ROW, bg_color=BG_COLOR)

    for res in sorted_res:
        if res.thumb_crop is None:
            continue

        panel = np.zeros((H_ROW, W_INFO, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        score_bgr = (0, 255, 0) if res.breakdown.score < 1.0 else (0, 165, 255)
        if res.breakdown.score > 2.0:
            score_bgr = (0, 0, 255)

        z_range = 0.0
        if res.height_map is not None:
            vals = res.height_map[np.isfinite(res.height_map)]
            if vals.size > 0:
                z_range = float(np.max(vals) - np.min(vals))


        kind = str(getattr(res, "height_map_kind", ""))
        kind_s = kind.replace("triangulated", "tri").replace("pixel", "px").replace("_gate", "G")

        cv2.putText(panel, f"PA: {res.pa:.6f}", (10, 30), font, 0.7, (255, 255, 255), 1)
        cv2.putText(panel, f"Score: {res.breakdown.score:.3f}", (10, 60), font, 0.7, score_bgr, 1)
        cv2.putText(panel, f"Pk-Pk: {z_range:.2f}px | {kind_s}", (10, 90), font, 0.45, (150, 150, 150), 1)

        t_raw = format_thumb(res.thumb_crop)
        t_mask = format_thumb(res.thumb_mask)
        t_track = format_thumb(res.thumb_track)

        if res.height_map is not None:
            t_plot = render_heightmap_plot(
                res.height_map,
                width_px=W_PLOT,
                height_px=H_ROW,
                limit_scale=global_limit,
            )
        else:
            t_plot = np.zeros((H_ROW, W_PLOT, 3), dtype=np.uint8)

        v_sep = np.full((H_ROW, BORDER, 3), BORDER_COLOR, dtype=np.uint8)
        row_img = np.hstack([panel, v_sep, t_raw, v_sep, t_mask, v_sep, t_track, v_sep, t_plot])

        rows.append(row_img)
        rows.append(np.full((BORDER, row_img.shape[1], 3), BORDER_COLOR, dtype=np.uint8))

    if not rows:
        return None

    return np.vstack(rows)


def _render_dashboard_grid(results: List) -> Optional[np.ndarray]:
    """Grid overview body for grid patterns: table + tiled heightmaps."""
    if not results:
        return None

    BORDER = 2
    BG_COLOR = (30, 30, 30)
    BORDER_COLOR = (100, 100, 100)
    TEXT_COLOR = (220, 220, 220)

    font = cv2.FONT_HERSHEY_SIMPLEX

    placed = [
        r
        for r in results
        if getattr(r, "grid_row", None) is not None and getattr(r, "grid_col", None) is not None
    ]
    if not placed:
        return None

    max_r = max(int(r.grid_row) for r in placed)
    max_c = max(int(r.grid_col) for r in placed)
    nrows = max_r + 1
    ncols = max_c + 1

    global_limit = _global_limit_from_results(results)

    W_TILE = 220
    H_TILE = 140
    LABEL_H = 20
    W_THUMB = 100
    H_THUMB = 120

    grid_w = ncols * W_TILE + (ncols + 1) * BORDER
    grid_h = nrows * H_TILE + (nrows + 1) * BORDER

    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid_img[:] = BG_COLOR

    thumbs_w = 3 * W_THUMB + 4 * BORDER
    thumbs_img = np.zeros((grid_h, thumbs_w, 3), dtype=np.uint8)
    thumbs_img[:] = BG_COLOR

    by_cell = {(int(r.grid_row), int(r.grid_col)): r for r in placed}
    for rr in range(nrows):
        for cc in range(ncols):
            x0 = BORDER + cc * (W_TILE + BORDER)
            y0 = BORDER + rr * (H_TILE + BORDER)
            cell = by_cell.get((rr, cc), None)

            if cell is not None and cell.height_map is not None:
                tile = render_heightmap_plot(
                    cell.height_map,
                    width_px=W_TILE,
                    height_px=H_TILE,
                    limit_scale=global_limit,
                )
            else:
                tile = np.zeros((H_TILE, W_TILE, 3), dtype=np.uint8)
                tile[:] = (20, 20, 20)

            overlay = tile.copy()
            cv2.rectangle(overlay, (0, 0), (W_TILE - 1, LABEL_H), (0, 0, 0), -1)
            tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)

            if cell is not None:
                score = float(cell.breakdown.score)
                score_bgr = (0, 255, 0) if score < 1.0 else (0, 165, 255)
                if score > 2.0:
                    score_bgr = (0, 0, 255)

                pa1 = float(cell.pa)
                pa2 = getattr(cell, "pa2", None)
                if pa2 is None:
                    lbl = f"r{rr} c{cc} | p={pa1:.4f} | s={score:.2f}"
                else:
                    lbl = f"r{rr} c{cc} | p={pa1:.4f},{float(pa2):.4f} | s={score:.2f}"

                cv2.putText(tile, lbl, (6, 15), font, 0.42, score_bgr, 1, cv2.LINE_AA)
            else:
                cv2.putText(tile, f"r{rr} c{cc}", (6, 15), font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

            grid_img[y0 : y0 + H_TILE, x0 : x0 + W_TILE] = tile

    row_cells: Dict[int, List[Any]] = {}
    for r in placed:
        row_cells.setdefault(int(r.grid_row), []).append(r)

    mid_col = (ncols - 1) / 2.0
    for rr in range(nrows):
        row_candidates = row_cells.get(rr, [])
        chosen = None
        if row_candidates:
            with_thumbs = [c for c in row_candidates if c.thumb_crop is not None]
            pick_from = with_thumbs or row_candidates
            chosen = min(pick_from, key=lambda c: abs(int(c.grid_col) - mid_col))

        thumbs = [
            getattr(chosen, "thumb_crop", None) if chosen is not None else None,
            getattr(chosen, "thumb_mask", None) if chosen is not None else None,
            getattr(chosen, "thumb_track", None) if chosen is not None else None,
        ]

        y0 = BORDER + rr * (H_TILE + BORDER)
        y_thumb = y0 + (H_TILE - H_THUMB) // 2
        x_thumb = BORDER
        for img in thumbs:
            tile = _center_crop_resize(img, W_THUMB, H_THUMB, bg_color=BG_COLOR)
            thumbs_img[y_thumb : y_thumb + H_THUMB, x_thumb : x_thumb + W_THUMB] = tile
            x_thumb += W_THUMB + BORDER

    # Table block.
    sorted_res = sorted(
        results,
        key=lambda r: (
            1 if (getattr(r, "grid_row", None) is None or getattr(r, "grid_col", None) is None) else 0,
            getattr(r, "grid_row", 0) or 0,
            getattr(r, "grid_col", 0) or 0,
            float(getattr(r, "pa", 0.0)),
        ),
    )

    W_TABLE = 360
    ROW_H = 22
    HEADER_H = 28
    needed_h = HEADER_H + ROW_H * (len(sorted_res) + 1) + BORDER
    table_h = max(grid_h, needed_h)

    def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] >= target_h:
            return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        pad[:] = BG_COLOR
        return np.vstack([img, pad])

    grid_img = _pad_h(grid_img, table_h)
    thumbs_img = _pad_h(thumbs_img, table_h)

    table = np.zeros((table_h, W_TABLE, 3), dtype=np.uint8)
    table[:] = BG_COLOR

    cv2.rectangle(table, (0, 0), (W_TABLE - 1, HEADER_H), (20, 20, 20), -1)
    cv2.putText(table, "r,c", (8, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "p1", (70, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "p2", (160, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "score", (250, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "k", (320, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    y = HEADER_H + ROW_H
    for r in sorted_res:
        rr = getattr(r, "grid_row", None)
        cc = getattr(r, "grid_col", None)
        rcs = "-" if rr is None or cc is None else f"{int(rr)},{int(cc)}"

        pa1 = float(getattr(r, "pa", 0.0))
        pa2 = getattr(r, "pa2", None)
        pa2_s = "" if pa2 is None else f"{float(pa2):.4f}"

        score = float(r.breakdown.score)
        score_bgr = (0, 255, 0) if score < 1.0 else (0, 165, 255)
        if score > 2.0:
            score_bgr = (0, 0, 255)

        cv2.putText(table, rcs, (8, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(table, f"{pa1:.4f}", (70, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(table, pa2_s, (160, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(table, f"{score:.3f}", (250, y), font, 0.45, score_bgr, 1, cv2.LINE_AA)

        kind = str(getattr(r, "height_map_kind", ""))
        kind_s = kind.replace("triangulated", "tri").replace("pixel", "px").replace("_gate", "G")
        cv2.putText(table, kind_s, (320, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        y += ROW_H
        if y + ROW_H > table_h:
            break

    v_sep = np.full((table_h, BORDER, 3), BORDER_COLOR, dtype=np.uint8)
    return np.hstack([table, v_sep, thumbs_img, v_sep, grid_img])


def save_dashboard(results: List, out_path: Path, *, debug: Optional[Dict[str, Any]] = None) -> None:
    """Save analysis overview image.

    For grid patterns (results containing grid_row/grid_col), the heightmaps are
    laid out as a grid (tiles) next to a summary table. For line patterns, the
    original per-line dashboard is used.

    Additionally, a debug panel is included between the score plot and the body.
    """
    if not results:
        return

    is_grid = any(
        (getattr(r, "grid_row", None) is not None and getattr(r, "grid_col", None) is not None)
        for r in results
    )

    body: Optional[np.ndarray]
    if is_grid:
        body = _render_dashboard_grid(results)
        if body is None:
            body = _render_dashboard_rows(results)
    else:
        body = _render_dashboard_rows(results)

    if body is None:
        return

    width = int(body.shape[1])

    if is_grid:
        xs = []
        ys = []
        zs = []
        for r in results:
            pa2 = getattr(r, "pa2", None)
            if pa2 is None:
                continue
            xs.append(float(getattr(r, "pa", 0.0)))
            ys.append(float(pa2))
            zs.append(float(getattr(r.breakdown, "score", 0.0)))
        plot = render_score_surface_plot(xs, ys, zs, width_px=width, height_px=width)
    else:
        xs = [float(getattr(r, "pa", 0.0)) for r in results]
        ys = [float(r.breakdown.score) for r in results]
        plot = render_score_lineplot(xs, ys, width_px=width, height_px=220)

    dbg = _render_debug_panel(results=results, debug=debug, width_px=width, height_px=220)

    BORDER = 2
    BORDER_COLOR = (100, 100, 100)
    h_sep = np.full((BORDER, width, 3), BORDER_COLOR, dtype=np.uint8)

    out = np.vstack([plot, h_sep, dbg, h_sep, body])
    cv2.imwrite(str(out_path), out)

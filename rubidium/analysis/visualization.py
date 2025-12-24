# rubidium/analysis/visualization.py
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

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


def _save_dashboard_rows(results: List, out_path: Path) -> None:
    """Original per-line dashboard (RAW/MASK/TRACK + heightmap per row)."""
    if not results:
        return

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

        cv2.putText(panel, f"PA: {res.pa:.6f}", (10, 30), font, 0.7, (255, 255, 255), 1)
        cv2.putText(panel, f"Score: {res.breakdown.score:.3f}", (10, 60), font, 0.7, score_bgr, 1)
        cv2.putText(panel, f"Pk-Pk: {z_range:.2f}px", (10, 90), font, 0.5, (150, 150, 150), 1)

        def format_thumb(img):
            h, w = img.shape[:2]
            s = H_ROW / float(h)
            new_w = int(w * s)
            resized = cv2.resize(img, (new_w, H_ROW), interpolation=cv2.INTER_AREA)
            out = np.zeros((H_ROW, W_IMG, 3), dtype=np.uint8)
            out[:] = BG_COLOR
            if new_w >= W_IMG:
                start = (new_w - W_IMG) // 2
                out = resized[:, start : start + W_IMG]
            else:
                out[:, 0:new_w] = resized
            return out

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
        return
    dashboard = np.vstack(rows)
    cv2.imwrite(str(out_path), dashboard)


def _save_dashboard_grid(results: List, out_path: Path) -> None:
    """Grid overview for grid patterns: table + tiled heightmaps."""
    if not results:
        return

    BORDER = 2
    BG_COLOR = (30, 30, 30)
    BORDER_COLOR = (100, 100, 100)
    TEXT_COLOR = (220, 220, 220)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Determine grid extents.
    placed = [r for r in results if getattr(r, "grid_row", None) is not None and getattr(r, "grid_col", None) is not None]
    if not placed:
        # fallback
        _save_dashboard_rows(results, out_path)
        return

    max_r = max(int(r.grid_row) for r in placed)
    max_c = max(int(r.grid_col) for r in placed)
    nrows = max_r + 1
    ncols = max_c + 1

    global_limit = _global_limit_from_results(results)

    # Tile sizes: keep readable but not enormous.
    W_TILE = 220
    H_TILE = 140
    LABEL_H = 20

    grid_w = ncols * W_TILE + (ncols + 1) * BORDER
    grid_h = nrows * H_TILE + (nrows + 1) * BORDER

    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid_img[:] = BG_COLOR

    # Place tiles.
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

            # Label strip (top overlay).
            overlay = tile.copy()
            cv2.rectangle(overlay, (0, 0), (W_TILE - 1, LABEL_H), (0, 0, 0), -1)
            alpha = 0.55
            tile = cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0)

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

    # Table block.
    sorted_res = sorted(results, key=lambda r: (
        1 if (getattr(r, "grid_row", None) is None or getattr(r, "grid_col", None) is None) else 0,
        getattr(r, "grid_row", 0) or 0,
        getattr(r, "grid_col", 0) or 0,
        float(getattr(r, "pa", 0.0)),
    ))

    W_TABLE = 360
    ROW_H = 22
    HEADER_H = 28
    needed_h = HEADER_H + ROW_H * (len(sorted_res) + 1) + BORDER
    table_h = max(grid_h, needed_h)

    table = np.zeros((table_h, W_TABLE, 3), dtype=np.uint8)
    table[:] = BG_COLOR

    # Header
    cv2.rectangle(table, (0, 0), (W_TABLE - 1, HEADER_H), (20, 20, 20), -1)
    cv2.putText(table, "r,c", (8, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "p1", (70, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "p2", (160, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(table, "score", (250, 20), font, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

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

        y += ROW_H
        if y + ROW_H > table_h:
            break

    # Top line plot (score vs param1).
    xs = [float(r.pa) for r in results]
    ys = [float(r.breakdown.score) for r in results]
    W_TOTAL = W_TABLE + BORDER + grid_w
    H_PLOT = 220
    plot = render_score_lineplot(xs, ys, width_px=W_TOTAL, height_px=H_PLOT)

    v_sep = np.full((table_h, BORDER, 3), BORDER_COLOR, dtype=np.uint8)
    body = np.hstack([table, v_sep, grid_img])

    h_sep = np.full((BORDER, W_TOTAL, 3), BORDER_COLOR, dtype=np.uint8)
    out = np.vstack([plot, h_sep, body])

    cv2.imwrite(str(out_path), out)


def save_dashboard(results: List, out_path: Path) -> None:
    """Save analysis overview image.

    For grid patterns (results containing grid_row/grid_col), the heightmaps are
    laid out as a grid (tiles) next to a summary table. For line patterns, the
    original per-line dashboard is used.
    """
    if not results:
        return

    is_grid = any(
        (getattr(r, "grid_row", None) is not None and getattr(r, "grid_col", None) is not None)
        for r in results
    )
    if is_grid:
        _save_dashboard_grid(results, out_path)
    else:
        _save_dashboard_rows(results, out_path)

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
    limit_scale: Optional[float] = None
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm_t = height_map.T 
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if limit_scale is None:
        vals = hm_t[np.isfinite(hm_t)]
        limit = np.percentile(np.abs(vals), 99) if vals.size > 0 else 1.0
        limit = max(limit, 0.1)
    else:
        limit = limit_scale

    ax.imshow(hm_t, aspect="auto", interpolation="nearest", vmin=-limit, vmax=limit, cmap="viridis")
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="png", dpi=dpi)
    plt.close(fig)
    io_buf.seek(0)
    
    file_bytes = np.asarray(bytearray(io_buf.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img.shape[0] != height_px or img.shape[1] != width_px:
        img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)
        
    return img

def create_debug_thumbnails(crop: np.ndarray, mask: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def save_dashboard(results: List, out_path: Path) -> None:
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
    
    all_abs_vals = []
    for res in results:
        if res.height_map is not None:
            valid = res.height_map[np.isfinite(res.height_map)]
            if valid.size > 0:
                all_abs_vals.append(np.abs(valid))
    
    global_limit = 1.0
    if all_abs_vals:
        concat = np.concatenate(all_abs_vals)
        if concat.size > 0:
            global_limit = float(np.percentile(concat, 99))
            if global_limit < 0.1: global_limit = 0.1

    rows = []
    total_w = W_INFO + 3*W_IMG + W_PLOT + 4*BORDER
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
        if res.thumb_crop is None: continue

        panel = np.zeros((H_ROW, W_INFO, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        
        score_bgr = (0, 255, 0) if res.breakdown.score < 1.0 else (0, 165, 255)
        if res.breakdown.score > 2.0: score_bgr = (0, 0, 255)

        z_range = 0.0
        if res.height_map is not None:
             vals = res.height_map[np.isfinite(res.height_map)]
             if vals.size > 0: z_range = np.max(vals) - np.min(vals)

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
                out = resized[:, start:start+W_IMG]
            else:
                out[:, 0:new_w] = resized
            return out

        t_raw = format_thumb(res.thumb_crop)
        t_mask = format_thumb(res.thumb_mask)
        t_track = format_thumb(res.thumb_track)

        if res.height_map is not None:
            t_plot = render_heightmap_plot(res.height_map, width_px=W_PLOT, height_px=H_ROW, limit_scale=global_limit)
        else:
            t_plot = np.zeros((H_ROW, W_PLOT, 3), dtype=np.uint8)

        v_sep = np.full((H_ROW, BORDER, 3), BORDER_COLOR, dtype=np.uint8)
        row_img = np.hstack([panel, v_sep, t_raw, v_sep, t_mask, v_sep, t_track, v_sep, t_plot])
        
        rows.append(row_img)
        rows.append(np.full((BORDER, row_img.shape[1], 3), BORDER_COLOR, dtype=np.uint8))

    if not rows: return
    dashboard = np.vstack(rows)
    cv2.imwrite(str(out_path), dashboard)
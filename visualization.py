import cv2
import numpy as np
from matplotlib import colormaps

# HSV lookup table point circles (plt.cm.get_cmap("hsv", 256));
_HSV_LUT = (np.array([colormaps["hsv"](i) for i in range(256)])[:, :3] * 255).astype(np.uint8)

def depth_to_color(depth_map, vmax):
    """Colorize a dense depth map:
    near = high index (red end), far = low index (violet end),
    normalized by vmax and vectorized over the whole map."""
    normalized = np.clip(depth_map, 0, vmax) / vmax
    idx = ((1.0 - normalized) * 255).astype(np.uint8)
    color = _HSV_LUT[idx].copy()
    color[depth_map <= 0] = 0  # invalid (no depth) -> blank
    return color

def overlay_points_on_image(image, pts_2d, depths, vmax, alpha=0.25):
    """Draw a filled circle at each 2D point, colored by depth.
    vmax: depth (m) at the far end of the color scale, deeper points get the same color.
    alpha: opacity of the dots, 1 = opaque."""
    out = image.copy()
    normalized = np.clip(depths, 0, vmax) / vmax
    idx = ((1.0 - normalized) * 255).astype(np.uint8)
    colors = _HSV_LUT[idx]
    for (x, y), color in zip(pts_2d, colors):
        center = (int(np.round(x)), int(np.round(y)))
        radius = 2
        thickness=-1 # fill the circle
        cv2.circle(out, center, radius, color=tuple(int(c) for c in color), thickness=thickness)
    # blend with the original: pixels without a dot are the same in both, so they stay unchanged
    return cv2.addWeighted(out, alpha, image, 1 - alpha, 0)

def _put_text(image, text, org, scale, color, outline=None):
    """Text starting at org = (x, y of the baseline) (in place).
    outline: color drawn around the letters, keeps the text readable over other content."""
    x, y = org
    if outline is not None:
        # the same text shifted by 1 px in all 8 directions; a thicker putText is wider in OpenCV 5, so it wouldn't line up
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            cv2.putText(image, text, (x + dx, y + dy), cv2.FONT_HERSHEY_PLAIN, scale, outline, 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, 1)

def draw_panel_label(image, text, color=(255, 255, 255), outline=None, legend=None):
    """Small label in a panel's top-left corner.
    outline: color drawn around the letters, keeps the label readable over other content."""
    out = image.copy()
    x, y, scale = 6, 18, 0.9
    _put_text(out, text, (x, y), scale, color, outline)
    if legend is not None:
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 1)[0][0] + 12
        draw_legend(out, legend, (x, y), color, outline)
    return out

LEGEND_SCALE = 0.7  # font scale of the legends, as the BEV grid labels
LEGEND_SWATCH = 8   # side of a color square and height of a color ramp (px)

def draw_legend(image, entries, org, color=(0, 0, 0), outline=None, ramp_width=50, size=1):
    """One-line legend (in place), starting at org = (x, y of the text baseline).
    entries: (swatch, text) pairs, each drawn as the swatch followed by the text.
    swatch: an RGB color for a square, (N, 3) colors for a ramp of ramp_width px, or None for text only.
    outline: color drawn around the letters and the swatches."""
    x, y = org
    font_scale, swatch_size = LEGEND_SCALE * size, round(LEGEND_SWATCH * size)
    for swatch, text in entries:
        if swatch is not None:
            swatch = np.asarray(swatch)
            is_ramp = swatch.ndim == 2
            width = ramp_width if is_ramp else swatch_size
            # a ramp is resampled to its width; a single color fills the square
            colors = swatch[np.linspace(0, len(swatch) - 1, width).astype(int)] if is_ramp else swatch
            if outline is not None:
                cv2.rectangle(image, (x - 1, y - swatch_size - 1), (x + width, y), outline, 1)
            image[y - swatch_size:y, x:x + width] = colors
            x += width + round(3 * size)
        _put_text(image, text, (x, y), font_scale, color, outline)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, 1)[0][0] + round(10 * size)

def draw_depth_legend(image, vmax, title):
    """Legend for overlay_points_on_image in the image's top-left corner (in place):
    the title, then the color ramp from 0 to vmax m. Twice the size of the panel legends."""
    draw_legend(image, [(None, f"{title} 0 m"), (_HSV_LUT[::-1], f"{vmax:g} m")], (6, 28),
                color=(255, 255, 255), outline=(0, 0, 0), ramp_width=280, size=2)

def draw_table(image, title, rows, scale=1.6, dim=0.25):
    """Table in white over the dimmed image, centered (returns a copy): the title, then the rows.
    rows: lists of strings, one per column; the first column is left-aligned, the others right-aligned.
    dim: brightness of the image under the table, 1 = unchanged."""
    out = (image * dim).astype(np.uint8)

    def text_width(text):
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, 1)[0][0]

    row_height, gap = round(26 * scale), round(28 * scale)  # between baselines, between columns (px)
    # each column is as wide as its widest cell
    col_widths = []
    for col in range(len(rows[0])):
        cells = [row[col] for row in rows]
        col_widths.append(max(text_width(cell) for cell in cells))
    x0 = (out.shape[1] - sum(col_widths) - gap * (len(col_widths) - 1)) // 2
    y = (out.shape[0] - row_height * len(rows)) // 2  # baseline of the title

    _put_text(out, title, (x0, y), scale, (255, 255, 255))
    for row in rows:
        y += row_height
        x = x0
        for col, (cell, col_width) in enumerate(zip(row, col_widths)):
            _put_text(out, cell, (x if col == 0 else x + col_width - text_width(cell), y), scale, (255, 255, 255))
            x += col_width + gap
    return out

CHART_COLORS = ((0, 0, 0), (0, 114, 178), (204, 121, 167))  # black, blue, purple: not used in the agreement panels
CHART_GRID_GREY = (230, 230, 230)
CHART_LABEL_GREY = (150, 150, 150)

def draw_iou_chart(ious, names, width, first_frame=0, height=150):
    """IoU chart (height, width, 3), RGB uint8: one line per source over the frames of a drive.
    ious: (frames, sources), nan where a frame has no value (yet), so the lines grow as the drive is processed.
    names: one per source for the legend, colors from CHART_COLORS.
    x: frame, counted from first_frame, y: IoU from 0 to 1. A dot marks each line's latest value."""
    chart = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 30, width - 12, 24, height - 16  # plot area (px)

    for value in (0, 0.25, 0.5, 0.75, 1):
        y = round(bottom - value * (bottom - top))
        cv2.line(chart, (left, y), (right, y), CHART_GRID_GREY, 1)
        if value in (0, 0.5, 1):
            _put_text(chart, f"{value:g}", (6, y + 4), LEGEND_SCALE, CHART_LABEL_GREY)
    last_frame = str(first_frame + len(ious) - 1)
    last_frame_width = cv2.getTextSize(last_frame, cv2.FONT_HERSHEY_PLAIN, LEGEND_SCALE, 1)[0][0]
    _put_text(chart, f"frame {first_frame}", (left, height - 4), LEGEND_SCALE, CHART_LABEL_GREY)
    _put_text(chart, last_frame, (right - last_frame_width, height - 4), LEGEND_SCALE, CHART_LABEL_GREY)

    xs = np.round(np.linspace(left, right, len(ious))).astype(int)
    for series, color in zip(ious.T, CHART_COLORS):
        valid = ~np.isnan(series)
        ys = np.round(bottom - np.nan_to_num(series) * (bottom - top)).astype(int)
        # a segment between neighboring frames that both have a value
        for i in np.flatnonzero(valid[:-1] & valid[1:]):
            cv2.line(chart, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), color, 2, cv2.LINE_AA)
        if valid.any():
            latest = np.flatnonzero(valid)[-1]
            cv2.circle(chart, (int(xs[latest]), int(ys[latest])), 3, color, -1, cv2.LINE_AA)

    draw_legend(chart, [(None, "IoU per frame, whole window")] + list(zip(CHART_COLORS, names)), (left, 14))
    return chart

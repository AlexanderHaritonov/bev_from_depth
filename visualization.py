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

def draw_panel_label(image, text, color=(255, 255, 255), outline=None):
    """Small label in a panel's top-left corner.
    outline: color drawn around the letters, keeps the label readable over other content."""
    out = image.copy()
    _put_text(out, text, (6, 18), 0.9, color, outline)
    return out

LEGEND_SCALE = 0.7  # font scale of the legends, as the BEV grid labels
LEGEND_SWATCH = 8   # side of a color square and height of a color ramp (px)

def draw_legend(image, entries, org, color=(0, 0, 0), outline=None, ramp_width=50):
    """One-line legend (in place), starting at org = (x, y of the text baseline).
    entries: (swatch, text) pairs, each drawn as the swatch followed by the text.
    swatch: an RGB color for a square, (N, 3) colors for a ramp of ramp_width px, or None for text only.
    outline: color drawn around the letters and the swatches."""
    x, y = org
    for swatch, text in entries:
        if swatch is not None:
            swatch = np.asarray(swatch)
            is_ramp = swatch.ndim == 2
            width = ramp_width if is_ramp else LEGEND_SWATCH
            # a ramp is resampled to its width; a single color fills the square
            colors = swatch[np.linspace(0, len(swatch) - 1, width).astype(int)] if is_ramp else swatch
            if outline is not None:
                cv2.rectangle(image, (x - 1, y - LEGEND_SWATCH - 1), (x + width, y), outline, 1)
            image[y - LEGEND_SWATCH:y, x:x + width] = colors
            x += width + 3
        _put_text(image, text, (x, y), LEGEND_SCALE, color, outline)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, LEGEND_SCALE, 1)[0][0] + 10

def draw_depth_legend(image, vmax, title):
    """Legend for overlay_points_on_image in the image's top-left corner (in place):
    the title, then the color ramp from 0 to vmax m."""
    draw_legend(image, [(None, f"{title} 0 m"), (_HSV_LUT[::-1], f"{vmax:g} m")], (6, 14),
                color=(255, 255, 255), outline=(0, 0, 0), ramp_width=140)

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

def draw_panel_label(image, text, color=(255, 255, 255), outline=None):
    """Small label in a panel's top-left corner.
    outline: color drawn around the letters, keeps the label readable over other content."""
    out = image.copy()
    if outline is not None:
        # the same text shifted by 1 px in all 8 directions; a thicker putText is wider in OpenCV 5, so it wouldn't line up
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            cv2.putText(out, text, (6 + dx, 18 + dy), cv2.FONT_HERSHEY_PLAIN, 0.9, outline, 1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
    return out

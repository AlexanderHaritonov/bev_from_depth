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

def overlay_points_on_image(image, pts_2d, depths, vmax):
    """Draw a filled circle at each 2D point, colored by depth."""
    out = image.copy()
    normalized = np.clip(depths, 0, vmax) / vmax
    idx = ((1.0 - normalized) * 255).astype(np.uint8)
    colors = _HSV_LUT[idx]
    for (x, y), color in zip(pts_2d, colors):
        cv2.circle(out, (int(np.round(x)), int(np.round(y))), 2,
                   color=tuple(int(c) for c in color), thickness=-1)
    return out

def draw_panel_label(image, text, color=(255, 255, 255)):
    """Small label in a panel's top-left corner."""
    out = image.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
    return out

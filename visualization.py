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

def draw_panel_label(image, text, color=(255, 255, 255)):
    """Small label in a panel's top-left corner."""
    out = image.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
    return out

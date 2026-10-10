import numpy as np

from constants import BEV_PARAMS

# Reference (no numpy), same result as depth_to_points below:
#
# def depth_to_points(depth, P):
#     fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
#     h, w = len(depth), len(depth[0])
#     points = [[None] * w for _ in range(h)]
#     for v in range(h):
#         for u in range(w):
#             z = depth[v][u]
#             x = (u - cx) * z / fx
#             y = (v - cy) * z / fy
#             points[v][u] = [x, y, z]
#     return points

def depth_to_points(depth, P):
    """Back-project a depth map (H, W) to camera-frame 3D points (H, W, 3) as x right, y down, z forward (m)."""
    # P: 3x4 camera projection matrix (intrinsics K + stereo baseline offset)
    
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)

def depth_jumps(depth, max_relative_jump):
    """Boolean mask (H, W) of pixels whose depth differs from any valid 4-neighbor by more than max_relative_jump.
    depth: (H, W) in meters, 0 = invalid."""
    
    assert depth.ndim == 2
    jumps = np.zeros(depth.shape, dtype=bool)

    # horizontal neighbors
    a, b = depth[:, :-1], depth[:, 1:]
    j = (a > 0) & (b > 0) & (np.abs(a - b) > max_relative_jump * np.minimum(a, b))
    jumps[:, :-1] |= j
    jumps[:, 1:] |= j

    # vertical neighbors
    a, b = depth[:-1, :], depth[1:, :]
    j = (a > 0) & (b > 0) & (np.abs(a - b) > max_relative_jump * np.minimum(a, b))
    jumps[:-1, :] |= j
    jumps[1:, :] |= j

    return jumps

CAM_HEIGHT = 1.65  # KITTI camera mounting height above ground (m)
MIN_HEIGHT = CAM_HEIGHT - 1.2  # keep points less than 1.2 m below the camera (m)
MAX_HEIGHT = np.inf            # no upper limit

def filter_points_for_bev(points, min_height=MIN_HEIGHT, max_height=MAX_HEIGHT, max_relative_jump=None):
    """Boolean mask (H, W) of points to keep.
    max_relative_jump: None skips the flying-pixel filter (not needed for ground truth; ~0.1 for mono/stereo)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    # valid depth (0 = invalid) and within BEV range
    mask = (z > 0) & (z <= BEV_PARAMS.MAX_Z) & (np.abs(x) <= BEV_PARAMS.MAX_X)

    # height band above ground: drops the road (y points down)
    height = CAM_HEIGHT - y
    mask &= (height > min_height) & (height < max_height)

    # flying pixels: interpolated depth between foreground and background at object edges
    if max_relative_jump is not None:
        mask &= ~depth_jumps(z, max_relative_jump)

    return mask


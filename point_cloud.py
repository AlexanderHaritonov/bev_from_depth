import numpy as np

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
    """Back-project a depth map (H, W) to camera-frame 3D points (H, W, 3): x right, y down, z forward."""
    # P: 3x4 camera projection matrix (intrinsics K + stereo baseline offset)
    
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


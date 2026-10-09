import cv2
import numpy as np
from matplotlib import colormaps

from constants import BEV_PARAMS, EGO_WIDTH, EGO_LENGTH, EGO_FRONT

BEV_MIN_Z, BEV_MAX_Z, BEV_MAX_X = BEV_PARAMS.MIN_Z, BEV_PARAMS.MAX_Y, BEV_PARAMS.MAX_X

BEV_RES = (BEV_MAX_Z - BEV_MIN_Z) / BEV_PARAMS.H  # m per pixel, ~0.19
BEV_W = round(2 * BEV_MAX_X / BEV_RES)            # 514 px

# viridis (matplotlib's default, as in project_depthmap) as a 256-entry RGB lookup table: low = purple, high = yellow
_VIRIDIS_LUT = (colormaps["viridis"](np.arange(256))[:, :3] * 255).astype(np.uint8)

# Reference: project_depthmap, which plots with plt.scatter:
#
# def project_depthmap(cam_points, rgb, cam_pos=-1.2):
#     max_longitudinal = 70
#     window_x = (-30, 30)
#     window_y = (-3, max_longitudinal)
#
#     x, y, z = cam_points
#     # flip the y-axis to positive upwards
#     y = - y
#
#     # We sample points for points less than 70m ahead and above ground
#     # Camera is mounted 1m above on an ego vehicle
#     ind = np.where((z < max_longitudinal) & (y > cam_pos))
#     bird_eye = cam_points[:3, ind]
#
#     # Color by pixels or radial distance
#     dists = np.sqrt(np.sum(bird_eye[0:2:2, :] ** 2, axis=0))
#     axes_limit = 10
#     colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(24, 12))
#     ...  # ax0: camera image
#     ax1.scatter(bird_eye[0, :], bird_eye[2, :], c=colors, s=0.1)
#     ax1.set_xlim(window_x)
#     ax1.set_ylim(window_y)
#     ax1.set_title('Bird Eye View')
#     plt.axis('off')
#     plt.gca().set_aspect('equal')
#     plt.show()

# Reference: project_topview, which colors each point by its camera pixel.
# Lines marked "# fix" differ from the notebook, which took the colors from rgb[y, x] with y, x in meters.
# In our code the same colors are cam[mask].
#
# cam_coords = np.zeros((height * width, 3))
# colors = np.zeros((height * width, 3))  # fix
#
# u0 = intrinsic[0, 2]
# v0 = intrinsic[1, 2]
# fx = intrinsic[0, 0]
# fy = intrinsic[1, 1]
# i = 0
#
# # Loop through each pixel in the image
# for v in range(height):
#     for u in range(width):
#         x = (u - u0) * depth[v, u] / fx
#         y = (v - v0) * depth[v, u] / fy
#         z = depth[v, u]
#         cam_coords[i] = (x, y, z)
#         colors[i] = image[v, u]  # fix: color of this point's pixel (image and depth same size)
#         i += 1
# cam_coords = cam_coords.T
#
# def project_topview(cam_points, rgb, colors):  # fix: colors from the loop
#     max_longitudinal = 70
#     window_x = (-50, 50)
#     window_y = (-3, max_longitudinal)
#
#     x, y, z = cam_points
#     # flip the y-axis to positive upwards
#     y = - y
#
#     # We sample points for points less than 70m ahead and above ground
#     # Camera is mounted 1m above on an ego vehicle
#     ind = np.where((z < max_longitudinal) & (y > -1.2) & (y<2))
#
#     bird_eye = cam_points[:3, ind]
#     filtered_colors = colors[ind[0]]/255.0  # fix
#
#     # Color by pixels or radial distance
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(24, 12))
#     ...  # ax0: camera image
#     ax1.scatter(bird_eye[0, :], bird_eye[2, :], c=filtered_colors, s=0.1)
#     ax1.set_xlim(window_x)
#     ax1.set_ylim(window_y)
#     ax1.set_title('Bird Eye View')
#     plt.axis('off')
#     plt.gca().set_aspect('equal')
#     plt.show()
#
# project_topview(cam_coords, image, colors)  # fix

def points_to_bev(points):
    """BEV image (BEV_PARAMS.H, BEV_W, 3), RGB uint8, from filtered camera-frame points (N, 3),
     drawn: white background, viridis colors by sideways distance.
     Forward (z) is up, x is right."""
    assert points.ndim == 2 and points.shape[1] == 3  # (N, 3), e.g. points[mask], not the (H, W, 3) grid
    x, z = points[:, 0], points[:, 2]

    # meters -> pixels; min() keeps points exactly on the edge (x = BEV_MAX_X) inside the image
    rows = np.minimum(((BEV_MAX_Z - z) / BEV_RES).astype(int), BEV_PARAMS.H - 1)
    cols = np.minimum(((x + BEV_MAX_X) / BEV_RES).astype(int), BEV_W - 1)

    # color by sideways distance |x|, full color at 10 * sqrt(2) = 14.1 m, as project_depthmap
    t = np.minimum(1, np.abs(x) / 10 / np.sqrt(2))

    # points sharing a cell have almost the same x, so it doesn't matter which one is drawn
    bev = np.full((BEV_PARAMS.H, BEV_W, 3), 255, dtype=np.uint8)
    bev[rows, cols] = _VIRIDIS_LUT[(t * 255).astype(int)]
    return bev

def draw_ego_car(bev):
    """Black box for the ego car at the bottom center, around the camera (in place)."""
    left = round((BEV_MAX_X - EGO_WIDTH / 2) / BEV_RES)
    right = round((BEV_MAX_X + EGO_WIDTH / 2) / BEV_RES)
    front = round((BEV_MAX_Z - EGO_FRONT) / BEV_RES)
    rear = round((BEV_MAX_Z - (EGO_FRONT - EGO_LENGTH)) / BEV_RES)
    cv2.rectangle(bev, (left, front), (right, rear), (0, 0, 0), 1)

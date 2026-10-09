from lidar_fusion import velo_to_cam
from mono_depth import get_mono_depth
from point_cloud import depth_to_points, filter_points
from stereo import get_depth_and_disparity

MAX_RELATIVE_JUMP = 0.1  # flying-pixel filter for stereo and mono depth

# Each function returns filtered camera-frame points (N, 3), ready for points_to_bev.

def gt_points(dl, gt_depth):
    points = depth_to_points(gt_depth, dl.P)
    return points[filter_points(points)]

def lidar_points(dl, pc_velo):
    """The raw scan, so wider than the camera view."""
    points = velo_to_cam(pc_velo, dl.P, dl.R0, dl.V2C)
    return points[filter_points(points)]

def stereo_points(dl, left, right, gt_depth):
    depth, _ = get_depth_and_disparity(left, right, dl.fx, dl.baseline)
    return _points_with_gt(dl, depth, gt_depth)

def mono_points(dl, left, mono_model, gt_depth):
    depth = get_mono_depth(left, mono_model)
    return _points_with_gt(dl, depth, gt_depth)

def _points_with_gt(dl, depth, gt_depth):
    """Points from a dense depth map, only at pixels that have ground truth."""
    points = depth_to_points(depth, dl.P)
    # flying pixels are found on the dense map, so before the ground-truth mask
    mask = filter_points(points, max_relative_jump=MAX_RELATIVE_JUMP) & (gt_depth > 0)
    return points[mask]

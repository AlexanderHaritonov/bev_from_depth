import numpy as np

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def velo_to_cam(pts_3d_velo, P, R0, V2C):
    """Velodyne points (N, 3) as x forward, y left, z up (m) to camera-frame points (N, 3) as x right, y down, z forward (m)."""
    # P with the intrinsics taken out, so the result stays in meters
    P_metric = np.linalg.inv(P[:, :3]) @ P # [I | shift to cam2] : [3x3] @ [3x4] -> [3x4]
    return project_velo_to_image_core(pts_3d_velo, P_metric, R0, V2C)

def project_velo_to_image_core(pts_3d_velo, P, R0, V2C):
    '''
    Input: 3D points in Velodyne Frame [nx3]
    Output: P*R0*R|T*X [nx3] -- (u * z_cam, v * z_cam, z_cam), pixels before the division by z_cam.
    z_cam (column 2) is the depth along the camera's own optical axis.
    '''
    R0_homo = np.vstack((R0, [0, 0, 0])) # Add Row of 0s (to be in homogeneous)
    R0_homo_2 = np.hstack((R0_homo, [[0], [0], [0], [1]]))# Add column of 0s (to be in homogeneous)
    p_r0 = P @ R0_homo_2 # P * R0 :                                           [3x4] @ [4x4] -> [3x4]
    p_r0_rt = p_r0 @ np.vstack((V2C, [0, 0, 0, 1])) # R*RO*R|T :               [3x4] @ [4x4] -> [3x4]

    pts_3d_homo = cart2hom(pts_3d_velo) # Add columns 0s(to be in homogeneous) [nx3] -> [nx4]
    p_r0_rt_x = p_r0_rt @ pts_3d_homo.T # P*R0*R|T*X (X being pts_3D)          [3x4] @ [4xn] -> [3xn]
    pts_2d = np.transpose(p_r0_rt_x)
    return pts_2d

def project_velo_to_image(pts_3d_velo, P, R0, V2C):
    pts_2d = project_velo_to_image_core(pts_3d_velo, P, R0, V2C)
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2] #Convert Back to Cartesian
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2] #Convert Back to Cartesian
    return pts_2d

def get_lidar_in_image_fov(pc_velo, P, R0, V2C, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = project_velo_to_image(pc_velo, P, R0, V2C)  # Project Velodyne to Camera Image
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # We don't want things that are closer to the clip distance (2m)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

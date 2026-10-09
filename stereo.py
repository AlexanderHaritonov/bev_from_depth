import cv2
import numpy as np

def create_stereo_matcher():
    """StereoSGBM matcher; parameters resemble the "good calibration" setup
    in advanced_calibration_starter.py (P1/P2 penalties scaled to block_size)."""
    block_size = 9
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 13,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=40,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

def compute_disparity_map(left, right):
    stereo = create_stereo_matcher()
    disparity_map = stereo.compute(left, right).astype(np.float32) / 16.0
    return disparity_map

def compute_depth_map(disparity_map, fx, baseline):
    depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
    valid = disparity_map > 0
    depth_map[valid] = (fx * baseline) / disparity_map[valid]
    return depth_map

def get_depth_and_disparity(left, right, fx, baseline):
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    disparity_map = compute_disparity_map(left_gray, right_gray)
    depth_map = compute_depth_map(disparity_map, fx, baseline)

    return depth_map, disparity_map

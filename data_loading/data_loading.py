import os

import cv2
import numpy as np

class DataLoader:
    """Loads calib and stereo image pairs from a KITTI sync folder."""

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.fx, self.baseline = self._get_fx_and_baseline()
        self._get_lidar_calib()

    def _get_lidar_calib(self):
        """P (cam2 projection),
           R0 (rectifying rotation),
           V2C (velodyne -> cam0 transform).

        Adapted from LiDAR2Camera.__init__ in AH_Week_7_Visual_Fusion.ipynb, which reads a
        single combined calib file (P2 / R0_rect / Tr_velo_to_cam)
        -- KITTI raw splits these across calib_cam_to_cam.txt and calib_velo_to_cam.txt instead."""
        
        calib_dir = os.path.join(os.path.dirname(self.root_folder), "calib")
        cam_to_cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
        velo_to_cam = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))

        P = cam_to_cam["P_rect_02"]
        self.P = np.reshape(P, (3, 4))

        # Rotation from reference camera (cam0) coord to rect camera coord
        R0 = cam_to_cam["R_rect_00"]
        self.R0 = np.reshape(R0, (3, 3))

        # Rigid transform from Velodyne coord to reference camera coord
        R = velo_to_cam["R"].reshape(3, 3)
        T = velo_to_cam["T"].reshape(3, 1)
        self.V2C = np.hstack((R, T))

    def _get_fx_and_baseline(self):
        """Focal length (fx) and stereo baseline (m) for the image_02/image_03 pair."""
        cam_to_cam_calib_file = os.path.join(os.path.dirname(self.root_folder), "calib", "calib_cam_to_cam.txt")
        calib = read_calib_file(cam_to_cam_calib_file)
        P_rect_02 = calib["P_rect_02"].reshape(3, 4)
        P_rect_03 = calib["P_rect_03"].reshape(3, 4)
        fx = P_rect_02[0, 0]
        baseline = (P_rect_02[0, 3] - P_rect_03[0, 3]) / fx
        return fx, baseline

    def load_stereo_pair(self, frame_number):
        """Load the matching left (image_02) / right (image_03) frame from the sync folder."""
        filename = f"{frame_number:010d}.png"
        left_path = os.path.join(self.root_folder, "image_02", "data", filename)
        right_path = os.path.join(self.root_folder, "image_03", "data", filename)
        return load_image(left_path), load_image(right_path)

    def load_point_cloud(self, frame_number):
        """Load a KITTI velodyne .bin scan (x, y, z, reflectance per point) as an Nx3 array."""
        filename = f"{frame_number:010d}.bin"
        path = os.path.join(self.root_folder, "velodyne_points", "data", filename)
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

    def load_depth_gt(self, frame_number):
        """Load the ground-truth depth map (meters, 0 = invalid) from KITTI Depth Completion."""
        filename = f"{frame_number:010d}.png"
        sync_name = os.path.basename(self.root_folder.rstrip("/"))
        drive_dir = os.path.dirname(self.root_folder)
        path = os.path.join(drive_dir, "depth_gt", "train", sync_name,
                             "proj_depth", "groundtruth", "image_02", filename)
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 256.0

def load_image(path):
    """Read an image file and return it as an RGB numpy array (H, W, 3)."""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # channel swap

def read_calib_file(cam_to_cam_calib_file):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(cam_to_cam_calib_file, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

import os

import cv2

from bev import draw_bev_panel, points_to_grey_bev
from constants import BEV_PARAMS, CAM_H, CAM_W
from data_loading.data_loading import DataLoader
from depth_sources.depth_sources import gt_points, lidar_points, mono_points, stereo_points
from depth_sources.lidar_fusion import get_lidar_in_image_fov
from depth_sources.mono_depth import load_mono_depth_model
from occupancy import draw_occupancy_panel, occupancy_grid
from visualization import overlay_points_on_image

GT_MARGIN = 5  # no depth ground truth for the first and last 5 frames of a drive

def _count_frames(root_folder):
    """Number of left camera frames (image_02) in the sync folder."""
    data_dir = os.path.join(root_folder, "image_02", "data")
    return len([f for f in os.listdir(data_dir) if f.endswith(".png")])

def _camera_image_with_lidar(dl, image, pc_velo):
    """Camera image with the LiDAR points inside it as dots, colored by distance along the optical axis."""
    _, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, dl.P, dl.R0, dl.V2C, 0, 0, CAM_W, CAM_H, return_more=True)
    pts_2d = pts_2d[fov_inds]
    return overlay_points_on_image(image, pts_2d[:, :2], pts_2d[:, 2], vmax=BEV_PARAMS.MAX_Z)

def _build_frame(dl, frame_number, mono_model):
    """Camera image on top, LiDAR | stereo | mono BEV below, then each source's occupancy grid
    against the ground truth's: 1242 x 1125 px."""
    left, right = dl.load_stereo_pair(frame_number)
    assert left.shape[:2] == (CAM_H, CAM_W), f"camera image {left.shape[:2]}, expected {(CAM_H, CAM_W)}"
    gt_depth = dl.load_depth_gt(frame_number)
    gt = gt_points(dl, gt_depth)
    gt_bev = points_to_grey_bev(gt)

    pc_velo = dl.load_point_cloud(frame_number)
    camera_image = _camera_image_with_lidar(dl, left, pc_velo)
    lidar = lidar_points(dl, pc_velo)
    stereo = stereo_points(dl, left, right, gt_depth)
    mono = mono_points(dl, left, mono_model, gt_depth)
    bev_row = cv2.hconcat([draw_bev_panel(lidar, "LiDAR"),
                           draw_bev_panel(stereo, "stereo", gt_bev),
                           draw_bev_panel(mono, "mono", gt_bev)])

    # LiDAR: only the points inside the camera image, so all three sources cover the same area
    lidar_in_image = lidar_points(dl, get_lidar_in_image_fov(pc_velo, dl.P, dl.R0, dl.V2C, 0, 0, CAM_W, CAM_H))
    gt_grid = occupancy_grid(gt)
    lidar_panel = draw_occupancy_panel(occupancy_grid(lidar_in_image), gt_grid, "LiDAR vs ground truth")
    stereo_panel = draw_occupancy_panel(occupancy_grid(stereo), gt_grid, "stereo vs ground truth")
    mono_panel = draw_occupancy_panel(occupancy_grid(mono), gt_grid, "mono vs ground truth")
    agreement_row = cv2.hconcat([lidar_panel, stereo_panel, mono_panel])
    return cv2.vconcat([camera_image, bev_row, agreement_row])

def make_video(root_folder, output_dir="output", fps=10):
    """Render camera / LiDAR | stereo | mono BEV / their occupancy against the ground truth for every frame
    with depth ground truth into output_dir/<sequence name>.mp4, writing frame by frame."""
    dl = DataLoader(root_folder)
    mono_model = load_mono_depth_model()
    frames_cnt = _count_frames(root_folder)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(root_folder.rstrip("/")) + ".mp4")

    out = None
    for idx in range(GT_MARGIN, frames_cnt - GT_MARGIN):
        print(idx + 1, "of", frames_cnt)
        try:
            frame = _build_frame(dl, idx, mono_model)
        except (FileNotFoundError, cv2.error):
            print(f"skipping frame {idx}: missing file")
            continue

        # the mp4 writer drops the last row/column of odd-sized frames, so repeat them to get even sizes
        frame = cv2.copyMakeBorder(frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2, cv2.BORDER_REPLICATE)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if out is not None:
        out.release()
    return output_path

if __name__ == "__main__":
    output_path = make_video("../kitty_data/drive1/2011_09_26_drive_0001_sync")
    print(f"wrote {output_path}")

import os

import cv2

from bev import draw_ego_car, points_to_bev, points_to_grey_bev
from constants import CAM_H, CAM_W
from data_loading.data_loading import DataLoader
from depth_sources.depth_sources import gt_points, lidar_points, mono_points, stereo_points
from depth_sources.mono_depth import load_mono_depth_model
from visualization import draw_panel_label

GT_MARGIN = 5  # no depth ground truth for the first and last 5 frames of a drive

def _count_frames(root_folder):
    """Number of left camera frames (image_02) in the sync folder."""
    data_dir = os.path.join(root_folder, "image_02", "data")
    return len([f for f in os.listdir(data_dir) if f.endswith(".png")])

def _panel(points, label, background=None):
    """BEV panel: points in viridis, on background (grey ground truth) if given, with ego car and label."""
    bev = points_to_bev(points, None if background is None else background.copy())
    draw_ego_car(bev)
    return draw_panel_label(bev, label, color=(0, 0, 0))

def _build_frame(dl, frame_number, mono_model):
    """Camera image on top, LiDAR | stereo | mono BEV below: 1242 x 750 px."""
    left, right = dl.load_stereo_pair(frame_number)
    assert left.shape[:2] == (CAM_H, CAM_W), f"camera image {left.shape[:2]}, expected {(CAM_H, CAM_W)}"
    gt_depth = dl.load_depth_gt(frame_number)
    gt_bev = points_to_grey_bev(gt_points(dl, gt_depth))

    lidar = _panel(lidar_points(dl, dl.load_point_cloud(frame_number)), "LiDAR")
    stereo = _panel(stereo_points(dl, left, right, gt_depth), "stereo", gt_bev)
    mono = _panel(mono_points(dl, left, mono_model, gt_depth), "mono", gt_bev)
    return cv2.vconcat([left, cv2.hconcat([lidar, stereo, mono])])

def make_video(root_folder, output_dir="output", fps=10):
    """Render camera / LiDAR | stereo | mono BEV for every frame with depth ground truth into output_dir/<sequence name>.mp4,
    writing frame by frame."""
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
    output_path = make_video("../kitty_data/drive14/2011_09_26_drive_0014_sync")
    print(f"wrote {output_path}")

import os

import cv2

from bev import draw_ego_car, points_to_bev
from data_loading.data_loading import DataLoader
from point_cloud import depth_to_points, filter_points_for_bev

GT_MARGIN = 5  # no depth ground truth for the first and last 5 frames of a drive

def _count_frames(root_folder):
    """Number of left camera frames (image_02) in the sync folder."""
    data_dir = os.path.join(root_folder, "image_02", "data")
    return len([f for f in os.listdir(data_dir) if f.endswith(".png")])

def _build_frame(dl, frame_number):
    """Camera image | BEV from the ground-truth depth, both 375 px high."""
    camera, _ = dl.load_stereo_pair(frame_number)
    points = depth_to_points(dl.load_depth_gt(frame_number), dl.P)
    bev = points_to_bev(points[filter_points_for_bev(points)])
    draw_ego_car(bev)
    return cv2.hconcat([camera, bev])

def make_video(root_folder, output_dir="output", fps=10):
    """Render camera | BEV for every frame with depth ground truth into output_dir/<sequence name>.mp4,
    writing frame by frame."""
    dl = DataLoader(root_folder)
    frames_cnt = _count_frames(root_folder)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(root_folder.rstrip("/")) + ".mp4")

    out = None
    for idx in range(GT_MARGIN, frames_cnt - GT_MARGIN):
        print(idx + 1, "of", frames_cnt)
        try:
            frame = _build_frame(dl, idx)
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

import os

import cv2
import numpy as np

from bev import draw_bev_panel, points_to_grey_bev
from constants import DISTANCE_BANDS, BEV_PARAMS, CAM_H, CAM_W, GT_MARGIN
from data_loading.data_loading import DataLoader
from depth_sources.depth_sources import gt_points, lidar_points, mono_points, stereo_points
from depth_sources.lidar_fusion import get_lidar_in_image_fov
from depth_sources.mono_depth import load_mono_depth_model
from occupancy import draw_occupancy_panel, iou_scores, iou_table, mean_iou_scores, occupancy_grid
from visualization import draw_depth_legend, draw_iou_chart, draw_table, overlay_points_on_image

SOURCES = ("LiDAR", "stereo (SGBM)", "mono (DepthAnything)")
END_CARD_SECONDS = 3  # the last frame with the mean IoU table, held at the end of the video

def _count_frames(root_folder):
    """Number of left camera frames (image_02) in the sync folder."""
    data_dir = os.path.join(root_folder, "image_02", "data")
    return len([f for f in os.listdir(data_dir) if f.endswith(".png")])

def _camera_image_with_lidar(dl, image, pc_velo):
    """Camera image with the LiDAR points inside it as dots, colored by distance along the optical axis, with legend."""
    _, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, dl.P, dl.R0, dl.V2C, 0, 0, CAM_W, CAM_H, return_more=True)
    pts_2d = pts_2d[fov_inds]
    camera_image = overlay_points_on_image(image, pts_2d[:, :2], pts_2d[:, 2], vmax=BEV_PARAMS.MAX_Z)
    draw_depth_legend(camera_image, BEV_PARAMS.MAX_Z, "LiDAR distance")
    return camera_image

def _build_frame(dl, frame_number, mono_model):
    """Camera image on top, LiDAR | stereo | mono BEV below, then each source's occupancy grid
    against the ground truth's: 1242 x 1125 px.
    Also returns the IoU scores (sources, scores): a row per source in SOURCES, as iou_scores."""
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
    lidar_caption, stereo_caption, mono_caption = SOURCES
    bev_row = cv2.hconcat([draw_bev_panel(lidar, lidar_caption),
                           draw_bev_panel(stereo, stereo_caption, gt_bev),
                           draw_bev_panel(mono, mono_caption, gt_bev)])

    # LiDAR: only the points inside the camera image, so all three sources cover the same area
    lidar_in_image = lidar_points(dl, get_lidar_in_image_fov(pc_velo, dl.P, dl.R0, dl.V2C, 0, 0, CAM_W, CAM_H))
    gt_grid = occupancy_grid(gt)
    lidar_grid = occupancy_grid(lidar_in_image)
    stereo_grid = occupancy_grid(stereo)
    mono_grid = occupancy_grid(mono)
    lidar_iou = iou_scores(lidar_grid, gt_grid)
    stereo_iou = iou_scores(stereo_grid, gt_grid)
    mono_iou = iou_scores(mono_grid, gt_grid)
    lidar_panel = draw_occupancy_panel(lidar_grid, gt_grid, "LiDAR", lidar_iou)
    stereo_panel = draw_occupancy_panel(stereo_grid, gt_grid, "stereo", stereo_iou)
    mono_panel = draw_occupancy_panel(mono_grid, gt_grid, "mono", mono_iou)
    agreement_row = cv2.hconcat([lidar_panel, stereo_panel, mono_panel])
    return cv2.vconcat([camera_image, bev_row, agreement_row]), np.array([lidar_iou, stereo_iou, mono_iou])

def make_video(root_folder, output_dir="output", fps=10):
    """Render camera / LiDAR | stereo | mono BEV / their occupancy against the ground truth / IoU chart for every frame
    with depth ground truth into output_dir/<sequence name>.mp4, writing frame by frame: 1242 x 1275 px.
    At the end, print the drive's mean IoU per source as a table and hold it as an end card for END_CARD_SECONDS."""
    dl = DataLoader(root_folder)
    mono_model = load_mono_depth_model()
    frames_cnt = _count_frames(root_folder)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(root_folder.rstrip("/")) + ".mp4")

    frames = range(GT_MARGIN, frames_cnt - GT_MARGIN)
    # IoU scores (frames, sources, scores), nan until a frame is done, and for skipped frames
    ious = np.full((len(frames), len(SOURCES), 1 + len(DISTANCE_BANDS)), np.nan)

    out = None
    for i, idx in enumerate(frames):
        print(idx + 1, "of", frames_cnt)
        try:
            frame, ious[i] = _build_frame(dl, idx, mono_model)
        except (FileNotFoundError, cv2.error):
            print(f"skipping frame {idx}: missing file")
            continue

        # the IoU over the whole window, for the frames so far
        chart = draw_iou_chart(ious[:, :, 0], SOURCES, frame.shape[1], first_frame=GT_MARGIN)
        frame = cv2.vconcat([frame, chart])

        # the mp4 writer drops the last row/column of odd-sized frames, so repeat them to get even sizes
        frame = cv2.copyMakeBorder(frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2, cv2.BORDER_REPLICATE)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # the drive's mean IoU per source: in the terminal, and as the end card of the video
    frames_done = int((~np.isnan(ious[:, 0, 0])).sum())
    title = f"Mean IoU with the ground truth over {frames_done} frames"
    table = iou_table(SOURCES, mean_iou_scores(ious))
    print(title)
    name_width = max(len(name) for name in SOURCES) + 2
    for row in table:
        print(f"{row[0]:{name_width}}" + "".join(f"{cell:>14}" for cell in row[1:]))

    if out is not None:
        # end card: the last frame, with the table over the dimmed camera image
        frame[:CAM_H] = draw_table(frame[:CAM_H], title, table)
        for _ in range(round(END_CARD_SECONDS * fps)):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    return output_path

if __name__ == "__main__":
    output_path = make_video("../kitty_data/drive14/2011_09_26_drive_0014_sync")
    print(f"wrote {output_path}")

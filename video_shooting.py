import os

import cv2

from data_loading import DataLoader

def _count_frames(root_folder):
    """Number of left camera frames (image_02) in the sync folder."""
    data_dir = os.path.join(root_folder, "image_02", "data")
    return len([f for f in os.listdir(data_dir) if f.endswith(".png")])

def _build_frame(dl, frame_number):
    left, _ = dl.load_stereo_pair(frame_number)
    return left

def make_video(root_folder, output_dir="output", fps=10):
    """Render every frame in root_folder into output_dir/<sequence name>.mp4, writing frame by frame."""
    dl = DataLoader(root_folder)
    frames_cnt = _count_frames(root_folder)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(root_folder.rstrip("/")) + ".mp4")

    out = None
    for idx in range(frames_cnt):
        print(idx + 1, "of", frames_cnt)
        try:
            frame = _build_frame(dl, idx)
        except (FileNotFoundError, cv2.error):
            print(f"skipping frame {idx}: missing file")
            continue

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

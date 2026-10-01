import os
import sys

from remotezip import RemoteZip

KITTI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kitty_data")
DRIVES = {
    "2011_09_26_drive_0001_sync": "drive1",
    "2011_09_26_drive_0009_sync": "drive9",
    "2011_09_26_drive_0014_sync": "drive14",
}


def download_depth_gt(url):
    with RemoteZip(url) as archive:
        for name in archive.namelist():
            if "groundtruth/image_02/" not in name or not name.endswith(".png"):
                continue
            for drive, folder in DRIVES.items():
                if drive in name:
                    out_dir = os.path.join(KITTI_DIR, folder, "depth_gt")
                    archive.extract(name, out_dir)
                    print(name)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python download_depth_gt.py <url of data_depth_annotated.zip>")
    download_depth_gt(sys.argv[1])


if __name__ == "__main__":
    main()

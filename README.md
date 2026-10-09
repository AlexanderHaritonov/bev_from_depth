## Data
To download the depth ground truth data:

Change KITTI_DIR and DRIVES in download_depth.gr.py, then:
```
pip install remotezip
python data_loading/download_depth_gt.py https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
```
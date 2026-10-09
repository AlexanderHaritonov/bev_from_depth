import cv2
from PIL import Image
from transformers import pipeline

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

def load_mono_depth_model():
    return pipeline("depth-estimation", model=MODEL_ID, device="cpu")

def get_mono_depth(image, model):
    """Monocular metric depth map (m) for a single RGB image, resized to match it."""
    h, w = image.shape[:2]
    result = model(Image.fromarray(image))
    depth_map = result["predicted_depth"].numpy()
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_map

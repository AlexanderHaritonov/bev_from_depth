import numpy as np

from constants import BEV_PARAMS

CELL = 1.0  # occupancy grid cell size (m)
BANDS = ((0, 20), (20, 40), (40, 70))  # distance bands ahead of the camera (m)

# cell edges lie on whole meters from the camera, so they line up with the 10 m grid lines and the band limits
GRID_MIN_X = -np.ceil(BEV_PARAMS.MAX_X / CELL) * CELL                     # left edge of the grid (m): -41
GRID_ROWS = int(np.ceil((BEV_PARAMS.MAX_Z - BEV_PARAMS.MIN_Z) / CELL))    # 73
GRID_COLS = int(2 * np.ceil(BEV_PARAMS.MAX_X / CELL))                     # 82

def occupancy_grid(points):
    """Occupancy grid (GRID_ROWS, GRID_COLS), bool, from filtered camera-frame points (N, 3) as x, y, z (m):
    cells (of CELL meters size) over the BEV window, a cell is occupied if it holds at least one point.
    Row 0 is the farthest and x grows with the columns, as in the BEV image."""
    rows = np.clip(((BEV_PARAMS.MAX_Z - points[:, 2]) / CELL).astype(int), 0, GRID_ROWS - 1)
    cols = np.clip(((points[:, 0] - GRID_MIN_X) / CELL).astype(int), 0, GRID_COLS - 1)
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    grid[rows, cols] = True
    return grid

def _band_rows(band):
    """Grid rows of a distance band (near, far) in m."""
    near, far = band
    return slice(int((BEV_PARAMS.MAX_Z - far) / CELL), int((BEV_PARAMS.MAX_Z - near) / CELL))

def iou(grid, gt_grid):
    """Cells occupied in both / cells occupied in either, nan if both grids are empty."""
    either = (grid | gt_grid).sum()
    return (grid & gt_grid).sum() / either if either else np.nan

def iou_scores(grid, gt_grid):
    """IoU of two occupancy grids over the whole window, then per band in BANDS: array (1 + len(BANDS),)."""
    bands = [iou(grid[_band_rows(band)], gt_grid[_band_rows(band)]) for band in BANDS]
    return np.array([iou(grid, gt_grid)] + bands)

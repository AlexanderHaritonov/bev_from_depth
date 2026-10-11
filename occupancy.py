import numpy as np

from bev import BEV_RES, BEV_W, GT_GREY, LEGEND_LEFT_X, LEGEND_RIGHT_X, LEGEND_Y, draw_ego_car, draw_grid
from constants import DISTANCE_BANDS, BEV_PARAMS, OCCUPANCY_CELL
from visualization import draw_legend, draw_panel_label

# agreement panel colors (colorblind-safe green and orange); only in the ground truth: GT_GREY, as in the BEV panels
BOTH_COLOR = (0, 158, 115)        # occupied in the source and in the ground truth
SOURCE_ONLY_COLOR = (213, 94, 0)  # occupied only in the source
BAND_LINE_GREY = (150, 150, 150)  # lines between the distance bands, darker than the grid

# cell edges lie on whole meters from the camera, so they line up with the 10 m grid lines and the band limits
GRID_MIN_X = -np.ceil(BEV_PARAMS.MAX_X / OCCUPANCY_CELL) * OCCUPANCY_CELL                     # left edge of the grid (m): -41
GRID_ROWS = int(np.ceil((BEV_PARAMS.MAX_Z - BEV_PARAMS.MIN_Z) / OCCUPANCY_CELL))    # 73
GRID_COLS = int(2 * np.ceil(BEV_PARAMS.MAX_X / OCCUPANCY_CELL))                     # 82

def occupancy_grid(points):
    """Occupancy grid (GRID_ROWS, GRID_COLS), bool, from filtered camera-frame points (N, 3) as x, y, z (m):
    cells (of CELL meters size) over the BEV window, a cell is occupied if it holds at least one point.
    Row 0 is the farthest and x grows with the columns, as in the BEV image."""
    rows = np.clip(((BEV_PARAMS.MAX_Z - points[:, 2]) / OCCUPANCY_CELL).astype(int), 0, GRID_ROWS - 1)
    cols = np.clip(((points[:, 0] - GRID_MIN_X) / OCCUPANCY_CELL).astype(int), 0, GRID_COLS - 1)
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    grid[rows, cols] = True
    return grid

def _band_rows(band):
    """Grid rows of a distance band (near, far) in m."""
    near, far = band
    return slice(int((BEV_PARAMS.MAX_Z - far) / OCCUPANCY_CELL), int((BEV_PARAMS.MAX_Z - near) / OCCUPANCY_CELL))

def iou(grid, gt_grid):
    """Cells occupied in both / cells occupied in either, nan if both grids are empty."""
    either = (grid | gt_grid).sum()
    return (grid & gt_grid).sum() / either if either else np.nan

def iou_scores(grid, gt_grid):
    """IoU of two occupancy grids over the whole window, then per band in BANDS: array (1 + len(BANDS),)."""
    bands = [iou(grid[_band_rows(band)], gt_grid[_band_rows(band)]) for band in DISTANCE_BANDS]
    return np.array([iou(grid, gt_grid)] + bands)

def _iou_text(score):
    """An IoU with two decimals, "-" where there is none (nan)."""
    return "-" if np.isnan(score) else f"{score:.2f}"

def iou_label(name, scores):
    """Panel label from a source's iou_scores: the whole window first, then the bands."""
    whole, *bands = [_iou_text(score) for score in scores]
    return f"{name} IoU {whole} ({' / '.join(bands)})"

def mean_iou_scores(ious):
    """Mean of iou_scores over the frames (axis 0), leaving out nan; nan where no frame has a value."""
    counts = (~np.isnan(ious)).sum(axis=0)
    return np.where(counts > 0, np.nansum(ious, axis=0) / np.maximum(counts, 1), np.nan)

def iou_table(names, scores):
    """IoU scores (sources, scores) as a table of strings: a header row, then a row per source.
    Columns: the source's name, the whole window, then the bands."""
    header = ["", "whole window"] + [f"{near}-{far} m" for near, far in DISTANCE_BANDS]
    table = [header]
    for name, row in zip(names, scores):
        cells = [_iou_text(score) for score in row]
        table.append([name] + cells)
    return table

def _draw_band_lines(bev):
    """Darker lines between the distance bands (in place), only on empty (white) pixels, like the grid."""
    for _, far in DISTANCE_BANDS[:-1]:
        line = bev[round((BEV_PARAMS.MAX_Z - far) / BEV_RES)]
        line[(line == 255).all(axis=1)] = BAND_LINE_GREY

def draw_occupancy_panel(grid, gt_grid, name, scores):
    """Agreement panel, same size and window as a BEV panel: a source's occupancy grid against the ground truth's.
    Cells occupied in both in green, only in the source in orange, only in the ground truth in grey,
    with grid, band lines, ego car, legend and the label from the source's name and iou_scores."""
    cells = np.full((GRID_ROWS, GRID_COLS, 3), 255, dtype=np.uint8)
    cells[grid & gt_grid] = BOTH_COLOR
    cells[grid & ~gt_grid] = SOURCE_ONLY_COLOR
    cells[~grid & gt_grid] = GT_GREY

    # the cell under each BEV pixel (pixel centers)
    rows = ((np.arange(BEV_PARAMS.H) + 0.5) * BEV_RES / OCCUPANCY_CELL).astype(int)
    x = (np.arange(BEV_W) + 0.5) * BEV_RES - BEV_PARAMS.MAX_X
    cols = ((x - GRID_MIN_X) / OCCUPANCY_CELL).astype(int)
    bev = cells[np.ix_(rows, cols)]

    _draw_band_lines(bev)  # before the grid, which leaves them alone
    draw_grid(bev)
    draw_ego_car(bev)
    draw_legend(bev, [(BOTH_COLOR, "both"), (SOURCE_ONLY_COLOR, f"only {name}")], (LEGEND_LEFT_X, LEGEND_Y))
    draw_legend(bev, [(GT_GREY, "only ground truth")], (LEGEND_RIGHT_X, LEGEND_Y))
    return draw_panel_label(bev, iou_label(name, scores), color=(0, 0, 0), outline=(255, 255, 255))

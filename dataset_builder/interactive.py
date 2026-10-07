"""Raster-based candidate coordinates for manual transmitter placement."""
from __future__ import annotations
import numpy as np

def _sample_mask_numpy(mask_hw: np.ndarray, xy_norm: np.ndarray) -> np.ndarray:
    h, w = mask_hw.shape
    j = np.clip((xy_norm[:, 0] * w).astype(np.int64), 0, w - 1)
    i = np.clip((xy_norm[:, 1] * h).astype(np.int64), 0, h - 1)
    return mask_hw[i, j]


def _to_norm_xy(xy_real: np.ndarray, map_x: float, map_y: float) -> np.ndarray:
    out = np.empty_like(xy_real, dtype=np.float64)
    out[:, 0] = (xy_real[:, 0] + map_x / 2.0) / map_x
    out[:, 1] = (xy_real[:, 1] + map_y / 2.0) / map_y
    return out


def build_deployable_points_from_mask(
    deployable_mask: np.ndarray,
    map_x: float,
    map_y: float,
    cell_x: float,
    cell_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_x <= 0 or cell_y <= 0:
        raise ValueError("cell-size must be > 0 in both x and y")
    xs = np.arange(-map_x / 2.0, map_x / 2.0 + 1e-9, float(cell_x), dtype=np.float64)
    ys = np.arange(-map_y / 2.0, map_y / 2.0 + 1e-9, float(cell_y), dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    all_xy = np.column_stack((gx.ravel(), gy.ravel()))
    norm = _to_norm_xy(all_xy, map_x, map_y)
    m = _sample_mask_numpy(np.asarray(deployable_mask, dtype=np.float32), norm)
    deployable_xy = all_xy[m > 0.5]
    return all_xy, deployable_xy


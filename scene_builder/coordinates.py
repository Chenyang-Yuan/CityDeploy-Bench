"""Canonical coordinate frame shared by scenes, datasets, models, and samplers.

Local metric coordinates are centered on the scene: x points east and y points
north. Model coordinates are dimensionless and live in [0, 1]^2. Raster row
zero corresponds to y_min, so normalized coordinates map directly to
``torch.grid_sample`` after the standard ``2*x-1`` conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COORDINATE_SCHEMA_VERSION = "citydeploy.coordinates.v1"
COORDINATE_CONVENTION = "local_centered_xy_m_to_unit_square_xy"


@dataclass(frozen=True)
class SceneFrame2D:
    map_x_m: float
    map_y_m: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.map_x_m) or not np.isfinite(self.map_y_m):
            raise ValueError("map dimensions must be finite")
        if self.map_x_m <= 0.0 or self.map_y_m <= 0.0:
            raise ValueError("map dimensions must be positive")

    @property
    def size_m(self) -> np.ndarray:
        return np.asarray([self.map_x_m, self.map_y_m], dtype=np.float64)

    def real_to_norm01(self, xy_real_m, *, check_bounds: bool = False) -> np.ndarray:
        xy = _xy_array(xy_real_m, "xy_real_m")
        out = xy / self.size_m + 0.5
        if check_bounds:
            validate_norm01(out)
        return out

    def norm01_to_real(self, xy_norm01, *, check_bounds: bool = False) -> np.ndarray:
        xy = _xy_array(xy_norm01, "xy_norm01")
        if check_bounds:
            validate_norm01(xy)
        return (xy - 0.5) * self.size_m


def _xy_array(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim < 1 or out.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [..., 2], got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def validate_norm01(xy_norm01, *, atol: float = 1e-6) -> None:
    xy = _xy_array(xy_norm01, "xy_norm01")
    if np.any(xy < -atol) or np.any(xy > 1.0 + atol):
        lo = float(np.min(xy))
        hi = float(np.max(xy))
        raise ValueError(f"xy_norm01 outside [0,1]: min={lo:.6g}, max={hi:.6g}")


def frame_from_metadata(metadata: dict) -> SceneFrame2D:
    return SceneFrame2D(float(metadata["map_x_m"]), float(metadata["map_y_m"]))


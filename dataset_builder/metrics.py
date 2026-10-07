"""Canonical conversion from Sionna path-gain maps to CityDeploy radio metrics."""

from __future__ import annotations

import numpy as np


SERVING_ASSOCIATION = "strongest_rsrp"
THROUGHPUT_MODEL = "shannon_uncapped"


def _tx_axis_last(path_gain: np.ndarray, num_tx: int) -> np.ndarray:
    array = np.asarray(path_gain, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError(f"Expected [num_tx, ..., ...] path gain, got shape={array.shape}")
    if array.shape[0] == num_tx:
        return np.moveaxis(array, 0, -1)
    matches = [axis for axis, size in enumerate(array.shape) if size == num_tx]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify a unique TX axis of size {num_tx} in shape={array.shape}")
    return np.moveaxis(array, matches[0], -1)


def tx_axis_last(path_gain: np.ndarray, num_tx: int) -> np.ndarray:
    """Return path gain as ``[H, W, N]`` using an unambiguous TX axis."""
    return _tx_axis_last(path_gain, num_tx)


def compute_radio_metrics(
    path_gain,
    num_tx: int,
    tx_power_dbm: float = 23.0,
    bandwidth_hz: float = 20e6,
    noise_figure_db: float = 9.0,
) -> dict[str, np.ndarray]:
    """Compute all metrics using one strongest-RSRP serving-TX association."""
    if num_tx <= 0:
        raise ValueError("num_tx must be positive")
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    pg = _tx_axis_last(np.asarray(path_gain), num_tx=num_tx)
    pg = np.where(np.isfinite(pg) & (pg > 0.0), pg, 0.0)

    tx_power_w = 10.0 ** ((float(tx_power_dbm) - 30.0) / 10.0)
    noise_dbm = -174.0 + 10.0 * np.log10(float(bandwidth_hz)) + float(noise_figure_db)
    noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    rx_power_all = pg * tx_power_w

    # Constant TX powers make strongest RSRP equivalent to strongest path gain.
    serving_idx = np.argmax(rx_power_all, axis=-1)
    serving_rx = np.take_along_axis(rx_power_all, serving_idx[..., None], axis=-1)[..., 0]
    serving_pg = np.take_along_axis(pg, serving_idx[..., None], axis=-1)[..., 0]
    interference = np.maximum(np.sum(rx_power_all, axis=-1) - serving_rx, 0.0)
    sinr = serving_rx / (interference + noise_w)

    path_loss_db = np.full(serving_pg.shape, np.nan, dtype=np.float64)
    rsrp_dbm = np.full(serving_rx.shape, np.nan, dtype=np.float64)
    sinr_db = np.full(sinr.shape, np.nan, dtype=np.float64)
    valid_pg = serving_pg > 0.0
    valid_rx = serving_rx > 0.0
    valid_sinr = sinr > 0.0
    path_loss_db[valid_pg] = -10.0 * np.log10(serving_pg[valid_pg])
    rsrp_dbm[valid_rx] = 10.0 * np.log10(serving_rx[valid_rx]) + 30.0
    sinr_db[valid_sinr] = 10.0 * np.log10(sinr[valid_sinr])
    throughput_mbps = float(bandwidth_hz) * np.log2(1.0 + sinr) / 1e6

    return {
        "serving_idx": serving_idx,
        "sinr_linear": sinr,
        "sinr_db": sinr_db,
        "throughput_mbps": throughput_mbps,
        "path_loss_db": path_loss_db,
        "rsrp_dbm": rsrp_dbm,
        "rsrp_dbm_best": rsrp_dbm,  # Compatibility key; now explicitly serving-cell RSRP.
    }


def compute_urban_radio_metrics(
    path_gain,
    num_tx: int,
    *,
    tx_power_dbm: float,
    bandwidth_hz: float,
    noise_figure_db: float,
    num_resource_blocks: int,
    subcarriers_per_resource_block: int,
    ssb_power_offset_db: float,
    implementation_efficiency: float,
    resource_share: float,
    max_spectral_efficiency_bps_hz: float,
) -> dict[str, np.ndarray]:
    """Compute the immutable V4 urban metrics from per-TX path gains."""
    metrics = compute_radio_metrics(
        path_gain,
        num_tx=num_tx,
        tx_power_dbm=tx_power_dbm,
        bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db,
    )
    active_subcarriers = int(num_resource_blocks) * int(subcarriers_per_resource_block)
    if active_subcarriers <= 0:
        raise ValueError("active subcarrier count must be positive")
    ss_rsrp_dbm = (
        np.asarray(metrics["rsrp_dbm_best"], dtype=np.float64)
        - 10.0 * np.log10(active_subcarriers)
        + float(ssb_power_offset_db)
    )
    spectral_efficiency = np.minimum(
        np.log2(1.0 + np.asarray(metrics["sinr_linear"], dtype=np.float64)),
        float(max_spectral_efficiency_bps_hz),
    )
    effective_throughput_mbps = (
        float(bandwidth_hz)
        * spectral_efficiency
        * float(implementation_efficiency)
        * float(resource_share)
        / 1e6
    )
    metrics.update(
        {
            "wideband_received_power_dbm": np.asarray(metrics["rsrp_dbm_best"]),
            "ss_rsrp_dbm": ss_rsrp_dbm,
            "effective_throughput_mbps": effective_throughput_mbps,
        }
    )
    return metrics


def summarize_urban_metrics(
    metrics: dict[str, np.ndarray],
    evaluation_mask: np.ndarray,
    thresholds: dict,
) -> tuple[dict[str, float | int | bool | None], dict[str, np.ndarray]]:
    """Summarize road-domain V4 labels and retain exact predicate masks."""
    road = np.asarray(evaluation_mask, dtype=bool)
    arrays = {
        "pathloss": np.asarray(metrics["path_loss_db"]),
        "ss_rsrp": np.asarray(metrics["ss_rsrp_dbm"]),
        "sinr": np.asarray(metrics["sinr_db"]),
        "effective_throughput": np.asarray(metrics["effective_throughput_mbps"]),
    }
    if any(value.shape != road.shape for value in arrays.values()):
        raise ValueError("metric maps and evaluation mask must have the same shape")
    conditions = {
        "pathloss": road & np.isfinite(arrays["pathloss"]) & (
            arrays["pathloss"] <= float(thresholds["pathloss_db_max"])
        ),
        "ss_rsrp": road & np.isfinite(arrays["ss_rsrp"]) & (
            arrays["ss_rsrp"] >= float(thresholds["ss_rsrp_dbm_min"])
        ),
        "sinr": road & np.isfinite(arrays["sinr"]) & (
            arrays["sinr"] >= float(thresholds["sinr_db_min"])
        ),
        "effective_throughput": road & np.isfinite(arrays["effective_throughput"]) & (
            arrays["effective_throughput"]
            >= float(thresholds["effective_throughput_mbps_min"])
        ),
    }
    conditions["joint"] = np.logical_and.reduce(tuple(conditions.values()))
    denominator = int(np.count_nonzero(road))
    if denominator <= 0:
        raise ValueError("road evaluation mask contains no cells")
    summary: dict[str, float | int | bool | None] = {"evaluation_cell_count": denominator}
    for name, condition in conditions.items():
        count = int(np.count_nonzero(condition))
        prefix = "joint" if name == "joint" else name
        summary[f"{prefix}_covered_count"] = count
        coverage_name = "joint_4metric_coverage" if name == "joint" else f"{name}_coverage"
        summary[coverage_name] = count / denominator
    summary["joint_target_met"] = bool(
        float(summary["joint_4metric_coverage"])
        >= float(thresholds["joint_coverage_target"])
    )
    for name, values in arrays.items():
        selected = values[road & np.isfinite(values)]
        for statistic in ("mean", "p05", "p50", "p95"):
            summary[f"{name}_{statistic}"] = None
        if selected.size:
            summary[f"{name}_mean"] = float(np.mean(selected))
            summary[f"{name}_p05"] = float(np.percentile(selected, 5))
            summary[f"{name}_p50"] = float(np.percentile(selected, 50))
            summary[f"{name}_p95"] = float(np.percentile(selected, 95))
    return summary, conditions

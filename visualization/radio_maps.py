"""Canonical plotting functions for road-domain radio maps."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


PAPER_STYLE = {
    "background": "#fafaf8",
    "building_fill": "#deddd8",
    "building_edge": "#4f504f",
    "road_edge": "#777a78",
    "uncovered": "#2b6ca3",
    "covered": "#df4438",
    "text": "#202322",
}

RADIO_MAP_TYPOGRAPHY = {
    "title": 15,
    "axis_label": 12,
    "axis_tick": 10,
    "colorbar_label": 12,
    "colorbar_tick": 10,
}

METRIC_MAP_LAYOUT = {
    "left": 0.10,
    "right": 0.88,
    "bottom": 0.10,
    "top": 0.91,
}


def save_metric_plot(
    path: Path,
    values: np.ndarray,
    road_mask: np.ndarray,
    building_mask: np.ndarray,
    tx_positions: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    colorbar_label: str,
    cmap_name: str = "turbo",
    dpi: int = 220,
    coverage_value: float | None = None,
) -> None:
    """Save one continuous road-domain metric with a legible city context."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    data = np.where(road_mask, values, np.nan)
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
        if math.isclose(float(vmin), float(vmax)):
            vmin, vmax = float(vmin) - 0.5, float(vmax) + 0.5
    else:
        vmin, vmax = None, None
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    fig.subplots_adjust(**METRIC_MAP_LAYOUT)
    ax.set_facecolor("#f7f7f4")
    building_layer = np.ma.masked_where(building_mask <= 0.5, building_mask)
    ax.imshow(
        building_layer,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#d9d8d3"]),
        vmin=0,
        vmax=1,
        alpha=0.72,
        zorder=0,
    )
    image = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=2,
    )
    if np.any(building_mask > 0.5):
        x = np.linspace(extent[0], extent[1], building_mask.shape[1])
        y = np.linspace(extent[2], extent[3], building_mask.shape[0])
        ax.contour(
            x,
            y,
            building_mask,
            levels=[0.5],
            colors="#4a4a48",
            linewidths=0.9,
            alpha=0.95,
            zorder=3,
        )
    _draw_transmitters(ax, tx_positions, text_color="#202322")
    suffix = f" · Coverage {coverage_value:.1%}" if coverage_value is not None else ""
    ax.set_title(f"{title}{suffix}", fontsize=RADIO_MAP_TYPOGRAPHY["title"], pad=12)
    ax.set_xlabel("x (m)", fontsize=RADIO_MAP_TYPOGRAPHY["axis_label"])
    ax.set_ylabel("y (m)", fontsize=RADIO_MAP_TYPOGRAPHY["axis_label"])
    ax.tick_params(labelsize=RADIO_MAP_TYPOGRAPHY["axis_tick"])
    ax.set_aspect("equal")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=RADIO_MAP_TYPOGRAPHY["colorbar_label"])
    colorbar.ax.tick_params(labelsize=RADIO_MAP_TYPOGRAPHY["colorbar_tick"])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_joint_coverage_plot(
    path: Path,
    joint_mask: np.ndarray,
    road_mask: np.ndarray,
    building_mask: np.ndarray,
    tx_positions: np.ndarray,
    extent: tuple[float, float, float, float],
    coverage_value: float | None = None,
    dpi: int = 220,
) -> None:
    """Save the canonical paper-style four-metric joint coverage map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    style = PAPER_STYLE
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    fig.patch.set_facecolor(style["background"])
    ax.set_facecolor(style["background"])
    building_layer = np.ma.masked_where(building_mask <= 0.5, building_mask)
    ax.imshow(
        building_layer,
        origin="lower",
        extent=extent,
        cmap=ListedColormap([style["building_fill"]]),
        vmin=0,
        vmax=1,
        alpha=0.96,
        zorder=0,
    )
    joint_display = np.ma.masked_where(~road_mask, joint_mask.astype(float))
    binary_cmap = ListedColormap([style["uncovered"], style["covered"]])
    binary_cmap.set_bad((0, 0, 0, 0))
    ax.imshow(
        joint_display,
        origin="lower",
        extent=extent,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )
    x = np.linspace(extent[0], extent[1], road_mask.shape[1])
    y = np.linspace(extent[2], extent[3], road_mask.shape[0])
    if np.any(building_mask):
        ax.contour(
            x,
            y,
            building_mask,
            levels=[0.5],
            colors=style["building_edge"],
            linewidths=1.0,
            zorder=3,
        )
    if np.any(road_mask):
        ax.contour(
            x,
            y,
            road_mask,
            levels=[0.5],
            colors=style["road_edge"],
            linewidths=0.35,
            alpha=0.8,
            zorder=3,
        )
    _draw_transmitters(ax, tx_positions, text_color=style["text"])
    suffix = f" · {coverage_value:.1%}" if coverage_value is not None else ""
    ax.set_title(
        f"Four-metric joint road coverage{suffix}",
        color=style["text"],
        fontsize=RADIO_MAP_TYPOGRAPHY["title"],
        pad=12,
    )
    ax.set_xlabel("x (m)", color=style["text"], fontsize=RADIO_MAP_TYPOGRAPHY["axis_label"])
    ax.set_ylabel("y (m)", color=style["text"], fontsize=RADIO_MAP_TYPOGRAPHY["axis_label"])
    ax.tick_params(colors=style["text"], labelsize=RADIO_MAP_TYPOGRAPHY["axis_tick"])
    for spine in ax.spines.values():
        spine.set_color(style["building_edge"])
    ax.set_aspect("equal")
    legend = ax.legend(
        handles=[
            Patch(facecolor=style["covered"], label="Joint covered"),
            Patch(facecolor=style["uncovered"], label="Not covered"),
            Patch(
                facecolor=style["building_fill"],
                edgecolor=style["building_edge"],
                label="Building",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.92,
    )
    legend.get_frame().set_facecolor(style["background"])
    legend.get_frame().set_edgecolor(style["building_edge"])
    for label in legend.get_texts():
        label.set_color(style["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_transmitters(ax, tx_positions: np.ndarray, text_color: str) -> None:
    for index, (x_pos, y_pos, _z_pos) in enumerate(tx_positions, start=1):
        ax.scatter(
            x_pos,
            y_pos,
            marker="*",
            s=210,
            c="#ffffff",
            edgecolors="#111111",
            linewidths=1.15,
            zorder=10,
        )
        ax.text(x_pos, y_pos, f" {index}", color=text_color, weight="bold", zorder=11)


def save_radio_map_outputs(
    run_dir: Path,
    metrics: dict,
    condition_masks: dict,
    road_mask: np.ndarray,
    building_mask: np.ndarray,
    tx_positions: np.ndarray,
    map_x: float,
    map_y: float,
    cmap: str = "turbo",
    dpi: int = 220,
) -> dict[str, str]:
    """Save all canonical continuous maps, the joint map, and raw arrays."""
    wideband_received_power = metrics.get("wideband_received_power_dbm")
    if wideband_received_power is None:
        wideband_received_power = metrics.get("wideband_rss_dbm")
    if wideband_received_power is None:
        raise KeyError(
            "metrics must contain wideband_received_power_dbm or its legacy "
            "wideband_rss_dbm alias"
        )
    extent = (-map_x / 2.0, map_x / 2.0, -map_y / 2.0, map_y / 2.0)
    specifications = (
        ("path_loss_db", "pathloss", "radio_map_pathloss.png", "Path Loss", "Path loss (dB)"),
        ("ss_rsrp_dbm", "ss_rsrp", "radio_map_ss_rsrp.png", "SS-RSRP", "SS-RSRP (dBm)"),
        ("sinr_db", "sinr", "radio_map_sinr.png", "SINR", "SINR (dB)"),
        (
            "effective_throughput_mbps",
            "effective_throughput",
            "radio_map_effective_throughput.png",
            "Effective Throughput",
            "Effective throughput (Mbit/s)",
        ),
    )
    output_paths: dict[str, str] = {}
    road_denominator = max(1, np.count_nonzero(road_mask))
    for key, condition_key, filename, title, label in specifications:
        path = run_dir / filename
        metric_coverage = float(np.count_nonzero(condition_masks[condition_key]) / road_denominator)
        save_metric_plot(
            path,
            np.asarray(metrics[key]),
            road_mask,
            building_mask,
            tx_positions,
            extent,
            title,
            label,
            cmap,
            dpi,
            metric_coverage,
        )
        output_paths[key] = str(path)
    joint_coverage = float(
        np.count_nonzero(condition_masks["joint"]) / max(1, np.count_nonzero(road_mask))
    )
    joint_path = run_dir / "radio_map_joint_coverage.png"
    save_joint_coverage_plot(
        joint_path,
        condition_masks["joint"],
        road_mask,
        building_mask,
        tx_positions,
        extent,
        joint_coverage,
        dpi,
    )
    output_paths["joint"] = str(joint_path)
    np.savez_compressed(
        run_dir / "radio_maps.npz",
        path_loss_db=np.asarray(metrics["path_loss_db"], dtype=np.float32),
        wideband_received_power_dbm=np.asarray(wideband_received_power, dtype=np.float32),
        wideband_rss_dbm=np.asarray(wideband_received_power, dtype=np.float32),
        ss_rsrp_dbm=np.asarray(metrics["ss_rsrp_dbm"], dtype=np.float32),
        sinr_db=np.asarray(metrics["sinr_db"], dtype=np.float32),
        effective_throughput_mbps=np.asarray(metrics["effective_throughput_mbps"], dtype=np.float32),
        serving_idx=np.asarray(metrics["serving_idx"], dtype=np.int32),
        road_evaluation_mask=np.asarray(road_mask, dtype=np.uint8),
        building_mask=np.asarray(building_mask, dtype=np.uint8),
        pathloss_covered=np.asarray(condition_masks["pathloss"], dtype=np.uint8),
        ss_rsrp_covered=np.asarray(condition_masks["ss_rsrp"], dtype=np.uint8),
        sinr_covered=np.asarray(condition_masks["sinr"], dtype=np.uint8),
        effective_throughput_covered=np.asarray(
            condition_masks["effective_throughput"], dtype=np.uint8
        ),
        joint_covered=np.asarray(condition_masks["joint"], dtype=np.uint8),
        tx_positions_m=np.asarray(tx_positions, dtype=np.float32),
    )
    output_paths["raw_npz"] = str(run_dir / "radio_maps.npz")
    return output_paths

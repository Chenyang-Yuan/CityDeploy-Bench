#!/usr/bin/env python3
"""Interactive, non-dataset UMi road-deployment workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import numpy as np


PROJECT_ROOT = workspace_root()
MANUAL_DEPLOYMENT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_builder.interactive import build_deployable_points_from_mask
from dataset_builder.metrics import compute_radio_metrics
from dataset_builder.runtime import require_supported_runtime
from scene_builder.geometry import get_scene_map_size
from scene_builder.package import load_scene_inputs_and_meta
from visualization.radio_maps import save_radio_map_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively place road-domain UMi TXs and save manual-deployment radio maps."
    )
    parser.add_argument("--scene", required=True, help="Scene name under datasets/scenes")
    parser.add_argument("--num-tx", type=int, default=None, help="Require exactly this many selected TXs")
    parser.add_argument(
        "--tx-points",
        type=float,
        nargs="+",
        default=None,
        metavar="XY",
        help="Optional non-interactive x y pairs; each point is snapped to the road grid",
    )
    parser.add_argument("--config", type=Path, default=MANUAL_DEPLOYMENT_DIR / "config.json")
    parser.add_argument("--scenes-root", type=Path, default=PROJECT_ROOT / "datasets" / "scenes")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "manual")
    parser.add_argument("--samples-per-tx", type=int, default=None, help="Override config for a faster smoke test")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    required = ("radio", "ray_tracing", "throughput", "coverage_thresholds", "deployment")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Experiment config is missing sections: {missing}")
    return config


def nearest_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a semantic mask at target cell centers with nearest-neighbour lookup."""
    src = np.asarray(mask)
    out_h, out_w = shape
    rows = np.minimum(((np.arange(out_h) + 0.5) * src.shape[0] / out_h).astype(int), src.shape[0] - 1)
    cols = np.minimum(((np.arange(out_w) + 0.5) * src.shape[1] / out_w).astype(int), src.shape[1] - 1)
    return src[np.ix_(rows, cols)]


def choose_tx_points(
    road_mask: np.ndarray,
    building_mask: np.ndarray,
    candidates: np.ndarray,
    map_x: float,
    map_y: float,
    required_count: int | None,
    minimum_distance_m: float,
    remove_radius_m: float,
) -> list[list[float]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.widgets import Button

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(candidates)
    except Exception:
        tree = None

    extent = (-map_x / 2.0, map_x / 2.0, -map_y / 2.0, map_y / 2.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(bottom=0.14)
    ax.imshow(building_mask, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1, alpha=0.70)
    road_layer = np.ma.masked_where(np.asarray(road_mask) <= 0.5, road_mask)
    ax.imshow(
        road_layer,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#39d98a"]),
        vmin=0,
        vmax=1,
        alpha=0.78,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Road-domain TX deployment: left add, right remove")
    status = fig.text(0.02, 0.04, "Select at least one TX", fontsize=10)
    selected: list[np.ndarray] = []
    artists = []
    confirmed = {"value": False}

    def redraw(message: str = "") -> None:
        while artists:
            artists.pop().remove()
        for index, point in enumerate(selected, start=1):
            artist = ax.scatter(
                point[0], point[1], marker="*", s=230, c="#ff2d2d",
                edgecolors="black", linewidths=0.8, zorder=10,
            )
            artists.append(artist)
            artists.append(ax.text(point[0], point[1], f" {index}", color="black", weight="bold", zorder=11))
        target = f" / required {required_count}" if required_count is not None else ""
        status.set_text(f"Selected {len(selected)}{target}. {message}")
        fig.canvas.draw_idle()

    def snap(x: float, y: float) -> np.ndarray:
        query = np.asarray([x, y], dtype=np.float64)
        if tree is not None:
            _, idx = tree.query(query, k=1)
            return candidates[int(idx)].copy()
        idx = int(np.argmin(np.sum((candidates - query) ** 2, axis=1)))
        return candidates[idx].copy()

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            point = snap(float(event.xdata), float(event.ydata))
            if selected:
                distances = np.linalg.norm(np.asarray(selected) - point, axis=1)
                if float(np.min(distances)) < minimum_distance_m:
                    redraw(f"Rejected: minimum TX spacing is {minimum_distance_m:g} m")
                    return
            if required_count is not None and len(selected) >= required_count:
                redraw("Remove a TX before adding another")
                return
            selected.append(point)
            redraw()
        elif event.button == 3 and selected:
            click = np.asarray([event.xdata, event.ydata], dtype=np.float64)
            distances = np.linalg.norm(np.asarray(selected) - click, axis=1)
            idx = int(np.argmin(distances))
            if float(distances[idx]) <= remove_radius_m:
                selected.pop(idx)
                redraw()

    def on_confirm(_event) -> None:
        if not selected:
            redraw("Select at least one TX")
            return
        if required_count is not None and len(selected) != required_count:
            redraw(f"Exactly {required_count} TXs are required")
            return
        confirmed["value"] = True
        plt.close(fig)

    def on_clear(_event) -> None:
        selected.clear()
        redraw()

    confirm_ax = fig.add_axes([0.72, 0.025, 0.20, 0.065])
    clear_ax = fig.add_axes([0.55, 0.025, 0.13, 0.065])
    confirm_button = Button(confirm_ax, "Confirm & Run", color="#9be7b4", hovercolor="#67d58f")
    clear_button = Button(clear_ax, "Clear", color="#eeeeee", hovercolor="#dddddd")
    confirm_button.on_clicked(on_confirm)
    clear_button.on_clicked(on_clear)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    return [point.tolist() for point in selected] if confirmed["value"] else []


def apply_scattering_coefficients(scene, configured: dict[str, float]) -> dict[str, float]:
    """Apply substring-matched coefficients and return the material values changed."""
    applied: dict[str, float] = {}
    materials = getattr(scene, "radio_materials", {})
    iterable = materials.items() if hasattr(materials, "items") else []
    for name, material in iterable:
        lower_name = str(name).lower()
        matches = [(token, value) for token, value in configured.items() if token.lower() in lower_name]
        if not matches:
            continue
        token, value = max(matches, key=lambda item: len(item[0]))
        try:
            material.scattering_coefficient = float(value)
            applied[str(name)] = float(value)
        except Exception as exc:
            print(f"[warn] Could not set scattering coefficient for {name} ({token}): {exc}")
    return applied


def compute_experiment_metrics(path_gain: np.ndarray, num_tx: int, config: dict) -> dict[str, np.ndarray]:
    radio = config["radio"]
    throughput = config["throughput"]
    metrics = compute_radio_metrics(
        path_gain,
        num_tx=num_tx,
        tx_power_dbm=float(radio["tx_power_dbm"]),
        bandwidth_hz=float(radio["bandwidth_hz"]),
        noise_figure_db=float(radio["noise_figure_db"]),
    )
    num_active_subcarriers = int(radio["num_resource_blocks"]) * int(radio["subcarriers_per_resource_block"])
    normalization_db = 10.0 * math.log10(num_active_subcarriers)
    ss_rsrp_dbm = (
        np.asarray(metrics["rsrp_dbm_best"], dtype=np.float64)
        - normalization_db
        + float(radio.get("ssb_power_offset_db", 0.0))
    )
    spectral_efficiency = np.minimum(
        np.log2(1.0 + np.asarray(metrics["sinr_linear"], dtype=np.float64)),
        float(throughput["max_spectral_efficiency_bps_hz"]),
    )
    effective_throughput_mbps = (
        float(radio["bandwidth_hz"])
        * spectral_efficiency
        * float(throughput["implementation_efficiency"])
        * float(throughput["resource_share"])
        / 1e6
    )
    metrics["wideband_rss_dbm"] = np.asarray(metrics["rsrp_dbm_best"])
    metrics["ss_rsrp_dbm"] = ss_rsrp_dbm
    metrics["effective_throughput_mbps"] = effective_throughput_mbps
    return metrics


def compute_road_coverages(metrics: dict, road_on_radio_grid: np.ndarray, thresholds: dict) -> tuple[dict, dict]:
    road = np.asarray(road_on_radio_grid, dtype=bool)
    path_loss = np.asarray(metrics["path_loss_db"])
    ss_rsrp = np.asarray(metrics["ss_rsrp_dbm"])
    sinr = np.asarray(metrics["sinr_db"])
    throughput = np.asarray(metrics["effective_throughput_mbps"])
    conditions = {
        "pathloss": road & np.isfinite(path_loss) & (path_loss <= float(thresholds["pathloss_db_max"])),
        "ss_rsrp": road & np.isfinite(ss_rsrp) & (ss_rsrp >= float(thresholds["ss_rsrp_dbm_min"])),
        "sinr": road & np.isfinite(sinr) & (sinr >= float(thresholds["sinr_db_min"])),
        "effective_throughput": road & np.isfinite(throughput) & (
            throughput >= float(thresholds["effective_throughput_mbps_min"])
        ),
    }
    joint = conditions["pathloss"] & conditions["ss_rsrp"] & conditions["sinr"] & conditions["effective_throughput"]
    denominator = int(np.count_nonzero(road))
    if denominator == 0:
        raise RuntimeError("The resampled road evaluation mask contains no cells.")
    summary = {"denominator_road_cells": denominator}
    for name, mask in conditions.items():
        count = int(np.count_nonzero(mask))
        summary[f"{name}_count"] = count
        summary[f"{name}_coverage"] = count / denominator
    summary["joint_count"] = int(np.count_nonzero(joint))
    summary["joint_4metric_coverage"] = summary["joint_count"] / denominator
    summary["joint_target_met"] = summary["joint_4metric_coverage"] >= float(
        thresholds["joint_coverage_target"]
    )
    return summary, {**conditions, "joint": joint}


def finite_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(selected)),
        "p05": float(np.percentile(selected, 5)),
        "p50": float(np.percentile(selected, 50)),
        "p95": float(np.percentile(selected, 95)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.samples_per_tx is not None:
        config["ray_tracing"]["samples_per_tx"] = int(args.samples_per_tx)
    if config["deployment"]["candidate_policy"] != "road_only" or config["deployment"]["evaluation_policy"] != "road_only":
        raise ValueError("This manual-deployment workflow requires road_only candidate and evaluation policies.")

    scene_folder = args.scenes_root.resolve() / args.scene
    scene_xml = scene_folder / "scene.xml"
    inputs_dir = scene_folder / "inputs"
    if not scene_xml.exists():
        raise FileNotFoundError(f"Scene not found: {scene_xml}")
    semantic_path = inputs_dir / "semantic_masks.npz"
    if not semantic_path.exists():
        raise FileNotFoundError(f"Semantic masks not found: {semantic_path}")

    building_occupancy, tx_feasible_mask, metadata = load_scene_inputs_and_meta(inputs_dir)
    with np.load(semantic_path) as semantic:
        road_mask = np.asarray(semantic["road_mask"], dtype=np.float32)
    deployable_mask = ((tx_feasible_mask > 0.5) & (road_mask > 0.5)).astype(np.float32)
    map_x, map_y = get_scene_map_size(scene_xml)
    if map_x is None or map_y is None:
        raise RuntimeError(f"Could not read map dimensions from {scene_xml}")
    map_x, map_y = float(map_x), float(map_y)
    if abs(map_x - float(metadata["map_x_m"])) > 1e-4 or abs(map_y - float(metadata["map_y_m"])) > 1e-4:
        raise RuntimeError("Scene XML and semantic-mask dimensions do not agree.")

    candidate_cell = config["deployment"]["candidate_grid_m"]
    _, candidates = build_deployable_points_from_mask(
        deployable_mask, map_x, map_y, float(candidate_cell[0]), float(candidate_cell[1])
    )
    if not candidates.size:
        raise RuntimeError("No road-domain TX candidates were found.")
    print(f"Scene: {scene_folder}")
    print(f"Map: {map_x:g} m x {map_y:g} m")
    print(f"Road TX candidates: {len(candidates):,}")
    if args.tx_points is not None:
        if len(args.tx_points) == 0 or len(args.tx_points) % 2 != 0:
            raise ValueError("--tx-points requires x y pairs")
        requested = np.asarray(args.tx_points, dtype=np.float64).reshape(-1, 2)
        selected_xy = []
        for point in requested:
            idx = int(np.argmin(np.sum((candidates - point) ** 2, axis=1)))
            selected_xy.append(candidates[idx].tolist())
        selected_array = np.asarray(selected_xy)
        if args.num_tx is not None and len(selected_xy) != args.num_tx:
            raise ValueError(f"--num-tx requires {args.num_tx} points, got {len(selected_xy)}")
        if len(selected_xy) > 1:
            delta = selected_array[:, None, :] - selected_array[None, :, :]
            distances = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(distances, np.inf)
            if float(np.min(distances)) < float(config["deployment"]["minimum_tx_distance_m"]):
                raise ValueError("Snapped --tx-points violate minimum_tx_distance_m")
    else:
        selected_xy = choose_tx_points(
            road_mask=road_mask,
            building_mask=building_occupancy,
            candidates=candidates,
            map_x=map_x,
            map_y=map_y,
            required_count=args.num_tx,
            minimum_distance_m=float(config["deployment"]["minimum_tx_distance_m"]),
            remove_radius_m=float(config["deployment"]["remove_click_radius_m"]),
        )
    if not selected_xy:
        print("No TX plan was confirmed; nothing was simulated or saved.")
        return

    radio = config["radio"]
    ray = config["ray_tracing"]
    tx_positions = np.column_stack(
        [np.asarray(selected_xy, dtype=np.float64), np.full(len(selected_xy), float(radio["tx_height_m"]))]
    )
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / f"{args.scene}_ntx{len(tx_positions)}_{run_stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = args.output_root.resolve() / f"{args.scene}_ntx{len(tx_positions)}_{run_stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)

    # Save the selected plan before loading native libraries: a fatal JIT error

    save_deployment_input(run_dir, args.scene, tx_positions, [map_x, map_y], config)

    runtime = require_supported_runtime(requested_device=args.device)
    from radio_backend.sionna_adapter import (
        configure_transmitters_receivers,
        generate_coverage_map,
        load_and_preview_scene,
    )

    print("\nConfirmed TX positions:")
    for index, position in enumerate(tx_positions, start=1):
        print(f"  TX{index}: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} m")
    print(f"Results directory: {run_dir}")
    print(f"Backend: {runtime['mitsuba_variant']}")
    print("Starting scene load and ray tracing ...")
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    scene = load_and_preview_scene(str(scene_folder), preview=False)
    scene_load_s = time.perf_counter() - load_started
    applied_scattering = apply_scattering_coefficients(scene, config.get("diffuse_scattering_coefficients", {}))
    configure_transmitters_receivers(
        scene,
        tx_positions=tx_positions.tolist(),
        tx_azimuths=[0.0] * len(tx_positions),
        frequency=float(radio["frequency_hz"]),
        tx_power_dbm=float(radio["tx_power_dbm"]),
        tx_pattern=str(radio.get("antenna_pattern", "iso")),
        tx_polarization=str(radio.get("polarization", "V")),
        rx_pattern=str(radio.get("antenna_pattern", "iso")),
        rx_polarization=str(radio.get("polarization", "V")),
    )
    trace_started = time.perf_counter()
    radio_map = generate_coverage_map(
        scene,
        max_depth=int(ray["max_depth"]),
        los=bool(ray["line_of_sight"]),
        cell_size=tuple(float(v) for v in ray["cell_size_m"]),
        size=[map_x, map_y],
        samples_per_tx=int(ray["samples_per_tx"]),
        use_planar=bool(ray["use_planar"]),
        device=args.device,
        tx_position=tx_positions[0].tolist(),
        specular_reflection=bool(ray["specular_reflection"]),
        diffuse_reflection=bool(ray["diffuse_reflection"]),
        refraction=bool(ray["refraction"]),
        diffraction=bool(ray["diffraction"]),
        edge_diffraction=bool(ray["edge_diffraction"]),
        seed=int(ray["seed"]),
    )
    raytracing_s = time.perf_counter() - trace_started

    metrics_started = time.perf_counter()
    metrics = compute_experiment_metrics(np.asarray(radio_map.path_gain), len(tx_positions), config)
    shape = np.asarray(metrics["path_loss_db"]).shape
    road_grid = nearest_mask(road_mask, shape) > 0.5
    building_grid = nearest_mask(building_occupancy, shape) > 0.5
    coverage, condition_masks = compute_road_coverages(metrics, road_grid, config["coverage_thresholds"])
    metric_compute_s = time.perf_counter() - metrics_started

    save_started = time.perf_counter()
    outputs = save_radio_map_outputs(
        run_dir=run_dir,
        metrics=metrics,
        condition_masks=condition_masks,
        road_mask=road_grid,
        building_mask=building_grid,
        tx_positions=tx_positions,
        map_x=map_x,
        map_y=map_y,
        cmap=str(config["visualization"].get("colormap", "turbo")),
        dpi=int(config["visualization"].get("dpi", 220)),
    )
    save_s = time.perf_counter() - save_started
    total_s = time.perf_counter() - total_started
    timings = {
        "scene_load_s": scene_load_s,
        "raytracing_s": raytracing_s,
        "metric_computation_s": metric_compute_s,
        "save_and_plot_s": save_s,
        "total_after_confirmation_s": total_s,
    }
    road_stats = {
        "path_loss_db": finite_stats(metrics["path_loss_db"], road_grid),
        "ss_rsrp_dbm": finite_stats(metrics["ss_rsrp_dbm"], road_grid),
        "sinr_db": finite_stats(metrics["sinr_db"], road_grid),
        "effective_throughput_mbps": finite_stats(metrics["effective_throughput_mbps"], road_grid),
    }
    resolved = {
        **config,
        "scene": args.scene,
        "scene_folder": str(scene_folder),
        "output_policy": "manual-deployment-only; never written to datasets",
        "runtime": runtime,
        "applied_scattering_coefficients": applied_scattering,
    }
    (run_dir / "configuration_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    summary = {
        "scene": args.scene,
        "map_size_m": [map_x, map_y],
        "num_tx": len(tx_positions),
        "tx_positions_m": tx_positions.tolist(),
        "evaluation_policy": "road_only",
        "thresholds": config["coverage_thresholds"],
        "coverage": coverage,
        "road_metric_statistics": road_stats,
        "timings": timings,
        "outputs": outputs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTiming:")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.3f} s")
    print("\nRoad-domain coverage:")
    print(f"  Path loss:            {coverage['pathloss_coverage']:.4%}")
    print(f"  SS-RSRP:              {coverage['ss_rsrp_coverage']:.4%}")
    print(f"  SINR:                 {coverage['sinr_coverage']:.4%}")
    print(f"  Effective throughput: {coverage['effective_throughput_coverage']:.4%}")
    print(f"  Joint four-metric:    {coverage['joint_4metric_coverage']:.4%}")
    print(f"  Joint target met:     {coverage['joint_target_met']}")
    print(f"  Denominator:          {coverage['denominator_road_cells']:,} road cells")
    print(f"\nSaved manual deployment run: {run_dir}")


def save_deployment_input(run_dir, scene, tx_positions, map_size_m, config):
    """Persist inputs only; this file is not evidence of a completed simulation."""
    payload = {
        "scene": scene,
        "num_tx": len(tx_positions),
        "tx_positions_m": np.asarray(tx_positions, dtype=float).tolist(),
        "map_size_m": list(map_size_m),
        "configuration": config,
        "note": "Selected inputs; completion is recorded separately in summary.json.",
    }
    (Path(run_dir) / "deployment_input.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

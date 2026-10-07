"""Build canonical model inputs and TX feasibility data from semantic masks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TX_FEASIBILITY_POLICIES = {
    "road_only",
    "road_and_explicit_open",
}


def _map_figure_size(map_x: float, map_y: float, base: float = 6.0) -> tuple[float, float]:
    aspect = max(float(map_x), 1e-6) / max(float(map_y), 1e-6)
    return (base * min(aspect, 2.2), base) if aspect >= 1.0 else (base, base * min(1.0 / aspect, 2.2))


def compose_tx_feasible_mask(
    semantic_masks: dict[str, np.ndarray],
    policy: str = "road_only",
    threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if policy not in TX_FEASIBILITY_POLICIES:
        raise ValueError(f"Unknown TX feasibility policy: {policy}")
    road = np.asarray(semantic_masks["road_mask"], dtype=np.float32) > threshold
    explicit_open = np.asarray(semantic_masks.get("explicit_open_mask", np.zeros_like(road))) > threshold
    feasible = road
    if policy == "road_and_explicit_open":
        feasible = road | explicit_open
    return feasible.astype(np.float32), {
        "road": road.astype(np.float32),
        "explicit_open": explicit_open.astype(np.float32),
    }


def build_tx_deployable_space(
    inputs_dir: Path,
    visualizations_dir: Path,
    threshold: float = 0.5,
    policy: str = "road_only",
) -> tuple[bool, str, dict]:
    semantic_npz = inputs_dir / "semantic_masks.npz"
    summary_json = inputs_dir / "summary.json"
    out_dir = inputs_dir / "tx_deployable"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not semantic_npz.exists() or not summary_json.exists():
        return False, f"Missing semantic inputs under {inputs_dir}", {}
    try:
        with np.load(semantic_npz) as loaded:
            sem = {key: np.asarray(loaded[key]) for key in loaded.files}
        road = np.asarray(sem["road_mask"], dtype=np.float32)
        deployable, components = compose_tx_feasible_mask(sem, policy, threshold)
    except Exception as exc:
        return False, f"Failed to load semantic masks: {exc}", {}
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    sem_meta = summary.get("semantic_raster", {})
    map_x = float(sem_meta.get("map_x_m_local", road.shape[1]))
    map_y = float(sem_meta.get("map_y_m_local", road.shape[0]))
    h, w = road.shape
    blocked = 1.0 - deployable
    xs = np.linspace(-map_x / 2 + 0.5 * map_x / w, map_x / 2 - 0.5 * map_x / w, w)
    ys = np.linspace(-map_y / 2 + 0.5 * map_y / h, map_y / 2 - 0.5 * map_y / h, h)
    gx, gy = np.meshgrid(xs, ys)
    deploy_bool = deployable > 0.5
    deploy_xy = np.column_stack([gx[deploy_bool], gy[deploy_bool]]).astype(np.float32)
    np.savez_compressed(
        out_dir / "masks.npz",
        tx_deployable_mask=deployable,
        tx_blocked_mask=blocked,
        tx_deployable_road_mask=components["road"],
        tx_deployable_explicit_open_mask=components["explicit_open"],
    )
    np.save(out_dir / "points_local_xy.npy", deploy_xy)
    extent = (-map_x / 2, map_x / 2, -map_y / 2, map_y / 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, values, title in zip(
        axes,
        (components["road"], components["explicit_open"], deployable),
        ("Road deployable", "Explicit-open candidate", f"TX feasible ({policy})"),
    ):
        ax.imshow(values, origin="lower", extent=extent, cmap="Greens", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(visualizations_dir / "tx_feasibility_components.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=_map_figure_size(map_x, map_y))
    ax.imshow(deployable, origin="lower", extent=extent, cmap="Greens", vmin=0, vmax=1)
    ax.set(title="TX deployable", xlabel="x (m)", ylabel="y (m)"); ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(visualizations_dir / "tx_deployable.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    stats = {
        "grid_h": h, "grid_w": w, "map_x_m_local": map_x, "map_y_m_local": map_y,
        "threshold": threshold, "policy": policy, "num_total_cells": h * w,
        "num_road_cells": int((road > threshold).sum()),
        "num_explicit_open_cells": int(components["explicit_open"].sum()),
        "num_deployable_cells": int(deploy_bool.sum()), "num_blocked_cells": int((~deploy_bool).sum()),
        "deployable_ratio": float(deploy_bool.mean()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return True, "", stats


def build_training_scene_inputs(
    scene_id: str,
    inputs_dir: Path,
    threshold: float = 0.5,
    tx_policy: str = "road_only",
) -> tuple[bool, str, dict]:
    semantic_npz, summary_json = inputs_dir / "semantic_masks.npz", inputs_dir / "summary.json"
    if not semantic_npz.exists() or not summary_json.exists():
        return False, f"Missing semantic inputs under {inputs_dir}", {}
    try:
        with np.load(semantic_npz) as loaded:
            sem = {key: np.asarray(loaded[key]) for key in loaded.files}
        building = np.asarray(sem["building_mask"], dtype=np.float32)
        deployable, _ = compose_tx_feasible_mask(sem, tx_policy, threshold)
        evaluation = np.asarray(sem.get("service_evaluation_mask", np.ones_like(building)), dtype=np.float32)
    except Exception as exc:
        return False, f"Failed to load semantic masks: {exc}", {}
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    sem_meta = summary.get("semantic_raster", {})
    map_x = float(sem_meta.get("map_x_m_local", building.shape[1]))
    map_y = float(sem_meta.get("map_y_m_local", building.shape[0]))
    h, w = building.shape
    occupancy = (building > threshold).astype(np.float32)
    evaluation = (evaluation > threshold).astype(np.float32)
    total_points = int((np.floor(map_x) + 1) * (np.floor(map_y) + 1))
    deployable_count = int(round(float(deployable.mean()) * total_points))
    np.savez_compressed(
        inputs_dir / "scene_inputs.npz",
        building_occupancy=occupancy,
        deployable_mask=deployable,
        tx_feasible_mask=deployable,
        evaluation_mask=evaluation,
    )
    metadata = {
        "schema_version": "citydeploy.scene.v3", "semantic_schema_version": summary.get("semantic_schema_version", "unknown"),
        "scene": scene_id, "map_x_m": map_x, "map_y_m": map_y, "grid_h": h, "grid_w": w,
        "normalization": {
            "real_to_norm": "x_norm=(x_real+map_x/2)/map_x; y_norm=(y_real+map_y/2)/map_y",
            "norm_to_real": "x_real=x_norm*map_x-map_x/2; y_real=y_norm*map_y-map_y/2",
            "grid_convention": "row 0 is y_min; columns increase from x_min to x_max",
        },
        "tx_candidate_point_grid_m": [1.0, 1.0], "num_total_tx_candidate_points": total_points,
        "num_deployable_points": deployable_count, "num_blocked_points": total_points - deployable_count,
        "deployable_points_ratio": float(deployable_count / total_points), "tx_feasibility_policy": tx_policy,
        "evaluation_mask_ratio": float(evaluation.mean()), "evaluation_policy": "all_non_water_cells",
    }
    (inputs_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return True, "", {"grid_h": h, "grid_w": w, "map_x_m": map_x, "map_y_m": map_y}

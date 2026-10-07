#!/usr/bin/env python3
"""Benchmark learned deployment rewards on seen and unseen urban scenes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path
from typing import Any

import pyarrow
import numpy as np
import torch

from dataset_builder.generate import _metric_kwargs, _resize_mask_nearest
from dataset_builder.metrics import compute_urban_radio_metrics, summarize_urban_metrics, tx_axis_last
from dataset_builder.raytracing import SceneRayTracer
from energy_model.dataset import load_scene_feature_tensor, set_seed
from energy_model.guidance import load_guidance_checkpoint, predict_reward
from samplers import deployment_constraint_penalty
from samplers.runner import add_sampler_arguments, config_to_dict, run_sampler, sampler_config
from energy_model.train import resolve_device
from scene_builder.coordinates import SceneFrame2D
from visualization.radio_maps import save_joint_coverage_plot


PROJECT_ROOT = workspace_root()
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "benchmarks"

BENCHMARK_SCENES = {
    "london": {
        128: {"seen": "london_10", "unseen": "london_09"},
        256: {"seen": "london_06", "unseen": "london_08"},
        512: {"seen": "london_01", "unseen": "london_03"},
    },
    "new_york": {
        128: {"seen": "new_york_09", "unseen": "new_york_10"},
        256: {"seen": "new_york_06", "unseen": "new_york_12"},
        512: {"seen": "new_york_03", "unseen": "new_york_01"},
    },
    "paris": {
        128: {"seen": "paris_09", "unseen": "paris_11"},
        256: {"seen": "paris_06", "unseen": "paris_08"},
        512: {"seen": "paris_01", "unseen": "paris_03"},
    },
    "shanghai": {
        128: {"seen": "shanghai_09", "unseen": "shanghai_10"},
        256: {"seen": "shanghai_06", "unseen": "shanghai_07"},
        512: {"seen": "shanghai_01", "unseen": "shanghai_02"},
    },
    "sydney": {
        128: {"seen": "sydney_09", "unseen": "sydney_10"},
        256: {"seen": "sydney_06", "unseen": "sydney_08"},
        512: {"seen": "sydney_01", "unseen": "sydney_11"},
    },
    "tokyo": {
        128: {"seen": "tokyo_09", "unseen": "tokyo_11"},
        256: {"seen": "tokyo_07", "unseen": "tokyo_06"},
        512: {"seen": "tokyo_03", "unseen": "tokyo_01"},
    },
}
NUM_TX_BY_SIZE = {128: 2, 256: 3, 512: 7}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Six-city seen/unseen deployment benchmark")
    parser.add_argument("--hypergraph-checkpoint", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--interaction-checkpoint", type=Path)
    parser.add_argument("--nri-checkpoint", type=Path)
    parser.add_argument("--scene-root", type=Path, default=PROJECT_ROOT / "datasets" / "scenes")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raytracing" / "urban_multicity",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-particles", type=int, default=100)
    add_sampler_arguments(parser)
    parser.add_argument("--min-tx-distance-m", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--ray-device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--max-cases", type=int, default=0, help="Smoke-test only; zero runs all cases")
    return parser


def _load_model(path: Path, device: torch.device) -> tuple[str, torch.nn.Module]:
    name, model, _checkpoint = load_guidance_checkpoint(path, device)
    return name, model


def _predict(
    name: str,
    model: torch.nn.Module,
    points: torch.Tensor,
    scene: torch.Tensor,
    map_size: torch.Tensor,
) -> torch.Tensor:
    mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    return predict_reward(name, model, points, mask, scene, map_size.expand(points.shape[0], -1))


def _sample_plan(
    name: str,
    model: torch.nn.Module,
    scene: torch.Tensor,
    map_size: torch.Tensor,
    feasible_mask: torch.Tensor,
    candidates: np.ndarray,
    num_tx: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, float, float, int | None]:
    set_seed(seed)

    def reward_fn(points: torch.Tensor) -> torch.Tensor:
        prediction = _predict(name, model, points, scene, map_size)
        _, components = deployment_constraint_penalty(
            points,
            feasible_mask,
            map_size.expand(points.shape[0], -1),
            args.min_tx_distance_m,
        )
        return prediction - 10.0 * components["nondeployable"] - 10.0 * components["minimum_distance"]

    started = time.perf_counter()
    result = run_sampler(
        reward_fn,
        sampler=args.sampler,
        num_particles=args.num_particles,
        num_tx=num_tx,
        device=scene.device,
        config=sampler_config(args),
    )
    inference_seconds = time.perf_counter() - started
    points = result["best_tx_xy_norm01"].detach().cpu().numpy()
    penalized = result["best_reward"].detach().cpu().numpy()
    order = np.argsort(-penalized)
    frame = SceneFrame2D(float(map_size[0, 0]), float(map_size[0, 1]))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(candidates)
        snap = lambda values: candidates[tree.query(values)[1]]
    except Exception:
        snap = lambda values: candidates[
            np.argmin(np.sum((values[:, None, :] - candidates[None, :, :]) ** 2, axis=-1), axis=1)
        ]
    for index in order:
        real = frame.norm01_to_real(points[index])
        snapped = np.asarray(snap(real), dtype=np.float64)
        if len(np.unique(snapped, axis=0)) != num_tx:
            continue
        distance = np.linalg.norm(snapped[:, None, :] - snapped[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        if float(distance.min()) + 1e-6 < args.min_tx_distance_m:
            continue
        norm = frame.real_to_norm01(snapped)
        tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).to(scene.device)
        with torch.no_grad():
            predicted = float(_predict(name, model, tensor, scene, map_size).item())
        return snapped, predicted, inference_seconds, result.get("num_reward_evaluations")
    raise RuntimeError("no sampled plan remained valid after snapping to road candidates")


def _dataset_distributions(dataset_root: Path) -> dict[tuple[str, int], np.ndarray]:
    import pyarrow.parquet as pq

    grouped: dict[tuple[str, int], list[float]] = {}
    for path in (dataset_root / "data").glob("*/*.parquet"):
        for row in pq.read_table(
            path, columns=["scene_id", "num_tx", "joint_4metric_coverage"]
        ).to_pylist():
            grouped.setdefault((str(row["scene_id"]), int(row["num_tx"])), []).append(
                float(row["joint_4metric_coverage"])
            )
    return {key: np.asarray(values) for key, values in grouped.items()}


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    result = {}
    for value, items in sorted(grouped.items()):
        actual = np.asarray([float(item["joint_4metric_coverage"]) for item in items])
        error = np.asarray([float(item["prediction_error"]) for item in items])
        result[value] = {
            "num_cases": len(items),
            "actual_joint_mean": float(actual.mean()),
            "prediction_mae": float(np.abs(error).mean()),
            "prediction_bias": float(error.mean()),
        }
    return result


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint_paths = [args.hypergraph_checkpoint, args.reward_checkpoint]
    checkpoint_paths.extend(
        path for path in (args.interaction_checkpoint, args.nri_checkpoint) if path is not None
    )
    loaded = [_load_model(path.resolve(), device) for path in checkpoint_paths]
    models = dict(loaded)
    if len(models) != len(loaded):
        raise RuntimeError("checkpoint arguments contain duplicate model kinds")
    plan = json.loads((args.dataset_root / "generation_plan.json").read_text(encoding="utf-8"))
    rf = json.loads(resolve_path(plan["rf_profile"]).read_text(encoding="utf-8"))
    distributions = _dataset_distributions(args.dataset_root)
    output_root = args.output_root.resolve() / args.sampler / "seen_unseen_six_city"
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cases = [
        (city, size, status, scene_id)
        for city, sizes in BENCHMARK_SCENES.items()
        for size, statuses in sizes.items()
        for status, scene_id in statuses.items()
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    for case_index, (city, size, status, scene_id) in enumerate(cases):
        num_tx = NUM_TX_BY_SIZE[size]
        scene_dir = args.scene_root.resolve() / scene_id
        metadata = json.loads((scene_dir / "inputs" / "metadata.json").read_text(encoding="utf-8"))
        if round(float(metadata["map_x_m"])) != size or round(float(metadata["map_y_m"])) != size:
            raise RuntimeError(f"{scene_id} is not a {size} m square scene")
        scene_array, _ = load_scene_feature_tensor(scene_dir)
        scene_tensor = torch.from_numpy(scene_array).unsqueeze(0).to(device)
        map_size = torch.tensor([[size, size]], dtype=torch.float32, device=device)
        with np.load(scene_dir / "inputs" / "scene_inputs.npz", allow_pickle=False) as loaded:
            feasible_mask = torch.from_numpy(
                np.asarray(loaded["tx_feasible_mask"], dtype=np.float32)
            ).to(device)
        with np.load(scene_dir / "inputs" / "semantic_masks.npz", allow_pickle=False) as loaded:
            road_source = np.asarray(loaded["road_mask"], dtype=bool)
            building_source = np.asarray(loaded["building_mask"], dtype=bool)
        candidates = np.asarray(
            np.load(scene_dir / "inputs" / "tx_deployable" / "points_local_xy.npy", allow_pickle=False),
            dtype=np.float64,
        )
        tracer = SceneRayTracer(scene_dir, rf, args.ray_device)
        for model_index, (model_name, model) in enumerate(models.items()):
            case_dir = output_root / model_name / status / f"{city}_{size}_{scene_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            result_path = case_dir / "result.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                print(f"skip complete: {model_name} {status} {scene_id}")
                continue
            case_seed = args.seed + case_index * 1009
            snapped, prediction, inference_seconds, reward_evaluations = _sample_plan(
                model_name,
                model,
                scene_tensor,
                map_size,
                feasible_mask,
                candidates,
                num_tx,
                args,
                case_seed,
            )
            tx_xyz = np.column_stack(
                [snapped, np.full(num_tx, float(rf["tx_height_m"]), dtype=np.float64)]
            )
            raw_path_gain, ray_seconds = tracer.trace(tx_xyz, (size, size), case_seed)
            path_gain = tx_axis_last(raw_path_gain, num_tx)
            metrics = compute_urban_radio_metrics(path_gain, num_tx, **_metric_kwargs(rf))
            road = _resize_mask_nearest(road_source, path_gain.shape[:2])
            building = _resize_mask_nearest(building_source, path_gain.shape[:2])
            summary, conditions = summarize_urban_metrics(metrics, road, rf["coverage_thresholds"])
            actual = float(summary["joint_4metric_coverage"])
            reference = distributions.get((scene_id, num_tx))
            percentile = None if reference is None else 100.0 * float(np.mean(reference <= actual))
            save_joint_coverage_plot(
                case_dir / "joint_coverage.png",
                conditions["joint"],
                road,
                building,
                tx_xyz,
                (-size / 2, size / 2, -size / 2, size / 2),
                actual,
            )
            row = {
                "search_method": args.sampler,
                "model": model_name,
                "city": city,
                "size_m": size,
                "status": status,
                "scene_id": scene_id,
                "num_tx": num_tx,
                "seed": case_seed,
                "predicted_reward": prediction,
                "joint_4metric_coverage": actual,
                "prediction_error": prediction - actual,
                "pathloss_coverage": float(summary["pathloss_coverage"]),
                "ss_rsrp_coverage": float(summary["ss_rsrp_coverage"]),
                "sinr_coverage": float(summary["sinr_coverage"]),
                "effective_throughput_coverage": float(summary["effective_throughput_coverage"]),
                "dataset_percentile": percentile,
                "tx_positions_m": tx_xyz.tolist(),
                "inference_seconds": inference_seconds,
                "num_reward_evaluations": reward_evaluations,
                "raytracing_seconds": ray_seconds,
                "joint_plot": str(case_dir / "joint_coverage.png"),
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(
                f"{len(rows):02d}/{len(cases)*len(models)} method={args.sampler} "
                f"model={model_name} status={status} "
                f"scene={scene_id} ntx={num_tx} pred={prediction:.4f} actual={actual:.4f}"
            )
        del tracer, scene_tensor, feasible_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = output_root / "results.csv"
    scalar_fields = [key for key, value in rows[0].items() if key not in {"tx_positions_m"}]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_fields} for row in rows)
    report = {
        "num_results": len(rows),
        "num_scene_conditions": len(rows) // max(len(models), 1),
        "sampling": {
            "method": args.sampler,
            "num_particles": args.num_particles,
            "config": config_to_dict(sampler_config(args)),
            "min_tx_distance_m": args.min_tx_distance_m,
        },
        "by_model": _aggregate(rows, "model"),
        "by_status": _aggregate(rows, "status"),
        "by_size_m": _aggregate(rows, "size_m"),
        "results_csv": str(csv_path),
    }
    (output_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

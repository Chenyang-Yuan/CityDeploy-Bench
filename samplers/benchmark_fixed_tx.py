#!/usr/bin/env python3
"""Run a resumable fixed-cardinality multi-planner benchmark with Sionna verification."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path
from typing import Any

import pyarrow  # Load before torch on Windows.
import numpy as np
import torch

from dataset_builder.generate import _metric_kwargs, _resize_mask_nearest
from dataset_builder.metrics import compute_urban_radio_metrics, summarize_urban_metrics, tx_axis_last
from energy_model.dataset import load_scene_feature_tensor, set_seed
from energy_model.guidance import load_guidance_checkpoint, predict_reward
from energy_model.train import resolve_device
from samplers.common import deployment_constraint_penalty
from samplers.runner import run_sampler, sampler_config_from_dict
from scene_builder.coordinates import SceneFrame2D
from visualization.radio_maps import save_joint_coverage_plot


PROJECT_ROOT = workspace_root()
DEFAULT_PLAN = config_path("experiments/fixed_tx_seen.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-TX multi-planner Sionna benchmark")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repeats", type=int, default=0, help="Override plan repeats; zero keeps plan")
    parser.add_argument("--only-planners", nargs="*", default=[])
    parser.add_argument("--only-scenes", nargs="*", default=[])
    parser.add_argument("--max-trials", type=int, default=0, help="Pilot limit; zero runs all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--ray-device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return resolve_path(path)


def _load_plan(path: Path) -> dict:
    plan = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if plan.get("schema") != "citydeploy.fixed-tx-benchmark":
        raise ValueError(f"unsupported plan schema: {plan.get('schema')}")
    return plan


def _dataset_distributions(dataset_root: Path) -> dict[tuple[str, int], np.ndarray]:
    import pyarrow.parquet as pq

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for path in (dataset_root / "data").glob("*/*.parquet"):
        table = pq.read_table(path, columns=["scene_id", "num_tx", "joint_4metric_coverage"])
        for row in table.to_pylist():
            grouped[(str(row["scene_id"]), int(row["num_tx"]))].append(
                float(row["joint_4metric_coverage"])
            )
    return {key: np.asarray(values) for key, values in grouped.items()}


def _validate_plan(plan: dict, load_models: bool, device: torch.device | None = None) -> dict:
    planners = plan.get("planners", [])
    conditions = plan.get("conditions", [])
    if not planners or not conditions:
        raise ValueError("plan must contain non-empty planners and conditions")
    planner_ids = [str(item["id"]) for item in planners]
    if len(set(planner_ids)) != len(planner_ids):
        raise ValueError("planner ids must be unique")
    dataset_root = _resolve(plan["dataset_root"])
    generation = json.loads((dataset_root / "generation_plan.json").read_text(encoding="utf-8"))
    assignments = generation.get("split", {}).get("scene_assignments", {})
    for condition in conditions:
        scene_id = str(condition["scene"])
        if assignments.get(scene_id) != "train":
            raise ValueError(f"fixed seen benchmark requires a train scene, got {scene_id}={assignments.get(scene_id)}")
        scene_dir = _resolve(plan["scene_root"]) / scene_id
        metadata = json.loads((scene_dir / "inputs" / "metadata.json").read_text(encoding="utf-8"))
        expected = int(condition["size_m"])
        actual = (round(float(metadata["map_x_m"])), round(float(metadata["map_y_m"])))
        if actual != (expected, expected):
            raise ValueError(f"{scene_id} expected {expected} m square, got {actual}")
    models = {}
    model_paths = {name: _resolve(value) for name, value in plan["models"].items()}
    for planner in planners:
        if planner["model"] not in model_paths:
            raise ValueError(f"planner {planner['id']} references unknown model {planner['model']}")
        sampler_config_from_dict(str(planner["sampler"]), dict(planner["config"]))
    if load_models:
        assert device is not None
        for alias, path in model_paths.items():
            kind, model, checkpoint = load_guidance_checkpoint(path, device)
            expected_kind = {
                "hpem": "hypergraph_potential",
                "interaction_network": "interaction_network",
                "nri": "nri",
            }.get(alias)
            if expected_kind is not None and kind != expected_kind:
                raise ValueError(f"model alias {alias} expected {expected_kind}, checkpoint contains {kind}")
            models[alias] = (kind, model, path, checkpoint)
    return {"models": models, "model_paths": model_paths, "generation": generation}


def _trial_specs(plan: dict, repeats: int, only_planners: set[str], only_scenes: set[str]) -> list[dict]:
    planners = [item for item in plan["planners"] if not only_planners or item["id"] in only_planners]
    specs = []
    # Keep the original condition index so filtering a run never changes its seeds.
    for condition_index, condition in enumerate(plan["conditions"]):
        if only_scenes and condition["scene"] not in only_scenes:
            continue
        for repeat in range(repeats):
            seed = int(plan["base_seed"]) + condition_index * 1009 + repeat * 9176
            for planner in planners:
                specs.append({"condition": condition, "planner": planner, "repeat": repeat, "seed": seed})
    return specs


def _constrained_snap(real: np.ndarray, candidates: np.ndarray, min_distance: float) -> np.ndarray | None:
    """Project an unordered TX set onto distinct, separated road candidates."""
    num_tx = len(real)
    squared_distance = np.sum((real[:, None] - candidates[None]) ** 2, axis=-1)
    nearest_order = np.argsort(squared_distance, axis=1)
    base = tuple(range(num_tx))
    tx_orders = [base[offset:] + base[:offset] for offset in range(num_tx)]
    reversed_base = tuple(reversed(base))
    tx_orders += [reversed_base[offset:] + reversed_base[:offset] for offset in range(num_tx)]

    best = None
    best_cost = float("inf")
    for tx_order in tx_orders:
        assignment: dict[int, int] = {}
        selected: list[np.ndarray] = []
        used: set[int] = set()
        for tx_index in tx_order:
            chosen = None
            for candidate_index in nearest_order[tx_index]:
                candidate_index = int(candidate_index)
                if candidate_index in used:
                    continue
                point = candidates[candidate_index]
                if selected and min(np.linalg.norm(point - other) for other in selected) + 1e-6 < min_distance:
                    continue
                chosen = candidate_index
                break
            if chosen is None:
                assignment = {}
                break
            assignment[tx_index] = chosen
            used.add(chosen)
            selected.append(candidates[chosen])
        if len(assignment) != num_tx:
            continue
        snapped = np.asarray([candidates[assignment[index]] for index in range(num_tx)], dtype=np.float64)
        cost = float(np.sum((snapped - real) ** 2))
        if cost < best_cost:
            best, best_cost = snapped, cost
    return best


def _snap_candidate(points_norm: np.ndarray, reward: np.ndarray, candidates: np.ndarray, frame: SceneFrame2D, min_distance: float):
    order = np.argsort(-reward)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(candidates)
        snap = lambda values: candidates[tree.query(values)[1]]
    except Exception:
        snap = lambda values: candidates[np.argmin(np.sum((values[:, None] - candidates[None]) ** 2, axis=-1), axis=1)]
    repaired_fallback = None
    for index in order:
        real = frame.norm01_to_real(points_norm[index])
        snapped = np.asarray(snap(real), dtype=np.float64)
        if len(np.unique(snapped, axis=0)) != points_norm.shape[1]:
            if repaired_fallback is None:
                repaired = _constrained_snap(real, candidates, min_distance)
                if repaired is not None:
                    repaired_fallback = (repaired, int(index))
            continue
        if len(snapped) > 1:
            distance = np.linalg.norm(snapped[:, None] - snapped[None], axis=-1)
            np.fill_diagonal(distance, np.inf)
            if float(distance.min()) + 1e-6 < min_distance:
                if repaired_fallback is None:
                    repaired = _constrained_snap(real, candidates, min_distance)
                    if repaired is not None:
                        repaired_fallback = (repaired, int(index))
                continue
        return snapped, int(index)
    if repaired_fallback is not None:
        return repaired_fallback
    raise RuntimeError("no sampled plan remained valid after snapping to deployable road points")


def _aggregate(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = {}
    for name, items in sorted(groups.items()):
        joint = np.asarray([item["joint_4metric_coverage"] for item in items], dtype=float)
        runtime = np.asarray([item["inference_seconds"] for item in items], dtype=float)
        evaluations = np.asarray([item["num_reward_evaluations"] for item in items], dtype=float)
        output[name] = {
            "trials": len(items),
            "joint_mean": float(joint.mean()),
            "joint_std": float(joint.std(ddof=1)) if len(joint) > 1 else 0.0,
            "inference_seconds_mean": float(runtime.mean()),
            "reward_evaluations_mean": float(evaluations.mean()),
        }
    return output


def _write_summary(output_root: Path, expected_trials: int) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in output_root.glob("*/*/*/seed_*/result.json")]
    rows.sort(key=lambda row: (row["planner_id"], row["city"], row["size_m"], row["seed"]))
    if not rows:
        return
    fields = [key for key in rows[0] if key not in {"tx_positions_m", "sampler_config", "checkpoint"}]
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    report = {
        "completed_trials": len(rows),
        "expected_trials": expected_trials,
        "complete": len(rows) == expected_trials,
        "by_planner": _aggregate(rows, "planner_id"),
        "by_size_m": _aggregate(rows, "size_m"),
        "by_city": _aggregate(rows, "city"),
        "results_csv": str(output_root / "results.csv"),
    }
    (output_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    plan = _load_plan(args.plan)
    repeats = args.repeats or int(plan["repeats"])
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    device = resolve_device(args.device)
    validated = _validate_plan(plan, load_models=not args.dry_run, device=device)
    specs = _trial_specs(plan, repeats, set(args.only_planners), set(args.only_scenes))
    expected_trials = len(plan["conditions"]) * len(plan["planners"]) * repeats
    if args.max_trials:
        specs = specs[: args.max_trials]
    inventory = {
        "experiment_id": plan["experiment_id"],
        "conditions": len({(x["condition"]["scene"], x["condition"]["num_tx"]) for x in specs}),
        "planners": sorted({x["planner"]["id"] for x in specs}),
        "repeats": repeats,
        "trials": len(specs),
        "size_tx": sorted({(x["condition"]["size_m"], x["condition"]["num_tx"]) for x in specs}),
        "models": {name: str(path) for name, path in validated["model_paths"].items()},
    }
    print(json.dumps(inventory, indent=2))
    if args.dry_run:
        return

    dataset_root = _resolve(plan["dataset_root"])
    scene_root = _resolve(plan["scene_root"])
    output_root = _resolve(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_plan.json").write_text(json.dumps(plan | {"resolved_inventory": inventory}, indent=2), encoding="utf-8")
    rf_path = _resolve(validated["generation"]["rf_profile"])
    rf = json.loads(rf_path.read_text(encoding="utf-8"))
    distributions = _dataset_distributions(dataset_root)
    models = validated["models"]
    condition_cache: dict[str, dict[str, Any]] = {}

    for trial_index, spec in enumerate(specs, start=1):
        condition, planner = spec["condition"], spec["planner"]
        scene_id = str(condition["scene"])
        size, num_tx, seed = int(condition["size_m"]), int(condition["num_tx"]), int(spec["seed"])
        trial_dir = output_root / planner["id"] / condition["city"] / f"{size}m_{scene_id}" / f"seed_{seed}"
        result_path = trial_dir / "result.json"
        failure_path = trial_dir / "failure.json"
        if result_path.exists():
            print(f"[{trial_index}/{len(specs)}] skip complete {planner['id']} {scene_id} seed={seed}")
            continue
        if failure_path.exists() and not args.retry_failed:
            print(f"[{trial_index}/{len(specs)}] skip failed {planner['id']} {scene_id} seed={seed}")
            continue
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            if scene_id not in condition_cache:
                condition_cache.clear()
                gc.collect()
                scene_dir = scene_root / scene_id
                scene_array, _ = load_scene_feature_tensor(scene_dir)
                with np.load(scene_dir / "inputs" / "scene_inputs.npz", allow_pickle=False) as loaded:
                    feasible = torch.from_numpy(np.asarray(loaded["tx_feasible_mask"], dtype=np.float32)).to(device)
                with np.load(scene_dir / "inputs" / "semantic_masks.npz", allow_pickle=False) as loaded:
                    road = np.asarray(loaded["road_mask"], dtype=bool)
                    building = np.asarray(loaded["building_mask"], dtype=bool)
                candidates = np.asarray(np.load(scene_dir / "inputs" / "tx_deployable" / "points_local_xy.npy", allow_pickle=False), dtype=np.float64)
                from dataset_builder.raytracing import SceneRayTracer
                condition_cache[scene_id] = {
                    "scene_dir": scene_dir,
                    "scene": torch.from_numpy(scene_array).unsqueeze(0).to(device),
                    "map_size": torch.tensor([[size, size]], dtype=torch.float32, device=device),
                    "feasible": feasible, "road": road, "building": building, "candidates": candidates,
                    "tracer": SceneRayTracer(scene_dir, rf, args.ray_device),
                }
            data = condition_cache[scene_id]
            model_kind, model, checkpoint_path, _ = models[planner["model"]]
            set_seed(seed)

            def reward_fn(points: torch.Tensor) -> torch.Tensor:
                batch, active_tx = points.shape[:2]
                mask = torch.ones((batch, active_tx), dtype=torch.bool, device=points.device)
                map_batch = data["map_size"].expand(batch, -1)
                prediction = predict_reward(model_kind, model, points, mask, data["scene"], map_batch)
                _, penalties = deployment_constraint_penalty(
                    points, data["feasible"], map_batch, float(plan["minimum_tx_distance_m"])
                )
                weights = plan["constraint_weights"]
                return prediction - float(weights["nondeployable"]) * penalties["nondeployable"] - float(weights["minimum_distance"]) * penalties["minimum_distance"]

            config = sampler_config_from_dict(planner["sampler"], dict(planner["config"]))
            started = time.perf_counter()
            sampled = run_sampler(
                reward_fn,
                sampler=planner["sampler"],
                num_particles=int(planner["num_particles"]),
                num_tx=num_tx,
                device=device,
                config=config,
            )
            inference_seconds = time.perf_counter() - started
            points_norm = sampled["best_tx_xy_norm01"].detach().cpu().numpy()
            rewards = sampled["best_reward"].detach().cpu().numpy()
            frame = SceneFrame2D(size, size)
            snapped, selected_index = _snap_candidate(
                points_norm, rewards, data["candidates"], frame, float(plan["minimum_tx_distance_m"])
            )
            snapped_norm = frame.real_to_norm01(snapped).astype(np.float32)
            tensor = torch.from_numpy(snapped_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                snapped_mask = torch.ones((1, num_tx), dtype=torch.bool, device=device)
                predicted = float(
                    predict_reward(
                        model_kind, model, tensor, snapped_mask, data["scene"], data["map_size"]
                    ).item()
                )
            tx_xyz = np.column_stack([snapped, np.full(num_tx, float(rf["tx_height_m"]))])
            raw_path_gain, ray_seconds = data["tracer"].trace(tx_xyz, (size, size), seed)
            path_gain = tx_axis_last(raw_path_gain, num_tx)
            metrics = compute_urban_radio_metrics(path_gain, num_tx, **_metric_kwargs(rf))
            road = _resize_mask_nearest(data["road"], path_gain.shape[:2])
            building = _resize_mask_nearest(data["building"], path_gain.shape[:2])
            summary, conditions = summarize_urban_metrics(metrics, road, rf["coverage_thresholds"])
            actual = float(summary["joint_4metric_coverage"])
            reference = distributions.get((scene_id, num_tx))
            percentile = None if reference is None else 100 * float(np.mean(reference <= actual))
            save_joint_coverage_plot(
                trial_dir / "joint_coverage.png", conditions["joint"], road, building, tx_xyz,
                (-size / 2, size / 2, -size / 2, size / 2), actual,
            )
            np.savez_compressed(
                trial_dir / "candidates.npz",
                tx_xy_norm01=points_norm.astype(np.float32),
                predicted_reward=rewards.astype(np.float32),
                selected_index=np.int64(selected_index),
            )
            (trial_dir / "trace.json").write_text(json.dumps(sampled["trace"], indent=2), encoding="utf-8")
            row = {
                "experiment_id": plan["experiment_id"], "planner_id": planner["id"],
                "planner_label": planner["label"], "sampler": planner["sampler"],
                "model_alias": planner["model"], "model_kind": model_kind,
                "city": condition["city"], "scene_id": scene_id, "split": "train",
                "size_m": size, "num_tx": num_tx, "repeat": spec["repeat"], "seed": seed,
                "predicted_reward": predicted, "joint_4metric_coverage": actual,
                "prediction_error": predicted - actual,
                "pathloss_coverage": float(summary["pathloss_coverage"]),
                "ss_rsrp_coverage": float(summary["ss_rsrp_coverage"]),
                "sinr_coverage": float(summary["sinr_coverage"]),
                "effective_throughput_coverage": float(summary["effective_throughput_coverage"]),
                "dataset_percentile": percentile,
                "inference_seconds": inference_seconds, "raytracing_seconds": ray_seconds,
                "num_reward_evaluations": int(sampled["num_reward_evaluations"]),
                "tx_positions_m": tx_xyz.tolist(), "checkpoint": str(checkpoint_path),
                "sampler_config": planner["config"], "joint_plot": str(trial_dir / "joint_coverage.png"),
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            failure_path.unlink(missing_ok=True)
            print(f"[{trial_index}/{len(specs)}] {planner['label']} {scene_id} ntx={num_tx} seed={seed} pred={predicted:.4f} actual={actual:.4f} evals={sampled['num_reward_evaluations']} time={inference_seconds:.1f}s")
        except Exception as error:
            failure = {"planner_id": planner["id"], "scene_id": scene_id, "seed": seed, "error": repr(error), "traceback": traceback.format_exc()}
            failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            print(f"[{trial_index}/{len(specs)}] FAILED {planner['id']} {scene_id} seed={seed}: {error}")
            if args.fail_fast:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_summary(output_root, expected_trials)


if __name__ == "__main__":
    main()

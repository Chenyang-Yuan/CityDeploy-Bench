#!/usr/bin/env python3
"""Run one reward-guided deployment method and verify its best plan with Sionna."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import pyarrow  # Load before torch on Windows.
import numpy as np
import torch

from dataset_builder.generate import _metric_kwargs, _resize_mask_nearest
from dataset_builder.metrics import compute_urban_radio_metrics, summarize_urban_metrics, tx_axis_last
from energy_model.dataset import DEFAULT_SCENE_ROOT, load_scene_feature_tensor, set_seed
from energy_model.guidance import assert_guidance_permutation_invariant, load_guidance_checkpoint, predict_reward
from energy_model.train import resolve_device
from samplers.benchmark_fixed_tx import _snap_candidate
from samplers.common import deployment_constraint_penalty
from samplers.runner import add_sampler_arguments, config_to_dict, run_sampler, sampler_config, sampler_config_from_dict
from scene_builder.coordinates import SceneFrame2D
from visualization.radio_maps import save_radio_map_outputs


PROJECT_ROOT = workspace_root()
DEFAULT_EVALUATION_DATASET = PROJECT_ROOT / "datasets" / "raytracing" / "urban_multicity"
REWARD_MODEL_LABELS = {
    "hypergraph_potential": "hpem",
    "reward_predictor": "reward_predictor",
    "interaction_network": "interaction_network",
    "nri": "nri",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one TX deployment method and verify the selected plan with Sionna RT"
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Run several reward checkpoints sequentially in the supplied order",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--num-tx", type=int, required=True)
    parser.add_argument("--num-particles", type=int, default=20)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET)
    parser.add_argument("--rf-profile", type=Path)
    add_sampler_arguments(parser)
    parser.add_argument("--sampler-config", type=Path, help="Exact sampler config JSON; overrides method hyperparameter flags")
    parser.add_argument("--run-dir", type=Path, help="Exact output directory (one model and one seed only)")
    parser.add_argument("--min-tx-distance-m", type=float, default=20.0)
    parser.add_argument("--nondeployable-weight", type=float, default=10.0)
    parser.add_argument("--minimum-distance-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Run this method repeatedly using seed, seed+step, ... (default: 1)",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment between consecutive seeds when --num-seeds is greater than one",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--ray-device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation"
    )
    return parser


def _resolve(path: Path) -> Path:
    return resolve_path(path)


def _load_rf_profile(args: argparse.Namespace) -> tuple[Path, dict]:
    if args.rf_profile is not None:
        path = _resolve(args.rf_profile)
    else:
        generation = json.loads((_resolve(args.dataset_root) / "generation_plan.json").read_text(encoding="utf-8"))
        path = _resolve(Path(generation["rf_profile"]))
    return path, json.loads(path.read_text(encoding="utf-8"))


def _reward_model_label(model_kind: str) -> str:
    """Return the stable, filesystem-safe reward-model name used in run folders."""
    return REWARD_MODEL_LABELS.get(model_kind, model_kind)


def _run_name_prefix(scene: str, num_tx: int, sampler: str, model_kind: str) -> str:
    return f"{scene}_ntx{num_tx}_{sampler}_{_reward_model_label(model_kind)}"


def _multi_seed_summary(
    args: argparse.Namespace,
    model_kind: str,
    checkpoint_path: Path,
    results: list[dict],
) -> dict:
    metric_keys = (
        "predicted_reward",
        "joint_4metric_coverage",
        "pathloss_coverage",
        "ss_rsrp_coverage",
        "sinr_coverage",
        "effective_throughput_coverage",
        "inference_seconds",
        "raytracing_seconds",
        "num_reward_evaluations",
    )
    statistics = {}
    for key in metric_keys:
        values = np.asarray([float(result[key]) for result in results], dtype=np.float64)
        statistics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "schema_version": "citydeploy.multi-seed-evaluation.v1",
        "scene_id": args.scene,
        "num_tx": args.num_tx,
        "sampler": args.sampler,
        "num_particles": args.num_particles,
        "model_kind": model_kind,
        "reward_model": _reward_model_label(model_kind),
        "checkpoint": str(checkpoint_path),
        "num_seeds": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "statistics": statistics,
        "runs": [
            {
                "seed": int(result["seed"]),
                "result_json": str(result["result_json"]),
                "joint_4metric_coverage": float(result["joint_4metric_coverage"]),
            }
            for result in results
        ],
    }


def _evaluate_checkpoint(args: argparse.Namespace, requested_checkpoint: Path) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = _resolve(requested_checkpoint)
    scene_dir = _resolve(args.scene_root) / args.scene
    output_root = _resolve(args.output_root)
    rf_path, rf = _load_rf_profile(args)

    model_kind, model, checkpoint = load_guidance_checkpoint(checkpoint_path, device)
    if args.num_tx > model.config.max_num_tx:
        raise ValueError(f"num_tx={args.num_tx} exceeds checkpoint max_num_tx={model.config.max_num_tx}")

    scene_array, metadata = load_scene_feature_tensor(scene_dir)
    scene = torch.from_numpy(scene_array).unsqueeze(0).to(device)
    frame = SceneFrame2D(float(metadata["map_x_m"]), float(metadata["map_y_m"]))
    map_size = torch.tensor([[frame.map_x_m, frame.map_y_m]], dtype=torch.float32, device=device)
    with np.load(scene_dir / "inputs" / "scene_inputs.npz", allow_pickle=False) as loaded:
        feasible = torch.from_numpy(np.asarray(loaded["tx_feasible_mask"], dtype=np.float32)).to(device)
    with np.load(scene_dir / "inputs" / "semantic_masks.npz", allow_pickle=False) as loaded:
        road = np.asarray(loaded["road_mask"], dtype=bool)
        building = np.asarray(loaded["building_mask"], dtype=bool)
    candidates = np.asarray(
        np.load(scene_dir / "inputs" / "tx_deployable" / "points_local_xy.npy", allow_pickle=False),
        dtype=np.float64,
    )

    invariant_points = torch.rand((1, args.num_tx, 2), device=device)
    invariant_mask = torch.ones((1, args.num_tx), dtype=torch.bool, device=device)
    assert_guidance_permutation_invariant(
        model_kind, model, invariant_points, invariant_mask, scene, map_size
    )

    def reward_fn(points: torch.Tensor) -> torch.Tensor:
        batch, active_tx = points.shape[:2]
        tx_mask = torch.ones((batch, active_tx), dtype=torch.bool, device=points.device)
        map_batch = map_size.expand(batch, -1)
        prediction = predict_reward(model_kind, model, points, tx_mask, scene, map_batch)
        _, penalties = deployment_constraint_penalty(
            points, feasible, map_batch, args.min_tx_distance_m
        )
        return (
            prediction
            - args.nondeployable_weight * penalties["nondeployable"]
            - args.minimum_distance_weight * penalties["minimum_distance"]
        )

    selected_config = (
        sampler_config_from_dict(args.sampler, json.loads(_resolve(args.sampler_config).read_text(encoding="utf-8")))
        if getattr(args, "sampler_config", None) is not None else sampler_config(args)
    )
    from dataset_builder.raytracing import SceneRayTracer

    tracer = SceneRayTracer(scene_dir, rf, args.ray_device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reward_model = _reward_model_label(model_kind)
    run_name_prefix = _run_name_prefix(args.scene, args.num_tx, args.sampler, model_kind)
    base_output = output_root / args.sampler / model_kind
    seeds = [args.seed + index * args.seed_step for index in range(args.num_seeds)]
    batch_dir = None
    if args.num_seeds > 1:
        batch_dir = base_output / (
            f"{run_name_prefix}_seedset{args.seed}_n{args.num_seeds}_{timestamp}"
        )
        batch_dir.mkdir(parents=True, exist_ok=False)

    results = []
    for run_index, current_seed in enumerate(seeds, start=1):
        set_seed(current_seed)
        print(f"\n=== Seed {current_seed} ({run_index}/{len(seeds)}) ===")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        sampled = run_sampler(
            reward_fn,
            sampler=args.sampler,
            num_particles=args.num_particles,
            num_tx=args.num_tx,
            device=device,
            config=selected_config,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - started
        sampled_points = sampled["best_tx_xy_norm01"].detach().cpu().numpy()
        sampled_rewards = sampled["best_reward"].detach().cpu().numpy()
        snapped, selected_index = _snap_candidate(
            sampled_points, sampled_rewards, candidates, frame, args.min_tx_distance_m
        )
        snapped_norm = frame.real_to_norm01(snapped).astype(np.float32)
        snapped_tensor = torch.from_numpy(snapped_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            snapped_mask = torch.ones((1, args.num_tx), dtype=torch.bool, device=device)
            predicted_reward = float(
                predict_reward(
                    model_kind, model, snapped_tensor, snapped_mask, scene, map_size
                ).item()
            )

        tx_xyz = np.column_stack(
            [snapped, np.full(args.num_tx, float(rf["tx_height_m"]))]
        )
        raw_path_gain, raytracing_seconds = tracer.trace(
            tx_xyz, (frame.map_x_m, frame.map_y_m), current_seed
        )
        path_gain = tx_axis_last(raw_path_gain, args.num_tx)
        metrics = compute_urban_radio_metrics(
            path_gain, args.num_tx, **_metric_kwargs(rf)
        )
        resized_road = _resize_mask_nearest(road, path_gain.shape[:2])
        resized_building = _resize_mask_nearest(building, path_gain.shape[:2])
        summary, coverage_conditions = summarize_urban_metrics(
            metrics, resized_road, rf["coverage_thresholds"]
        )

        if getattr(args, "run_dir", None) is not None:
            output_dir = _resolve(args.run_dir)
        elif batch_dir is None:
            output_dir = base_output / (
                f"{run_name_prefix}_seed{current_seed}_{timestamp}"
            )
        else:
            output_dir = batch_dir / f"{run_name_prefix}_seed{current_seed}"
        output_dir.mkdir(parents=True, exist_ok=False)
        radio_map_outputs = save_radio_map_outputs(
            output_dir,
            metrics,
            coverage_conditions,
            resized_road,
            resized_building,
            tx_xyz,
            frame.map_x_m,
            frame.map_y_m,
        )
        np.savez_compressed(
            output_dir / "candidates.npz",
            tx_xy_norm01=sampled_points.astype(np.float32),
            predicted_reward=sampled_rewards.astype(np.float32),
            selected_index=np.int64(selected_index),
        )
        (output_dir / "trace.json").write_text(
            json.dumps(sampled["trace"], indent=2), encoding="utf-8"
        )
        result_path = output_dir / "result.json"
        result = {
            "scene_id": args.scene,
            "num_tx": args.num_tx,
            "seed": current_seed,
            "ray_seed": current_seed,
            "sampler": args.sampler,
            "num_particles": args.num_particles,
            "model_kind": model_kind,
            "reward_model": reward_model,
            "model_schema_version": checkpoint.get("model_schema_version"),
            "checkpoint": str(checkpoint_path),
            "rf_profile": str(rf_path),
            "rf_profile_snapshot": rf,
            "raytracing_runtime": tracer.runtime_info,
            "sampler_config": config_to_dict(selected_config),
            "predicted_reward": predicted_reward,
            "joint_4metric_coverage": float(summary["joint_4metric_coverage"]),
            "pathloss_coverage": float(summary["pathloss_coverage"]),
            "ss_rsrp_coverage": float(summary["ss_rsrp_coverage"]),
            "sinr_coverage": float(summary["sinr_coverage"]),
            "effective_throughput_coverage": float(
                summary["effective_throughput_coverage"]
            ),
            "inference_seconds": inference_seconds,
            "raytracing_seconds": raytracing_seconds,
            "num_reward_evaluations": int(sampled["num_reward_evaluations"]),
            "num_final_reward_evaluations": 1,
            "num_invariance_check_evaluations": 2 if run_index == 1 else 0,
            "total_model_evaluations": int(sampled["num_reward_evaluations"]) + 1 + (2 if run_index == 1 else 0),
            "tx_positions_m": tx_xyz.tolist(),
            "tx_xy_norm01": snapped_norm.tolist(),
            "map_size_m": [frame.map_x_m, frame.map_y_m],
            "radio_maps": radio_map_outputs,
            "joint_plot": radio_map_outputs["joint"],
            "result_json": str(result_path),
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps(result, indent=2))
        print(f"Saved seed {current_seed} evaluation to {output_dir}")

    aggregate = None
    summary_path = None
    if batch_dir is not None:
        aggregate = _multi_seed_summary(
            args, model_kind, checkpoint_path, results
        )
        summary_path = batch_dir / "summary.json"
        summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print("\n=== Multi-seed summary ===")
        print(json.dumps(aggregate, indent=2))
        print(f"Saved multi-seed evaluation to {batch_dir}")
    return {
        "model_kind": model_kind,
        "reward_model": reward_model,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(batch_dir or Path(results[0]["result_json"]).parent),
        "summary_json": str(summary_path) if summary_path is not None else None,
        "result_json": results[0]["result_json"] if batch_dir is None else None,
        "statistics": aggregate["statistics"] if aggregate is not None else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.seed_step <= 0:
        raise ValueError("--seed-step must be positive")
    requested = [args.checkpoint] if args.checkpoint is not None else list(args.checkpoints)
    if args.run_dir is not None and (len(requested) != 1 or args.num_seeds != 1):
        raise ValueError("--run-dir requires exactly one checkpoint and one seed")
    resolved = [_resolve(path) for path in requested]
    if len({str(path).lower() for path in resolved}) != len(resolved):
        raise ValueError("checkpoint list contains duplicates")

    outcomes = []
    for index, checkpoint_path in enumerate(resolved, start=1):
        print(f"\n######## Reward model {index}/{len(resolved)}: {checkpoint_path} ########")
        outcomes.append(_evaluate_checkpoint(args, checkpoint_path))

    if len(outcomes) > 1:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = _resolve(args.output_root)
        sweep_dir = output_root / "model_sweeps" / (
            f"{args.scene}_ntx{args.num_tx}_{args.sampler}_model_sweep{len(outcomes)}_"
            f"seedset{args.seed}_n{args.num_seeds}_{timestamp}"
        )
        sweep_dir.mkdir(parents=True, exist_ok=False)
        sweep = {
            "schema_version": "citydeploy.reward-model-sweep.v1",
            "scene_id": args.scene,
            "num_tx": args.num_tx,
            "sampler": args.sampler,
            "num_particles": args.num_particles,
            "num_seeds_per_model": args.num_seeds,
            "seeds": [
                args.seed + seed_index * args.seed_step
                for seed_index in range(args.num_seeds)
            ],
            "model_order": [outcome["reward_model"] for outcome in outcomes],
            "models": outcomes,
        }
        sweep_path = sweep_dir / "summary.json"
        sweep_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        print("\n######## Reward-model sweep complete ########")
        print(json.dumps(sweep, indent=2))
        print(f"Saved reward-model sweep index to {sweep_path}")


if __name__ == "__main__":
    main()

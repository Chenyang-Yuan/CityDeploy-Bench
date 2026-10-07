#!/usr/bin/env python3
"""Generate multi-TX plans with either learned model and any registered sampler."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import numpy as np
import torch

from energy_model.dataset import DEFAULT_SCENE_ROOT, load_scene_feature_tensor, set_seed
from energy_model.guidance import (
    assert_guidance_permutation_invariant,
    load_guidance_checkpoint,
    predict_reward,
)
from energy_model.train import resolve_device
from samplers.common import deployment_constraint_penalty
from samplers.runner import add_sampler_arguments, config_to_dict, run_sampler, sampler_config
from scene_builder.coordinates import (
    COORDINATE_CONVENTION,
    COORDINATE_SCHEMA_VERSION,
    SceneFrame2D,
)


PROJECT_ROOT = workspace_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward-guided TX deployment samplers")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--num-tx", type=int, required=True)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--num-particles", type=int, default=30)
    add_sampler_arguments(parser)
    parser.add_argument("--min-tx-distance-m", type=float, default=35.0)
    parser.add_argument("--nondeployable-weight", type=float, default=10.0)
    parser.add_argument("--minimum-distance-weight", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "planning")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_kind, model, checkpoint = load_guidance_checkpoint(args.checkpoint, device)
    model_schema_version = checkpoint.get("model_schema_version")
    model_config = model.config
    if args.num_tx > model_config.max_num_tx:
        raise ValueError(f"num_tx={args.num_tx} exceeds trained max_num_tx={model_config.max_num_tx}")

    scene_dir = args.scene_root / args.scene
    scene_array, metadata = load_scene_feature_tensor(scene_dir)
    scene = torch.from_numpy(scene_array).unsqueeze(0).to(device)
    frame = SceneFrame2D(float(metadata["map_x_m"]), float(metadata["map_y_m"]))
    map_size = torch.tensor([[frame.map_x_m, frame.map_y_m]], dtype=torch.float32, device=device)
    with np.load(scene_dir / "inputs" / "scene_inputs.npz", allow_pickle=False) as scene_inputs:
        feasible_mask = torch.from_numpy(np.asarray(scene_inputs["tx_feasible_mask"], dtype=np.float32)).to(device)

    tx_mask_check = torch.ones((1, args.num_tx), dtype=torch.bool, device=device)
    invariance_points = torch.rand((1, args.num_tx, 2), device=device)
    assert_guidance_permutation_invariant(
        model_kind, model, invariance_points, tx_mask_check, scene, map_size
    )

    def reward_fn(points: torch.Tensor) -> torch.Tensor:
        batch = points.shape[0]
        active_num_tx = points.shape[1]
        tx_mask = torch.ones((batch, active_num_tx), dtype=torch.bool, device=points.device)
        map_batch = map_size.expand(batch, -1)
        predicted_reward = predict_reward(model_kind, model, points, tx_mask, scene, map_batch)
        _constraint, components = deployment_constraint_penalty(
            points,
            feasible_mask,
            map_batch,
            args.min_tx_distance_m,
        )
        return (
            predicted_reward
            - args.nondeployable_weight * components["nondeployable"]
            - args.minimum_distance_weight * components["minimum_distance"]
        )

    selected_config = sampler_config(args)
    result = run_sampler(
        reward_fn,
        sampler=args.sampler,
        num_particles=args.num_particles,
        num_tx=args.num_tx,
        device=device,
        config=selected_config,
    )
    reward = result["best_reward"].detach().cpu().numpy()
    points_norm = result["best_tx_xy_norm01"].detach().cpu().numpy()
    order = np.argsort(-reward)
    reward = reward[order]
    points_norm = points_norm[order]
    # A deterministic within-set order makes files and diffs reproducible while
    # preserving the unordered-set semantics used by the model.
    for candidate in range(points_norm.shape[0]):
        canonical = np.lexsort((points_norm[candidate, :, 1], points_norm[candidate, :, 0]))
        points_norm[candidate] = points_norm[candidate, canonical]
    points_real = frame.norm01_to_real(points_norm)

    output_dir = (
        args.output_root
        / args.sampler
        / model_kind
        / f"{args.scene}_ntx{args.num_tx}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "candidates.npz",
        tx_xy_norm01=points_norm.astype(np.float32),
        tx_xy_real_m=points_real.astype(np.float32),
        reward=reward.astype(np.float32),
    )
    top_k = min(max(1, args.top_k), len(reward))
    with (output_dir / "topk.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank", "reward"]
        for index in range(args.num_tx):
            fields.extend(
                [
                    f"tx{index + 1}_x_norm",
                    f"tx{index + 1}_y_norm",
                    f"tx{index + 1}_x_real",
                    f"tx{index + 1}_y_real",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank in range(top_k):
            row: dict[str, float | int] = {"rank": rank + 1, "reward": float(reward[rank])}
            for index in range(args.num_tx):
                row[f"tx{index + 1}_x_norm"] = float(points_norm[rank, index, 0])
                row[f"tx{index + 1}_y_norm"] = float(points_norm[rank, index, 1])
                row[f"tx{index + 1}_x_real"] = float(points_real[rank, index, 0])
                row[f"tx{index + 1}_y_real"] = float(points_real[rank, index, 1])
            writer.writerow(row)
    (output_dir / "trace.json").write_text(json.dumps(result["trace"], indent=2), encoding="utf-8")
    run_config = {
        "model_schema_version": model_schema_version,
        "model_kind": model_kind,
        "checkpoint": str(args.checkpoint.resolve()),
        "scene": args.scene,
        "num_tx": args.num_tx,
        "map_size_m": [frame.map_x_m, frame.map_y_m],
        "coordinates": {
            "schema_version": COORDINATE_SCHEMA_VERSION,
            "convention": COORDINATE_CONVENTION,
            "model_domain": "[0,1]^2",
        },
        "sampler": args.sampler,
        "sampler_config": config_to_dict(selected_config),
        "nondeployable_weight": args.nondeployable_weight,
        "minimum_distance_weight": args.minimum_distance_weight,
        "min_tx_distance_m": args.min_tx_distance_m,
        "device": device.type,
        "num_reward_evaluations": result.get("num_reward_evaluations"),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Saved {len(reward)} candidates to {output_dir}")


if __name__ == "__main__":
    main()

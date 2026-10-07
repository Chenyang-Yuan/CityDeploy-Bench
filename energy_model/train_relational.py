#!/usr/bin/env python3
"""Train Interaction-Network or NRI-style deployment reward baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import pyarrow  # Load before torch on Windows.
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from energy_model.data import SceneCardinalityBatchSampler, indices_for_split
from energy_model.dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SCENE_ROOT,
    SCENE_FEATURE_CHANNELS,
    TxDeploymentDataset,
    collate_deployments,
    set_seed,
)
from energy_model.relational_models import (
    INTERACTION_NETWORK_SCHEMA,
    NRI_SCHEMA,
    InteractionNetworkReward,
    NRIRelationalReward,
    RelationalRewardConfig,
)
from energy_model.train import resolve_device
from energy_model.train_reward import regression_metrics


PROJECT_ROOT = workspace_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a relational TX deployment reward model")
    parser.add_argument("--model", choices=["interaction_network", "nri"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-num-tx", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-types", type=int, default=4)
    parser.add_argument("--gumbel-temperature", type=float, default=0.5)
    parser.add_argument("--nri-kl-weight", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training" / "relational")
    return parser


def run_epoch(model, loader, scene_bank, map_size_bank, device, optimizer, gradient_clip, kl_weight):
    training = optimizer is not None
    model.train(training)
    targets, predictions = [], []
    loss_sum = 0.0
    samples = 0
    context_manager = torch.enable_grad() if training else torch.no_grad()
    with context_manager:
        for batch in loader:
            if not bool((batch["scene_index"] == batch["scene_index"][0]).all()):
                raise RuntimeError("each batch must contain one shared scene")
            batch = {key: value.to(device) for key, value in batch.items()}
            scene_id = int(batch["scene_index"][0].item())
            scene = scene_bank[scene_id].unsqueeze(0)
            map_size = map_size_bank[scene_id : scene_id + 1].expand(batch["tx_xy_norm01"].shape[0], -1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if isinstance(model, NRIRelationalReward):
                prediction, auxiliary = model.forward_with_aux(
                    batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size
                )
                loss = F.mse_loss(prediction, batch["joint"]) + kl_weight * auxiliary["edge_kl"].mean()
            else:
                prediction = model(batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size)
                loss = F.mse_loss(prediction, batch["joint"])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            count = prediction.numel()
            loss_sum += float(loss.detach()) * count
            samples += count
            targets.append(batch["joint"].detach().cpu().numpy())
            predictions.append(prediction.detach().cpu().numpy())
    metrics = regression_metrics(np.concatenate(targets), np.concatenate(predictions))
    metrics["objective"] = loss_sum / max(samples, 1)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = TxDeploymentDataset(
        args.dataset_root,
        args.scene_root,
        max_samples=args.max_samples or None,
        permute_tx_order=True,
        expand_family_memberships=False,
    )
    split_indices = {name: indices_for_split(dataset, name) for name in ("train", "validation", "test")}
    if any(not values for values in split_indices.values()):
        raise RuntimeError("explicit non-empty train, validation, and test splits are required")
    batch_samplers = {
        name: SceneCardinalityBatchSampler(
            dataset, values, args.batch_size, shuffle=name == "train", seed=args.seed
        )
        for name, values in split_indices.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate_deployments)
        for name, sampler in batch_samplers.items()
    }
    scene_bank = [dataset.scene_tensor_map[name].to(device) for name in dataset.scene_names]
    map_size_bank = torch.tensor(
        [[dataset.scene_meta_map[name]["map_x_m"], dataset.scene_meta_map[name]["map_y_m"]] for name in dataset.scene_names],
        dtype=torch.float32,
        device=device,
    )
    config = RelationalRewardConfig(
        max_num_tx=args.max_num_tx,
        dropout=args.dropout,
        edge_types=args.edge_types,
        gumbel_temperature=args.gumbel_temperature,
    )
    if args.model == "interaction_network":
        model = InteractionNetworkReward(config).to(device)
        schema = INTERACTION_NETWORK_SCHEMA
    else:
        model = NRIRelationalReward(config).to(device)
        schema = NRI_SCHEMA
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6
    )
    run_dir = args.save_dir / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    serializable_args = vars(args) | {
        "dataset_root": str(args.dataset_root), "scene_root": str(args.scene_root), "save_dir": str(args.save_dir)
    }
    (run_dir / "config.json").write_text(json.dumps({
        "model_schema_version": schema,
        "model_kind": args.model,
        "model": config.to_dict(),
        "target": "joint_4metric_coverage",
        "adaptation": "static_scene_conditioned_relational_reward",
        "scene_feature_channels": list(SCENE_FEATURE_CHANNELS),
        "split_counts": {name: len(values) for name, values in split_indices.items()},
        "training": serializable_args,
    }, indent=2), encoding="utf-8")
    fields = ["epoch", "train_objective", "train_mse", "validation_objective", "validation_mse", "validation_mae", "validation_spearman", "learning_rate"]
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            batch_samplers["train"].set_epoch(epoch)
            train = run_epoch(model, loaders["train"], scene_bank, map_size_bank, device, optimizer, args.gradient_clip, args.nri_kl_weight)
            validation = run_epoch(model, loaders["validation"], scene_bank, map_size_bank, device, None, args.gradient_clip, args.nri_kl_weight)
            scheduler.step(validation["mse"])
            writer.writerow({
                "epoch": epoch,
                "train_objective": train["objective"], "train_mse": train["mse"],
                "validation_objective": validation["objective"], "validation_mse": validation["mse"],
                "validation_mae": validation["mae"], "validation_spearman": validation["spearman"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            handle.flush()
            checkpoint = {"model_schema_version": schema, "model_config": config.to_dict(), "model_state": model.state_dict(), "epoch": epoch, "validation_metrics": validation}
            torch.save(checkpoint, run_dir / "checkpoint_last.pt")
            if validation["mse"] < best_validation - args.min_delta:
                best_validation, best_epoch, stale = validation["mse"], epoch, 0
                torch.save(checkpoint, run_dir / "checkpoint_best.pt")
            else:
                stale += 1
            print(f"epoch={epoch:03d} train_mse={train['mse']:.6f} val_mse={validation['mse']:.6f} val_mae={validation['mae']:.4f} val_spearman={validation['spearman']:.4f}")
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break
    best = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    test = run_epoch(model, loaders["test"], scene_bank, map_size_bank, device, None, args.gradient_clip, args.nri_kl_weight)
    report = {"best_epoch": int(best["epoch"]), "validation": best["validation_metrics"], "test": test}
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved relational reward run: {run_dir}")


if __name__ == "__main__":
    main()

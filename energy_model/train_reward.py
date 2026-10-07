#!/usr/bin/env python3
"""Train and evaluate the standalone scene-conditioned reward predictor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import pyarrow  # Load before PyTorch on Windows.
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
from energy_model.reward_model import (
    REWARD_MODEL_SCHEMA_VERSION,
    RewardPredictor,
    RewardPredictorConfig,
    assert_reward_permutation_invariant,
)
from energy_model.train import resolve_device


PROJECT_ROOT = workspace_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the standalone coverage reward predictor")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-num-tx", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training" / "regressor"
    )
    return parser


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    mse = float(np.mean(error**2))
    variance = float(np.sum((target - target.mean()) ** 2))
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if len(target) > 1 else float("nan")
    target_rank, prediction_rank = _rankdata(target), _rankdata(prediction)
    spearman = (
        float(np.corrcoef(target_rank, prediction_rank)[0, 1])
        if len(target) > 1
        else float("nan")
    )
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - float(np.sum(error**2)) / max(variance, 1e-12),
        "pearson": pearson,
        "spearman": spearman,
        "bias": float(np.mean(error)),
        "num_samples": int(len(target)),
    }


def run_epoch(
    model: RewardPredictor,
    loader: DataLoader,
    dataset: TxDeploymentDataset,
    scene_bank: list[torch.Tensor],
    map_size_bank: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            scene_index = batch["scene_index"]
            if not bool((scene_index == scene_index[0]).all()):
                raise RuntimeError("every reward batch must contain one shared scene")
            batch = {key: value.to(device) for key, value in batch.items()}
            scene_id = int(batch["scene_index"][0].item())
            scene = scene_bank[scene_id].unsqueeze(0)
            map_size = map_size_bank[scene_id : scene_id + 1].expand(
                batch["tx_xy_norm01"].shape[0], -1
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size)
            loss = F.mse_loss(prediction, batch["joint"])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            targets.append(batch["joint"].detach().cpu().numpy())
            predictions.append(prediction.detach().cpu().numpy())
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    metrics = regression_metrics(target, prediction)
    return metrics["mse"], metrics


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
    train_indices = indices_for_split(dataset, "train")
    validation_indices = indices_for_split(dataset, "validation")
    test_indices = indices_for_split(dataset, "test")
    if not train_indices or not validation_indices or not test_indices:
        raise RuntimeError("explicit train, validation, and test splits are required")
    samplers = {
        "train": SceneCardinalityBatchSampler(
            dataset, train_indices, args.batch_size, shuffle=True, seed=args.seed
        ),
        "validation": SceneCardinalityBatchSampler(
            dataset, validation_indices, args.batch_size, shuffle=False, seed=args.seed
        ),
        "test": SceneCardinalityBatchSampler(
            dataset, test_indices, args.batch_size, shuffle=False, seed=args.seed
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_deployments,
        )
        for name, sampler in samplers.items()
    }
    scene_bank = [dataset.scene_tensor_map[name].to(device) for name in dataset.scene_names]
    map_size_bank = torch.tensor(
        [
            [dataset.scene_meta_map[name]["map_x_m"], dataset.scene_meta_map[name]["map_y_m"]]
            for name in dataset.scene_names
        ],
        dtype=torch.float32,
        device=device,
    )
    config = RewardPredictorConfig(
        max_num_tx=args.max_num_tx,
        dropout=args.dropout,
    )
    model = RewardPredictor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6
    )

    first = dataset[train_indices[0]]
    count = int(first["num_tx"])
    tx = first["tx_xy_norm01"].unsqueeze(0).to(device)
    mask = torch.ones((1, count), dtype=torch.bool, device=device)
    scene_id = int(first["scene_index"])
    assert_reward_permutation_invariant(
        model,
        tx,
        mask,
        scene_bank[scene_id].unsqueeze(0),
        map_size_bank[scene_id : scene_id + 1],
    )

    run_dir = args.save_dir / f"regressor_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "model_schema_version": REWARD_MODEL_SCHEMA_VERSION,
        "model": config.to_dict(),
        "target": "joint_4metric_coverage",
        "sample_unit": "unique_physical_deployment",
        "scene_feature_channels": list(SCENE_FEATURE_CHANNELS),
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "training": vars(args)
        | {
            "dataset_root": str(args.dataset_root),
            "scene_root": str(args.scene_root),
            "save_dir": str(args.save_dir),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    history_path = run_dir / "history.csv"
    best_validation = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_mse",
                "validation_mse",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
                "validation_spearman",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            samplers["train"].set_epoch(epoch)
            train_mse, _ = run_epoch(
                model,
                loaders["train"],
                dataset,
                scene_bank,
                map_size_bank,
                device,
                optimizer,
                args.gradient_clip,
            )
            validation_mse, validation_metrics = run_epoch(
                model,
                loaders["validation"],
                dataset,
                scene_bank,
                map_size_bank,
                device,
                None,
                args.gradient_clip,
            )
            scheduler.step(validation_mse)
            row = {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "validation_spearman": validation_metrics["spearman"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            checkpoint = {
                "model_schema_version": REWARD_MODEL_SCHEMA_VERSION,
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
            }
            torch.save(checkpoint, run_dir / "checkpoint_last.pt")
            if validation_mse < best_validation - args.min_delta:
                best_validation = validation_mse
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(checkpoint, run_dir / "checkpoint_best.pt")
            else:
                epochs_without_improvement += 1
            print(
                f"epoch={epoch:03d} train_mse={train_mse:.6f} "
                f"val_mse={validation_mse:.6f} val_mae={validation_metrics['mae']:.4f} "
                f"val_spearman={validation_metrics['spearman']:.4f}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    best = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    _, test_metrics = run_epoch(
        model,
        loaders["test"],
        dataset,
        scene_bank,
        map_size_bank,
        device,
        None,
        args.gradient_clip,
    )
    final_report = {
        "best_epoch": int(best["epoch"]),
        "validation": best["validation_metrics"],
        "test": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))
    print(f"Saved reward predictor run: {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the unified V3/V4 hypergraph potential model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import pyarrow  # Load Arrow's OpenMP runtime before PyTorch on Windows.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from energy_model.dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SCENE_ROOT,
    SCENE_FEATURE_CHANNELS,
    TxDeploymentDataset,
    collate_deployments,
    grouped_ranking_loss,
    set_seed,
)
from energy_model.data import (
    SceneSplit,
    SceneCardinalityBatchSampler,
    indices_for_scenes,
    indices_for_split,
    make_scene_split,
)
from energy_model.model import (
    MODEL_SCHEMA_VERSION,
    HypergraphPotentialConfig,
    HypergraphPotentialModel,
)
from scene_builder.coordinates import COORDINATE_CONVENTION, COORDINATE_SCHEMA_VERSION


PROJECT_ROOT = workspace_root()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_orders(raw: str) -> tuple[int, ...]:
    return tuple(sorted(set(int(value.strip()) for value in raw.split(",") if value.strip())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the unified order-1..4 hypergraph potential model")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--orders", default="1,2,3,4", help="Enabled hyperedge orders, e.g. 1,2,3,4 or 1,2")
    parser.add_argument("--max-num-tx", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-ranking", type=float, default=0.25)
    parser.add_argument("--w-aux", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--ranking-epsilon", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training" / "hpem")
    return parser


def _loss(
    output: dict,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    energy = output["energy"]
    auxiliary = output["aux"]
    joint = batch["joint"]
    energy_loss = F.mse_loss(-energy, joint)
    ranking_loss = grouped_ranking_loss(
        energy,
        joint,
        args.ranking_margin,
        args.ranking_epsilon,
    )
    auxiliary_loss = torch.stack(
        [
            F.mse_loss(auxiliary["joint"], batch["joint"]),
            F.mse_loss(auxiliary["pathloss"], batch["pathloss"]),
            F.mse_loss(auxiliary["sinr"], batch["sinr"]),
            F.mse_loss(auxiliary["throughput"], batch["throughput"]),
            F.mse_loss(auxiliary["rsrp"], batch["rsrp"]),
        ]
    ).mean()
    total = args.w_energy * energy_loss + args.w_ranking * ranking_loss + args.w_aux * auxiliary_loss
    return total, {"energy": energy_loss, "ranking": ranking_loss, "auxiliary": auxiliary_loss}


def run_epoch(
    model: HypergraphPotentialModel,
    loader: DataLoader,
    dataset: TxDeploymentDataset,
    scene_bank: list[torch.Tensor],
    map_size_bank: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "energy": 0.0, "ranking": 0.0, "auxiliary": 0.0, "samples": 0.0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            scene_index = batch["scene_index"]
            if not bool((scene_index == scene_index[0]).all()):
                raise RuntimeError("V3 loader must group every batch by scene")
            batch = {key: value.to(device) for key, value in batch.items()}
            scene_id = int(batch["scene_index"][0].item())
            scene = scene_bank[scene_id].unsqueeze(0)
            map_size = map_size_bank[scene_id : scene_id + 1].expand(batch["tx_xy_norm01"].shape[0], -1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size)
            loss, parts = _loss(output, batch, args)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
            count = float(batch["tx_xy_norm01"].shape[0])
            totals["loss"] += float(loss.item()) * count
            for name, value in parts.items():
                totals[name] += float(value.item()) * count
            totals["samples"] += count
    denominator = max(1.0, totals["samples"])
    return {name: value / denominator for name, value in totals.items() if name != "samples"}


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    orders = parse_orders(args.orders)
    dataset = TxDeploymentDataset(
        args.dataset_root,
        args.scene_root,
        max_samples=args.max_samples or None,
        permute_tx_order=True,
    )
    if dataset.has_explicit_splits:
        train_indices = indices_for_split(dataset, "train")
        validation_indices = indices_for_split(dataset, "validation")
        test_indices = indices_for_split(dataset, "test")
        scene_split = SceneSplit(
            tuple(sorted({dataset.records[index].scene for index in train_indices})),
            tuple(sorted({dataset.records[index].scene for index in validation_indices})),
            tuple(sorted({dataset.records[index].scene for index in test_indices})),
        )
    else:
        scene_split = make_scene_split(
            dataset.scene_names,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_indices = indices_for_scenes(dataset, scene_split.train_scenes)
        validation_indices = indices_for_scenes(dataset, scene_split.validation_scenes)
    if not train_indices or not validation_indices:
        raise RuntimeError("scene split produced an empty train or validation set")

    train_sampler = SceneCardinalityBatchSampler(
        dataset, train_indices, args.batch_size, shuffle=True, seed=args.seed
    )
    validation_sampler = SceneCardinalityBatchSampler(
        dataset, validation_indices, args.batch_size, shuffle=False, seed=args.seed
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_deployments,
    )
    validation_loader = DataLoader(
        dataset,
        batch_sampler=validation_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_deployments,
    )

    scene_bank = [dataset.scene_tensor_map[name].to(device) for name in dataset.scene_names]
    map_size_bank = torch.tensor(
        [[dataset.scene_meta_map[name]["map_x_m"], dataset.scene_meta_map[name]["map_y_m"]] for name in dataset.scene_names],
        dtype=torch.float32,
        device=device,
    )
    config = HypergraphPotentialConfig(orders=orders, max_num_tx=args.max_num_tx)
    model = HypergraphPotentialModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_id = time.strftime("hpem_%Y%m%d_%H%M%S")
    run_dir = args.save_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model": config.to_dict(),
        "coordinates": {
            "schema_version": COORDINATE_SCHEMA_VERSION,
            "convention": COORDINATE_CONVENTION,
        },
        "scene_feature_channels": list(SCENE_FEATURE_CHANNELS),
        "scene_shapes_hw": {
            name: list(dataset.scene_tensor_map[name].shape[-2:]) for name in dataset.scene_names
        },
        "scene_split": scene_split.to_dict(),
        "training": vars(args) | {"dataset_root": str(args.dataset_root), "scene_root": str(args.scene_root), "save_dir": str(args.save_dir)},
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "scene_split.json").write_text(json.dumps(scene_split.to_dict(), indent=2), encoding="utf-8")

    history_path = run_dir / "history.csv"
    best_validation = float("inf")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_loss", "train_energy", "train_ranking", "train_auxiliary", "learning_rate"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(model, train_loader, dataset, scene_bank, map_size_bank, device, args, optimizer)
            validation_metrics = run_epoch(
                model, validation_loader, dataset, scene_bank, map_size_bank, device, args, None
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_energy": train_metrics["energy"],
                "train_ranking": train_metrics["ranking"],
                "train_auxiliary": train_metrics["auxiliary"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            checkpoint = {
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_metrics["loss"],
                "scene_split": scene_split.to_dict(),
            }
            torch.save(checkpoint, run_dir / "checkpoint_last.pt")
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                torch.save(checkpoint, run_dir / "checkpoint_best.pt")
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                f"validation={validation_metrics['loss']:.6f} device={device.type}"
            )
    print(f"Saved HPEM run: {run_dir}")


if __name__ == "__main__":
    main()

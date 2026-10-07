#!/usr/bin/env python3
"""Continuously adapt reward models on fixed target-scene deployment data.

The 30-percent checkpoint and the full-data checkpoint are saved without
reinitializing either the model or optimizer. A disjoint target-scene subset is
reserved for comparing zero-shot, 30-percent adaptation, and full adaptation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import numpy as np
import pyarrow  # Load Arrow's OpenMP runtime before PyTorch on Windows.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from energy_model.data import SceneCardinalityBatchSampler
from energy_model.dataset import (
    TxDeploymentDataset,
    collate_deployments,
    grouped_ranking_loss,
    set_seed,
)
from energy_model.guidance import load_guidance_checkpoint
from energy_model.relational_models import NRIRelationalReward
from energy_model.train import resolve_device
from energy_model.train_reward import regression_metrics


PROJECT_ROOT = workspace_root()
DEFAULT_PLAN = config_path("energy_model/reward_adaptation_three_targets.json")
MODEL_KINDS = ("hypergraph_potential", "reward_predictor", "interaction_network", "nri")


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return resolve_path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_key(record) -> str:
    return f"{record.path.name}:{record.row_index}:{record.family_id}"


def _make_loader(dataset, indices, batch_size, shuffle, seed, num_workers):
    sampler = SceneCardinalityBatchSampler(
        dataset, indices, batch_size, shuffle=shuffle, seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_deployments,
    )
    return loader, sampler


def _forward_loss(model_kind, model, batch, scene, map_size, config):
    if model_kind == "hypergraph_potential":
        output = model(batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size)
        prediction = output["reward"]
        energy_loss = F.mse_loss(prediction, batch["joint"])
        ranking_loss = grouped_ranking_loss(
            output["energy"], batch["joint"], config["ranking_margin"], config["ranking_epsilon"]
        )
        auxiliary_loss = torch.stack(
            [
                F.mse_loss(output["aux"]["joint"], batch["joint"]),
                F.mse_loss(output["aux"]["pathloss"], batch["pathloss"]),
                F.mse_loss(output["aux"]["sinr"], batch["sinr"]),
                F.mse_loss(output["aux"]["throughput"], batch["throughput"]),
                F.mse_loss(output["aux"]["rsrp"], batch["rsrp"]),
            ]
        ).mean()
        weights = config["hpem_loss_weights"]
        loss = (
            weights["energy"] * energy_loss
            + weights["ranking"] * ranking_loss
            + weights["auxiliary"] * auxiliary_loss
        )
        parts = {
            "energy_loss": float(energy_loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
            "auxiliary_loss": float(auxiliary_loss.detach()),
        }
        return prediction, loss, parts
    if model_kind == "nri" and isinstance(model, NRIRelationalReward):
        prediction, auxiliary = model.forward_with_aux(
            batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size
        )
        kl = auxiliary["edge_kl"].mean()
        loss = F.mse_loss(prediction, batch["joint"]) + config["nri_kl_weight"] * kl
        return prediction, loss, {"edge_kl": float(kl.detach())}
    prediction = model(batch["tx_xy_norm01"], batch["tx_mask"], scene, map_size)
    loss = F.mse_loss(prediction, batch["joint"])
    return prediction, loss, {}


def _run_epoch(model_kind, model, loader, sampler, dataset, device, config, optimizer, epoch):
    training = optimizer is not None
    model.train(training)
    if training:
        sampler.set_epoch(epoch)
    scene_bank = [dataset.scene_tensor_map[name].to(device) for name in dataset.scene_names]
    map_sizes = torch.tensor(
        [[dataset.scene_meta_map[name]["map_x_m"], dataset.scene_meta_map[name]["map_y_m"]]
         for name in dataset.scene_names],
        dtype=torch.float32,
        device=device,
    )
    targets, predictions = [], []
    objective_sum = 0.0
    part_sums: dict[str, float] = {}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scene_id = int(batch["scene_index"][0].item())
            count = int(batch["joint"].numel())
            scene = scene_bank[scene_id].unsqueeze(0)
            map_size = map_sizes[scene_id : scene_id + 1].expand(count, -1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, loss, parts = _forward_loss(
                model_kind, model, batch, scene, map_size, config
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
            objective_sum += float(loss.detach()) * count
            for name, value in parts.items():
                part_sums[name] = part_sums.get(name, 0.0) + value * count
            samples += count
            targets.append(batch["joint"].detach().cpu().numpy())
            predictions.append(prediction.detach().cpu().numpy())
    metrics = regression_metrics(np.concatenate(targets), np.concatenate(predictions))
    metrics["objective"] = objective_sum / max(samples, 1)
    metrics.update({name: value / max(samples, 1) for name, value in part_sums.items()})
    return metrics


def _save_checkpoint(path, model_kind, model, base_checkpoint, optimizer, target, stage, epoch, metrics):
    checkpoint = {
        "model_schema_version": base_checkpoint["model_schema_version"],
        "model_config": base_checkpoint["model_config"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "adaptation_stage": stage,
        "target": target,
        "evaluation_metrics": metrics,
    }
    torch.save(checkpoint, path)


def _train_stage(model_kind, model, dataset, indices, config, device, optimizer, epochs, epoch_offset, history_writer):
    loader, sampler = _make_loader(
        dataset, indices, config["batch_size"], True, config["seed"], config["num_workers"]
    )
    last = {}
    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        last = _run_epoch(
            model_kind, model, loader, sampler, dataset, device, config, optimizer, epoch
        )
        history_writer.writerow({
            "epoch": epoch,
            "stage": "few_shot_30" if epoch_offset == 0 else "in_distribution",
            "num_training_samples": len(indices),
            "objective": last["objective"],
            "mse": last["mse"],
            "mae": last["mae"],
            "spearman": last["spearman"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"epoch={epoch:03d} stage={'few30' if epoch_offset == 0 else 'full'} "
            f"train_mse={last['mse']:.6f} train_spearman={last['spearman']:.4f}"
        )
    return last


def _adapt_one(config, target, requested_model, device):
    scene, num_tx = target["scene"], int(target["num_tx"])
    target_name = f"{scene}_ntx{num_tx}"
    dataset = TxDeploymentDataset(
        _absolute(config["dataset_root"]),
        _absolute(config["scene_root"]),
        include_scenes=[scene],
        include_num_tx=[num_tx],
        permute_tx_order=True,
        expand_family_memberships=False,
    )
    needed = config["adaptation_samples_per_target"] + config["evaluation_samples_per_target"]
    if len(dataset) < needed:
        raise RuntimeError(f"{target_name} has {len(dataset)} samples; {needed} are required")
    indices = list(range(len(dataset)))
    random.Random(config["seed"] + sum(ord(c) for c in target_name)).shuffle(indices)
    adaptation = indices[: config["adaptation_samples_per_target"]]
    evaluation = indices[config["adaptation_samples_per_target"] : needed]
    few_count = int(round(len(adaptation) * config["fewshot_fraction"]))
    fewshot = adaptation[:few_count]
    target_root = _absolute(config["output_root"]) / target_name
    target_root.mkdir(parents=True, exist_ok=True)
    split_path = target_root / "data_split.json"
    split_payload = {
        "target": target,
        "seed": config["seed"],
        "fewshot_fraction": config["fewshot_fraction"],
        "counts": {"few_shot_30": len(fewshot), "in_distribution": len(adaptation), "evaluation": len(evaluation)},
        "few_shot_30_keys": [_record_key(dataset.records[index]) for index in fewshot],
        "in_distribution_keys": [_record_key(dataset.records[index]) for index in adaptation],
        "evaluation_keys": [_record_key(dataset.records[index]) for index in evaluation],
    }
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != split_payload:
        raise RuntimeError(f"existing split disagrees with current plan: {split_path}")
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    checkpoint_path = _absolute(config["base_checkpoints"][requested_model])
    model_kind, model, base_checkpoint = load_guidance_checkpoint(checkpoint_path, device)
    if model_kind != requested_model:
        raise RuntimeError(f"expected {requested_model}, checkpoint contains {model_kind}")
    model_root = target_root / model_kind
    model_root.mkdir(parents=True, exist_ok=True)
    full_checkpoint_path = model_root / "in_distribution" / "checkpoint.pt"
    if full_checkpoint_path.exists():
        print(f"skip complete: {target_name} {model_kind}")
        return
    learning_rate = float(config["learning_rates"][model_kind])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"]
    )
    eval_loader, eval_sampler = _make_loader(
        dataset, evaluation, config["batch_size"], False, config["seed"], config["num_workers"]
    )
    zero_metrics = _run_epoch(
        model_kind, model, eval_loader, eval_sampler, dataset, device, config, None, 0
    )
    reference = {
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": _sha256(checkpoint_path),
        "evaluation_metrics": zero_metrics,
    }
    (model_root / "zero_shot_reference.json").write_text(
        json.dumps(reference, indent=2), encoding="utf-8"
    )
    history_path = model_root / "history.csv"
    fields = ["epoch", "stage", "num_training_samples", "objective", "mse", "mae", "spearman", "learning_rate"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        _train_stage(
            model_kind, model, dataset, fewshot, config, device, optimizer,
            config["fewshot_epochs"], 0, writer,
        )
        few_metrics = _run_epoch(
            model_kind, model, eval_loader, eval_sampler, dataset, device, config, None,
            config["fewshot_epochs"],
        )
        few_dir = model_root / "few_shot_30"
        few_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            few_dir / "checkpoint.pt", model_kind, model, base_checkpoint, optimizer,
            target, "few_shot_30", config["fewshot_epochs"], few_metrics,
        )
        (few_dir / "metrics.json").write_text(json.dumps(few_metrics, indent=2), encoding="utf-8")
        handle.flush()
        _train_stage(
            model_kind, model, dataset, adaptation, config, device, optimizer,
            config["in_distribution_epochs"], config["fewshot_epochs"], writer,
        )
    full_metrics = _run_epoch(
        model_kind, model, eval_loader, eval_sampler, dataset, device, config, None,
        config["fewshot_epochs"] + config["in_distribution_epochs"],
    )
    full_dir = model_root / "in_distribution"
    full_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        full_dir / "checkpoint.pt", model_kind, model, base_checkpoint, optimizer,
        target, "in_distribution", config["fewshot_epochs"] + config["in_distribution_epochs"],
        full_metrics,
    )
    (full_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    run_config = dict(config)
    run_config["base_checkpoints"] = {model_kind: str(checkpoint_path)}
    (model_root / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"saved: {model_root}")


def _validate_plan(config):
    if config.get("schema_version") != "citydeploy.reward-adaptation-plan.v1":
        raise ValueError("unsupported adaptation plan schema")
    fraction = float(config["fewshot_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("fewshot_fraction must be between zero and one")
    missing = set(MODEL_KINDS) - set(config["base_checkpoints"])
    if missing:
        raise ValueError(f"missing base checkpoints: {sorted(missing)}")
    for target in config["targets"]:
        scene_path = _absolute(config["scene_root"]) / target["scene"]
        if not scene_path.is_dir():
            raise FileNotFoundError(f"scene not found: {scene_path}")
    for value in config["base_checkpoints"].values():
        if not _absolute(value).is_file():
            raise FileNotFoundError(f"checkpoint not found: {_absolute(value)}")


def main():
    parser = argparse.ArgumentParser(description="Adapt all reward models to fixed target conditions")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--target", action="append", help="Optional target name, e.g. paris_11_ntx2")
    parser.add_argument("--model", action="append", choices=MODEL_KINDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(_absolute(args.plan).read_text(encoding="utf-8"))
    _validate_plan(config)
    selected_targets = [
        target for target in config["targets"]
        if not args.target or f"{target['scene']}_ntx{target['num_tx']}" in args.target
    ]
    selected_models = args.model or list(MODEL_KINDS)
    if not selected_targets:
        raise ValueError("no target matched --target")
    summary = {
        "targets": [f"{target['scene']}_ntx{target['num_tx']}" for target in selected_targets],
        "models": selected_models,
        "few_shot_samples": round(config["adaptation_samples_per_target"] * config["fewshot_fraction"]),
        "full_samples": config["adaptation_samples_per_target"],
        "evaluation_samples": config["evaluation_samples_per_target"],
        "checkpoints_to_create": len(selected_targets) * len(selected_models) * 2,
        "output_root": str(_absolute(config["output_root"])),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    set_seed(config["seed"])
    device = resolve_device(args.device)
    started = time.perf_counter()
    for target in selected_targets:
        for model_kind in selected_models:
            print(f"\n=== {target['scene']} ntx={target['num_tx']} model={model_kind} ===")
            _adapt_one(config, target, model_kind, device)
    print(f"all adaptations complete in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate CityDeploy V4 deployment families from reusable master TX path gains."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path
from typing import Any

import numpy as np

from dataset_builder.metrics import (
    compute_urban_radio_metrics,
    summarize_urban_metrics,
    tx_axis_last,
)
from dataset_builder.runtime import require_supported_runtime
from dataset_builder.raytracing import SceneRayTracer
from dataset_builder.storage import (
    CONTRACT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    FamilyMembershipShardWriter,
    ParquetShardWriter,
    build_split_manifest,
    canonical_json_sha256,
    existing_sample_ids,
    existing_family_ids,
    existing_membership_family_ids,
    existing_membership_family_counts,
    existing_membership_ids,
    family_id,
    family_membership_id,
    file_sha256,
    sample_id,
)
from energy_model.dataset import SCENE_FEATURE_CHANNELS, load_scene_feature_tensor
from scene_builder.coordinates import COORDINATE_CONVENTION, COORDINATE_SCHEMA_VERSION, SceneFrame2D


PROJECT_ROOT = workspace_root()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scene_generation_profile(plan: dict, metadata: dict) -> dict[str, Any]:
    """Resolve and validate the TX cardinality plan for one metric scene size."""
    map_x = float(metadata["map_x_m"])
    map_y = float(metadata["map_y_m"])
    width = int(round(map_x))
    height = int(round(map_y))
    if abs(map_x - width) > 1e-6 or abs(map_y - height) > 1e-6:
        raise ValueError(f"scene size must be integral metres, got {map_x}x{map_y}")
    key = f"{width}x{height}"
    profiles = plan.get("scene_size_profiles")
    if profiles is None:
        profile = {
            "num_tx_values": plan["num_tx_values"],
            "master_pool_size": plan["master_pool_size"],
            "master_pools_per_scene": plan["master_pools_per_scene"],
        }
    else:
        if not isinstance(profiles, dict) or key not in profiles:
            raise ValueError(f"no scene_size_profiles entry for {key}")
        profile = dict(profiles[key])
    cardinalities = sorted(set(int(value) for value in profile["num_tx_values"]))
    master_size = int(profile["master_pool_size"])
    pool_count = int(profile.get("master_pools_per_scene", plan.get("master_pools_per_scene", 0)))
    if not cardinalities or cardinalities[0] < 1:
        raise ValueError(f"{key}: num_tx_values must contain positive integers")
    if master_size < max(cardinalities):
        raise ValueError(f"{key}: master_pool_size must cover every requested cardinality")
    if pool_count < 1:
        raise ValueError(f"{key}: master_pools_per_scene must be positive")
    return {
        "scene_size_key": key,
        "map_size_m": [map_x, map_y],
        "num_tx_values": cardinalities,
        "master_pool_size": master_size,
        "master_pools_per_scene": pool_count,
    }


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask)
    if source.shape == shape:
        return source.astype(bool)
    rows = np.clip(np.floor(np.arange(shape[0]) * source.shape[0] / shape[0]).astype(int), 0, source.shape[0] - 1)
    columns = np.clip(np.floor(np.arange(shape[1]) * source.shape[1] / shape[1]).astype(int), 0, source.shape[1] - 1)
    return source[np.ix_(rows, columns)].astype(bool)


def _candidate_points(scene_dir: Path) -> np.ndarray:
    preferred = scene_dir / "inputs" / "tx_deployable" / "points_local_xy.npy"
    if preferred.exists():
        return np.asarray(np.load(preferred, allow_pickle=False), dtype=np.float64)
    with np.load(scene_dir / "inputs" / "scene_inputs.npz", allow_pickle=False) as loaded:
        mask = np.asarray(loaded["tx_feasible_mask"], dtype=bool)
    meta = _load_json(scene_dir / "inputs" / "metadata.json")
    frame = SceneFrame2D(float(meta["map_x_m"]), float(meta["map_y_m"]))
    row, column = np.nonzero(mask)
    norm = np.column_stack(((column + 0.5) / mask.shape[1], (row + 0.5) / mask.shape[0]))
    return frame.norm01_to_real(norm)


def _sample_pool(
    points: np.ndarray,
    count: int,
    min_distance_m: float,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if count > len(points):
        raise RuntimeError(f"pool size {count} exceeds {len(points)} candidates")
    if strategy == "clustered":
        center = points[int(rng.integers(len(points)))]
        order = np.argsort(np.linalg.norm(points - center, axis=1))
    elif strategy == "maximin":
        selected = [int(rng.integers(len(points)))]
        available = np.ones(len(points), dtype=bool)
        available[selected[0]] = False
        while len(selected) < count:
            candidate_ids = np.flatnonzero(available)
            if candidate_ids.size == 0:
                break
            if candidate_ids.size > 4096:
                candidate_ids = rng.choice(candidate_ids, size=4096, replace=False)
            distances = np.linalg.norm(
                points[candidate_ids, None, :] - points[np.asarray(selected)][None, :, :], axis=-1
            ).min(axis=1)
            chosen = int(candidate_ids[int(np.argmax(distances))])
            if float(distances.max()) < min_distance_m:
                break
            selected.append(chosen)
            available[chosen] = False
        if len(selected) == count:
            return np.asarray(sorted(selected), dtype=np.int64)
        strategy = "uniform"
    if strategy == "uniform":
        order = rng.permutation(len(points))
    selected: list[int] = []
    for candidate in order:
        if not selected or np.linalg.norm(points[selected] - points[int(candidate)], axis=1).min() >= min_distance_m:
            selected.append(int(candidate))
            if len(selected) == count:
                return np.asarray(sorted(selected), dtype=np.int64)
    raise RuntimeError(f"could not sample {count} TXs with {min_distance_m:g} m spacing using {strategy}")


def _strategy_schedule(weights: dict[str, float], count: int, seed: int) -> list[str]:
    """Return a deterministic prefix-balanced schedule with exact final quotas."""
    if count < 1:
        raise ValueError("strategy schedule count must be positive")
    names = list(weights)
    values = [float(weights[name]) for name in names]
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("sampling strategy weights must be non-negative and non-zero")
    total = sum(values)
    exact = [value / total * count for value in values]
    quotas = [int(math.floor(value)) for value in exact]
    remainder = count - sum(quotas)
    order = sorted(range(len(names)), key=lambda index: (-(exact[index] - quotas[index]), index))
    for index in order[:remainder]:
        quotas[index] += 1
    # Smooth weighted scheduling keeps early pilot prefixes diverse while the
    # complete schedule still has exact quotas. Pool 0 intentionally uses the
    # highest-weight strategy, so a one-pool pilot remains compatible with all
    # later expansions of the same family sequence.
    del seed  # Kept in the API because strategy schedules are part of seeded generation.
    used = [0] * len(names)
    schedule: list[str] = []
    for step in range(count):
        candidates = [index for index, quota in enumerate(quotas) if used[index] < quota]
        selected = max(
            candidates,
            key=lambda index: ((step + 1) * quotas[index] / count - used[index], -index),
        )
        schedule.append(names[selected])
        used[selected] += 1
    return schedule


def _expanded_strategy_schedule(
    weights: dict[str, float], base_count: int, expansion_factor: int, seed: int
) -> list[str]:
    """Extend generation in stable blocks without changing the original prefix."""
    if expansion_factor < 1:
        raise ValueError("expansion factor must be at least one")
    return [
        strategy
        for block_index in range(expansion_factor)
        for strategy in _strategy_schedule(
            weights,
            base_count,
            seed + block_index * 10_000_019,
        )
    ]


def _family_subsets(
    master_size: int,
    cardinalities: list[int],
    subsets_per_cardinality: int,
    hypergraph_subsets_per_family: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """Build reproducible subsets, including nested chains where combinations are sampled."""
    chosen: dict[int, set[tuple[int, ...]]] = {}
    requested_by_cardinality: dict[int, int] = {}
    sampled_cardinalities: list[int] = []
    for num_tx in cardinalities:
        available = math.comb(master_size, num_tx)
        requested = min(
            max(subsets_per_cardinality, int(hypergraph_subsets_per_family.get(str(num_tx), 0))),
            available,
        )
        requested_by_cardinality[num_tx] = requested
        if requested == available:
            chosen[num_tx] = set(itertools.combinations(range(master_size), num_tx))
        else:
            chosen[num_tx] = set()
            sampled_cardinalities.append(num_tx)

    # Prefixes from the same permutation form explicit nested hypergraph chains,
    # e.g. S5 ⊂ S6 ⊂ S7 ⊂ S8 for a nine-TX master family.
    attempts = 0
    while any(len(chosen[num_tx]) < requested_by_cardinality[num_tx] for num_tx in sampled_cardinalities):
        permutation = rng.permutation(master_size)
        active = [
            num_tx
            for num_tx in sampled_cardinalities
            if len(chosen[num_tx]) < requested_by_cardinality[num_tx]
        ]
        candidate_chain = {
            num_tx: tuple(sorted(int(value) for value in permutation[:num_tx]))
            for num_tx in active
        }
        # Accept or reject the active chain as a unit. This preserves the
        # cross-cardinality containment relation after uniqueness filtering.
        if all(candidate_chain[num_tx] not in chosen[num_tx] for num_tx in active):
            for num_tx in active:
                chosen[num_tx].add(candidate_chain[num_tx])
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not construct the requested unique nested family subsets")

    return [
        (num_tx, np.asarray(subset, dtype=np.int64))
        for num_tx in cardinalities
        for subset in sorted(chosen[num_tx])
    ]


def _metric_kwargs(rf: dict) -> dict[str, Any]:
    throughput = rf["throughput"]
    return {
        "tx_power_dbm": float(rf["tx_power_dbm"]),
        "bandwidth_hz": float(rf["bandwidth_hz"]),
        "noise_figure_db": float(rf["noise_figure_db"]),
        "num_resource_blocks": int(rf["num_resource_blocks"]),
        "subcarriers_per_resource_block": int(rf["subcarriers_per_resource_block"]),
        "ssb_power_offset_db": float(rf["ssb_power_offset_db"]),
        "implementation_efficiency": float(throughput["implementation_efficiency"]),
        "resource_share": float(throughput["resource_share"]),
        "max_spectral_efficiency_bps_hz": float(throughput["max_spectral_efficiency_bps_hz"]),
    }


def _save_rich_map(
    root: Path,
    split: str,
    identifier: str,
    metrics: dict[str, np.ndarray],
    conditions: dict[str, np.ndarray],
    evaluation_mask: np.ndarray,
    selected_path_gain: np.ndarray | None,
) -> str:
    relative = Path("radio_maps") / split / f"{identifier}.npz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path_loss_db": np.asarray(metrics["path_loss_db"], dtype=np.float32),
        "ss_rsrp_dbm": np.asarray(metrics["ss_rsrp_dbm"], dtype=np.float32),
        "sinr_db": np.asarray(metrics["sinr_db"], dtype=np.float32),
        "effective_throughput_mbps": np.asarray(metrics["effective_throughput_mbps"], dtype=np.float32),
        "serving_tx_index": np.asarray(metrics["serving_idx"], dtype=np.int16),
        "road_evaluation_mask": np.asarray(evaluation_mask, dtype=np.uint8),
        **{f"{name}_covered": np.asarray(value, dtype=np.uint8) for name, value in conditions.items()},
    }
    if selected_path_gain is not None:
        payload["per_tx_path_gain"] = np.asarray(selected_path_gain, dtype=np.float32)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return relative.as_posix()


def _write_dataset_card(root: Path, manifest: dict) -> None:
    text = f"""---
pretty_name: CityDeploy Urban Multi-TX Ray-Tracing Dataset V4
license: other
tags:
- geospatial
- tabular
- ray-tracing
- wireless
- sionna
- hypergraph
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.parquet
  - split: validation
    path: data/validation/*.parquet
  - split: test
    path: data/test/*.parquet
---

# CityDeploy Urban Multi-TX Ray-Tracing Dataset V4

Scene-conditioned, unordered transmitter deployments generated with Sionna RT.
The primary reward is cell-wise four-metric joint road coverage. TX sets are
grouped into reusable deployment families to support ranking, counterfactual,
and order-1-through-4 hypergraph learning.

## Contract

- Dataset schema: `{manifest['schema_version']}`
- Contract: `{manifest['contract_id']}`
- Evaluation domain: road cells only
- TX deployment domain: road cells only
- Cardinalities: {', '.join(str(value) for value in manifest['num_tx_values'])}

The four marginal labels are path-loss, SS-RSRP, SINR, and effective-throughput
coverage. `joint_4metric_coverage` is their cell-wise intersection, not a mean
or product. Full provenance and thresholds are in `dataset_contract.json`.

## Files

Deployment rows are Zstandard-compressed Parquet shards. A deterministic small
subset has complete radio maps under `radio_maps/`. Scene tensors are stored
once under `scenes/features/`.

## License and attribution

The public data license must be selected before release. Geographic source
attribution: © OpenStreetMap contributors. See individual scene manifests for
source provenance and limitations.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _initialize(plan: dict, plan_path: Path, rf: dict, scenes: list[Path]) -> tuple[Path, dict, dict]:
    root = (PROJECT_ROOT / plan["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene_meta = {scene.name: _load_json(scene / "inputs" / "metadata.json") for scene in scenes}
    scene_profiles = {
        scene.name: _scene_generation_profile(plan, scene_meta[scene.name])
        for scene in scenes
    }
    all_cardinalities = sorted(
        {value for profile in scene_profiles.values() for value in profile["num_tx_values"]}
    )
    split_cfg = plan["split"]
    split_manifest = build_split_manifest(
        scene_meta,
        train_fraction=float(split_cfg["train_fraction"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        seed=int(split_cfg["seed"]),
        scene_assignments=split_cfg.get("scene_assignments"),
    )
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "rf_profile": rf,
        "coordinates": {
            "schema_version": COORDINATE_SCHEMA_VERSION,
            "convention": COORDINATE_CONVENTION,
            "real_units": "m",
            "normalized_domain": "[0,1]^2",
        },
        "deployment_policy": "road_only",
        "evaluation_policy": "road_only",
        "minimum_tx_distance_m": float(plan["sampling"]["minimum_tx_distance_m"]),
        "num_tx_values": all_cardinalities,
        "metric_names": ["path_loss_db", "ss_rsrp_dbm", "sinr_db", "effective_throughput_mbps"],
        "primary_reward": "joint_4metric_coverage",
        "energy_target_definition": "-joint_4metric_coverage",
    }

    # size-specific plans freeze the resolved profile for every scene.
    if plan.get("scene_size_profiles") is not None:
        contract["scene_generation_profiles"] = scene_profiles
    contract_id = canonical_json_sha256(contract)
    contract["contract_id"] = contract_id
    scene_registry = {
        "schema_version": "citydeploy.scene-registry.v1",
        "scenes": {
            scene.name: {
                "manifest_sha256": file_sha256(scene / "manifest.json"),
                "scene_xml_sha256": file_sha256(scene / "scene.xml"),
                "scene_inputs_sha256": file_sha256(scene / "inputs" / "scene_inputs.npz"),
                "semantic_masks_sha256": file_sha256(scene / "inputs" / "semantic_masks.npz"),
                "tx_candidate_points_sha256": file_sha256(
                    scene / "inputs" / "tx_deployable" / "points_local_xy.npy"
                ),
            }
            for scene in scenes
        },
    }
    scene_registry["sha256"] = canonical_json_sha256(scene_registry)
    for filename, content in (
        ("dataset_contract.json", contract),
        ("split_manifest.json", split_manifest),
    ):
        path = root / filename
        if path.exists() and _load_json(path) != content:
            raise RuntimeError(f"immutable V4 file differs: {path}")
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    scene_registry_path = root / "scene_registry.json"
    if scene_registry_path.exists() and _load_json(scene_registry_path) != scene_registry:
        raise RuntimeError("scene registry changed; refusing to append to the existing V4 dataset")
    scene_registry_path.write_text(json.dumps(scene_registry, indent=2), encoding="utf-8")
    shutil.copy2(plan_path, root / "generation_plan.json")
    rf_snapshot = root / "rf_profile.json"
    rf_snapshot.write_text(json.dumps(rf, indent=2, ensure_ascii=False), encoding="utf-8")
    features_root = root / "scenes" / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    for scene in scenes:
        array, metadata = load_scene_feature_tensor(scene)
        np.savez_compressed(
            features_root / f"{scene.name}.npz",
            features=array.astype(np.uint8),
            channel_names=np.asarray(SCENE_FEATURE_CHANNELS),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            tx_candidate_points_xy_m=_candidate_points(scene).astype(np.float32),
        )
        scene_rows.append(
            {
                "scene_id": scene.name,
                "geographic_group_id": split_manifest["scenes"][scene.name]["geographic_group_id"],
                "split": split_manifest["scenes"][scene.name]["split"],
                "map_x_m": float(metadata["map_x_m"]),
                "map_y_m": float(metadata["map_y_m"]),
                "grid_h": int(metadata["grid_h"]),
                "grid_w": int(metadata["grid_w"]),
                "scene_manifest_sha256": file_sha256(scene / "manifest.json"),
                "scene_inputs_sha256": file_sha256(scene / "inputs" / "scene_inputs.npz"),
                "center_lon": float(metadata["center_wgs84"]["lon"]),
                "center_lat": float(metadata["center_wgs84"]["lat"]),
            }
        )
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(scene_rows), root / "scenes" / "scenes.parquet", compression="zstd")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": str(plan["dataset_id"]),
        "contract_id": contract_id,
        "license": None,
        "num_samples_total": 0,
        "num_family_memberships": 0,
        "num_families_completed": 0,
        "scene_ids": [scene.name for scene in scenes],
        "num_tx_values": contract["num_tx_values"],
        "split_manifest_sha256": split_manifest["sha256"],
        "scene_registry_sha256": scene_registry["sha256"],
        "storage": {
            "deployments": "data/<split>/part-*.parquet",
            "family_memberships": "family_memberships/<split>/part-*.parquet",
            "scene_features": "scenes/features/<scene_id>.npz",
            "rich_radio_maps": "radio_maps/<split>/<sample_id>.npz",
        },
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if existing.get("contract_id") != contract_id:
            raise RuntimeError("existing V4 dataset uses a different contract")
        manifest.update(
            {
                key: existing.get(key, manifest[key])
                for key in (
                    "num_samples_total",
                    "num_family_memberships",
                    "num_families_completed",
                )
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    schema_description = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "parquet_schema": str(__import__("dataset_builder.storage", fromlist=["parquet_schema"]).parquet_schema()),
        "primary_reward": "joint_4metric_coverage",
        "sample_unit": "one unordered transmitter deployment",
    }
    (root / "schema.json").write_text(json.dumps(schema_description, indent=2), encoding="utf-8")
    citation_path = root / "CITATION.cff"
    if not citation_path.exists():
        citation_path.write_text(
            "cff-version: 1.2.0\nmessage: Please cite the CityDeploy dataset and associated paper.\n"
            "title: CityDeploy Urban Multi-TX Ray-Tracing Dataset\ntype: dataset\nversion: 4.0.0\n",
            encoding="utf-8",
        )
    pending_license = root / "LICENSE_PENDING.md"
    if not pending_license.exists():
        pending_license.write_text(
            "# Dataset license pending\n\nSelect and record the public dataset license before upload. "
            "Retain © OpenStreetMap contributors attribution and audit all scene-source terms.\n",
            encoding="utf-8",
        )
    _write_dataset_card(root, manifest)
    return root, contract, split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CityDeploy V4 master-pool deployment families")
    parser.add_argument("--plan", type=Path, default=config_path("datasets/urban_multicity.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--scene", action="append", help="Restrict to one or more scene IDs")
    parser.add_argument("--pools-per-scene", type=int, help="Override the plan for pilot runs")
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=1,
        help="Generate stable additional pool blocks; 3 means three times the configured pools",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and initialize without ray tracing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expansion_factor < 1:
        raise ValueError("--expansion-factor must be at least 1")
    if args.pools_per_scene is not None and args.expansion_factor != 1:
        raise ValueError("--pools-per-scene and --expansion-factor cannot be used together")
    plan_path = args.plan.resolve()
    plan = _load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise RuntimeError(f"expected {PLAN_SCHEMA_VERSION}, got {plan.get('schema_version')}")
    rf_path = resolve_path(plan["rf_profile"])
    rf = _load_json(rf_path)
    scenes_root = (PROJECT_ROOT / plan["scenes_root"]).resolve()
    configured_scenes = plan.get("scenes", "*")
    configured_set = None if configured_scenes == "*" else set(str(value) for value in configured_scenes)
    all_scenes = sorted(
        path for path in scenes_root.iterdir()
        if path.is_dir()
        and (configured_set is None or path.name in configured_set)
        and (path / "inputs" / "semantic_masks.npz").exists()
    )
    selected = set(args.scene or ())
    scenes = [path for path in all_scenes if not selected or path.name in selected]
    missing = selected.difference(path.name for path in all_scenes)
    if missing:
        raise RuntimeError(f"unknown or unconfigured scenes: {sorted(missing)}")
    if not all_scenes or not scenes:
        raise RuntimeError("no matching V3 scene packages found")
    root, contract, split_manifest = _initialize(plan, plan_path, rf, all_scenes)
    if args.dry_run:
        print(f"Initialized V4 dataset without ray tracing: {root}")
        return
    runtime = require_supported_runtime(args.device, require_cuda=args.require_cuda)
    storage = plan["storage"]
    writers = {
        split: ParquetShardWriter(
            root / "data" / split,
            int(storage["parquet_rows_per_shard"]),
            int(storage["parquet_row_group_size"]),
            str(storage["compression"]),
        )
        for split in ("train", "validation", "test")
    }
    membership_writers = {
        split: FamilyMembershipShardWriter(
            root / "family_memberships" / split,
            int(storage["parquet_rows_per_shard"]),
            int(storage["parquet_row_group_size"]),
            str(storage["compression"]),
        )
        for split in ("train", "validation", "test")
    }
    known_ids = existing_sample_ids(root / "data")
    known_memberships = existing_membership_ids(root / "family_memberships")
    membership_family_ids = existing_membership_family_ids(root / "family_memberships")
    membership_family_counts = existing_membership_family_counts(root / "family_memberships")

    # must first be migrated with repair_family_memberships so duplicates are
    # represented rather than silently skipped.
    known_families = membership_family_ids or existing_family_ids(root / "data")
    base_seed = int(plan["sampling"]["seed"])
    total_new = 0
    families_completed = 0
    started_all = time.perf_counter()
    try:
        scene_seed_index = {scene.name: index for index, scene in enumerate(all_scenes)}
        for scene_dir in scenes:
            scene_index = scene_seed_index[scene_dir.name]
            profile = contract.get("scene_generation_profiles", {}).get(scene_dir.name)
            if profile is None:
                metadata_for_profile = _load_json(scene_dir / "inputs" / "metadata.json")
                profile = _scene_generation_profile(plan, metadata_for_profile)
            configured_pool_count = int(profile["master_pools_per_scene"])
            pool_count = int(
                args.pools_per_scene
                if args.pools_per_scene is not None
                else configured_pool_count * args.expansion_factor
            )
            if args.pools_per_scene is not None and pool_count > configured_pool_count:
                raise ValueError(
                    f"{scene_dir.name}: --pools-per-scene={pool_count} exceeds "
                    f"the frozen profile limit {configured_pool_count}"
                )
            master_size = int(profile["master_pool_size"])
            cardinalities = [int(value) for value in profile["num_tx_values"]]
            expected_memberships_per_family = len(
                _family_subsets(
                    master_size,
                    cardinalities,
                    int(plan.get("subsets_per_cardinality", 1)),
                    plan.get("hypergraph_subsets_per_family", {}),
                    np.random.default_rng(0),
                )
            )
            strategy_schedule = _expanded_strategy_schedule(
                plan["sampling"]["strategies"],
                configured_pool_count,
                args.expansion_factor,
                base_seed + scene_index * 1_000_003,
            )
            if args.pools_per_scene is not None:
                strategy_schedule = strategy_schedule[:pool_count]
            pending_pool_indices = []
            for pool_index in range(pool_count):
                seed = base_seed + scene_index * 1_000_003 + pool_index
                identifier = family_id(contract["contract_id"], scene_dir.name, pool_index, seed)
                if (
                    identifier not in known_families
                    or (
                        membership_family_ids
                        and membership_family_counts.get(identifier, 0)
                        != expected_memberships_per_family
                    )
                ):
                    pending_pool_indices.append(pool_index)
            if not pending_pool_indices:
                print(f"scene={scene_dir.name} all {pool_count} requested families already complete; skipped")
                continue
            metadata = _load_json(scene_dir / "inputs" / "metadata.json")
            map_size = (float(metadata["map_x_m"]), float(metadata["map_y_m"]))
            frame = SceneFrame2D(*map_size)
            candidates = _candidate_points(scene_dir)
            with np.load(scene_dir / "inputs" / "semantic_masks.npz", allow_pickle=False) as semantic:
                road_mask_source = np.asarray(semantic["road_mask"], dtype=bool)
            tracer = SceneRayTracer(scene_dir, rf, args.device)
            scene_split = split_manifest["scenes"][scene_dir.name]
            split = str(scene_split["split"])
            for pool_index in pending_pool_indices:
                seed = base_seed + scene_index * 1_000_003 + pool_index
                np_rng = np.random.default_rng(seed)
                strategy = strategy_schedule[pool_index]
                pool_indices = _sample_pool(
                    candidates,
                    master_size,
                    float(plan["sampling"]["minimum_tx_distance_m"]),
                    strategy,
                    np_rng,
                )
                pool_xy = candidates[pool_indices]
                pool_xyz = np.column_stack(
                    [pool_xy, np.full(master_size, float(rf["tx_height_m"]), dtype=np.float64)]
                )
                current_family = family_id(contract["contract_id"], scene_dir.name, pool_index, seed)
                path_gain_raw, trace_seconds = tracer.trace(pool_xyz, map_size, seed + int(rf["ray_tracing"]["base_seed"]))
                pool_path_gain = tx_axis_last(path_gain_raw, master_size)
                road_mask = _resize_mask_nearest(road_mask_source, pool_path_gain.shape[:2])
                subsets = _family_subsets(
                    master_size,
                    cardinalities,
                    int(plan.get("subsets_per_cardinality", 1)),
                    plan.get("hypergraph_subsets_per_family", {}),
                    np_rng,
                )
                anchor_candidates = pool_indices
                anchor_id = sample_id(contract["contract_id"], scene_dir.name, anchor_candidates)
                for num_tx, subset in subsets:
                    candidate_ids = np.asarray(pool_indices[subset], dtype=np.int64)
                    identifier = sample_id(contract["contract_id"], scene_dir.name, candidate_ids)
                    membership_identifier = family_membership_id(current_family, identifier)
                    if membership_identifier not in known_memberships:
                        membership_writers[split].append(
                            {
                                "membership_id": membership_identifier,
                                "family_id": current_family,
                                "sample_id": identifier,
                                "scene_id": scene_dir.name,
                                "split": split,
                                "contract_id": contract["contract_id"],
                                "pool_index": pool_index,
                                "master_sample_id": anchor_id,
                                "family_role": "anchor" if identifier == anchor_id else ("hyperedge_subset" if num_tx <= 4 else "cardinality_subset"),
                                "subset_indices_in_family": subset.astype(np.int16).tolist(),
                                "sampling_strategy": strategy,
                                "simulation_seed": seed,
                                "num_tx": num_tx,
                            }
                        )
                        known_memberships.add(membership_identifier)
                    if identifier in known_ids:
                        continue
                    selected_pg = pool_path_gain[..., subset]
                    metrics = compute_urban_radio_metrics(selected_pg, num_tx, **_metric_kwargs(rf))
                    summary, conditions = summarize_urban_metrics(metrics, road_mask, rf["coverage_thresholds"])
                    tx_xyz = pool_xyz[subset]
                    tx_norm = frame.real_to_norm01(tx_xyz[:, :2], check_bounds=True)
                    selector = int(identifier[:16], 16) / float(16**16 - 1)
                    rich_fraction = float(storage["rich_radio_map_fraction"])
                    path_gain_fraction = float(storage["per_tx_path_gain_fraction"])
                    rich_path = None
                    if selector < rich_fraction:
                        rich_path = _save_rich_map(
                            root,
                            split,
                            identifier,
                            metrics,
                            conditions,
                            road_mask,
                            selected_pg if selector < path_gain_fraction else None,
                        )
                    row = {
                        "sample_id": identifier,
                        "scene_id": scene_dir.name,
                        "geographic_group_id": scene_split["geographic_group_id"],
                        "split": split,
                        "contract_id": contract["contract_id"],
                        "family_id": current_family,
                        "parent_sample_id": None if identifier == anchor_id else anchor_id,
                        "family_role": "anchor" if identifier == anchor_id else ("hyperedge_subset" if num_tx <= 4 else "cardinality_subset"),
                        "subset_indices_in_family": subset.astype(np.int16).tolist(),
                        "sampling_strategy": strategy,
                        "simulation_seed": seed,
                        "num_tx": num_tx,
                        "tx_density_per_km2": float(num_tx / (map_size[0] * map_size[1] / 1e6)),
                        "tx_candidate_indices": candidate_ids.astype(np.int32).tolist(),
                        "tx_xyz_m": tx_xyz.astype(np.float32).tolist(),
                        "tx_xy_norm01": tx_norm.astype(np.float32).tolist(),
                        **summary,
                        "runtime_seconds": float(trace_seconds),
                        "backend": str(runtime["mitsuba_variant"]),
                        "rich_map_path": rich_path,
                        "valid": True,
                        "warning_flags": [],
                    }
                    writers[split].append(row)
                    known_ids.add(identifier)
                    total_new += 1
                families_completed += 1
                known_families.add(current_family)
                elapsed = time.perf_counter() - started_all
                print(
                    f"scene={scene_dir.name} family={pool_index + 1}/{pool_count} "
                    f"strategy={strategy} new_rows={total_new} elapsed={elapsed:.1f}s"
                )
    finally:
        for writer in writers.values():
            writer.flush()
        for writer in membership_writers.values():
            writer.flush()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest["num_samples_total"] = len(existing_sample_ids(root / "data"))
    manifest["num_family_memberships"] = len(existing_membership_ids(root / "family_memberships"))
    manifest["num_families_completed"] = len(existing_membership_family_ids(root / "family_memberships"))
    manifest["generation_expansion_factor"] = int(args.expansion_factor)
    manifest["last_runtime"] = runtime
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from dataset_builder.validate import validate_dataset_v4, write_distribution_qa

    report = validate_dataset_v4(root)
    qa_root = root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    (qa_root / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_distribution_qa(root, qa_root / "distributions.parquet")
    if not report["passed"]:
        raise RuntimeError("V4 QA failed after generation; inspect qa/qa_report.json")
    print(f"Saved {total_new} new V4 deployment rows to {root}")


if __name__ == "__main__":
    main()

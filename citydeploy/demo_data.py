"""Create an analytic, non-radio dataset for testing the learning interface."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataset_builder.storage import (ParquetShardWriter, FamilyMembershipShardWriter,
    canonical_json_sha256, sample_id, family_id, family_membership_id, METRIC_PREFIXES)
from energy_model.dataset import SCENE_FEATURE_CHANNELS


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def analytic_coverage(points, cells):
    """Fraction of evaluation cells within a fixed normalized service radius."""
    distance = np.linalg.norm(cells[:, None, :] - points[None, :, :], axis=-1)
    return distance.min(axis=1) < 0.28


def create_dataset(output: Path, rows_per_split=24, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory.")
    if not 4 <= rows_per_split <= 200:
        raise ValueError("Use 4 to 200 rows per split for this small interface demonstration.")
    rng = np.random.default_rng(seed)
    root = output / "dataset"
    scene_root = output / "scenes"
    splits = {f"synthetic_{i}": {"split": split, "geographic_group_id": f"synthetic-{i}"}
              for i, split in enumerate(("train", "validation", "test"))}
    contract = {"schema_version": "citydeploy.dataset-contract.v4", "label_source": "analytic_demo_not_radio",
                "minimum_tx_distance_m": 0.0,
                "scene_generation_profiles": {s: {"master_pool_size": 2, "num_tx_values": [2]} for s in splits}}
    contract["contract_id"] = canonical_json_sha256(contract)
    split_manifest = {"schema_version": "citydeploy.geographic-split.v1", "scenes": splits}
    split_manifest["sha256"] = canonical_json_sha256(split_manifest)
    registry = {"scenes": list(splits), "label_source": "analytic_demo_not_radio"}
    registry["sha256"] = canonical_json_sha256(registry)
    write_json(root / "dataset_contract.json", contract)
    write_json(root / "split_manifest.json", split_manifest)
    write_json(root / "scene_registry.json", registry)
    write_json(root / "generation_plan.json", {"subsets_per_cardinality": 1, "hypergraph_subsets_per_family": {}})
    table = []
    for scene_index, (scene, assignment) in enumerate(splits.items()):
        features = np.zeros((len(SCENE_FEATURE_CHANNELS), 16, 16), dtype=np.uint8)
        features[0, 4:9, 4 + scene_index:9 + scene_index] = 1
        road = 1 - features[0]
        features[1] = features[-2] = features[-1] = road
        yy, xx = np.nonzero(road)
        cells = np.column_stack(((xx + .5) / 16, (yy + .5) / 16))
        candidates = (cells - .5) * 128
        metadata = {"map_x_m": 128.0, "map_y_m": 128.0, "scene_id": scene,
                    "label_source": "analytic_demo_not_radio"}
        feature_dir = root / "scenes" / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feature_dir / f"{scene}.npz", features=features,
                            channel_names=np.asarray(SCENE_FEATURE_CHANNELS),
                            metadata_json=np.asarray(json.dumps(metadata)), tx_candidate_points_xy_m=candidates)
        inputs = scene_root / scene / "inputs"
        inputs.mkdir(parents=True)
        write_json(inputs / "metadata.json", metadata)
        np.savez_compressed(inputs / "scene_inputs.npz", building_occupancy=features[0],
                            deployable_mask=road, tx_feasible_mask=road, evaluation_mask=road)
        np.savez_compressed(inputs / "semantic_masks.npz",
                            **{name: features[i] for i, name in enumerate(SCENE_FEATURE_CHANNELS[:-2])})
        table.append(metadata)
        split = assignment["split"]
        writer = ParquetShardWriter(root / "data" / split, rows_per_split, rows_per_split)
        memberships = FamilyMembershipShardWriter(root / "family_memberships" / split, rows_per_split, rows_per_split)
        seen = set()
        for i in range(rows_per_split):
            while True:
                indices = tuple(sorted(rng.choice(len(cells), 2, replace=False).tolist()))
                if indices not in seen:
                    seen.add(indices)
                    break
            points = cells[list(indices)]
            covered = int(analytic_coverage(points, cells).sum())
            fraction = covered / len(cells)
            sid = sample_id(contract["contract_id"], scene, indices)
            fid = family_id(contract["contract_id"], scene, i, seed)
            row = dict(sample_id=sid, scene_id=scene, **assignment, contract_id=contract["contract_id"],
                       family_id=fid, parent_sample_id=None, family_role="anchor",
                       subset_indices_in_family=[0, 1], sampling_strategy="uniform", simulation_seed=seed,
                       num_tx=2, tx_density_per_km2=2 / .128 ** 2, tx_candidate_indices=list(indices),
                       tx_xyz_m=np.column_stack((candidates[list(indices)], [10., 10.])).tolist(),
                       tx_xy_norm01=points.tolist(), evaluation_cell_count=len(cells),
                       joint_covered_count=covered, joint_4metric_coverage=fraction, joint_target_met=fraction >= .9,
                       runtime_seconds=0.0, backend="analytic_demo", rich_map_path=None, valid=True,
                       warning_flags=["Not radio simulation; all metric labels repeat the analytic service fraction."])
            for metric in METRIC_PREFIXES:
                row[f"{metric}_covered_count"] = covered
                row[f"{metric}_coverage"] = fraction
                for statistic in ("mean", "p05", "p50", "p95"):
                    row[f"{metric}_{statistic}"] = None
            writer.append(row)
            memberships.append(dict(membership_id=family_membership_id(fid, sid), family_id=fid,
                sample_id=sid, scene_id=scene, split=split, contract_id=contract["contract_id"], pool_index=i,
                master_sample_id=sid, family_role="anchor", subset_indices_in_family=[0, 1],
                sampling_strategy="uniform", simulation_seed=seed, num_tx=2))
        writer.flush()
        memberships.flush()
    pq.write_table(pa.Table.from_pylist(table), root / "scenes" / "scenes.parquet")
    write_json(root / "manifest.json", {"schema_version": "citydeploy.dataset.v4", "dataset_id": "synthetic-demo",
        "label_source": "analytic_demo_not_radio", "num_samples_total": 3 * rows_per_split,
        "num_family_memberships": 3 * rows_per_split, "scene_registry_sha256": registry["sha256"]})
    return root, scene_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_data"))
    parser.add_argument("--rows-per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, scenes = create_dataset(args.output, args.rows_per_split, args.seed)
    from dataset_builder.validate import validate_dataset_v4
    report = validate_dataset_v4(root)
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps({"dataset": str(root), "scenes": str(scenes), "qa_passed": True,
                      "label_source": "analytic_demo_not_radio"}, indent=2))


if __name__ == "__main__":
    main()

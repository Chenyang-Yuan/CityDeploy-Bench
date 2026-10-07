#!/usr/bin/env python3
"""Validate CityDeploy Dataset V4 contracts, Parquet rows, splits, and rich maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from citydeploy.schema import canonical_schema

from dataset_builder.storage import (
    COUNT_FIELDS,
    COVERAGE_FIELDS,
    DATASET_SCHEMA_VERSION,
    canonical_json_sha256,
    family_membership_schema,
    parquet_schema,
)
from scene_builder.coordinates import SceneFrame2D


def validate_dataset_v4(root: Path, max_rows: int | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    manifest_path = root / "manifest.json"
    contract_path = root / "dataset_contract.json"
    split_path = root / "split_manifest.json"
    scene_registry_path = root / "scene_registry.json"
    check("manifest_exists", manifest_path.exists(), str(manifest_path))
    check("contract_exists", contract_path.exists(), str(contract_path))
    check("split_manifest_exists", split_path.exists(), str(split_path))
    check("scene_registry_exists", scene_registry_path.exists(), str(scene_registry_path))
    if not all(path.exists() for path in (manifest_path, contract_path, split_path, scene_registry_path)):
        return {"schema_version": "citydeploy.dataset-qa.v4", "passed": False, "checks": checks}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    scene_registry = json.loads(scene_registry_path.read_text(encoding="utf-8"))
    check("schema", canonical_schema(manifest.get("schema_version")) == DATASET_SCHEMA_VERSION, str(manifest.get("schema_version")))
    contract_body = {key: value for key, value in contract.items() if key != "contract_id"}
    check("contract_hash", canonical_json_sha256(contract_body) == contract.get("contract_id"), str(contract.get("contract_id")))
    split_body = {key: value for key, value in splits.items() if key != "sha256"}
    check("split_hash", canonical_json_sha256(split_body) == splits.get("sha256"), str(splits.get("sha256")))
    registry_body = {key: value for key, value in scene_registry.items() if key != "sha256"}
    registry_hash = canonical_json_sha256(registry_body)
    check("scene_registry_hash", registry_hash == scene_registry.get("sha256"), registry_hash)
    check("manifest_scene_registry", manifest.get("scene_registry_sha256") == scene_registry.get("sha256"), str(manifest.get("scene_registry_sha256")))
    group_splits: dict[str, set[str]] = {}
    for item in splits.get("scenes", {}).values():
        group_splits.setdefault(str(item["geographic_group_id"]), set()).add(str(item["split"]))
    leaking = {group: values for group, values in group_splits.items() if len(values) != 1}
    check("geographic_split_isolation", not leaking, f"leaking_groups={leaking}")

    scene_table_path = root / "scenes" / "scenes.parquet"
    check("scene_table_exists", scene_table_path.exists(), str(scene_table_path))
    scene_meta: dict[str, tuple[float, float]] = {}
    scene_candidates: dict[str, np.ndarray] = {}
    if scene_table_path.exists():
        for row in pq.read_table(scene_table_path).to_pylist():
            scene_id = str(row["scene_id"])
            scene_meta[scene_id] = (float(row["map_x_m"]), float(row["map_y_m"]))
            feature_path = root / "scenes" / "features" / f"{scene_id}.npz"
            check(f"scene_feature:{scene_id}", feature_path.exists(), str(feature_path))
            if feature_path.exists():
                with np.load(feature_path, allow_pickle=False) as loaded:
                    if "tx_candidate_points_xy_m" in loaded:
                        scene_candidates[scene_id] = np.asarray(loaded["tx_candidate_points_xy_m"], dtype=np.float64)

    files = sorted((root / "data").glob("*/*.parquet"))
    declared_samples = int(manifest.get("num_samples_total", -1))
    initialized_empty = declared_samples == 0 and not files
    check(
        "parquet_files",
        bool(files) or initialized_empty,
        f"count={len(files)}, declared_samples={declared_samples}",
    )
    expected_names = set(parquet_schema().names)
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    deployment_meta: dict[str, tuple[str, str, int]] = {}
    total = 0
    stop = False
    for path in files:
        table = pq.read_table(path)
        missing = expected_names.difference(table.column_names)
        if missing:
            invalid.append({"file": str(path), "errors": [f"missing columns {sorted(missing)}"]})
            continue
        for row_index, row in enumerate(table.to_pylist()):
            if max_rows is not None and total >= max_rows:
                stop = True
                break
            total += 1
            errors: list[str] = []
            identifier = str(row["sample_id"])
            if identifier in seen:
                errors.append("duplicate sample_id")
            seen.add(identifier)
            scene_id = str(row["scene_id"])
            deployment_meta[identifier] = (scene_id, str(row["split"]), int(row["num_tx"]))
            expected_split = splits.get("scenes", {}).get(scene_id, {}).get("split")
            if row["split"] != expected_split or path.parent.name != expected_split:
                errors.append("row/file split disagrees with split manifest")
            indices = [int(value) for value in row["tx_candidate_indices"]]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                errors.append("TX candidate indices are not a canonical unique set")
            num_tx = int(row["num_tx"])
            xyz = np.asarray(row["tx_xyz_m"], dtype=np.float64)
            norm = np.asarray(row["tx_xy_norm01"], dtype=np.float64)
            if xyz.shape != (num_tx, 3) or norm.shape != (num_tx, 2) or len(indices) != num_tx:
                errors.append("TX array shape disagrees with num_tx")
            if np.any(norm < 0.0) or np.any(norm > 1.0):
                errors.append("normalized TX coordinate outside [0,1]")
            if scene_id in scene_meta and xyz.shape == (num_tx, 3):
                reconstructed = SceneFrame2D(*scene_meta[scene_id]).norm01_to_real(norm)
                if not np.allclose(reconstructed, xyz[:, :2], atol=1e-3, rtol=0.0):
                    errors.append("real/normalized TX coordinates disagree")
            if xyz.shape == (num_tx, 3) and num_tx > 1:
                distance = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=-1)
                np.fill_diagonal(distance, np.inf)
                minimum = float(contract.get("minimum_tx_distance_m", 0.0))
                if float(distance.min()) + 1e-4 < minimum:
                    errors.append("minimum TX distance is violated")
            candidates = scene_candidates.get(scene_id)
            if candidates is not None and xyz.shape == (num_tx, 3):
                if any(value < 0 or value >= len(candidates) for value in indices):
                    errors.append("TX candidate index is out of range")
                elif not np.allclose(candidates[np.asarray(indices)], xyz[:, :2], atol=1e-3, rtol=0.0):
                    errors.append("TX positions disagree with canonical road candidates")
            denominator = int(row["evaluation_cell_count"])
            if denominator <= 0:
                errors.append("evaluation_cell_count is not positive")
            for count_name, coverage_name in zip(COUNT_FIELDS, COVERAGE_FIELDS):
                if denominator > 0 and abs(float(row[coverage_name]) - int(row[count_name]) / denominator) > 2e-6:
                    errors.append(f"{coverage_name} disagrees with count")
            if float(row["joint_4metric_coverage"]) > min(float(row[name]) for name in COVERAGE_FIELDS[:-1]) + 1e-6:
                errors.append("joint coverage exceeds a marginal")
            rich_path = row.get("rich_map_path")
            if rich_path and not (root / rich_path).exists():
                errors.append("rich radio map is missing")
            if errors:
                invalid.append({"file": str(path), "row": row_index, "sample_id": identifier, "errors": errors})
        if stop:
            break
    check("rows_valid", not invalid, f"checked={total}, invalid={len(invalid)}")
    if max_rows is None:
        check("manifest_row_count", total == int(manifest.get("num_samples_total", -1)), f"rows={total}")

    membership_files = sorted((root / "family_memberships").glob("*/*.parquet"))
    check(
        "family_membership_files",
        bool(membership_files) or initialized_empty,
        f"count={len(membership_files)}",
    )
    expected_membership_names = set(family_membership_schema().names)
    membership_invalid: list[dict[str, Any]] = []
    membership_ids: set[str] = set()
    membership_pairs: set[tuple[str, str]] = set()
    referenced_samples: set[str] = set()
    family_rows: dict[str, list[dict[str, Any]]] = {}
    membership_total = 0
    for path in membership_files:
        table = pq.read_table(path)
        missing = expected_membership_names.difference(table.column_names)
        if missing:
            membership_invalid.append(
                {"file": str(path), "errors": [f"missing columns {sorted(missing)}"]}
            )
            continue
        for row_index, row in enumerate(table.to_pylist()):
            membership_total += 1
            errors: list[str] = []
            membership_id = str(row["membership_id"])
            family_identifier = str(row["family_id"])
            sample_identifier = str(row["sample_id"])
            pair = (family_identifier, sample_identifier)
            if membership_id in membership_ids:
                errors.append("duplicate membership_id")
            if pair in membership_pairs:
                errors.append("duplicate family/sample membership")
            membership_ids.add(membership_id)
            membership_pairs.add(pair)
            referenced_samples.add(sample_identifier)
            deployment = deployment_meta.get(sample_identifier)
            if deployment is None:
                errors.append("membership references a missing deployment")
            else:
                scene_id, split, num_tx = deployment
                if str(row["scene_id"]) != scene_id or str(row["split"]) != split:
                    errors.append("membership scene/split disagrees with deployment")
                if int(row["num_tx"]) != num_tx:
                    errors.append("membership num_tx disagrees with deployment")
                if path.parent.name != split:
                    errors.append("membership file partition disagrees with split")
            if str(row["contract_id"]) != str(contract.get("contract_id")):
                errors.append("membership contract_id disagrees with contract")
            if len(row["subset_indices_in_family"]) != int(row["num_tx"]):
                errors.append("membership subset length disagrees with num_tx")
            family_rows.setdefault(family_identifier, []).append(row)
            if errors:
                membership_invalid.append(
                    {
                        "file": str(path),
                        "row": row_index,
                        "membership_id": membership_id,
                        "errors": errors,
                    }
                )

    plan_path = root / "generation_plan.json"
    expected_family_counts: dict[str, int] = {}
    if plan_path.exists():
        generation_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        base_subsets = int(generation_plan.get("subsets_per_cardinality", 1))
        hypergraph_counts = generation_plan.get("hypergraph_subsets_per_family", {})
        for family_identifier, rows in family_rows.items():
            scene_id = str(rows[0]["scene_id"])
            profile = contract.get("scene_generation_profiles", {}).get(scene_id)
            if profile is None:
                membership_invalid.append(
                    {"family_id": family_identifier, "errors": ["missing scene generation profile"]}
                )
                continue
            master_size = int(profile["master_pool_size"])
            expected = sum(
                min(
                    max(base_subsets, int(hypergraph_counts.get(str(num_tx), 0))),
                    math.comb(master_size, int(num_tx)),
                )
                for num_tx in profile["num_tx_values"]
            )
            expected_family_counts[family_identifier] = expected
            expected_by_cardinality = {
                int(num_tx): min(
                    max(base_subsets, int(hypergraph_counts.get(str(num_tx), 0))),
                    math.comb(master_size, int(num_tx)),
                )
                for num_tx in profile["num_tx_values"]
            }
            actual_by_cardinality: dict[int, int] = {}
            for row in rows:
                cardinality = int(row["num_tx"])
                actual_by_cardinality[cardinality] = actual_by_cardinality.get(cardinality, 0) + 1
            if len(rows) != expected or actual_by_cardinality != expected_by_cardinality:
                membership_invalid.append(
                    {
                        "family_id": family_identifier,
                        "errors": [
                            "incomplete family: "
                            f"memberships={len(rows)}, expected={expected}, "
                            f"by_cardinality={actual_by_cardinality}, "
                            f"expected_by_cardinality={expected_by_cardinality}"
                        ],
                    }
                )
    elif family_rows:
        membership_invalid.append(
            {"file": str(plan_path), "errors": ["generation plan required for family completeness QA"]}
        )
    orphaned = seen.difference(referenced_samples)
    if orphaned:
        membership_invalid.append(
            {"errors": [f"{len(orphaned)} deployments have no family membership"], "sample_ids": sorted(orphaned)[:20]}
        )
    check(
        "family_memberships_valid",
        not membership_invalid,
        f"memberships={membership_total}, families={len(family_rows)}, invalid={len(membership_invalid)}",
    )
    if max_rows is None and membership_files:
        check(
            "manifest_membership_count",
            membership_total == int(manifest.get("num_family_memberships", -1)),
            f"memberships={membership_total}",
        )
    return {
        "schema_version": "citydeploy.dataset-qa.v4",
        "dataset_id": manifest.get("dataset_id"),
        "passed": all(item["passed"] for item in checks),
        "rows_checked": total,
        "checks": checks,
        "invalid_rows": invalid[:100],
        "invalid_memberships": membership_invalid[:100],
    }


def write_distribution_qa(root: Path, output_path: Path) -> Path | None:
    """Write scene/cardinality label distributions for imbalance inspection."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((Path(root) / "data").glob("*/*.parquet"))
    if not files:
        return None
    columns = ["scene_id", "split", "num_tx", *COVERAGE_FIELDS]
    rows = []
    for path in files:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["split"]), int(row["num_tx"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (scene_id, split, num_tx), items in sorted(groups.items()):
        record: dict[str, Any] = {
            "scene_id": scene_id,
            "split": split,
            "num_tx": num_tx,
            "num_samples": len(items),
        }
        for field in COVERAGE_FIELDS:
            values = np.asarray([float(item[field]) for item in items], dtype=np.float64)
            record[f"{field}_min"] = float(values.min())
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_p50"] = float(np.median(values))
            record[f"{field}_max"] = float(values.max())
        output.append(record)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output), output_path, compression="zstd")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    report = validate_dataset_v4(args.dataset, args.max_rows)
    path = args.dataset.resolve() / "qa" / "qa_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.max_rows is None:
        write_distribution_qa(args.dataset, args.dataset.resolve() / "qa" / "distributions.parquet")
    print(json.dumps({"passed": report["passed"], "report": str(path)}))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

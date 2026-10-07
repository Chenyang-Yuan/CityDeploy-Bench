"""Schema, geographic splitting, and Parquet storage for CityDeploy Dataset V4."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DATASET_SCHEMA_VERSION = "citydeploy.dataset.v4"
CONTRACT_SCHEMA_VERSION = "citydeploy.dataset-contract.v4"
PLAN_SCHEMA_VERSION = "citydeploy.dataset-plan.v4"
SPLIT_SCHEMA_VERSION = "citydeploy.geographic-split.v1"

METRIC_PREFIXES = ("pathloss", "ss_rsrp", "sinr", "effective_throughput")
COVERAGE_FIELDS = tuple(f"{name}_coverage" for name in METRIC_PREFIXES) + (
    "joint_4metric_coverage",
)
COUNT_FIELDS = tuple(f"{name}_covered_count" for name in METRIC_PREFIXES) + (
    "joint_covered_count",
)
STAT_FIELDS = tuple(
    f"{metric}_{statistic}"
    for metric in METRIC_PREFIXES
    for statistic in ("mean", "p05", "p50", "p95")
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(contract_id: str, scene_id: str, candidate_indices: Iterable[int]) -> str:
    canonical = {
        "contract_id": str(contract_id),
        "scene_id": str(scene_id),
        "candidate_indices": sorted(int(value) for value in candidate_indices),
    }
    return canonical_json_sha256(canonical)[:24]


def family_id(contract_id: str, scene_id: str, pool_index: int, seed: int) -> str:
    return canonical_json_sha256(
        {
            "contract_id": contract_id,
            "scene_id": scene_id,
            "pool_index": int(pool_index),
            "seed": int(seed),
        }
    )[:24]


def _bbox_overlaps(a: dict, b: dict) -> bool:
    return not (
        float(a["max_lon"]) <= float(b["min_lon"])
        or float(b["max_lon"]) <= float(a["min_lon"])
        or float(a["max_lat"]) <= float(b["min_lat"])
        or float(b["max_lat"]) <= float(a["min_lat"])
    )


def geographic_groups(scene_metadata: dict[str, dict]) -> dict[str, str]:
    """Group all directly or transitively overlapping scene bounding boxes."""
    scene_ids = sorted(scene_metadata)
    parent = {scene_id: scene_id for scene_id in scene_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(scene_ids):
        left_bbox = scene_metadata[left].get("source_bbox_wgs84")
        if not left_bbox:
            continue
        for right in scene_ids[index + 1 :]:
            right_bbox = scene_metadata[right].get("source_bbox_wgs84")
            if right_bbox and _bbox_overlaps(left_bbox, right_bbox):
                union(left, right)
    roots = {scene_id: find(scene_id) for scene_id in scene_ids}
    root_order = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    return {
        scene_id: f"geo-{root_order[root]:03d}"
        for scene_id, root in roots.items()
    }


def build_split_manifest(
    scene_metadata: dict[str, dict],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    scene_assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0, abs_tol=1e-8):
        raise ValueError("split fractions must sum to one")
    groups = geographic_groups(scene_metadata)
    grouped: dict[str, list[str]] = {}
    for scene_id, group_id in groups.items():
        grouped.setdefault(group_id, []).append(scene_id)
    group_ids = sorted(grouped)
    if len(group_ids) < 3:
        raise RuntimeError("at least three non-overlapping geographic groups are required")

    valid_splits = {"train", "validation", "test"}
    assignments: dict[str, str] = {}
    if scene_assignments is not None:
        if set(scene_assignments) != set(scene_metadata):
            missing = sorted(set(scene_metadata).difference(scene_assignments))
            extra = sorted(set(scene_assignments).difference(scene_metadata))
            raise ValueError(f"explicit split assignments disagree with scenes: missing={missing}, extra={extra}")
        invalid = sorted({str(value) for value in scene_assignments.values()}.difference(valid_splits))
        if invalid:
            raise ValueError(f"invalid explicit split names: {invalid}")
        for group_id, scene_ids in grouped.items():
            selected = {str(scene_assignments[scene_id]) for scene_id in scene_ids}
            if len(selected) != 1:
                raise ValueError(
                    f"overlapping geographic group {group_id} crosses explicit splits: "
                    f"{sorted(scene_ids)} -> {sorted(selected)}"
                )
            assignments[group_id] = selected.pop()
        if set(assignments.values()) != valid_splits:
            raise ValueError("explicit assignments must populate train, validation, and test")
    else:
        random.Random(seed).shuffle(group_ids)
        targets = {
            "train": train_fraction * len(scene_metadata),
            "validation": validation_fraction * len(scene_metadata),
            "test": test_fraction * len(scene_metadata),
        }
        counts = {name: 0 for name in targets}
        # Seed every split, then greedily minimize normalized target overfill.
        initial = ("validation", "test", "train")
        for group_id, split in zip(group_ids[:3], initial):
            assignments[group_id] = split
            counts[split] += len(grouped[group_id])
        for group_id in group_ids[3:]:
            split = min(
                targets,
                key=lambda name: (counts[name] / max(targets[name], 1e-9), name),
            )
            assignments[group_id] = split
            counts[split] += len(grouped[group_id])
    scenes = {
        scene_id: {
            "geographic_group_id": group_id,
            "split": assignments[group_id],
        }
        for scene_id, group_id in groups.items()
    }
    result = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "unit": "overlapping_geographic_group",
        "stratification": ["city", "map_size_m"] if scene_assignments is not None else [],
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "groups": {
            group_id: {"split": assignments[group_id], "scenes": sorted(scene_ids)}
            for group_id, scene_ids in sorted(grouped.items())
        },
        "scenes": scenes,
    }
    result["sha256"] = canonical_json_sha256(result)
    return result


def parquet_schema():
    import pyarrow as pa

    fields = [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("scene_id", pa.string(), nullable=False),
        pa.field("geographic_group_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("contract_id", pa.string(), nullable=False),
        pa.field("family_id", pa.string(), nullable=False),
        pa.field("parent_sample_id", pa.string()),
        pa.field("family_role", pa.string(), nullable=False),
        pa.field("subset_indices_in_family", pa.list_(pa.int16()), nullable=False),
        pa.field("sampling_strategy", pa.string(), nullable=False),
        pa.field("simulation_seed", pa.int64(), nullable=False),
        pa.field("num_tx", pa.int16(), nullable=False),
        pa.field("tx_density_per_km2", pa.float32(), nullable=False),
        pa.field("tx_candidate_indices", pa.list_(pa.int32()), nullable=False),
        pa.field("tx_xyz_m", pa.list_(pa.list_(pa.float32(), 3)), nullable=False),
        pa.field("tx_xy_norm01", pa.list_(pa.list_(pa.float32(), 2)), nullable=False),
        pa.field("evaluation_cell_count", pa.int32(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.int32(), nullable=False) for name in COUNT_FIELDS)
    fields.extend(pa.field(name, pa.float32(), nullable=False) for name in COVERAGE_FIELDS)
    fields.append(pa.field("joint_target_met", pa.bool_(), nullable=False))
    fields.extend(pa.field(name, pa.float32()) for name in STAT_FIELDS)
    fields.extend(
        [
            pa.field("runtime_seconds", pa.float32(), nullable=False),
            pa.field("backend", pa.string(), nullable=False),
            pa.field("rich_map_path", pa.string()),
            pa.field("valid", pa.bool_(), nullable=False),
            pa.field("warning_flags", pa.list_(pa.string()), nullable=False),
        ]
    )
    return pa.schema(fields)


def family_membership_schema():
    """Many-to-many links between unique deployments and sampled TX families."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("membership_id", pa.string(), nullable=False),
            pa.field("family_id", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("scene_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("contract_id", pa.string(), nullable=False),
            pa.field("pool_index", pa.int32(), nullable=False),
            pa.field("master_sample_id", pa.string(), nullable=False),
            pa.field("family_role", pa.string(), nullable=False),
            pa.field("subset_indices_in_family", pa.list_(pa.int16()), nullable=False),
            pa.field("sampling_strategy", pa.string(), nullable=False),
            pa.field("simulation_seed", pa.int64(), nullable=False),
            pa.field("num_tx", pa.int16(), nullable=False),
        ]
    )


def family_membership_id(family_identifier: str, sample_identifier: str) -> str:
    return canonical_json_sha256(
        {"family_id": str(family_identifier), "sample_id": str(sample_identifier)}
    )[:24]


@dataclass
class ParquetShardWriter:
    root: Path
    rows_per_shard: int
    row_group_size: int
    compression: str = "zstd"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        existing = sorted(self.root.glob("part-*.parquet"))
        self.shard_index = 0 if not existing else int(existing[-1].stem.split("-")[-1]) + 1

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root / f"part-{self.shard_index:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.rows, schema=parquet_schema())
        options = {
            "compression": self.compression,
            "row_group_size": self.row_group_size,
            "write_page_index": True,
        }
        try:
            pq.write_table(table, temporary, use_content_defined_chunking=True, **options)
        except TypeError:
            pq.write_table(table, temporary, **options)
        temporary.replace(path)
        self.rows.clear()
        self.shard_index += 1
        return path


@dataclass
class FamilyMembershipShardWriter:
    root: Path
    rows_per_shard: int
    row_group_size: int
    compression: str = "zstd"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        existing = sorted(self.root.glob("part-*.parquet"))
        self.shard_index = 0 if not existing else int(existing[-1].stem.split("-")[-1]) + 1

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        if not self.rows:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root / f"part-{self.shard_index:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.rows, schema=family_membership_schema())
        options = {
            "compression": self.compression,
            "row_group_size": self.row_group_size,
            "write_page_index": True,
        }
        try:
            pq.write_table(table, temporary, use_content_defined_chunking=True, **options)
        except TypeError:
            pq.write_table(table, temporary, **options)
        temporary.replace(path)
        self.rows.clear()
        self.shard_index += 1
        return path


def existing_sample_ids(data_root: Path) -> set[str]:
    import pyarrow.parquet as pq

    result: set[str] = set()
    for path in sorted(Path(data_root).glob("*/*.parquet")):
        result.update(str(value) for value in pq.read_table(path, columns=["sample_id"])["sample_id"].to_pylist())
    return result


def existing_family_ids(data_root: Path) -> set[str]:
    import pyarrow.parquet as pq

    result: set[str] = set()
    for path in sorted(Path(data_root).glob("*/*.parquet")):
        result.update(str(value) for value in pq.read_table(path, columns=["family_id"])["family_id"].to_pylist())
    return result


def existing_membership_ids(membership_root: Path) -> set[str]:
    import pyarrow.parquet as pq

    result: set[str] = set()
    for path in sorted(Path(membership_root).glob("*/*.parquet")):
        result.update(
            str(value)
            for value in pq.read_table(path, columns=["membership_id"])["membership_id"].to_pylist()
        )
    return result


def existing_membership_family_ids(membership_root: Path) -> set[str]:
    import pyarrow.parquet as pq

    result: set[str] = set()
    for path in sorted(Path(membership_root).glob("*/*.parquet")):
        result.update(
            str(value)
            for value in pq.read_table(path, columns=["family_id"])["family_id"].to_pylist()
        )
    return result


def existing_membership_family_counts(membership_root: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    result: dict[str, int] = {}
    for path in sorted(Path(membership_root).glob("*/*.parquet")):
        for value in pq.read_table(path, columns=["family_id"])["family_id"].to_pylist():
            identifier = str(value)
            result[identifier] = result.get(identifier, 0) + 1
    return result

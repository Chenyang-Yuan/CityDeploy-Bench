#!/usr/bin/env python3
"""Safely repartition an initialized V4 dataset without rerunning ray tracing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

from dataset_builder.storage import ParquetShardWriter, build_split_manifest
from dataset_builder.validate import validate_dataset_v4, write_distribution_qa


PROJECT_ROOT = workspace_root()
SPLITS = ("train", "validation", "test")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_manifest(plan: dict) -> tuple[Path, dict]:
    scenes_root = (PROJECT_ROOT / plan["scenes_root"]).resolve()
    configured = [str(value) for value in plan["scenes"]]
    metadata = {
        scene_id: _load(scenes_root / scene_id / "inputs" / "metadata.json")
        for scene_id in configured
    }
    split = plan["split"]
    manifest = build_split_manifest(
        metadata,
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
        test_fraction=float(split["test_fraction"]),
        seed=int(split["seed"]),
        scene_assignments=split.get("scene_assignments"),
    )
    return (PROJECT_ROOT / plan["output_root"]).resolve(), manifest


def _remove_exact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def repartition(plan_path: Path, apply: bool) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    plan_path = plan_path.resolve()
    plan = _load(plan_path)
    root, new_split = _new_manifest(plan)
    old_split_path = root / "split_manifest.json"
    if not old_split_path.exists():
        raise FileNotFoundError(f"dataset is not initialized: {root}")
    old_split = _load(old_split_path)
    changed_scenes = sorted(
        scene_id
        for scene_id, item in new_split["scenes"].items()
        if old_split.get("scenes", {}).get(scene_id, {}).get("split") != item["split"]
    )
    parquet_paths = sorted((root / "data").glob("*/*.parquet"))
    rows = [row for path in parquet_paths for row in pq.read_table(path).to_pylist()]
    summary = {
        "dataset": str(root),
        "rows": len(rows),
        "changed_scenes": len(changed_scenes),
        "scene_counts": dict(Counter(item["split"] for item in new_split["scenes"].values())),
        "row_counts": dict(Counter(new_split["scenes"][row["scene_id"]]["split"] for row in rows)),
        "apply": bool(apply),
    }
    if not changed_scenes:
        summary["status"] = "already_partitioned"
        return summary
    if not apply:
        summary["status"] = "dry_run"
        summary["changed_scene_ids"] = changed_scenes
        return summary

    stage = root / ".repartition_stage"
    backup = root / ".repartition_backup"
    if stage.exists() or backup.exists():
        raise RuntimeError("repartition staging or backup directory already exists; inspect it before retrying")
    if (root / "radio_maps" / "shards").exists() or (root / "radio_maps" / "index.parquet").exists():
        raise RuntimeError("repartition before packing radio-map shards")
    stage.mkdir()
    backup.mkdir()
    storage = plan["storage"]
    writers = {
        split: ParquetShardWriter(
            stage / "data" / split,
            int(storage["parquet_rows_per_shard"]),
            int(storage["parquet_row_group_size"]),
            str(storage["compression"]),
        )
        for split in SPLITS
    }
    try:
        for original in rows:
            row = dict(original)
            assignment = new_split["scenes"][str(row["scene_id"])]
            row["split"] = assignment["split"]
            row["geographic_group_id"] = assignment["geographic_group_id"]
            rich_path = row.get("rich_map_path")
            if rich_path:
                source = root / str(rich_path)
                if not source.exists():
                    raise FileNotFoundError(f"rich radio map is missing: {source}")
                relative = Path("radio_maps") / row["split"] / source.name
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                row["rich_map_path"] = relative.as_posix()
            writers[row["split"]].append(row)
        for writer in writers.values():
            writer.flush()

        scene_table_path = root / "scenes" / "scenes.parquet"
        scene_table = pq.read_table(scene_table_path)
        scene_rows = []
        for original in scene_table.to_pylist():
            row = dict(original)
            assignment = new_split["scenes"][str(row["scene_id"])]
            row["split"] = assignment["split"]
            row["geographic_group_id"] = assignment["geographic_group_id"]
            scene_rows.append(row)
        staged_scene_table = stage / "scenes" / "scenes.parquet"
        staged_scene_table.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(scene_rows, schema=scene_table.schema), staged_scene_table, compression="zstd")

        manifest = _load(root / "manifest.json")
        manifest["split_manifest_sha256"] = new_split["sha256"]
        (stage / "split_manifest.json").write_text(
            json.dumps(new_split, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.copy2(plan_path, stage / "generation_plan.json")

        relative_targets = (
            Path("data"),
            Path("radio_maps"),
            Path("split_manifest.json"),
            Path("manifest.json"),
            Path("generation_plan.json"),
            Path("scenes") / "scenes.parquet",
        )
        installed: list[Path] = []
        try:
            for relative in relative_targets:
                current = root / relative
                if current.exists():
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    current.replace(saved)
            for relative in relative_targets:
                replacement = stage / relative
                if replacement.exists():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    replacement.replace(target)
                    installed.append(target)
            report = validate_dataset_v4(root)
            if not report["passed"]:
                raise RuntimeError("repartitioned dataset failed validation")
        except Exception:
            for target in reversed(installed):
                _remove_exact(target)
            for relative in relative_targets:
                saved = backup / relative
                if saved.exists():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    saved.replace(target)
            raise

        qa_root = root / "qa"
        qa_root.mkdir(parents=True, exist_ok=True)
        (qa_root / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_distribution_qa(root, qa_root / "distributions.parquet")
        shutil.rmtree(backup)
        shutil.rmtree(stage)
        summary["status"] = "repartitioned"
        summary["qa_passed"] = True
        return summary
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if backup.exists() and not any(backup.iterdir()):
            backup.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=config_path("datasets/urban_multicity.json"),
    )
    parser.add_argument("--apply", action="store_true", help="Apply the validated repartition operation.")
    args = parser.parse_args()
    print(json.dumps(repartition(args.plan, args.apply), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

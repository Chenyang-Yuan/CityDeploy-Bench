#!/usr/bin/env python3
"""Validate a CityDeploy v3 scene package and emit a machine-readable QA report."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xml_extent(scene_xml: Path) -> tuple[float, float]:
    values = {
        item.get("name"): item.get("value")
        for item in ET.parse(scene_xml).getroot().findall("default")
    }
    return float(values["scenegen_bbox_width"]), float(values["scenegen_bbox_length"])


def validate_manifest(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_root = manifest_path.parent
    checks: list[dict] = []

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else package_root / path

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("schema", manifest.get("schema_version") == "citydeploy.scene.v3", str(manifest.get("schema_version")))
    paths = manifest["paths"]
    required = {
        "scene_xml": resolve(paths["scene_xml"]),
        "semantic_masks": resolve(paths["input_dir"]) / "semantic_masks.npz",
        "scene_inputs": resolve(paths["scene_inputs_npz"]),
        "metadata": resolve(paths["scene_inputs_metadata_json"]),
        "osm_snapshot": resolve(manifest["source"]["snapshot"]),
        "query_record": resolve(manifest["source"]["query_record"]),
        "material_profile": resolve(manifest["radio_material_profile"]["profile_path"]),
        "tx_masks": resolve(paths["tx_deployable_dir"]) / "masks.npz",
        "tx_points": resolve(paths["tx_deployable_dir"]) / "points_local_xy.npy",
        "tx_metadata": resolve(paths["tx_deployable_dir"]) / "metadata.json",
    }
    for name, path in required.items():
        check(f"exists:{name}", path.exists(), str(path))

    if all(path.exists() for path in required.values()):
        hash_keys = {
            "scene_xml": "scene_xml",
            "semantic_masks": "semantic_masks",
            "scene_inputs": "scene_inputs",
            "osm_snapshot": "osm_snapshot",
            "query_record": "query_record",
            "material_profile": "material_profile",
        }
        for name, manifest_key in hash_keys.items():
            actual = _sha256(required[name])
            expected = manifest.get("sha256", {}).get(manifest_key)
            check(f"sha256:{name}", actual == expected, f"expected={expected}, actual={actual}")

        width, height = _xml_extent(required["scene_xml"])
        expected_width = float(manifest["map_x_m"])
        expected_height = float(manifest["map_y_m"])
        check("extent:xml_width", abs(width - expected_width) <= 1e-6, f"{width} vs {expected_width}")
        check("extent:xml_height", abs(height - expected_height) <= 1e-6, f"{height} vs {expected_height}")

        xml_root = ET.parse(required["scene_xml"]).getroot()
        mesh_refs = [
            (required["scene_xml"].parent / str(node.get("value"))).resolve()
            for node in xml_root.findall(".//string[@name='filename']")
            if node.get("value")
        ]
        missing_meshes = [str(path) for path in mesh_refs if not path.exists()]
        check("mesh:references_present", bool(mesh_refs), f"references={len(mesh_refs)}")
        check("mesh:references_exist", not missing_meshes, f"missing={len(missing_meshes)}")

        metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
        check("metadata:schema", metadata.get("schema_version") == "citydeploy.scene.v3", str(metadata.get("schema_version")))
        check("metadata:scene", metadata.get("scene") == manifest.get("scene"), f"{metadata.get('scene')} vs {manifest.get('scene')}")
        check("extent:metadata_width", abs(float(metadata["map_x_m"]) - expected_width) <= 1e-6, str(metadata["map_x_m"]))
        check("extent:metadata_height", abs(float(metadata["map_y_m"]) - expected_height) <= 1e-6, str(metadata["map_y_m"]))

        with np.load(required["semantic_masks"]) as loaded:
            sem = {key: np.asarray(loaded[key]) for key in loaded.files}
        canonical_channels = [
            "building_mask",
            "road_mask",
            "green_mask",
            "water_mask",
            "obstacle_mask",
            "built_up_mask",
            "explicit_open_mask",
            "unknown_feature_mask",
            "background_mask",
            "service_evaluation_mask",
        ]
        missing = [key for key in canonical_channels if key not in sem]
        check("semantic:channels", not missing, f"missing={missing}")
        if not missing:
            shapes = {sem[key].shape for key in canonical_channels}
            check("semantic:shape", len(shapes) == 1, str(sorted(shapes)))
            primary = canonical_channels[:7]
            overlap = np.sum([(sem[key] > 0.5).astype(np.uint8) for key in primary], axis=0)
            check("semantic:primary_exclusive", int((overlap > 1).sum()) == 0, f"overlap_cells={int((overlap > 1).sum())}")

        with np.load(required["scene_inputs"]) as loaded:
            inputs = {key: np.asarray(loaded[key]) for key in loaded.files}
        input_missing = [key for key in ("building_occupancy", "tx_feasible_mask", "evaluation_mask") if key not in inputs]
        check("inputs:channels", not input_missing, f"missing={input_missing}")
        if not input_missing and not missing:
            tx = inputs["tx_feasible_mask"] > 0.5
            collision = tx & ((sem["building_mask"] > 0.5) | (sem["water_mask"] > 0.5))
            check("tx:no_building_or_water", int(collision.sum()) == 0, f"collision_cells={int(collision.sum())}")
            check("inputs:shape_matches_semantics", tx.shape == sem["building_mask"].shape, f"{tx.shape}")
            expected_shape = (int(metadata["grid_h"]), int(metadata["grid_w"]))
            check("inputs:shape_matches_metadata", tx.shape == expected_shape, f"{tx.shape} vs {expected_shape}")

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "citydeploy.scene-qa.v1",
        "scene": manifest.get("scene"),
        "validated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to scene manifest.json")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    report_path = args.report or args.manifest.parent / "qa" / "scene_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

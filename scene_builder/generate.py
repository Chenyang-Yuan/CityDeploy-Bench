#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import numpy as np
from pyproj import Transformer

PROJECT_ROOT = workspace_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_builder.scene_inputs import (  # noqa: E402
    TX_FEASIBILITY_POLICIES,
    build_training_scene_inputs,
    build_tx_deployable_space,
)
from scene_builder.geo import get_utm_epsg_code_from_gps  # noqa: E402
from scene_builder.material_profiles import load_material_profile  # noqa: E402
from scene_builder.naming import (  # noqa: E402
    allocate_location_scene_dir,
    reverse_geocode_place,
    user_place_identity,
)
from scene_builder.validation import validate_manifest  # noqa: E402

DEFAULT_WATER_MATERIAL_TYPE = "mat-itu_wet_ground"
DEFAULT_MATERIAL_PROFILE = config_path("scenes/urban_materials.json")
DEFAULT_OVERPASS_URLS = (
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
)


def _normalise_overpass_url(value: str) -> str:
    """Return the OSMnx base URL, accepting an interpreter URL as input."""
    url = str(value).strip().rstrip("/")
    if url.endswith("/interpreter"):
        url = url[: -len("/interpreter")]
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Invalid Overpass URL: {value!r}")
    return url


def _configured_overpass_urls(cli_values: list[str] | None) -> tuple[str, ...]:
    raw_values = list(cli_values or [])
    if not raw_values:
        env_value = os.environ.get("CityDeploy_OVERPASS_URLS", "")
        raw_values = [item for item in env_value.split(",") if item.strip()]
    if not raw_values:
        raw_values = list(DEFAULT_OVERPASS_URLS)
    # Preserve order while avoiding duplicate requests to the same endpoint.
    return tuple(dict.fromkeys(_normalise_overpass_url(item) for item in raw_values))


def _compute_bbox_metric_extent(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> dict:
    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)
    utm_crs = get_utm_epsg_code_from_gps(center_lon, center_lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    corner_lon = [min_lon, max_lon, max_lon, min_lon]
    corner_lat = [min_lat, min_lat, max_lat, max_lat]
    corner_xy = np.asarray(
        [to_utm.transform(lo, la) for lo, la in zip(corner_lon, corner_lat)],
        dtype=np.float64,
    )
    xmin_u = float(corner_xy[:, 0].min())
    ymin_u = float(corner_xy[:, 1].min())
    xmax_u = float(corner_xy[:, 0].max())
    ymax_u = float(corner_xy[:, 1].max())
    return {
        "center_lon": float(center_lon),
        "center_lat": float(center_lat),
        "utm_crs": str(utm_crs),
        "bbox_utm": {
            "min_x": xmin_u,
            "min_y": ymin_u,
            "max_x": xmax_u,
            "max_y": ymax_u,
        },
        "map_x_m": float(xmax_u - xmin_u),
        "map_y_m": float(ymax_u - ymin_u),
    }


def _compute_fixed_metric_frame(
    center_lon: float,
    center_lat: float,
    width_m: float,
    height_m: float,
) -> tuple[list[tuple[float, float]], dict]:
    """Build an exact axis-aligned local UTM frame and its WGS84 polygon."""
    utm_crs = get_utm_epsg_code_from_gps(center_lon, center_lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_gps = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    min_x, max_x = cx - width_m / 2.0, cx + width_m / 2.0
    min_y, max_y = cy - height_m / 2.0, cy + height_m / 2.0
    points = [
        to_gps.transform(min_x, min_y),
        to_gps.transform(max_x, min_y),
        to_gps.transform(max_x, max_y),
        to_gps.transform(min_x, max_y),
    ]
    return [(float(lon), float(lat)) for lon, lat in points], {
        "center_lon": float(center_lon),
        "center_lat": float(center_lat),
        "utm_crs": str(utm_crs),
        "bbox_utm": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
        },
        "map_x_m": float(width_m),
        "map_y_m": float(height_m),
    }


def _bbox_to_polygon_points(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> list[tuple[float, float]]:
    # Keep the input bbox itself as the scene boundary in WGS84.
    return [
        (float(min_lon), float(min_lat)),
        (float(max_lon), float(min_lat)),
        (float(max_lon), float(max_lat)),
        (float(min_lon), float(max_lat)),
    ]


def _generate_scene_assets_exact_bbox(
    polygon_points: list[tuple[float, float]],
    output_dir: Path,
    materials: dict[str, str],
    overpass_urls: tuple[str, ...] = DEFAULT_OVERPASS_URLS,
    attempts_per_endpoint: int = 2,
    request_timeout_seconds: int = 90,
) -> tuple[bool, str]:
    from scene_builder.mesh_builder import Scene  # type: ignore

    attempt_dir = output_dir / ".scene_build_attempt"
    errors: list[str] = []
    total_attempts = len(overpass_urls) * attempts_per_endpoint
    attempt_number = 0
    try:
        for round_number in range(1, attempts_per_endpoint + 1):
            for endpoint in overpass_urls:
                attempt_number += 1
                if attempt_dir.exists():
                    shutil.rmtree(attempt_dir)
                attempt_dir.mkdir(parents=True, exist_ok=False)
                print(
                    f"[INFO] OSM/Overpass attempt {attempt_number}/{total_attempts}: "
                    f"{endpoint} (timeout={request_timeout_seconds}s)"
                )
                try:
                    Scene()(
                        points=polygon_points,
                        data_dir=str(attempt_dir),
                        hag_tiff_path=None,
                        osm_server_addr=endpoint,
                        osm_requests_timeout=request_timeout_seconds,
                        osm_use_cache=True,
                        lidar_calibration=False,
                        generate_building_map=False,
                        ground_scale=1.0,
                        ground_material_type=materials["ground"],
                        empty_land_material_type=materials["empty_land"],
                        road_material_type=materials["road"],
                        water_material_type=materials["water"],
                        green_material_type=materials["green"],
                        obstacle_material_type=materials["obstacle"],
                        rooftop_material_type=materials["rooftop"],
                        wall_material_type=materials["wall"],
                        lidar_terrain=False,
                        dem_terrain=False,
                    )
                    for child in attempt_dir.iterdir():
                        destination = output_dir / child.name
                        if destination.exists():
                            if destination.is_dir():
                                shutil.rmtree(destination)
                            else:
                                destination.unlink()
                        shutil.move(str(child), str(destination))
                    attempt_dir.rmdir()
                    return True, ""
                except Exception as exc:
                    error = f"{endpoint} round {round_number}: {exc}"
                    errors.append(error)
                    print(f"[WARN] OSM/Overpass attempt failed: {error}")
                    if attempt_number < total_attempts:
                        time.sleep(min(2 * attempt_number, 8))
    finally:
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
    return False, "All configured Overpass attempts failed. " + " | ".join(errors)


def _quarantine_failed_scene(scene_dir: Path, save_root: Path, reason: str) -> Path:
    """Move an incomplete package aside, preserving it for diagnosis/recovery."""
    failed_root = save_root.parent / "staging" / "failed_scenes"
    failed_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = failed_root / f"{scene_dir.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = failed_root / f"{scene_dir.name}_{stamp}_{suffix}"
        suffix += 1
    (scene_dir / "failure.json").write_text(
        json.dumps(
            {
                "scene_id": scene_dir.name,
                "failed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": reason,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    shutil.move(str(scene_dir), str(destination))
    return destination


def _patch_scene_xml_extent(scene_xml: Path, map_x_m: float, map_y_m: float) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    found_x = False
    found_y = False
    for default in root.findall("default"):
        name = default.get("name", "")
        if name == "scenegen_bbox_width":
            default.set("value", f"{map_x_m:.12f}")
            found_x = True
        elif name == "scenegen_bbox_length":
            default.set("value", f"{map_y_m:.12f}")
            found_y = True
    if not found_x:
        ET.SubElement(root, "default", name="scenegen_bbox_width", value=f"{map_x_m:.12f}")
    if not found_y:
        ET.SubElement(root, "default", name="scenegen_bbox_length", value=f"{map_y_m:.12f}")
    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)


def _patch_summary_extent(summary_json: Path, extent_info: dict, grid_res_m: float) -> None:
    if not summary_json.exists():
        return
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    semantic = summary.setdefault("semantic_raster", {})
    semantic["grid_mode"] = "auto_from_bbox"
    semantic["grid_res_m"] = float(grid_res_m)
    semantic["map_x_m_local"] = float(extent_info["map_x_m"])
    semantic["map_y_m_local"] = float(extent_info["map_y_m"])
    summary["bbox_wgs84"] = {
        "min_lon": float(extent_info["bbox_wgs84"]["min_lon"]),
        "min_lat": float(extent_info["bbox_wgs84"]["min_lat"]),
        "max_lon": float(extent_info["bbox_wgs84"]["max_lon"]),
        "max_lat": float(extent_info["bbox_wgs84"]["max_lat"]),
    }
    summary["center_wgs84"] = {
        "lon": float(extent_info["center_lon"]),
        "lat": float(extent_info["center_lat"]),
    }
    summary["utm_crs"] = str(extent_info["utm_crs"])
    summary.setdefault("osm_source", {})["path"] = "../source/osm_features.geojson"
    summary["output_files"] = {
        "semantic_masks_npz": "semantic_masks.npz",
        "semantic_masks_check_png": "../visualizations/semantic_masks_check.png",
        "semantic_overlay_png": "../visualizations/semantic_overlay.png",
        "semantic_overlay_pdf": "../visualizations/semantic_overlay.pdf",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _patch_scene_inputs_metadata(
    metadata_json: Path,
    scene_id: str,
    extent_info: dict,
    grid_h: int | None = None,
    grid_w: int | None = None,
    place_identity: dict | None = None,
) -> None:
    meta = json.loads(metadata_json.read_text(encoding="utf-8"))
    meta["scene"] = scene_id
    meta["map_x_m"] = float(extent_info["map_x_m"])
    meta["map_y_m"] = float(extent_info["map_y_m"])
    if grid_h is not None:
        meta["grid_h"] = int(grid_h)
    if grid_w is not None:
        meta["grid_w"] = int(grid_w)
    meta["source_bbox_wgs84"] = {
        "min_lon": float(extent_info["bbox_wgs84"]["min_lon"]),
        "min_lat": float(extent_info["bbox_wgs84"]["min_lat"]),
        "max_lon": float(extent_info["bbox_wgs84"]["max_lon"]),
        "max_lat": float(extent_info["bbox_wgs84"]["max_lat"]),
    }
    meta["center_wgs84"] = {
        "lon": float(extent_info["center_lon"]),
        "lat": float(extent_info["center_lat"]),
    }
    meta["utm_crs"] = str(extent_info["utm_crs"])
    if place_identity is not None:
        meta["location"] = place_identity
    meta["consistency_note"] = (
        "This metadata is unified to the exact input bbox metric extent. "
        "scene.xml, summary.json, scene_inputs.npz and tx deployable data should use the same map_x_m/map_y_m."
    )
    metadata_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _organize_visualizations(inputs_dir: Path, visualizations_dir: Path) -> None:
    """Keep derived images out of numerical inputs without losing QA previews."""
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf", "*.svg"):
        for source in inputs_dir.rglob(pattern):
            target = visualizations_dir / source.name
            if target.exists():
                target = visualizations_dir / f"{source.parent.name}_{source.name}"
            source.replace(target)


def _write_unified_manifest(
    save_root: Path,
    scene_id: str,
    scene_dir: Path,
    input_dir: Path,
    extent_info: dict,
    grid_res_m: float,
    tx_feasibility_policy: str,
    material_profile: dict,
    place_identity: dict,
    parent_bbox: list[float] | None = None,
    requested_size_m: list[float] | None = None,
) -> None:
    def sha256_file(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    package_versions = {}
    for package in ("osmnx", "geopandas", "shapely", "pyproj", "rasterio", "sionna"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    source_snapshot = scene_dir / "source" / "osm_features.geojson"
    query_record = scene_dir / "source" / "query.json"
    material_profile_snapshot = scene_dir / "source" / "material_profile.json"
    scene_xml = scene_dir / "scene.xml"
    semantic_npz = input_dir / "semantic_masks.npz"
    scene_inputs_npz = input_dir / "scene_inputs.npz"
    xml_material_assignments = {}
    if scene_xml.exists():
        xml_root = ET.parse(scene_xml).getroot()
        for item in xml_root.findall("default"):
            name = item.get("name", "")
            if name.startswith("scenegen_") and name.endswith("_material"):
                xml_material_assignments[name.removeprefix("scenegen_").removesuffix("_material")] = item.get("value")
    manifest = {
        "schema_version": "citydeploy.scene.v3",
        "scene": scene_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scene_mode": "arbitrary_bbox",
        "location": place_identity,
        "bbox_wgs84": extent_info["bbox_wgs84"],
        "center_wgs84": {
            "lon": float(extent_info["center_lon"]),
            "lat": float(extent_info["center_lat"]),
        },
        "utm_crs": str(extent_info["utm_crs"]),
        "bbox_utm": extent_info["bbox_utm"],
        "map_x_m": float(extent_info["map_x_m"]),
        "map_y_m": float(extent_info["map_y_m"]),
        "grid_res_m": float(grid_res_m),
        "tx_feasibility_policy": tx_feasibility_policy,
        "height_model": {
            "priority": ["lidar_hag", "building:height", "height", "building:levels", "deterministic_type_prior"],
            "level_height_m": 3.2,
            "random_fallback": False,
        },
        "radio_material_profile": {
            "schema_version": material_profile.get("schema_version"),
            "name": material_profile.get("name"),
            "profile_path": "source/material_profile.json",
            "assignment_kind": "semantic_to_radio_material_prior",
            "materials": xml_material_assignments,
            "warning": "OSM semantics do not identify measured facade composition.",
        },
        "source": {
            "provider": "OpenStreetMap",
            "license": "ODbL-1.0",
            "attribution": "© OpenStreetMap contributors",
            "snapshot": "source/osm_features.geojson",
            "query_record": "source/query.json",
            "material_profile": "source/material_profile.json",
        },
        "software": {"python": platform.python_version(), "packages": package_versions},
        "sha256": {
            "osm_snapshot": sha256_file(source_snapshot),
            "query_record": sha256_file(query_record),
            "scene_xml": sha256_file(scene_xml),
            "semantic_masks": sha256_file(semantic_npz),
            "scene_inputs": sha256_file(scene_inputs_npz),
            "material_profile": sha256_file(material_profile_snapshot),
        },
        "paths": {
            "scene_dir": ".",
            "scene_xml": "scene.xml",
            "mesh_dir": "mesh",
            "input_dir": "inputs",
            "summary_json": "inputs/summary.json",
            "scene_inputs_npz": "inputs/scene_inputs.npz",
            "scene_inputs_metadata_json": "inputs/metadata.json",
            "tx_deployable_dir": "inputs/tx_deployable",
            "visualizations_dir": "visualizations",
        },
        "consistency_contract": [
            "Input bbox itself is the scene extent.",
            "No 512x512 sub-scene splitting is used.",
            "scene.xml scenegen_bbox_width/length are patched to exact bbox metric extent.",
            "inspect summary and scene_inputs metadata are patched to the same exact extent.",
            "TX deployable and future consumers should use map_x_m/map_y_m from this unified extent.",
        ],
    }
    if parent_bbox is not None:
        manifest["selection"] = {
            "parent_bbox_wgs84": {
                "min_lon": float(parent_bbox[0]),
                "min_lat": float(parent_bbox[1]),
                "max_lon": float(parent_bbox[2]),
                "max_lat": float(parent_bbox[3]),
            },
            "requested_size_m": (
                {"width": float(requested_size_m[0]), "height": float(requested_size_m[1])}
                if requested_size_m is not None
                else None
            ),
            "selection_mode": "fixed_metric_window" if requested_size_m is not None else "exact_parent_bbox",
        }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    (scene_dir / "manifest.json").write_text(payload, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a self-contained scene package directly from the exact input bbox. "
            "No 512m sub-scene split. No 3D viewer. "
            "scene.xml, inspect summary, scene_inputs and TX deployable metadata are unified to one metric extent."
        )
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Exact bbox in WGS84 (EPSG:4326). This bbox itself becomes the scene extent.",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scenes",
        help="Scene package root. Default: <project>/datasets/scenes",
    )
    parser.add_argument(
        "--grid-res-m",
        type=float,
        default=1.0,
        help="Meters per pixel for semantic raster auto grid. Default: 1.0",
    )
    parser.add_argument(
        "--max-raster-cells",
        type=int,
        default=4_194_304,
        help="Auto-grid memory guard; large arbitrary bboxes are coarsened to this cell budget.",
    )
    parser.add_argument(
        "--top-values",
        type=int,
        default=20,
        help="Forwarded to inspect_osm_features_bbox.py",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Forwarded to inspect_osm_features_bbox.py",
    )
    parser.add_argument(
        "--tx-feasibility-policy",
        choices=sorted(TX_FEASIBILITY_POLICIES),
        default="road_only",
        help="TX action-space policy. Default: road_only.",
    )
    parser.add_argument(
        "--material-profile",
        type=Path,
        default=DEFAULT_MATERIAL_PROFILE,
        help="Versioned semantic-to-radio-material profile JSON.",
    )
    parser.add_argument(
        "--place-name",
        type=str,
        default=None,
        help="Optional location-name override. Default: cached OSM reverse lookup at bbox center.",
    )
    parser.add_argument(
        "--parent-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Optional parent research bbox recorded by the interactive selector.",
    )
    parser.add_argument(
        "--requested-size-m",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional fixed-window size recorded by the interactive selector.",
    )
    parser.add_argument(
        "--local-frame-center",
        type=float,
        nargs=2,
        default=None,
        metavar=("LON", "LAT"),
        help="Center of an exact projected fixed-size frame; requires --requested-size-m.",
    )
    parser.add_argument(
        "--overpass-url",
        action="append",
        default=None,
        help=(
            "Overpass API base URL. Repeat to define failover order. "
            "Default: overpass-api.de, then overpass.kumi.systems."
        ),
    )
    parser.add_argument(
        "--overpass-attempts",
        type=int,
        default=2,
        help="Rounds across all configured Overpass endpoints. Default: 2.",
    )
    parser.add_argument(
        "--overpass-timeout-seconds",
        type=int,
        default=90,
        help="Timeout for each Overpass request. Default: 90.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_lon, min_lat, max_lon, max_lat = args.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid --bbox: require min < max for lon/lat.")
    if args.grid_res_m <= 0:
        raise ValueError("--grid-res-m must be > 0.")
    if args.requested_size_m is not None and any(value <= 0 for value in args.requested_size_m):
        raise ValueError("--requested-size-m values must be > 0.")
    if (args.requested_size_m is None) != (args.local_frame_center is None):
        raise ValueError("--requested-size-m and --local-frame-center must be provided together.")
    if args.overpass_attempts < 1:
        raise ValueError("--overpass-attempts must be >= 1.")
    if args.overpass_timeout_seconds < 1:
        raise ValueError("--overpass-timeout-seconds must be >= 1.")
    overpass_urls = _configured_overpass_urls(args.overpass_url)

    save_root = args.save_root.resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    material_profile = load_material_profile(args.material_profile)

    if args.requested_size_m is not None:
        polygon_points, extent_info = _compute_fixed_metric_frame(
            center_lon=float(args.local_frame_center[0]),
            center_lat=float(args.local_frame_center[1]),
            width_m=float(args.requested_size_m[0]),
            height_m=float(args.requested_size_m[1]),
        )
    else:
        polygon_points = _bbox_to_polygon_points(min_lon, min_lat, max_lon, max_lat)
        extent_info = _compute_bbox_metric_extent(min_lon, min_lat, max_lon, max_lat)
    extent_info["bbox_wgs84"] = {
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
    }
    if args.place_name:
        place_identity = user_place_identity(
            args.place_name,
            lon=float(extent_info["center_lon"]),
            lat=float(extent_info["center_lat"]),
        )
    else:
        place_identity = reverse_geocode_place(
            lon=float(extent_info["center_lon"]),
            lat=float(extent_info["center_lat"]),
            cache_path=save_root / "geocoding_cache.json",
        )
    scene_id, scene_dir, input_dir = allocate_location_scene_dir(
        save_root,
        place_identity["place_slug"],
    )

    print(f"[INFO] Save root: {save_root}")
    print(f"[INFO] Location: {place_identity['place_name']} ({place_identity['source']})")
    print(f"[INFO] Scene id: {scene_id}")
    print(
        f"[INFO] Exact bbox extent: "
        f"{min_lon} {min_lat} {max_lon} {max_lat}"
    )
    print(
        f"[INFO] Unified metric extent: "
        f"{extent_info['map_x_m']:.3f} x {extent_info['map_y_m']:.3f} m "
        f"(UTM: {extent_info['utm_crs']})"
    )

    ok_scene, scene_err = _generate_scene_assets_exact_bbox(
        polygon_points=polygon_points,
        output_dir=scene_dir,
        materials=material_profile["materials"],
        overpass_urls=overpass_urls,
        attempts_per_endpoint=int(args.overpass_attempts),
        request_timeout_seconds=int(args.overpass_timeout_seconds),
    )
    if not ok_scene:
        quarantined = _quarantine_failed_scene(scene_dir, save_root, scene_err)
        raise RuntimeError(
            f"Scene generation failed: {scene_err}\n"
            f"Incomplete package preserved at: {quarantined}"
        )
    print(f"[INFO] Scene assets written: {scene_dir}")
    material_profile_snapshot = scene_dir / "source" / "material_profile.json"
    material_profile_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(material_profile["profile_path"]), material_profile_snapshot)

    scene_xml = scene_dir / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"scene.xml not found: {scene_xml}")
    _patch_scene_xml_extent(
        scene_xml=scene_xml,
        map_x_m=float(extent_info["map_x_m"]),
        map_y_m=float(extent_info["map_y_m"]),
    )
    print("[INFO] scene.xml width/length patched to unified exact extent.")

    cmd = [
        sys.executable,
        "-m",
        "scene_builder.osm_semantics",
        "--bbox",
        str(min_lon),
        str(min_lat),
        str(max_lon),
        str(max_lat),
        "--save-dir",
        str(input_dir),
        "--osm-snapshot",
        str(scene_dir / "source" / "osm_features.geojson"),
        "--top-values",
        str(args.top_values),
        "--point-size",
        str(args.point_size),
        "--grid-res-m",
        str(args.grid_res_m),
        "--max-raster-cells",
        str(args.max_raster_cells),
        "--minimal-output",
    ]
    if args.requested_size_m is not None:
        cmd.extend(
            [
                "--local-frame-center",
                str(extent_info["center_lon"]),
                str(extent_info["center_lat"]),
                "--local-frame-size",
                str(extent_info["map_x_m"]),
                str(extent_info["map_y_m"]),
            ]
        )
    print("[INFO] Building inspect semantic data on exact bbox ...")
    result = subprocess.run(cmd, capture_output=False, check=False)
    if int(result.returncode) != 0:
        raise RuntimeError(f"Inspect semantic generation failed with exit code={result.returncode}")

    summary_json = input_dir / "summary.json"
    _patch_summary_extent(
        summary_json=summary_json,
        extent_info=extent_info,
        grid_res_m=float(args.grid_res_m),
    )

    ok_tx, tx_err, _ = build_tx_deployable_space(
        inputs_dir=input_dir,
        visualizations_dir=scene_dir / "visualizations",
        threshold=0.5,
        policy=args.tx_feasibility_policy,
    )
    if not ok_tx:
        raise RuntimeError(f"TX deployable generation failed: {tx_err}")

    ok_compat, compat_err, compat_stats = build_training_scene_inputs(
        scene_id=scene_id,
        inputs_dir=input_dir,
        threshold=0.5,
        tx_policy=args.tx_feasibility_policy,
    )
    if not ok_compat:
        raise RuntimeError(f"scene_inputs export failed: {compat_err}")

    metadata_json = input_dir / "metadata.json"
    _patch_scene_inputs_metadata(
        metadata_json=metadata_json,
        scene_id=scene_id,
        extent_info=extent_info,
        grid_h=int(compat_stats.get("grid_h", 0)) if compat_stats else None,
        grid_w=int(compat_stats.get("grid_w", 0)) if compat_stats else None,
        place_identity=place_identity,
    )
    _organize_visualizations(input_dir, scene_dir / "visualizations")

    _write_unified_manifest(
        save_root=save_root,
        scene_id=scene_id,
        scene_dir=scene_dir,
        input_dir=input_dir,
        extent_info=extent_info,
        grid_res_m=float(args.grid_res_m),
        tx_feasibility_policy=args.tx_feasibility_policy,
        material_profile=material_profile,
        place_identity=place_identity,
        parent_bbox=args.parent_bbox,
        requested_size_m=args.requested_size_m,
    )

    manifest_path = scene_dir / "manifest.json"
    qa_report = validate_manifest(manifest_path)
    qa_path = scene_dir / "qa" / "scene_report.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not qa_report["passed"]:
        failed = [item["name"] for item in qa_report["checks"] if not item["passed"]]
        raise RuntimeError(f"Scene package QA failed: {failed}. Report: {qa_path}")

    print(f"[INFO] Scene package inputs written: {input_dir}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] QA passed: {qa_path}")
    print("[INFO] This workflow does not open the 3D viewer.")


if __name__ == "__main__":
    main()

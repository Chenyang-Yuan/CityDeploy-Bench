#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

try:
    from .geo import get_utm_epsg_code_from_gps
    from .semantics import (
        OSM_QUERY_TAGS,
        SCHEMA_VERSION as SEMANTIC_SCHEMA_VERSION,
        highway_half_width_m,
        primary_semantic_class,
    )
except ImportError:
    import sys

    current_dir = Path(__file__).parent.absolute()
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from scene_builder.geo import get_utm_epsg_code_from_gps
    from scene_builder.semantics import (
        OSM_QUERY_TAGS,
        SCHEMA_VERSION as SEMANTIC_SCHEMA_VERSION,
        highway_half_width_m,
        primary_semantic_class,
    )


PROJECT_ROOT = workspace_root()


def _map_figure_size(map_x: float, map_y: float, base: float = 6.0) -> tuple[float, float]:
    aspect = max(float(map_x), 1e-6) / max(float(map_y), 1e-6)
    if aspect >= 1.0:
        return base * min(aspect, 2.2), base
    return base, base * min(1.0 / aspect, 2.2)


DEFAULT_TAGS = OSM_QUERY_TAGS


CATEGORY_PRIORITY = [
    "building",
    "highway",
    "railway",
    "waterway",
    "natural",
    "landuse",
    "leisure",
    "amenity",
    "man_made",
    "power",
    "barrier",
]


CATEGORY_COLORS = {
    "building": "#d73027",
    "highway": "#4575b4",
    "railway": "#7b3294",
    "water": "#2c7fb8",
    "natural": "#1a9850",
    "landuse": "#66bd63",
    "leisure": "#5aae61",
    "amenity": "#fdae61",
    "man_made": "#8c510a",
    "power": "#4d4d4d",
    "barrier": "#998ec3",
    "other": "#bdbdbd",
}

def coarse_layer(category: str) -> str:
    if category == "building":
        return "building"
    if category in {"highway", "railway"}:
        return "transport"
    if category == "water":
        return "water"
    if category in {"natural", "landuse", "leisure"}:
        return "green"
    return "other"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect what OSM features can be recognized in a bbox and visualize them."
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="Bounding box in WGS84 (EPSG:4326).",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/osm_inspection/<timestamp>",
    )
    p.add_argument(
        "--top-values",
        type=int,
        default=20,
        help="Top-N tag values per key in summary output.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Marker size for point geometries in plot.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Optional fixed raster grid size (H=W). If omitted, auto-match bbox size.",
    )
    p.add_argument(
        "--grid-res-m",
        type=float,
        default=1.0,
        help="Meters per pixel for auto grid mode (used when --grid-size is not set).",
    )
    p.add_argument(
        "--minimal-output",
        action="store_true",
        help=(
            "Only save files required by downstream batch pipeline "
            "(semantic_masks.npz, semantic_masks_check.png, summary.json)."
        ),
    )
    p.add_argument(
        "--max-raster-cells",
        type=int,
        default=4_194_304,
        help=(
            "Maximum H*W in auto-grid mode. Resolution is coarsened rather than silently "
            "allocating an unsafe raster. Default: 4194304."
        ),
    )
    p.add_argument(
        "--osm-snapshot",
        type=Path,
        default=None,
        help=(
            "Read the authoritative OSM GeoJSON/GPKG snapshot instead of querying Overpass. "
            "Recommended when a 3-D scene has already written source/osm_features.geojson."
        ),
    )
    p.add_argument(
        "--local-frame-center",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        help="Center of the authoritative projected scene frame.",
    )
    p.add_argument(
        "--local-frame-size",
        type=float,
        nargs=2,
        metavar=("WIDTH_M", "HEIGHT_M"),
        help="Exact metric size of the authoritative scene frame.",
    )
    return p


def validate_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> None:
    if min_lon >= max_lon:
        raise ValueError("Invalid bbox: min_lon must be < max_lon.")
    if min_lat >= max_lat:
        raise ValueError("Invalid bbox: min_lat must be < max_lat.")
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError("Longitude must be in [-180, 180].")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError("Latitude must be in [-90, 90].")


def classify_category(row: pd.Series) -> str:
    for k in CATEGORY_PRIORITY:
        if k in row and pd.notna(row[k]) and str(row[k]).strip() != "":
            if k == "natural" and str(row[k]).lower() in {"water", "wetland"}:
                return "water"
            if k == "waterway":
                return "water"
            return k
    return "other"


def build_value_summary(gdf: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for key in CATEGORY_PRIORITY:
        if key not in gdf.columns:
            continue
        vals = gdf[key].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if vals.empty:
            continue
        vc = vals.value_counts().head(top_n)
        for value, count in vc.items():
            rows.append({"tag_key": key, "tag_value": value, "count": int(count)})
    if not rows:
        return pd.DataFrame(columns=["tag_key", "tag_value", "count"])
    return pd.DataFrame(rows).sort_values(["tag_key", "count"], ascending=[True, False])


def _pick_uncategorized_label(row: pd.Series) -> str:
    key_order = [
        "landuse",
        "leisure",
        "amenity",
        "natural",
        "man_made",
        "power",
        "barrier",
        "building",
        "waterway",
        "highway",
        "railway",
    ]
    for key in key_order:
        raw = row.get(key, np.nan)
        if pd.isna(raw):
            continue
        val = str(raw).strip().lower()
        if val != "":
            return f"{key}={val}"
    return "other=untyped"


def _sanitize_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch in {"=", " ", "/", "\\"}:
            out.append("_")
        else:
            out.append("_")
    name = "".join(out).strip("_")
    return name if name else "untitled"


def _rasterize_geoms(geoms, h: int, w: int, extent_local: tuple[float, float, float, float]) -> np.ndarray:
    if not geoms:
        return np.zeros((h, w), dtype=np.float32)
    xmin, xmax, ymin, ymax = extent_local
    shapes = [(g, 1) for g in geoms if g is not None and (not g.is_empty) and g.is_valid]
    if not shapes:
        return np.zeros((h, w), dtype=np.float32)
    tfm = from_bounds(xmin, ymin, xmax, ymax, w, h)
    arr = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=tfm,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    # Rasterio row-0 is top; flip to row-0 is bottom.
    return np.flipud(arr).astype(np.float32)


def _plot_semantic_check(
    out_path: Path,
    building: np.ndarray,
    road: np.ndarray,
    green: np.ndarray,
    water: np.ndarray,
    obstacle: np.ndarray,
    built_up: np.ndarray,
    explicit_open: np.ndarray,
    unknown_feature: np.ndarray,
    map_x: float,
    map_y: float,
) -> None:
    h, w = building.shape
    extent_real = (-map_x / 2.0, map_x / 2.0, -map_y / 2.0, map_y / 2.0)

    # Build a crisp semantic overlay with explicit class colors:
    # background=white(other), green=green, water=blue, building=red, road=black, obstacle=purple.
    bg = np.array([1.00, 1.00, 1.00], dtype=np.float32)
    c_green = np.array([0.60, 0.86, 0.60], dtype=np.float32)
    c_water = np.array([0.25, 0.45, 0.90], dtype=np.float32)
    c_building = np.array([0.90, 0.20, 0.20], dtype=np.float32)
    c_road = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    c_obstacle = np.array([0.68, 0.20, 0.80], dtype=np.float32)
    c_built_up = np.array([0.96, 0.72, 0.54], dtype=np.float32)
    c_explicit_open = np.array([0.92, 0.90, 0.62], dtype=np.float32)
    c_unknown = np.array([0.70, 0.70, 0.70], dtype=np.float32)

    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:] = bg

    green_m = green > 0.5
    water_m = water > 0.5
    building_m = building > 0.5
    road_m = road > 0.5
    obstacle_m = obstacle > 0.5
    built_up_m = built_up > 0.5
    explicit_open_m = explicit_open > 0.5
    unknown_m = unknown_feature > 0.5

    overlay[unknown_m] = c_unknown
    overlay[explicit_open_m] = c_explicit_open
    overlay[built_up_m] = c_built_up
    overlay[green_m] = c_green
    overlay[water_m] = c_water
    overlay[building_m] = c_building
    overlay[road_m] = c_road
    overlay[obstacle_m] = c_obstacle

    overlay_png = out_path.with_name("semantic_overlay.png")
    overlay_pdf = out_path.with_name("semantic_overlay.pdf")
    fig_overlay, ax_overlay = plt.subplots(figsize=_map_figure_size(map_x, map_y, base=6.0))
    ax_overlay.imshow(overlay, origin="lower", extent=extent_real)
    ax_overlay.set_title("Semantic overlay")
    ax_overlay.set_xlabel("x_real (m)")
    ax_overlay.set_ylabel("y_real (m)")
    ax_overlay.set_aspect("equal")
    fig_overlay.tight_layout()
    fig_overlay.savefig(overlay_png, dpi=300, bbox_inches="tight")
    fig_overlay.savefig(overlay_pdf, bbox_inches="tight")
    plt.close(fig_overlay)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    ax = axes[0]
    ax.imshow(building, cmap="Greys", vmin=0, vmax=1, origin="lower", extent=extent_real)
    ax.set_title("Building mask")
    ax.set_xlabel("x_real (m)")
    ax.set_ylabel("y_real (m)")
    ax.set_aspect("equal")

    ax = axes[1]
    # Roads shown as black area on white, easier to read.
    road_vis = 1.0 - road
    ax.imshow(road_vis, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent_real)
    ax.set_title("Road mask (area)")
    ax.set_xlabel("x_real (m)")
    ax.set_ylabel("y_real (m)")
    ax.set_aspect("equal")

    ax = axes[2]
    ax.imshow(overlay, origin="lower", extent=extent_real)
    ax.set_title("Semantic overlay")
    ax.set_xlabel("x_real (m)")
    ax.set_ylabel("y_real (m)")
    ax.set_aspect("equal")

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=c_building, edgecolor="none", label="building"),
        Patch(facecolor=c_road, edgecolor="none", label="road"),
        Patch(facecolor=c_water, edgecolor="none", label="water"),
        Patch(facecolor=c_green, edgecolor="none", label="green"),
        Patch(facecolor=c_obstacle, edgecolor="none", label="obstacle"),
        Patch(facecolor=c_built_up, edgecolor="none", label="built-up context"),
        Patch(facecolor=c_explicit_open, edgecolor="none", label="explicit open"),
        Patch(facecolor=c_unknown, edgecolor="none", label="unknown feature"),
        Patch(facecolor=bg, edgecolor="#aaaaaa", label="unannotated background"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.9)

    plt.suptitle(
        f"OSM semantic raster check | Grid {h}x{w} | Local map {map_x:.1f}x{map_y:.1f} m\n"
        "Kept panels: building mask + road mask + clear semantic overlay."
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_single_mask_png(
    out_path: Path,
    mask: np.ndarray,
    map_x: float,
    map_y: float,
    title: str,
) -> None:
    extent_real = (-map_x / 2.0, map_x / 2.0, -map_y / 2.0, map_y / 2.0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent_real)
    ax.set_title(title)
    ax.set_xlabel("x_real (m)")
    ax.set_ylabel("y_real (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_uncategorized_labeled_map(
    out_path: Path,
    uncategorized_df,
    map_x: float,
    map_y: float,
) -> None:
    extent_real = (-map_x / 2.0, map_x / 2.0, -map_y / 2.0, map_y / 2.0)
    fig, ax = plt.subplots(figsize=(11.5, 9.5))

    if uncategorized_df.empty:
        ax.set_xlim(extent_real[0], extent_real[1])
        ax.set_ylim(extent_real[2], extent_real[3])
        ax.set_aspect("equal")
        ax.set_title("Uncategorized Features (none)")
        ax.set_xlabel("x_real (m)")
        ax.set_ylabel("y_real (m)")
        ax.text(
            0.5 * (extent_real[0] + extent_real[1]),
            0.5 * (extent_real[2] + extent_real[3]),
            "No uncategorized features.",
            ha="center",
            va="center",
            fontsize=12,
            color="#333333",
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    labels = sorted(uncategorized_df["uncategorized_label"].dropna().astype(str).unique().tolist())
    n = max(1, len(labels))
    cmap = plt.get_cmap("tab20", n)
    # Use hex color strings for geopandas/matplotlib compatibility across versions.
    color_map = {label: "#{:02x}{:02x}{:02x}".format(*[int(round(v * 255)) for v in cmap(i)[:3]]) for i, label in enumerate(labels)}

    for label in labels:
        sub = uncategorized_df[uncategorized_df["uncategorized_label"] == label]
        sub.plot(
            ax=ax,
            color=color_map[label],
            linewidth=1.0,
            markersize=10.0,
            alpha=0.75,
            label=label,
        )
        rp = sub.geometry.unary_union.representative_point()
        ax.text(
            float(rp.x),
            float(rp.y),
            label,
            fontsize=7.5,
            color="#111111",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
        )

    ax.set_xlim(extent_real[0], extent_real[1])
    ax.set_ylim(extent_real[2], extent_real[3])
    ax.set_aspect("equal")
    ax.set_title("Uncategorized Features (Labeled)")
    ax.set_xlabel("x_real (m)")
    ax.set_ylabel("y_real (m)")
    if len(labels) <= 18:
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if (args.local_frame_center is None) != (args.local_frame_size is None):
        raise ValueError("--local-frame-center and --local-frame-size must be provided together")
    if args.local_frame_size is not None and min(args.local_frame_size) <= 0:
        raise ValueError("--local-frame-size values must be positive")
    min_lon, min_lat, max_lon, max_lat = args.bbox
    validate_bbox(min_lon, min_lat, max_lon, max_lat)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.save_dir
        if args.save_dir is not None
        else PROJECT_ROOT / "outputs" / "osm_inspection" / f"osm_inspect_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    minimal_output = bool(args.minimal_output)

    # Match the existing project convention used in scene generation.
    bbox_tuple = (min_lon, min_lat, max_lon, max_lat)
    if args.osm_snapshot is not None:
        if not args.osm_snapshot.exists():
            raise FileNotFoundError(f"OSM snapshot not found: {args.osm_snapshot}")
        gdf = gpd.read_file(args.osm_snapshot)
        osm_source = {"mode": "snapshot", "path": str(args.osm_snapshot.resolve())}
    else:
        gdf = ox.features.features_from_bbox(bbox=bbox_tuple, tags=DEFAULT_TAGS)
        osm_source = {
            "mode": "live_overpass",
            "overpass_url": str(ox.settings.overpass_url),
            "warning": "Live query is not frozen; retain the exported source snapshot for reproducibility.",
        }
    if gdf.empty:
        raise RuntimeError("OSM query returned no features in this bbox.")

    # Normalize to WGS84 then project to local UTM for area/length stats.
    gdf = gdf.to_crs("EPSG:4326")
    # Strictly keep geometries inside the selected bbox.
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
    gdf["geometry"] = gdf.geometry.intersection(bbox_geom)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise RuntimeError("No OSM features remain after clipping to bbox.")

    if args.local_frame_center is not None:
        center_lon, center_lat = map(float, args.local_frame_center)
    else:
        center_lon = 0.5 * (min_lon + max_lon)
        center_lat = 0.5 * (min_lat + max_lat)
    utm_crs = get_utm_epsg_code_from_gps(center_lon, center_lat)
    gdf_m = gdf.to_crs(utm_crs)

    gdf = gdf.copy()
    gdf["category"] = gdf.apply(classify_category, axis=1)
    gdf["coarse_layer"] = gdf["category"].map(coarse_layer)
    gdf["geom_type"] = gdf.geometry.geom_type
    gdf_m = gdf_m.copy()
    gdf_m["category"] = gdf["category"].values
    gdf_m["coarse_layer"] = gdf["coarse_layer"].values

    # Metric stats
    is_poly = gdf_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    is_line = gdf_m.geometry.geom_type.isin(["LineString", "MultiLineString"])
    gdf_m["area_m2"] = np.where(is_poly, gdf_m.geometry.area, 0.0)
    gdf_m["length_m"] = np.where(is_line, gdf_m.geometry.length, 0.0)

    raw_geojson: Path | None = None
    recognized_map_png: Path | None = None
    recognized_map_simple_png: Path | None = None
    summary_by_category_csv: Path | None = None
    summary_by_tag_values_csv: Path | None = None

    # Draw bbox boundary coordinates are reused by optional plots.
    xs = [min_lon, max_lon, max_lon, min_lon, min_lon]
    ys = [min_lat, min_lat, max_lat, max_lat, min_lat]

    if not minimal_output:
        # Save raw features as GeoJSON for inspection in GIS tools.
        raw_geojson = out_dir / "osm_features.geojson"
        gdf.to_file(raw_geojson, driver="GeoJSON")

        # Summary by category
        summary_cat = (
            pd.DataFrame(
                {
                    "category": gdf["category"],
                    "geom_type": gdf["geom_type"],
                    "area_m2": gdf_m["area_m2"],
                    "length_m": gdf_m["length_m"],
                }
            )
            .groupby("category", as_index=False)
            .agg(
                feature_count=("category", "size"),
                polygon_area_m2=("area_m2", "sum"),
                line_length_m=("length_m", "sum"),
            )
            .sort_values("feature_count", ascending=False)
        )
        summary_by_category_csv = out_dir / "summary_by_category.csv"
        summary_cat.to_csv(summary_by_category_csv, index=False, encoding="utf-8")

        summary_values = build_value_summary(gdf, top_n=args.top_values)
        summary_by_tag_values_csv = out_dir / "summary_by_tag_values.csv"
        summary_values.to_csv(summary_by_tag_values_csv, index=False, encoding="utf-8")

        # Visualization
        fig, ax = plt.subplots(figsize=(11, 10))
        order = list(CATEGORY_COLORS.keys())
        for cat in order:
            sub = gdf[gdf["category"] == cat]
            if sub.empty:
                continue
            color = CATEGORY_COLORS[cat]
            sub.plot(ax=ax, color=color, linewidth=1.0, markersize=args.point_size, alpha=0.75, label=cat)

        ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.2, label="bbox")
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

        ax.set_title("OSM Recognized Features in Selected BBox")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        recognized_map_png = out_dir / "recognized_features_map.png"
        fig.savefig(recognized_map_png, dpi=180)
        plt.close(fig)

        # A lighter simplified map for manual inspection.
        gdf_simple = gdf.copy()
        gdf_simple["coarse_layer"] = gdf_simple["category"].map(coarse_layer)
        fig2, ax2 = plt.subplots(figsize=(11, 10))
        layer_colors = {
            "building": "#d9d9d9",
            "transport": "#e7b074",
            "water": "#b8d9ef",
            "green": "#cfe8be",
            "other": "#e5e5e5",
        }
        for layer in ["other", "green", "water", "transport", "building"]:
            sub = gdf_simple[gdf_simple["coarse_layer"] == layer]
            if sub.empty:
                continue
            sub.plot(
                ax=ax2,
                color=layer_colors[layer],
                linewidth=0.8,
                markersize=max(4.0, args.point_size * 0.8),
                alpha=0.75,
                label=layer if layer not in ax2.get_legend_handles_labels()[1] else None,
            )
        ax2.plot(xs, ys, color="black", linestyle="--", linewidth=1.2, label="bbox")
        ax2.set_xlim(min_lon, max_lon)
        ax2.set_ylim(min_lat, max_lat)
        ax2.set_title("OSM Features (Simplified 2D)")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.set_aspect("equal")
        ax2.legend(loc="lower left", fontsize=8, ncol=2, frameon=True, framealpha=0.9)
        plt.tight_layout()
        recognized_map_simple_png = out_dir / "recognized_features_map_simple.png"
        fig2.savefig(recognized_map_simple_png, dpi=200)
        plt.close(fig2)

    # Semantic raster outputs (area masks): building / road / green / water.
    tfm_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    corner_lon = [min_lon, max_lon, max_lon, min_lon]
    corner_lat = [min_lat, min_lat, max_lat, max_lat]
    corner_xy = np.asarray([tfm_to_utm.transform(lo, la) for lo, la in zip(corner_lon, corner_lat)], dtype=np.float64)
    if args.local_frame_size is not None:
        cx_u, cy_u = tfm_to_utm.transform(center_lon, center_lat)
        map_x_m, map_y_m = map(float, args.local_frame_size)
    else:
        xmin_u, ymin_u = corner_xy[:, 0].min(), corner_xy[:, 1].min()
        xmax_u, ymax_u = corner_xy[:, 0].max(), corner_xy[:, 1].max()
        cx_u, cy_u = 0.5 * (xmin_u + xmax_u), 0.5 * (ymin_u + ymax_u)
        map_x_m, map_y_m = float(xmax_u - xmin_u), float(ymax_u - ymin_u)

    gdf_local = gdf_m.copy()
    gdf_local["geometry"] = gdf_local.geometry.translate(xoff=-cx_u, yoff=-cy_u)
    local_box = box(-map_x_m / 2.0, -map_y_m / 2.0, map_x_m / 2.0, map_y_m / 2.0)
    gdf_local = gdf_local[gdf_local.geometry.intersects(local_box)].copy()
    gdf_local["geometry"] = gdf_local.geometry.intersection(local_box)
    gdf_local = gdf_local[~gdf_local.geometry.is_empty].copy()

    geom_type = gdf_local.geometry.geom_type
    is_poly_local = geom_type.isin(["Polygon", "MultiPolygon"])
    is_line_local = geom_type.isin(["LineString", "MultiLineString"])
    is_point_local = geom_type.isin(["Point", "MultiPoint"])

    # A single, versioned primary classifier is shared with the 3-D builder.
    # In particular, landuse=residential/industrial is *not* a building.
    primary_class = gdf_local.apply(primary_semantic_class, axis=1)
    is_building_sem = primary_class == "building"
    is_water_sem = primary_class == "water"
    is_transport_sem = primary_class == "transport"
    is_obstacle_sem = primary_class == "obstacle"
    is_green_sem = primary_class == "green"
    is_built_up_sem = primary_class == "built_up"
    is_explicit_open_sem = primary_class == "explicit_open"
    is_uncategorized_sem = primary_class == "unknown_feature"

    build_geoms = list(gdf_local[is_building_sem & is_poly_local].geometry.values)

    green_geoms = list(gdf_local[is_green_sem & is_poly_local].geometry.values)
    water_geoms = list(gdf_local[is_water_sem & is_poly_local].geometry.values)
    obstacle_geoms = list(gdf_local[is_obstacle_sem & is_poly_local].geometry.values)
    built_up_geoms = list(gdf_local[is_built_up_sem & is_poly_local].geometry.values)
    explicit_open_geoms = list(gdf_local[is_explicit_open_sem & is_poly_local].geometry.values)

    # Roads as area: buffer transport lines by width.
    road_geoms = list(gdf_local[is_transport_sem & is_poly_local].geometry.values)
    t_line = gdf_local[is_transport_sem & is_line_local]
    for _, row in t_line.iterrows():
        if pd.notna(row.get("railway", np.nan)):
            w_buf = 2.0
        else:
            w_buf = highway_half_width_m(row)
        road_geoms.append(row.geometry.buffer(w_buf))

    # Water line/point features -> small areas.
    water_line = gdf_local[is_water_sem & is_line_local]
    for _, row in water_line.iterrows():
        water_geoms.append(row.geometry.buffer(1.5))
    water_point = gdf_local[is_water_sem & is_point_local]
    for _, row in water_point.iterrows():
        water_geoms.append(row.geometry.buffer(2.0))

    # Point/line green features -> tiny areas.
    green_line = gdf_local[is_green_sem & is_line_local]
    for _, row in green_line.iterrows():
        green_geoms.append(row.geometry.buffer(0.9))
    green_point = gdf_local[is_green_sem & is_point_local]
    for _, row in green_point.iterrows():
        green_geoms.append(row.geometry.buffer(0.8))

    # Obstacle line/point features -> area masks.
    obstacle_line = gdf_local[is_obstacle_sem & is_line_local]
    for _, row in obstacle_line.iterrows():
        obstacle_geoms.append(row.geometry.buffer(0.8))
    obstacle_point = gdf_local[is_obstacle_sem & is_point_local]
    for _, row in obstacle_point.iterrows():
        obstacle_geoms.append(row.geometry.buffer(0.8))

    # Building line/point features are buffered into building mask.
    build_line = gdf_local[is_building_sem & is_line_local]
    for _, row in build_line.iterrows():
        build_geoms.append(row.geometry.buffer(0.9))
    build_point = gdf_local[is_building_sem & is_point_local]
    for _, row in build_point.iterrows():
        build_geoms.append(row.geometry.buffer(1.0))

    # Group uncategorized features into one mask per tag-value label.
    uncategorized_geo_map: dict[str, list] = {}
    uncategorized_feature_count: dict[str, int] = {}
    uncategorized_df = gdf_local[is_uncategorized_sem].copy()
    if not uncategorized_df.empty:
        uncategorized_df["uncategorized_label"] = uncategorized_df.apply(_pick_uncategorized_label, axis=1)
        for label, sub in uncategorized_df.groupby("uncategorized_label", sort=True):
            geoms = list(sub[sub.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].geometry.values)
            sub_line = sub[sub.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            for _, row in sub_line.iterrows():
                geoms.append(row.geometry.buffer(0.9))
            sub_point = sub[sub.geometry.geom_type.isin(["Point", "MultiPoint"])]
            for _, row in sub_point.iterrows():
                geoms.append(row.geometry.buffer(1.0))
            uncategorized_geo_map[label] = geoms
            uncategorized_feature_count[label] = int(len(sub))

    if args.grid_size is not None:
        h = w = int(max(32, args.grid_size))
        effective_res_x_m = map_x_m / w
        effective_res_y_m = map_y_m / h
    else:
        if args.grid_res_m <= 0:
            raise ValueError("--grid-res-m must be > 0")
        h = int(max(32, round(map_y_m / float(args.grid_res_m))))
        w = int(max(32, round(map_x_m / float(args.grid_res_m))))
        if args.max_raster_cells < 1024:
            raise ValueError("--max-raster-cells must be >= 1024")
        requested_cells = h * w
        if requested_cells > args.max_raster_cells:
            scale = np.sqrt(requested_cells / float(args.max_raster_cells))
            h = max(32, int(np.floor(h / scale)))
            w = max(32, int(np.floor(w / scale)))
        effective_res_x_m = map_x_m / w
        effective_res_y_m = map_y_m / h
    extent_local = (-map_x_m / 2.0, map_x_m / 2.0, -map_y_m / 2.0, map_y_m / 2.0)
    building_mask = _rasterize_geoms(build_geoms, h=h, w=w, extent_local=extent_local)
    road_mask = _rasterize_geoms(road_geoms, h=h, w=w, extent_local=extent_local)
    green_mask = _rasterize_geoms(green_geoms, h=h, w=w, extent_local=extent_local)
    water_mask = _rasterize_geoms(water_geoms, h=h, w=w, extent_local=extent_local)
    obstacle_mask = _rasterize_geoms(obstacle_geoms, h=h, w=w, extent_local=extent_local)
    built_up_mask = _rasterize_geoms(built_up_geoms, h=h, w=w, extent_local=extent_local)
    explicit_open_mask = _rasterize_geoms(explicit_open_geoms, h=h, w=w, extent_local=extent_local)
    # Enforce the canonical geometric priority across distinct, overlapping OSM
    # features: building > water > transport > obstacle > green > context.
    is_building = building_mask > 0.5
    water_mask[is_building] = 0.0
    is_water = water_mask > 0.5
    road_mask[is_building | is_water] = 0.0
    is_road = road_mask > 0.5
    obstacle_mask[is_building | is_water | is_road] = 0.0
    is_obstacle = obstacle_mask > 0.5
    green_mask[is_building | is_water | is_road | is_obstacle] = 0.0
    occupied_main = is_building | is_water | is_road | is_obstacle | (green_mask > 0.5)
    built_up_mask[occupied_main] = 0.0
    explicit_open_mask[occupied_main | (built_up_mask > 0.5)] = 0.0

    uncategorized_masks: dict[str, np.ndarray] = {}
    for label, geoms in uncategorized_geo_map.items():
        m = _rasterize_geoms(geoms, h=h, w=w, extent_local=extent_local)
        m[
            (road_mask > 0.5)
            | (water_mask > 0.5)
            | (green_mask > 0.5)
            | (obstacle_mask > 0.5)
            | (building_mask > 0.5)
            | (built_up_mask > 0.5)
            | (explicit_open_mask > 0.5)
        ] = 0.0
        if float(m.sum()) > 0:
            uncategorized_masks[label] = m

    used_main = np.clip(
        building_mask + road_mask + green_mask + water_mask + obstacle_mask
        + built_up_mask + explicit_open_mask,
        0.0,
        1.0,
    )
    uncategorized_union = np.zeros_like(used_main, dtype=np.float32)
    for m in uncategorized_masks.values():
        uncategorized_union = np.clip(uncategorized_union + m, 0.0, 1.0)
    unknown_feature_mask = uncategorized_union.astype(np.float32)
    background_mask = (1.0 - np.clip(used_main + unknown_feature_mask, 0.0, 1.0)).astype(np.float32)
    # Service evaluation is deliberately independent from TX feasibility.
    service_evaluation_mask = (water_mask <= 0.5).astype(np.float32)

    npz_payload: dict[str, np.ndarray] = {
        "building_mask": building_mask.astype(np.float32),
        "road_mask": road_mask.astype(np.float32),
        "green_mask": green_mask.astype(np.float32),
        "water_mask": water_mask.astype(np.float32),
        "obstacle_mask": obstacle_mask.astype(np.float32),
        "built_up_mask": built_up_mask.astype(np.float32),
        "explicit_open_mask": explicit_open_mask.astype(np.float32),
        "unknown_feature_mask": unknown_feature_mask,
        "background_mask": background_mask,
        "service_evaluation_mask": service_evaluation_mask,
        # Deprecated compatibility alias.  It must not be interpreted as
        # verified empty/deployable land by new code.
        "other_mask": background_mask,
    }
    for label, mask in uncategorized_masks.items():
        npz_payload[f"uncategorized__{_sanitize_name(label)}"] = mask.astype(np.float32)
    np.savez_compressed(out_dir / "semantic_masks.npz", **npz_payload)

    main_mask_dir: Path | None = None
    uncategorized_dir: Path | None = None
    uncategorized_masks_index_csv: Path | None = None
    uncategorized_labeled_map_png: Path | None = None
    uncategorized_details_csv: Path | None = None
    if not minimal_output:
        main_mask_dir = out_dir / "semantic_main_masks"
        main_mask_dir.mkdir(parents=True, exist_ok=True)
        _save_single_mask_png(main_mask_dir / "building_mask.png", building_mask, map_x_m, map_y_m, "Building mask")
        _save_single_mask_png(main_mask_dir / "road_mask.png", road_mask, map_x_m, map_y_m, "Road mask")
        _save_single_mask_png(main_mask_dir / "water_mask.png", water_mask, map_x_m, map_y_m, "Water mask")
        _save_single_mask_png(main_mask_dir / "green_mask.png", green_mask, map_x_m, map_y_m, "Green mask")
        _save_single_mask_png(main_mask_dir / "obstacle_mask.png", obstacle_mask, map_x_m, map_y_m, "Obstacle mask")

        uncategorized_dir = out_dir / "semantic_uncategorized_masks"
        uncategorized_dir.mkdir(parents=True, exist_ok=True)
        uncategorized_rows: list[dict] = []
        for label, mask in sorted(uncategorized_masks.items(), key=lambda kv: kv[0]):
            filename = f"{_sanitize_name(label)}.png"
            file_path = uncategorized_dir / filename
            _save_single_mask_png(file_path, mask, map_x_m, map_y_m, f"Uncategorized: {label}")
            uncategorized_rows.append(
                {
                    "label": label,
                    "feature_count": int(uncategorized_feature_count.get(label, 0)),
                    "pixel_count": int(mask.sum()),
                    "mask_png": str(file_path),
                }
            )
        uncategorized_masks_index_csv = out_dir / "uncategorized_masks_index.csv"
        pd.DataFrame(uncategorized_rows).to_csv(
            uncategorized_masks_index_csv, index=False, encoding="utf-8"
        )
        uncategorized_labeled_map_png = out_dir / "uncategorized_labeled_map.png"
        _plot_uncategorized_labeled_map(
            out_path=uncategorized_labeled_map_png,
            uncategorized_df=uncategorized_df,
            map_x=map_x_m,
            map_y=map_y_m,
        )

        # Per-feature detail table with approximate location (both local meters and WGS84).
        uncategorized_details_rows: list[dict] = []
        tfm_from_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        for idx, row in uncategorized_df.iterrows():
            label = str(row.get("uncategorized_label", "other=untyped"))
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            rp = geom.representative_point()
            x_local = float(rp.x)
            y_local = float(rp.y)
            x_utm = x_local + cx_u
            y_utm = y_local + cy_u
            lon, lat = tfm_from_utm.transform(x_utm, y_utm)
            uncategorized_details_rows.append(
                {
                    "feature_id": str(idx),
                    "label": label,
                    "geometry_type": str(geom.geom_type),
                    "x_local_m": x_local,
                    "y_local_m": y_local,
                    "lon_wgs84": float(lon),
                    "lat_wgs84": float(lat),
                }
            )
        uncategorized_details_csv = out_dir / "uncategorized_feature_details.csv"
        pd.DataFrame(uncategorized_details_rows).to_csv(
            uncategorized_details_csv, index=False, encoding="utf-8"
        )

    _plot_semantic_check(
        out_path=out_dir / "semantic_masks_check.png",
        building=building_mask,
        road=road_mask,
        green=green_mask,
        water=water_mask,
        obstacle=obstacle_mask,
        built_up=built_up_mask,
        explicit_open=explicit_open_mask,
        unknown_feature=unknown_feature_mask,
        map_x=map_x_m,
        map_y=map_y_m,
    )

    meta = {
        "schema_version": "citydeploy.scene-inspect.v3",
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "bbox_wgs84": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
        "center_wgs84": {"lon": center_lon, "lat": center_lat},
        "utm_crs": str(utm_crs),
        "query_tags": DEFAULT_TAGS,
        "osm_source": osm_source,
        "total_features": int(len(gdf)),
        "semantic_raster": {
            "grid_h": h,
            "grid_w": w,
            "grid_mode": "fixed_square" if args.grid_size is not None else "auto_from_bbox",
            "grid_res_m": float(args.grid_res_m),
            "requested_grid_res_m": float(args.grid_res_m),
            "effective_grid_res_m": {
                "x": float(effective_res_x_m),
                "y": float(effective_res_y_m),
            },
            "max_raster_cells": int(args.max_raster_cells),
            "map_x_m_local": map_x_m,
            "map_y_m_local": map_y_m,
            "coordinate_convention": "x=east, y=north, raster row 0=y_min",
            "frame_source": "explicit_projected_scene_frame" if args.local_frame_size is not None else "wgs84_bbox_envelope",
            "channels": [
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
                "other_mask",
                "uncategorized__*",
            ],
            "note": (
                "Buildings require building/building:part tags. Built-up land-use, explicit open land, "
                "unclassified features and unannotated background remain distinct. other_mask is a deprecated "
                "alias for background_mask and is never evidence of TX feasibility."
            ),
        },
        "output_files": {
            "semantic_masks_npz": str(out_dir / "semantic_masks.npz"),
            "semantic_masks_check_png": str(out_dir / "semantic_masks_check.png"),
            "semantic_overlay_png": str(out_dir / "semantic_overlay.png"),
            "semantic_overlay_pdf": str(out_dir / "semantic_overlay.pdf"),
        },
    }
    if not minimal_output:
        meta["output_files"].update(
            {
                "geojson": str(raw_geojson),
                "summary_by_category_csv": str(summary_by_category_csv),
                "summary_by_tag_values_csv": str(summary_by_tag_values_csv),
                "recognized_map_png": str(recognized_map_png),
                "recognized_map_simple_png": str(recognized_map_simple_png),
                "semantic_main_masks_dir": str(main_mask_dir),
                "semantic_uncategorized_masks_dir": str(uncategorized_dir),
                "uncategorized_masks_index_csv": str(uncategorized_masks_index_csv),
                "uncategorized_labeled_map_png": str(uncategorized_labeled_map_png),
                "uncategorized_feature_details_csv": str(uncategorized_details_csv),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OSM features processed: {len(gdf)} features")
    print(f"Output directory: {out_dir}")
    print(
        f"Semantic raster: {h}x{w} "
        + ("(fixed --grid-size)" if args.grid_size is not None else f"(auto, ~{args.grid_res_m:g}m/px)")
    )
    print("Output files:")
    print(f"  - {out_dir / 'semantic_masks_check.png'}")
    print(f"  - {out_dir / 'semantic_masks.npz'}")
    print(f"  - {out_dir / 'summary.json'}")
    if not minimal_output:
        print(f"  - {recognized_map_png}")
        print(f"  - {recognized_map_simple_png}")
        print(f"  - {main_mask_dir} (building/road/water/green/obstacle individual masks)")
        print(f"  - {uncategorized_dir} (uncategorized class masks)")
        print(f"  - {uncategorized_masks_index_csv}")
        print(f"  - {uncategorized_labeled_map_png} (uncategorized class locations)")
        print(f"  - {uncategorized_details_csv} (uncategorized feature coordinates)")
        print(f"  - {summary_by_category_csv}")
        print(f"  - {summary_by_tag_values_csv}")
        print(f"  - {raw_geojson}")


if __name__ == "__main__":
    main()


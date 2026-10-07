"""Canonical OSM semantics shared by scene meshes and dataset rasters.

This module deliberately contains no geopandas/osmnx imports.  The rules are
therefore cheap to unit-test and can be used by both the 3-D scene builder and
the raster exporter without maintaining two classification implementations.

OSM tags describe volunteered map semantics, not measured radio materials.
The primary class returned here is consequently a geometry/land-use class;
radio-material assignment is a separate, versioned modelling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "citydeploy.osm-semantics.v3"

OSM_QUERY_TAGS = {
    "building": True,
    "building:part": True,
    "highway": True,
    "railway": True,
    "water": True,
    "waterway": True,
    "natural": True,
    "landuse": True,
    "leisure": True,
    "amenity": True,
    "man_made": True,
    "power": True,
    "barrier": True,
}

WATER_NATURAL_VALUES = {"water", "wetland", "bay", "strait"}
WATER_LANDUSE_VALUES = {"basin", "reservoir", "salt_pond"}
GREEN_NATURAL_VALUES = {
    "tree",
    "tree_row",
    "scrub",
    "wood",
    "grassland",
    "heath",
}
GREEN_LANDUSE_VALUES = {
    "allotments",
    "farmland",
    "farmyard",
    "forest",
    "grass",
    "meadow",
    "orchard",
    "plant_nursery",
    "vineyard",
}
GREEN_LEISURE_VALUES = {"garden", "golf_course", "nature_reserve", "park", "pitch"}
BUILT_UP_LANDUSE_VALUES = {
    "commercial",
    "construction",
    "industrial",
    "religious",
    "residential",
    "retail",
}

# These are area semantics that may represent genuinely open ground.  They are
# exported separately, but are not automatically declared TX-safe.
OPEN_LANDUSE_VALUES = {"brownfield", "greenfield", "recreation_ground"}
OPEN_LEISURE_VALUES = {"common", "playground"}


def clean_tag(value: Any) -> str:
    """Return a normalized scalar OSM value; missing/NaN-like values become ''."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN without importing numpy/pandas
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return "" if text in {"", "nan", "none", "<na>"} else text


def has_tag(tags: Mapping[str, Any], key: str) -> bool:
    return clean_tag(tags.get(key)) not in {"", "no", "false", "0"}


@dataclass(frozen=True)
class SemanticFlags:
    building: bool
    transport: bool
    water: bool
    green: bool
    obstacle: bool
    built_up: bool
    explicit_open: bool


def semantic_flags(tags: Mapping[str, Any]) -> SemanticFlags:
    """Classify one OSM feature without conflating land use with a building."""
    natural = clean_tag(tags.get("natural"))
    landuse = clean_tag(tags.get("landuse"))
    leisure = clean_tag(tags.get("leisure"))
    return SemanticFlags(
        building=has_tag(tags, "building") or has_tag(tags, "building:part"),
        transport=has_tag(tags, "highway") or has_tag(tags, "railway"),
        water=(
            has_tag(tags, "water")
            or has_tag(tags, "waterway")
            or natural in WATER_NATURAL_VALUES
            or landuse in WATER_LANDUSE_VALUES
        ),
        green=(
            natural in GREEN_NATURAL_VALUES
            or landuse in GREEN_LANDUSE_VALUES
            or leisure in GREEN_LEISURE_VALUES
        ),
        obstacle=has_tag(tags, "barrier"),
        built_up=landuse in BUILT_UP_LANDUSE_VALUES,
        explicit_open=(
            landuse in OPEN_LANDUSE_VALUES or leisure in OPEN_LEISURE_VALUES
        ),
    )


def primary_semantic_class(tags: Mapping[str, Any]) -> str:
    """Return one deterministic class using the canonical overlap priority.

    Buildings win over enclosing land-use polygons.  Water then wins over
    transport crossings, followed by transport, obstacles, vegetation,
    built-up land-use context, and explicit open land.
    """
    flags = semantic_flags(tags)
    for name in (
        "building",
        "water",
        "transport",
        "obstacle",
        "green",
        "built_up",
        "explicit_open",
    ):
        if getattr(flags, name):
            return name
    return "unknown_feature"


def highway_half_width_m(tags: Mapping[str, Any]) -> float:
    """Estimate road half-width, preferring explicit OSM width and lanes."""
    raw_width = clean_tag(tags.get("width"))
    if raw_width:
        # Common OSM values include units such as "6 m" and decimal commas.
        token = raw_width.replace(",", ".").split()[0]
        try:
            width = float(token)
            if 0.5 <= width <= 60.0:
                return width / 2.0
        except ValueError:
            pass

    raw_lanes = clean_tag(tags.get("lanes"))
    if raw_lanes:
        try:
            lanes = int(float(raw_lanes))
            if 1 <= lanes <= 12:
                return lanes * 3.2 / 2.0
        except ValueError:
            pass

    highway = clean_tag(tags.get("highway"))
    half_width_by_class = {
        "motorway": 7.0,
        "motorway_link": 3.5,
        "trunk": 6.0,
        "trunk_link": 3.5,
        "primary": 5.5,
        "primary_link": 3.5,
        "secondary": 5.0,
        "secondary_link": 3.5,
        "tertiary": 4.5,
        "tertiary_link": 3.25,
        "unclassified": 4.0,
        "residential": 4.0,
        "living_street": 3.5,
        "service": 3.0,
        "pedestrian": 3.0,
        "footway": 1.0,
        "path": 0.9,
        "cycleway": 1.25,
        "steps": 0.8,
        "track": 2.0,
    }
    return half_width_by_class.get(highway, 2.5)


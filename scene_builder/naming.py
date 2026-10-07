"""Location-aware, collision-safe names for generated scene packages."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests


DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "CityDeploy/0.1 (academic wireless scene generation)"
PLACE_KEYS = ("city", "town", "municipality", "village", "borough", "county", "state")
CITY_SLUG_ALIASES = {
    "city_of_westminster": ("london", "London"),
    "greater_london": ("london", "London"),
    "london": ("london", "London"),
    "hongkou_district": ("shanghai", "Shanghai"),
    "pudong": ("shanghai", "Shanghai"),
    "shanghai": ("shanghai", "Shanghai"),
    "jersey_city": ("new_york", "New York"),
    "hoboken": ("new_york", "New York"),
    "new_york": ("new_york", "New York"),
    "chuo": ("tokyo", "Tokyo"),
    "koto": ("tokyo", "Tokyo"),
    "shibuya": ("tokyo", "Tokyo"),
    "sumida": ("tokyo", "Tokyo"),
    "taito": ("tokyo", "Tokyo"),
    "tokyo": ("tokyo", "Tokyo"),
    "paris": ("paris", "Paris"),
}


def _ascii_slug(value: str) -> str:
    """Return an ASCII-only filesystem slug, or an empty string."""
    text = unicodedata.normalize("NFKD", str(value)).strip().casefold()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\\/:*?\"<>|]+", "-", text)
    text = re.sub(r"[^a-z0-9_\- ]+", "", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:64]


def sanitize_place_slug(value: str) -> str:
    """Return a Windows/Open3D-safe, ASCII-only filesystem slug."""
    return _ascii_slug(value) or "unknown_location"


def canonical_city_identity(place_slug: str, place_name: str) -> tuple[str, str]:
    """Map district/metro labels to the project-wide canonical city ID."""
    return CITY_SLUG_ALIASES.get(place_slug, (place_slug, place_name))


def _fallback_place_name(lon: float, lat: float) -> str:
    lat_token = f"{lat:+.5f}".replace("+", "p").replace("-", "m").replace(".", "p")
    lon_token = f"{lon:+.5f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"lat_{lat_token}_lon_{lon_token}"


def _pick_place_name(payload: dict[str, Any], lon: float, lat: float) -> str:
    address = payload.get("address") or {}
    for key in PLACE_KEYS:
        value = address.get(key)
        if value:
            return str(value)
    if payload.get("name"):
        return str(payload["name"])
    return _fallback_place_name(lon, lat)


def _request_nominatim(
    endpoint: str,
    lon: float,
    lat: float,
    language: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "lat": f"{lat:.8f}",
        "lon": f"{lon:.8f}",
        "format": "jsonv2",
        "addressdetails": 1,
        "namedetails": 1,
        "zoom": 10,
    }
    if language:
        params["accept-language"] = language
    response = requests.get(
        endpoint,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Nominatim response is not a JSON object")
    return payload


def reverse_geocode_place(
    lon: float,
    lat: float,
    cache_path: Path,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Resolve one user-confirmed scene center and cache the response locally."""
    endpoint = endpoint or os.environ.get("CityDeploy_NOMINATIM_URL", DEFAULT_NOMINATIM_URL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    key = f"{lat:.5f},{lon:.5f}"
    payload = cache.get(key)
    source = "nominatim_cache"
    error = None
    if not isinstance(payload, dict):
        try:
            payload = _request_nominatim(endpoint, lon, lat)
            cache[key] = payload
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            source = "nominatim"
        except Exception as exc:  # A coordinate-based name keeps generation usable offline.
            payload = {}
            source = "coordinate_fallback"
            error = str(exc)

    place_name = _pick_place_name(payload, lon, lat)
    place_name_en: str | None = None
    place_slug = _ascii_slug(place_name)
    if not place_slug:
        english_key = f"{key}|en"
        english_payload = cache.get(english_key)
        if not isinstance(english_payload, dict):
            try:
                english_payload = _request_nominatim(endpoint, lon, lat, language="en")
                cache[english_key] = english_payload
                cache_path.write_text(
                    json.dumps(cache, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                english_payload = {}
                message = f"English-name lookup failed: {exc}"
                error = f"{error}; {message}" if error else message
        if english_payload:
            place_name_en = _pick_place_name(english_payload, lon, lat)
            place_slug = _ascii_slug(place_name_en)
    if not place_slug:
        place_slug = sanitize_place_slug(_fallback_place_name(lon, lat))
    source_place_name = place_name
    place_slug, place_name = canonical_city_identity(place_slug, place_name)
    if source_place_name != place_name:
        place_name_en = place_name
    return {
        "place_name": place_name,
        "place_name_en": place_name_en,
        "place_slug": place_slug,
        "source_place_name": source_place_name if source_place_name != place_name else None,
        "source": source,
        "center_wgs84": {"lon": float(lon), "lat": float(lat)},
        "display_name": payload.get("display_name") if isinstance(payload, dict) else None,
        "osm_attribution": "© OpenStreetMap contributors",
        "endpoint": endpoint,
        "error": error,
    }


def user_place_identity(place_name: str, lon: float, lat: float) -> dict[str, Any]:
    source_place_name = str(place_name).strip()
    place_slug = _ascii_slug(place_name)
    if not place_slug:
        place_slug = sanitize_place_slug(_fallback_place_name(lon, lat))
    place_slug, canonical_name = canonical_city_identity(place_slug, source_place_name)
    return {
        "place_name": canonical_name,
        "place_name_en": canonical_name if _ascii_slug(canonical_name) else None,
        "place_slug": place_slug,
        "source_place_name": source_place_name if source_place_name != canonical_name else None,
        "source": "user_override",
        "center_wgs84": {"lon": float(lon), "lat": float(lat)},
        "display_name": None,
        "osm_attribution": None,
        "endpoint": None,
        "error": None,
    }


def allocate_location_scene_dir(
    save_root: Path,
    place_slug: str,
) -> tuple[str, Path, Path]:
    scenes_root = save_root
    scenes_root.mkdir(parents=True, exist_ok=True)

    slug = sanitize_place_slug(place_slug)
    pattern = re.compile(rf"^{re.escape(slug)}_(\d+)$")
    indices: list[int] = []
    for path in scenes_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices, default=0) + 1
    scene_id = f"{slug}_{next_index:02d}"
    scene_dir = scenes_root / scene_id
    input_dir = scene_dir / "inputs"
    scene_dir.mkdir(parents=True, exist_ok=False)
    input_dir.mkdir(parents=True, exist_ok=False)
    return scene_id, scene_dir, input_dir

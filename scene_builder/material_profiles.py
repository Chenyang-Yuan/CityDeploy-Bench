"""Load and validate versioned semantic-to-radio-material priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .itu_materials import ITU_MATERIALS


MATERIAL_KEYS = {
    "ground",
    "empty_land",
    "road",
    "water",
    "green",
    "obstacle",
    "rooftop",
    "wall",
}


def load_material_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "citydeploy.material-profile.v1":
        raise ValueError(f"Unsupported material profile schema: {profile.get('schema_version')}")
    materials = profile.get("materials")
    if not isinstance(materials, dict) or set(materials) != MATERIAL_KEYS:
        raise ValueError(f"Material profile must define exactly: {sorted(MATERIAL_KEYS)}")
    unknown = {value for value in materials.values() if value not in ITU_MATERIALS}
    if unknown:
        raise ValueError(f"Unknown material ids: {sorted(unknown)}")
    frequency_hz = float(profile["intended_frequency_hz"])
    for semantic, material_id in materials.items():
        limits = ITU_MATERIALS[material_id]
        lower = limits["lower_freq_limit"]
        upper = limits["upper_freq_limit"]
        if isinstance(lower, list) or isinstance(upper, list):
            continue
        if not (float(lower) <= frequency_hz <= float(upper)):
            raise ValueError(
                f"{semantic}={material_id} is outside its declared frequency range at {frequency_hz} Hz"
            )
    profile["profile_path"] = str(profile_path)
    return profile

"""Canonical self-contained scene package paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import numpy as np


PROJECT_ROOT = workspace_root()
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_SCENES_ROOT = DEFAULT_DATASETS_ROOT / "scenes"


@dataclass(frozen=True)
class ScenePackage:
    """Resolve every artifact belonging to one scene from a single directory."""

    root: Path

    @classmethod
    def resolve(cls, scene: str | Path, scenes_root: str | Path = DEFAULT_SCENES_ROOT) -> "ScenePackage":
        raw = Path(scene)
        root = raw if raw.is_absolute() or len(raw.parts) > 1 else Path(scenes_root) / raw
        return cls(root.resolve())

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def scene_xml(self) -> Path:
        return self.root / "scene.xml"

    @property
    def mesh_dir(self) -> Path:
        return self.root / "mesh"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def visualizations_dir(self) -> Path:
        return self.root / "visualizations"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def scene_inputs(self) -> Path:
        return self.inputs_dir / "scene_inputs.npz"

    @property
    def semantic_masks(self) -> Path:
        return self.inputs_dir / "semantic_masks.npz"

    @property
    def metadata(self) -> Path:
        return self.inputs_dir / "metadata.json"

    def require(self) -> "ScenePackage":
        required = (self.manifest, self.scene_xml, self.scene_inputs, self.semantic_masks, self.metadata)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete scene package {self.root}: missing {missing}")
        return self


def load_scene_inputs_and_meta(scene_or_inputs_dir: str | Path):
    """Load model inputs from either a scene package or its ``inputs`` folder."""
    root = Path(scene_or_inputs_dir)
    inputs_dir = root / "inputs" if (root / "inputs").is_dir() else root
    npz_path = inputs_dir / "scene_inputs.npz"
    meta_path = inputs_dir / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Scene inputs not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Scene metadata not found: {meta_path}")
    with np.load(npz_path) as data:
        building = np.asarray(data["building_occupancy"])
        deployable = np.asarray(data["deployable_mask"])
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return building, deployable, metadata

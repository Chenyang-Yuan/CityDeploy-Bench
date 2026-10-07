"""V3/V4 dataset adapter for scene-conditioned unordered transmitter sets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path
from typing import Sequence

import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import Dataset

from scene_builder.package import load_scene_inputs_and_meta
from citydeploy.schema import canonical_schema


PROJECT_ROOT = workspace_root()
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "raytracing"
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "datasets" / "scenes"
SCENE_FEATURE_CHANNELS = (
    "building_mask",
    "road_mask",
    "green_mask",
    "water_mask",
    "obstacle_mask",
    "built_up_mask",
    "explicit_open_mask",
    "unknown_feature_mask",
    "background_mask",
    "tx_feasible_mask",
    "evaluation_mask",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scene_feature_tensor(scene_dir: Path) -> tuple[np.ndarray, dict]:
    scene_dir = Path(scene_dir)
    inputs_dir = scene_dir / "inputs" if (scene_dir / "inputs").is_dir() else scene_dir
    building, _deployable, metadata = load_scene_inputs_and_meta(inputs_dir)
    semantic_path = inputs_dir / "semantic_masks.npz"
    scene_inputs_path = inputs_dir / "scene_inputs.npz"
    if not semantic_path.exists() or not scene_inputs_path.exists():
        raise FileNotFoundError(f"V3 scene inputs missing under {inputs_dir}")
    with np.load(semantic_path, allow_pickle=False) as semantic, np.load(
        scene_inputs_path, allow_pickle=False
    ) as scene_inputs:
        zeros = np.zeros_like(building, dtype=np.float32)
        channels = [
            np.asarray(semantic[name], dtype=np.float32) if name in semantic else zeros
            for name in SCENE_FEATURE_CHANNELS[:-2]
        ]
        channels.extend(
            [
                np.asarray(scene_inputs["tx_feasible_mask"], dtype=np.float32),
                np.asarray(scene_inputs["evaluation_mask"], dtype=np.float32),
            ]
        )
    shapes = {tuple(channel.shape) for channel in channels}
    if len(shapes) != 1:
        raise RuntimeError(f"scene channel shapes disagree: {sorted(shapes)}")
    return np.stack(channels).astype(np.float32), metadata


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    scene: str
    num_tx: int
    split: str = ""
    row_index: int | None = None
    dataset_dir: Path | None = None
    family_id: str = ""


class TxDeploymentDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
        scene_root: Path = DEFAULT_SCENE_ROOT,
        *,
        include_scenes: Sequence[str] | None = None,
        include_num_tx: Sequence[int] | None = None,
        include_splits: Sequence[str] | None = None,
        max_samples: int | None = None,
        permute_tx_order: bool = True,
        expand_family_memberships: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.scene_root = Path(scene_root)
        self.permute_tx_order = bool(permute_tx_order)
        self.expand_family_memberships = bool(expand_family_memberships)
        scene_filter = set(include_scenes or ())
        count_filter = set(int(value) for value in (include_num_tx or ()))
        split_filter = set(str(value) for value in (include_splits or ()))
        self.records: list[SampleRecord] = []
        self.scene_tensor_map: dict[str, torch.Tensor] = {}
        self.scene_meta_map: dict[str, dict] = {}
        self._parquet_cache: dict[Path, list[dict]] = {}
        self._v4_feature_paths: dict[str, Path] = {}
        self.has_explicit_splits = False
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.dataset_root}")
        if (self.dataset_root / "manifest.json").exists():
            dataset_dirs = [self.dataset_root]
        else:
            dataset_dirs = sorted(path for path in self.dataset_root.iterdir() if path.is_dir())
        for dataset_dir in dataset_dirs:
            manifest_path = dataset_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            schema_version = canonical_schema(manifest.get("schema_version"))
            if schema_version not in {"citydeploy.dataset.v3", "citydeploy.dataset.v4"}:
                continue
            if schema_version == "citydeploy.dataset.v4":
                self.has_explicit_splits = True
                deployment_records: dict[str, SampleRecord] = {}
                for feature_path in (dataset_dir / "scenes" / "features").glob("*.npz"):
                    self._v4_feature_paths[feature_path.stem] = feature_path
                for path in sorted((dataset_dir / "data").glob("*/*.parquet")):
                    split = path.parent.name
                    if split_filter and split not in split_filter:
                        continue
                    index_table = pq.read_table(path, columns=["sample_id", "scene_id", "num_tx", "family_id"])
                    for row_index, row in enumerate(index_table.to_pylist()):
                        scene = str(row["scene_id"])
                        num_tx = int(row["num_tx"])
                        if scene_filter and scene not in scene_filter:
                            continue
                        if count_filter and num_tx not in count_filter:
                            continue
                        deployment_records[str(row["sample_id"])] = SampleRecord(
                            path,
                            scene,
                            num_tx,
                            split=split,
                            row_index=row_index,
                            dataset_dir=dataset_dir,
                            family_id=str(row["family_id"]),
                        )
                membership_files = sorted((dataset_dir / "family_memberships").glob("*/*.parquet"))
                if membership_files and self.expand_family_memberships:
                    for membership_path in membership_files:
                        membership_split = membership_path.parent.name
                        if split_filter and membership_split not in split_filter:
                            continue
                        table = pq.read_table(membership_path, columns=["sample_id", "family_id"])
                        for membership in table.to_pylist():
                            base = deployment_records.get(str(membership["sample_id"]))
                            if base is None:
                                continue
                            self.records.append(
                                SampleRecord(
                                    base.path,
                                    base.scene,
                                    base.num_tx,
                                    split=base.split,
                                    row_index=base.row_index,
                                    dataset_dir=base.dataset_dir,
                                    family_id=str(membership["family_id"]),
                                )
                            )
                else:
                    self.records.extend(deployment_records.values())
            else:
                scene = str(manifest.get("scene", ""))
                num_tx = int(manifest.get("num_tx", 0))
                if not scene or num_tx <= 0:
                    continue
                if scene_filter and scene not in scene_filter:
                    continue
                if count_filter and num_tx not in count_filter:
                    continue
                for path in sorted((dataset_dir / "samples").glob("*.npz")):
                    self.records.append(SampleRecord(path, scene, num_tx, dataset_dir=dataset_dir))
                    if max_samples and len(self.records) >= max_samples:
                        break
            if max_samples and len(self.records) >= max_samples and not self.has_explicit_splits:
                break
        if max_samples and self.has_explicit_splits and len(self.records) > max_samples:
            by_split: dict[str, list[SampleRecord]] = {}
            for record in self.records:
                by_split.setdefault(record.split, []).append(record)
            selected_records: list[SampleRecord] = []
            positions = {split: 0 for split in by_split}
            while len(selected_records) < max_samples:
                progressed = False
                for split in sorted(by_split):
                    position = positions[split]
                    if position < len(by_split[split]) and len(selected_records) < max_samples:
                        selected_records.append(by_split[split][position])
                        positions[split] += 1
                        progressed = True
                if not progressed:
                    break
            self.records = selected_records
        if not self.records:
            raise RuntimeError(f"no CityDeploy V3/V4 samples found under {self.dataset_root}")
        for scene in sorted({record.scene for record in self.records}):
            feature_path = self._v4_feature_paths.get(scene)
            if feature_path is not None:
                with np.load(feature_path, allow_pickle=False) as loaded:
                    array = np.asarray(loaded["features"], dtype=np.float32)
                    metadata = json.loads(str(loaded["metadata_json"].item()))
            else:
                array, metadata = load_scene_feature_tensor(self.scene_root / scene)
            self.scene_tensor_map[scene] = torch.from_numpy(array)
            self.scene_meta_map[scene] = metadata
        self.scene_names = sorted(self.scene_tensor_map)
        self.scene_to_index = {name: index for index, name in enumerate(self.scene_names)}
        family_names = sorted({record.family_id for record in self.records if record.family_id})
        self.family_to_index = {name: index for index, name in enumerate(family_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if record.row_index is not None:
            if record.path not in self._parquet_cache:
                columns = [
                    "tx_xy_norm01", "tx_xyz_m", "joint_4metric_coverage",
                    "pathloss_coverage", "ss_rsrp_coverage", "sinr_coverage",
                    "effective_throughput_coverage",
                ]
                self._parquet_cache[record.path] = pq.read_table(record.path, columns=columns).to_pylist()
            sample = self._parquet_cache[record.path][record.row_index]
            xy = np.asarray(sample["tx_xy_norm01"], dtype=np.float32)
            z = np.asarray(sample["tx_xyz_m"], dtype=np.float32)[:, 2]
            labels = {
                "joint": float(sample["joint_4metric_coverage"]),
                "pathloss": float(sample["pathloss_coverage"]),
                "sinr": float(sample["sinr_coverage"]),
                "throughput": float(sample["effective_throughput_coverage"]),
                "rsrp": float(sample["ss_rsrp_coverage"]),
            }
        else:
            with np.load(record.path, allow_pickle=False) as sample:
                xy = np.asarray(sample["tx_xy_norm"], dtype=np.float32)
                z = np.asarray(sample["tx_z_real"], dtype=np.float32)
                labels = {
                    "joint": float(sample["joint_4metric_coverage"]),
                    "pathloss": float(sample["pathloss_coverage"]),
                    "sinr": float(sample["sinr_coverage"]),
                    "throughput": float(sample["throughput_coverage"]),
                    "rsrp": float(sample["rsrp_coverage"]),
                }
        if xy.shape != (record.num_tx, 2) or z.shape != (record.num_tx,):
            raise RuntimeError(f"invalid TX shape in {record.path}")
        if self.permute_tx_order and record.num_tx > 1:
            permutation = np.random.permutation(record.num_tx)
            xy, z = xy[permutation], z[permutation]
        return {
            "tx_xy_norm01": torch.from_numpy(xy),
            "tx_z_real": torch.from_numpy(z),
            "tx_mask": torch.ones(record.num_tx, dtype=torch.bool),
            **{name: torch.tensor(value, dtype=torch.float32) for name, value in labels.items()},
            "scene_index": torch.tensor(self.scene_to_index[record.scene], dtype=torch.long),
            "num_tx": torch.tensor(record.num_tx, dtype=torch.long),
            "family_index": torch.tensor(self.family_to_index.get(record.family_id, -1), dtype=torch.long),
        }


def collate_deployments(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_num_tx = max(int(item["tx_xy_norm01"].shape[0]) for item in batch)
    xy = torch.zeros((batch_size, max_num_tx, 2), dtype=torch.float32)
    z = torch.zeros((batch_size, max_num_tx), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_num_tx), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = int(item["tx_xy_norm01"].shape[0])
        xy[index, :count] = item["tx_xy_norm01"]
        z[index, :count] = item["tx_z_real"]
        mask[index, :count] = True
    keys = ("joint", "pathloss", "sinr", "throughput", "rsrp", "scene_index", "num_tx", "family_index")
    return {
        "tx_xy_norm01": xy,
        "tx_z_real": z,
        "tx_mask": mask,
        **{key: torch.stack([item[key] for item in batch]) for key in keys},
    }


def grouped_ranking_loss(
    energy: torch.Tensor,
    target_reward: torch.Tensor,
    margin: float,
    difference_epsilon: float,
) -> torch.Tensor:
    if energy.numel() < 2:
        return energy.new_zeros(())
    i, j = torch.triu_indices(energy.numel(), energy.numel(), offset=1, device=energy.device)
    difference = target_reward[i] - target_reward[j]
    keep = difference.abs() > float(difference_epsilon)
    if not bool(keep.any()):
        return energy.new_zeros(())
    signed = torch.sign(difference[keep])
    # Higher reward must have lower energy.
    return torch.relu(float(margin) + signed * (energy[i[keep]] - energy[j[keep]])).mean()


def require_common_scene_shape(scene_tensor_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    shapes = {name: tuple(tensor.shape[-2:]) for name, tensor in scene_tensor_map.items()}
    unique = sorted(set(shapes.values()))
    if len(unique) != 1:
        raise RuntimeError(
            "mixed raster shapes cannot be resized without changing metric scale; "
            f"use fixed-size training tiles or shape buckets: {shapes}"
        )
    return int(unique[0][0]), int(unique[0][1])

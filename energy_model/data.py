"""Scene-level splits and grouped batches for hypergraph-potential training."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from energy_model.dataset import TxDeploymentDataset


@dataclass(frozen=True)
class SceneSplit:
    train_scenes: tuple[str, ...]
    validation_scenes: tuple[str, ...]
    test_scenes: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train_scenes": list(self.train_scenes),
            "validation_scenes": list(self.validation_scenes),
            "test_scenes": list(self.test_scenes),
        }


def make_scene_split(
    scene_names: Sequence[str],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> SceneSplit:
    scenes = sorted(set(str(name) for name in scene_names))
    if len(scenes) < 2:
        raise RuntimeError("scene-level validation requires at least two distinct scenes")
    if validation_ratio <= 0.0 or test_ratio < 0.0 or validation_ratio + test_ratio >= 1.0:
        raise ValueError("require validation_ratio>0, test_ratio>=0, and their sum <1")
    random.Random(seed).shuffle(scenes)
    num_validation = max(1, int(round(len(scenes) * validation_ratio)))
    num_test = int(round(len(scenes) * test_ratio))
    if test_ratio > 0.0:
        num_test = max(1, num_test)
    while num_validation + num_test >= len(scenes):
        if num_test > 0:
            num_test -= 1
        elif num_validation > 1:
            num_validation -= 1
        else:
            break
    validation = tuple(sorted(scenes[:num_validation]))
    test = tuple(sorted(scenes[num_validation : num_validation + num_test]))
    train = tuple(sorted(scenes[num_validation + num_test :]))
    return SceneSplit(train, validation, test)


def indices_for_scenes(dataset: TxDeploymentDataset, scenes: Sequence[str]) -> list[int]:
    selected = set(scenes)
    return [index for index, record in enumerate(dataset.records) if record.scene in selected]


def indices_for_split(dataset: TxDeploymentDataset, split: str) -> list[int]:
    """Select rows from an immutable V4 geographic split."""
    return [index for index, record in enumerate(dataset.records) if record.split == str(split)]


class SceneCardinalityBatchSampler(Sampler[list[int]]):
    """Create batches containing a single (scene, number-of-TX) group.

    This guarantees that the within-batch ranking objective has comparable
    deployments instead of frequently returning zero under global shuffling.
    """

    def __init__(
        self,
        dataset: TxDeploymentDataset,
        indices: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least two for ranking")
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index in self.indices:
            record = dataset.records[index]
            groups[(record.scene, record.num_tx)].append(index)
        self.groups = dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for key in sorted(self.groups):
            group = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for group in self.groups.values():
            full, remainder = divmod(len(group), self.batch_size)
            total += full + (1 if remainder and not self.drop_last else 0)
        return total

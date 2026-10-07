"""Permutation-invariant hypergraph potential model.

Hyperedge orders one through four share a scene encoder and combine through
order-specific potential terms. The active orders are configurable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from energy_model.dataset import SCENE_FEATURE_CHANNELS


MODEL_SCHEMA_VERSION = "citydeploy.hypergraph-potential.v3"


@dataclass(frozen=True)
class HypergraphPotentialConfig:
    scene_channels: int = len(SCENE_FEATURE_CHANNELS)
    scene_dim: int = 128
    local_dim: int = 48
    hidden_dim: int = 192
    hyper_hidden_dim: int = 160
    orders: tuple[int, ...] = (1, 2, 3, 4)
    max_num_tx: int = 15
    fourier_frequencies: tuple[float, ...] = (1.0, 2.0, 4.0)
    order_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        orders = tuple(sorted(set(int(order) for order in self.orders)))
        if not orders or any(order not in {1, 2, 3, 4} for order in orders):
            raise ValueError("orders must be a non-empty subset of (1,2,3,4)")
        if self.max_num_tx < max(orders):
            raise ValueError("max_num_tx must be at least the largest enabled order")
        if len(self.order_weights) != 4:
            raise ValueError("order_weights must contain weights for orders 1..4")
        object.__setattr__(self, "orders", orders)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["orders"] = list(self.orders)
        payload["fourier_frequencies"] = list(self.fourier_frequencies)
        payload["order_weights"] = list(self.order_weights)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "HypergraphPotentialConfig":
        data = dict(payload)
        for key in ("orders", "fourier_frequencies", "order_weights"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


class DenseSceneEncoder(nn.Module):
    """Fully convolutional scene encoder with a dense queryable feature field."""

    def __init__(self, in_channels: int, local_dim: int, global_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, local_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, local_dim),
            nn.SiLU(),
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(local_dim, global_dim),
            nn.SiLU(),
        )

    def forward(self, scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense = self.stem(scene)
        return dense, self.global_head(dense)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class HypergraphPotentialModel(nn.Module):
    """Scene-conditioned scalar potential over an unordered variable-size TX set."""

    def __init__(self, config: HypergraphPotentialConfig | None = None) -> None:
        super().__init__()
        self.config = config or HypergraphPotentialConfig()
        cfg = self.config
        self.scene_encoder = DenseSceneEncoder(cfg.scene_channels, cfg.local_dim, cfg.scene_dim)

        pos_dim = 2 + 4 * len(cfg.fourier_frequencies)
        map_dim = 4
        node_in_dim = pos_dim + cfg.local_dim + cfg.scene_dim + map_dim + 1
        self.node_encoder = _mlp(node_in_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.order1_head = _mlp(cfg.hidden_dim, cfg.hyper_hidden_dim, 1)

        # Generic symmetric hyperedge representation used for orders 2..4.
        self.member_encoders = nn.ModuleDict()
        self.hyper_feature_heads = nn.ModuleDict()
        self.hyper_energy_heads = nn.ModuleDict()
        geometry_dim = 2 + 2 + 4
        hyper_in_dim = (
            2 * cfg.hyper_hidden_dim
            + geometry_dim
            + cfg.local_dim
            + cfg.scene_dim
            + map_dim
            + 1
        )
        for order in cfg.orders:
            if order == 1:
                continue
            key = str(order)
            self.member_encoders[key] = _mlp(cfg.hidden_dim, cfg.hyper_hidden_dim, cfg.hyper_hidden_dim)
            self.hyper_feature_heads[key] = _mlp(hyper_in_dim, cfg.hyper_hidden_dim, cfg.hyper_hidden_dim)
            self.hyper_energy_heads[key] = _mlp(cfg.hyper_hidden_dim, cfg.hyper_hidden_dim, 1)

        global_in_dim = cfg.hidden_dim + cfg.scene_dim + map_dim + 1 + 4
        self.global_head = _mlp(global_in_dim, cfg.hidden_dim, 1)
        self.aux_trunk = _mlp(global_in_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(cfg.hidden_dim, 1) for name in ("joint", "pathloss", "sinr", "throughput", "rsrp")}
        )
        self.register_buffer(
            "order_weights",
            torch.tensor(cfg.order_weights, dtype=torch.float32),
            persistent=True,
        )

    @property
    def schema_version(self) -> str:
        return MODEL_SCHEMA_VERSION

    def _position_features(self, xy_norm01: torch.Tensor) -> torch.Tensor:
        features = [xy_norm01]
        for frequency in self.config.fourier_frequencies:
            phase = 2.0 * math.pi * float(frequency) * xy_norm01
            features.extend([torch.sin(phase), torch.cos(phase)])
        return torch.cat(features, dim=-1)

    @staticmethod
    def _map_features(map_size_m: torch.Tensor) -> torch.Tensor:
        if map_size_m.ndim != 2 or map_size_m.shape[-1] != 2:
            raise ValueError(f"map_size_m must have shape [B,2], got {tuple(map_size_m.shape)}")
        size = map_size_m.clamp_min(1e-6)
        diagonal = torch.linalg.vector_norm(size, dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([torch.log(size / 512.0), size / diagonal], dim=-1)

    @staticmethod
    def _sample_dense(dense: torch.Tensor, xy_norm01: torch.Tensor) -> torch.Tensor:
        grid = (xy_norm01 * 2.0 - 1.0).unsqueeze(2)
        sampled = F.grid_sample(
            dense,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)

    @staticmethod
    def _combination_index(n: int, order: int, device: torch.device) -> torch.Tensor:
        combos = list(itertools.combinations(range(n), order))
        if not combos:
            return torch.empty((0, order), dtype=torch.long, device=device)
        return torch.tensor(combos, dtype=torch.long, device=device)

    @staticmethod
    def _hyper_geometry(
        points_norm: torch.Tensor,
        map_size_m: torch.Tensor,
    ) -> torch.Tensor:
        # points_norm [B,C,K,2]. All reductions are symmetric in K.
        centroid = points_norm.mean(dim=2)
        spread = points_norm.std(dim=2, unbiased=False)
        real = (points_norm - 0.5) * map_size_m[:, None, None, :]
        k = points_norm.shape[2]
        pair_index = list(itertools.combinations(range(k), 2))
        pair_dist = torch.stack(
            [torch.linalg.vector_norm(real[:, :, i] - real[:, :, j], dim=-1) for i, j in pair_index],
            dim=-1,
        )
        diagonal = torch.linalg.vector_norm(map_size_m, dim=-1, keepdim=True).clamp_min(1e-6)
        pair_dist = pair_dist / diagonal[:, None, :]
        distance_stats = torch.stack(
            [
                pair_dist.mean(dim=-1),
                pair_dist.std(dim=-1, unbiased=False),
                pair_dist.amin(dim=-1),
                pair_dist.amax(dim=-1),
            ],
            dim=-1,
        )
        return torch.cat([centroid, spread, distance_stats], dim=-1)

    def forward(
        self,
        tx_xy_norm01: torch.Tensor,
        tx_mask: torch.Tensor,
        scene: torch.Tensor,
        map_size_m: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor] | dict[str, torch.Tensor]]:
        if tx_xy_norm01.ndim != 3 or tx_xy_norm01.shape[-1] != 2:
            raise ValueError("tx_xy_norm01 must have shape [B,N,2]")
        if tx_mask.shape != tx_xy_norm01.shape[:2]:
            raise ValueError("tx_mask must have shape [B,N]")
        if scene.shape[0] not in {1, tx_xy_norm01.shape[0]}:
            raise ValueError("scene batch must be one shared scene or match the TX batch")
        if scene.shape[1] != self.config.scene_channels:
            raise ValueError(
                f"scene has {scene.shape[1]} channels, expected {self.config.scene_channels}"
            )

        batch, n, _ = tx_xy_norm01.shape
        dense, scene_global = self.scene_encoder(scene)
        if dense.shape[0] == 1 and batch > 1:
            dense = dense.expand(batch, -1, -1, -1)
            scene_global = scene_global.expand(batch, -1)
        local = self._sample_dense(dense, tx_xy_norm01)
        map_features = self._map_features(map_size_m)
        cardinality = tx_mask.sum(dim=1, keepdim=True).to(tx_xy_norm01.dtype)
        cardinality_feature = cardinality / float(self.config.max_num_tx)
        node_input = torch.cat(
            [
                self._position_features(tx_xy_norm01),
                local,
                scene_global[:, None, :].expand(batch, n, -1),
                map_features[:, None, :].expand(batch, n, -1),
                cardinality_feature[:, None, :].expand(batch, n, -1),
            ],
            dim=-1,
        )
        node = self.node_encoder(node_input)
        node_pool = self._masked_mean(node, tx_mask, dim=1)

        zero = tx_xy_norm01.new_zeros((batch,))
        order_energies: dict[int, torch.Tensor] = {order: zero.clone() for order in range(1, 5)}
        if 1 in self.config.orders:
            order1_each = self.order1_head(node).squeeze(-1)
            order_energies[1] = self._masked_mean(order1_each, tx_mask, dim=1)

        for order in self.config.orders:
            if order == 1 or n < order:
                continue
            combinations = self._combination_index(n, order, tx_xy_norm01.device)
            member = node[:, combinations, :]
            member_mask = tx_mask[:, combinations].all(dim=-1)
            member_encoded = self.member_encoders[str(order)](member)
            member_mean = member_encoded.mean(dim=2)
            member_max = member_encoded.amax(dim=2)
            hyper_points = tx_xy_norm01[:, combinations, :]
            geometry = self._hyper_geometry(hyper_points, map_size_m)
            centroid_local = self._sample_dense(dense, hyper_points.mean(dim=2))
            count_feature = cardinality_feature[:, None, :].expand(batch, combinations.shape[0], -1)
            hyper_input = torch.cat(
                [
                    member_mean,
                    member_max,
                    geometry,
                    centroid_local,
                    scene_global[:, None, :].expand(batch, combinations.shape[0], -1),
                    map_features[:, None, :].expand(batch, combinations.shape[0], -1),
                    count_feature,
                ],
                dim=-1,
            )
            hyper_feature = self.hyper_feature_heads[str(order)](hyper_input)
            hyper_energy_each = self.hyper_energy_heads[str(order)](hyper_feature).squeeze(-1)
            order_energies[order] = self._masked_mean(hyper_energy_each, member_mask, dim=1)

        order_vector = torch.stack([order_energies[order] for order in range(1, 5)], dim=-1)
        global_input = torch.cat(
            [node_pool, scene_global, map_features, cardinality_feature, order_vector], dim=-1
        )
        global_energy = self.global_head(global_input).squeeze(-1)
        enabled = tx_xy_norm01.new_tensor(
            [1.0 if order in self.config.orders else 0.0 for order in range(1, 5)]
        )
        energy = global_energy + (order_vector * self.order_weights * enabled).sum(dim=-1)
        aux_feature = self.aux_trunk(global_input)
        aux = {name: head(aux_feature).squeeze(-1) for name, head in self.aux_heads.items()}
        return {
            "energy": energy,
            "reward": -energy,
            "global_energy": global_energy,
            "order_energies": order_energies,
            "aux": aux,
        }


def assert_permutation_invariant(
    model: HypergraphPotentialModel,
    tx_xy_norm01: torch.Tensor,
    tx_mask: torch.Tensor,
    scene: torch.Tensor,
    map_size_m: torch.Tensor,
    permutations: Iterable[torch.Tensor],
    atol: float = 1e-6,
) -> None:
    reference = model(tx_xy_norm01, tx_mask, scene, map_size_m)["energy"]
    for permutation in permutations:
        candidate = model(
            tx_xy_norm01[:, permutation],
            tx_mask[:, permutation],
            scene,
            map_size_m,
        )["energy"]
        if not torch.allclose(reference, candidate, atol=atol, rtol=0.0):
            difference = float(torch.max(torch.abs(reference - candidate)).item())
            raise RuntimeError(f"permutation invariance violated: max_abs_difference={difference}")

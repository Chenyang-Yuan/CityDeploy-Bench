"""Scene-conditioned relational reward models for unordered TX deployments.

These are task adaptations of Interaction Networks and Neural Relational
Inference.  They predict a scalar deployment reward rather than object
trajectories: transmitters are nodes and pairwise radio interactions are edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from energy_model.dataset import SCENE_FEATURE_CHANNELS
from energy_model.model import DenseSceneEncoder


INTERACTION_NETWORK_SCHEMA = "citydeploy.interaction-network-reward"
NRI_SCHEMA = "citydeploy.nri-relational-reward"


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass(frozen=True)
class RelationalRewardConfig:
    scene_channels: int = len(SCENE_FEATURE_CHANNELS)
    scene_dim: int = 96
    local_dim: int = 48
    node_dim: int = 128
    relation_dim: int = 128
    hidden_dim: int = 192
    max_num_tx: int = 9
    dropout: float = 0.2
    edge_types: int = 4
    gumbel_temperature: float = 0.5
    hard_edges: bool = False
    symmetric_edges: bool = True

    def __post_init__(self) -> None:
        if self.max_num_tx < 1:
            raise ValueError("max_num_tx must be positive")
        if self.edge_types < 2:
            raise ValueError("edge_types must be at least two")
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RelationalRewardConfig":
        return cls(**value)


class _RelationalRewardBase(nn.Module):
    def __init__(self, config: RelationalRewardConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config
        self.scene_encoder = DenseSceneEncoder(cfg.scene_channels, cfg.local_dim, cfg.scene_dim)
        node_input = 2 + cfg.local_dim + cfg.scene_dim + 3
        self.node_encoder = _mlp(node_input, cfg.node_dim, cfg.node_dim)

    def _inputs(
        self,
        tx_xy_norm01: torch.Tensor,
        tx_mask: torch.Tensor,
        scene: torch.Tensor,
        map_size_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tx_xy_norm01.ndim != 3 or tx_xy_norm01.shape[-1] != 2:
            raise ValueError("tx_xy_norm01 must have shape [batch, num_tx, 2]")
        batch, num_tx, _ = tx_xy_norm01.shape
        if num_tx > self.config.max_num_tx:
            raise ValueError(f"num_tx={num_tx} exceeds max_num_tx={self.config.max_num_tx}")
        if tx_mask.shape != (batch, num_tx):
            raise ValueError("tx_mask shape disagrees with TX coordinates")
        if scene.shape[0] not in {1, batch}:
            raise ValueError("scene batch must be one shared scene or match the TX batch")
        if map_size_m.shape not in {(1, 2), (batch, 2)}:
            raise ValueError("map_size_m must have shape [1,2] or [batch,2]")
        dense, scene_global = self.scene_encoder(scene)
        if dense.shape[0] == 1 and batch > 1:
            dense = dense.expand(batch, -1, -1, -1)
            scene_global = scene_global.expand(batch, -1)
        if map_size_m.shape[0] == 1 and batch > 1:
            map_size_m = map_size_m.expand(batch, -1)
        grid = tx_xy_norm01.mul(2.0).sub(1.0).unsqueeze(2)
        local = F.grid_sample(
            dense, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(-1).transpose(1, 2)
        size = torch.log2(map_size_m.clamp_min(1.0) / 128.0) / 2.0
        cardinality = tx_mask.sum(1, keepdim=True).to(tx_xy_norm01.dtype)
        cardinality = cardinality / float(self.config.max_num_tx)
        context = torch.cat([scene_global, size, cardinality], dim=-1)
        node_context = context.unsqueeze(1).expand(-1, num_tx, -1)
        nodes = self.node_encoder(torch.cat([tx_xy_norm01, local, node_context], dim=-1))
        eye = torch.eye(num_tx, dtype=torch.bool, device=tx_mask.device).unsqueeze(0)
        pair_mask = tx_mask.unsqueeze(2) & tx_mask.unsqueeze(1) & ~eye
        return nodes, context, tx_xy_norm01, tx_mask, pair_mask

    @staticmethod
    def _pair_features(nodes: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        receiver = nodes.unsqueeze(2).expand(-1, -1, nodes.shape[1], -1)
        sender = nodes.unsqueeze(1).expand(-1, nodes.shape[1], -1, -1)
        delta = xy.unsqueeze(1) - xy.unsqueeze(2)  # sender minus receiver
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return torch.cat([receiver, sender, delta, delta.abs(), distance], dim=-1)

    @staticmethod
    def _masked_pool(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = mask.unsqueeze(-1)
        mean = (values * expanded).sum(1) / expanded.sum(1).clamp_min(1)
        maximum = values.masked_fill(~expanded, -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum


class InteractionNetworkReward(_RelationalRewardBase):
    """Interaction-Network-style scalar reward predictor over a complete TX graph."""

    def __init__(self, config: RelationalRewardConfig | None = None) -> None:
        super().__init__(config or RelationalRewardConfig())
        cfg = self.config
        pair_dim = 2 * cfg.node_dim + 5
        self.relation_model = _mlp(pair_dim, cfg.hidden_dim, cfg.relation_dim)
        self.object_model = _mlp(
            cfg.node_dim + cfg.relation_dim + cfg.scene_dim + 3,
            cfg.hidden_dim,
            cfg.node_dim,
        )
        head_dim = 2 * cfg.node_dim + cfg.relation_dim + cfg.scene_dim + 3
        self.reward_head = nn.Sequential(
            _mlp(head_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, tx_xy_norm01, tx_mask, scene, map_size_m):
        nodes, context, xy, mask, pair_mask = self._inputs(
            tx_xy_norm01, tx_mask, scene, map_size_m
        )
        relations = self.relation_model(self._pair_features(nodes, xy))
        relations = relations * pair_mask.unsqueeze(-1)
        degree = pair_mask.sum(2, keepdim=True).clamp_min(1)
        effects = relations.sum(2) / degree
        updated = self.object_model(
            torch.cat([nodes, effects, context.unsqueeze(1).expand(-1, nodes.shape[1], -1)], -1)
        )
        node_mean, node_max = self._masked_pool(updated, mask)
        pair_count = pair_mask.sum((1, 2), keepdim=False).unsqueeze(-1).clamp_min(1)
        relation_mean = relations.sum((1, 2)) / pair_count
        return self.reward_head(torch.cat([node_mean, node_max, relation_mean, context], -1)).squeeze(-1)


class NRIRelationalReward(_RelationalRewardBase):
    """NRI-style latent-edge reward predictor adapted to static TX sets.

    Edge type zero is an explicit no-interaction edge.  During training the
    categorical posterior uses Gumbel-Softmax; evaluation uses deterministic
    probabilities so inference and coordinate gradients remain stable.
    """

    def __init__(self, config: RelationalRewardConfig | None = None) -> None:
        super().__init__(config or RelationalRewardConfig())
        cfg = self.config
        pair_dim = 2 * cfg.node_dim + 5
        self.edge_encoder = _mlp(pair_dim, cfg.hidden_dim, cfg.edge_types)
        self.message_models = nn.ModuleList(
            [_mlp(pair_dim, cfg.hidden_dim, cfg.relation_dim) for _ in range(cfg.edge_types - 1)]
        )
        self.node_decoder = _mlp(
            cfg.node_dim + cfg.relation_dim + cfg.scene_dim + 3,
            cfg.hidden_dim,
            cfg.node_dim,
        )
        head_dim = 2 * cfg.node_dim + cfg.relation_dim + cfg.scene_dim + 3
        self.reward_head = nn.Sequential(
            _mlp(head_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_with_aux(self, tx_xy_norm01, tx_mask, scene, map_size_m):
        nodes, context, xy, mask, pair_mask = self._inputs(
            tx_xy_norm01, tx_mask, scene, map_size_m
        )
        pair_features = self._pair_features(nodes, xy)
        logits = self.edge_encoder(pair_features)
        if self.config.symmetric_edges:
            logits = 0.5 * (logits + logits.transpose(1, 2))
        if self.training:
            probabilities = F.gumbel_softmax(
                logits,
                tau=self.config.gumbel_temperature,
                hard=self.config.hard_edges,
                dim=-1,
            )
        else:
            probabilities = logits.softmax(-1)
        messages = torch.zeros(
            *pair_features.shape[:3], self.config.relation_dim,
            dtype=pair_features.dtype,
            device=pair_features.device,
        )
        for edge_type, message_model in enumerate(self.message_models, start=1):
            messages = messages + probabilities[..., edge_type : edge_type + 1] * message_model(pair_features)
        messages = messages * pair_mask.unsqueeze(-1)
        degree = pair_mask.sum(2, keepdim=True).clamp_min(1)
        aggregated = messages.sum(2) / degree
        decoded = self.node_decoder(
            torch.cat([nodes, aggregated, context.unsqueeze(1).expand(-1, nodes.shape[1], -1)], -1)
        )
        node_mean, node_max = self._masked_pool(decoded, mask)
        pair_count = pair_mask.sum((1, 2), keepdim=False).unsqueeze(-1).clamp_min(1)
        message_mean = messages.sum((1, 2)) / pair_count
        reward = self.reward_head(
            torch.cat([node_mean, node_max, message_mean, context], -1)
        ).squeeze(-1)
        log_probabilities = probabilities.clamp_min(1e-8).log()
        log_uniform = -torch.log(probabilities.new_tensor(float(self.config.edge_types)))
        edge_kl = (probabilities * (log_probabilities - log_uniform)).sum(-1)
        edge_kl = (edge_kl * pair_mask).sum((1, 2)) / pair_mask.sum((1, 2)).clamp_min(1)
        entropy = -(probabilities * log_probabilities).sum(-1)
        entropy = (entropy * pair_mask).sum((1, 2)) / pair_mask.sum((1, 2)).clamp_min(1)
        return reward, {"edge_kl": edge_kl, "edge_entropy": entropy, "edge_probabilities": probabilities}

    def forward(self, tx_xy_norm01, tx_mask, scene, map_size_m):
        return self.forward_with_aux(tx_xy_norm01, tx_mask, scene, map_size_m)[0]


def assert_relational_permutation_invariant(
    model: nn.Module,
    tx_xy_norm01: torch.Tensor,
    tx_mask: torch.Tensor,
    scene: torch.Tensor,
    map_size_m: torch.Tensor,
    atol: float = 2e-6,
) -> None:
    model.eval()
    with torch.no_grad():
        reference = model(tx_xy_norm01, tx_mask, scene, map_size_m)
        permutation = torch.randperm(tx_xy_norm01.shape[1], device=tx_xy_norm01.device)
        candidate = model(tx_xy_norm01[:, permutation], tx_mask[:, permutation], scene, map_size_m)
    if not torch.allclose(reference, candidate, atol=atol, rtol=0.0):
        difference = float(torch.max(torch.abs(reference - candidate)).item())
        raise AssertionError(f"relational reward is not permutation invariant; max diff={difference}")

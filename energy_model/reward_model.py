"""Differentiable permutation-invariant reward predictor baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from energy_model.dataset import SCENE_FEATURE_CHANNELS


REWARD_MODEL_SCHEMA_VERSION = "citydeploy.reward-predictor.v1"


@dataclass(frozen=True)
class RewardPredictorConfig:
    scene_channels: int = len(SCENE_FEATURE_CHANNELS)
    scene_feature_dim: int = 96
    tx_hidden_dim: int = 128
    head_hidden_dim: int = 256
    max_num_tx: int = 9
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RewardPredictorConfig":
        return cls(**value)


class RewardPredictor(nn.Module):
    """Predict joint road coverage from a scene and an unordered TX set."""

    def __init__(self, config: RewardPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or RewardPredictorConfig()
        cfg = self.config
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(cfg.scene_channels, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, cfg.scene_feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, cfg.scene_feature_dim),
            nn.SiLU(),
        )
        self.tx_encoder = nn.Sequential(
            nn.Linear(2 + cfg.scene_feature_dim, cfg.tx_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.tx_hidden_dim, cfg.tx_hidden_dim),
            nn.SiLU(),
        )
        head_input = 2 * cfg.tx_hidden_dim + cfg.scene_feature_dim + 3
        self.reward_head = nn.Sequential(
            nn.Linear(head_input, cfg.head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tx_xy_norm01: torch.Tensor,
        tx_mask: torch.Tensor,
        scene: torch.Tensor,
        map_size_m: torch.Tensor,
    ) -> torch.Tensor:
        if tx_xy_norm01.ndim != 3 or tx_xy_norm01.shape[-1] != 2:
            raise ValueError("tx_xy_norm01 must have shape [batch, num_tx, 2]")
        batch, num_tx, _ = tx_xy_norm01.shape
        if num_tx > self.config.max_num_tx:
            raise ValueError(f"num_tx={num_tx} exceeds max_num_tx={self.config.max_num_tx}")
        if tx_mask.shape != (batch, num_tx):
            raise ValueError("tx_mask shape disagrees with TX coordinates")
        if scene.shape[0] not in {1, batch}:
            raise ValueError("scene batch must be one shared scene or match the TX batch")
        dense = self.scene_encoder(scene)
        scene_global = F.adaptive_avg_pool2d(dense, 1).flatten(1)
        if dense.shape[0] == 1 and batch > 1:
            dense = dense.expand(batch, -1, -1, -1)
            scene_global = scene_global.expand(batch, -1)
        grid = tx_xy_norm01.mul(2.0).sub(1.0).unsqueeze(2)
        local = F.grid_sample(
            dense,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(-1).transpose(1, 2)
        encoded = self.tx_encoder(torch.cat([tx_xy_norm01, local], dim=-1))
        mask = tx_mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1)
        pooled_mean = (encoded * mask).sum(dim=1) / denominator
        pooled_max = encoded.masked_fill(~mask, -torch.inf).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        cardinality = tx_mask.sum(dim=1, keepdim=True).to(tx_xy_norm01.dtype)
        cardinality = cardinality / float(self.config.max_num_tx)
        size_feature = torch.log2(map_size_m.clamp_min(1.0) / 128.0) / 2.0
        features = torch.cat(
            [pooled_mean, pooled_max, scene_global, cardinality, size_feature], dim=-1
        )
        return self.reward_head(features).squeeze(-1)


def assert_reward_permutation_invariant(
    model: RewardPredictor,
    tx_xy_norm01: torch.Tensor,
    tx_mask: torch.Tensor,
    scene: torch.Tensor,
    map_size_m: torch.Tensor,
    atol: float = 1e-6,
) -> None:
    model.eval()
    with torch.no_grad():
        reference = model(tx_xy_norm01, tx_mask, scene, map_size_m)
        permutation = torch.randperm(tx_xy_norm01.shape[1], device=tx_xy_norm01.device)
        candidate = model(
            tx_xy_norm01[:, permutation], tx_mask[:, permutation], scene, map_size_m
        )
    if not torch.allclose(reference, candidate, atol=atol, rtol=0.0):
        difference = float(torch.max(torch.abs(reference - candidate)).item())
        raise AssertionError(f"reward predictor is not permutation invariant; max diff={difference}")

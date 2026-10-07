"""Checkpoint registry shared by training, inference, and samplers."""

from __future__ import annotations

from pathlib import Path
from citydeploy.schema import canonical_schema

import torch

from energy_model.model import MODEL_SCHEMA_VERSION, HypergraphPotentialConfig, HypergraphPotentialModel
from energy_model.relational_models import (
    INTERACTION_NETWORK_SCHEMA,
    NRI_SCHEMA,
    InteractionNetworkReward,
    NRIRelationalReward,
    RelationalRewardConfig,
)
from energy_model.reward_model import REWARD_MODEL_SCHEMA_VERSION, RewardPredictor, RewardPredictorConfig


def build_guidance_model(schema: str, config: dict) -> tuple[str, torch.nn.Module]:
    schema = canonical_schema(schema)
    if schema == MODEL_SCHEMA_VERSION:
        return "hypergraph_potential", HypergraphPotentialModel(HypergraphPotentialConfig.from_dict(config))
    if schema == REWARD_MODEL_SCHEMA_VERSION:
        return "reward_predictor", RewardPredictor(RewardPredictorConfig.from_dict(config))
    if schema == INTERACTION_NETWORK_SCHEMA:
        return "interaction_network", InteractionNetworkReward(RelationalRewardConfig.from_dict(config))
    if schema == NRI_SCHEMA:
        return "nri", NRIRelationalReward(RelationalRewardConfig.from_dict(config))
    supported = [MODEL_SCHEMA_VERSION, REWARD_MODEL_SCHEMA_VERSION, INTERACTION_NETWORK_SCHEMA, NRI_SCHEMA]
    raise RuntimeError(f"unsupported checkpoint schema {schema!r}; supported: {supported}")


def load_guidance_checkpoint(path: Path, device: torch.device) -> tuple[str, torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    name, model = build_guidance_model(
        str(checkpoint.get("model_schema_version")), checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return name, model.to(device).eval(), checkpoint


def predict_reward(name: str, model: torch.nn.Module, *args) -> torch.Tensor:
    result = model(*args)
    return result["reward"] if name == "hypergraph_potential" else result


def assert_guidance_permutation_invariant(
    name: str,
    model: torch.nn.Module,
    tx_xy_norm01: torch.Tensor,
    tx_mask: torch.Tensor,
    scene: torch.Tensor,
    map_size_m: torch.Tensor,
    atol: float = 2e-6,
) -> None:
    model.eval()
    with torch.no_grad():
        reference = predict_reward(name, model, tx_xy_norm01, tx_mask, scene, map_size_m)
        permutation = torch.randperm(tx_xy_norm01.shape[1], device=tx_xy_norm01.device)
        candidate = predict_reward(
            name,
            model,
            tx_xy_norm01[:, permutation],
            tx_mask[:, permutation],
            scene,
            map_size_m,
        )
    if not torch.allclose(reference, candidate, atol=atol, rtol=0.0):
        difference = float(torch.max(torch.abs(reference - candidate)).item())
        raise AssertionError(f"{name} is not permutation invariant; max diff={difference}")

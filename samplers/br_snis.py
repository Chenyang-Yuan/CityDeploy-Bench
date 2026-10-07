"""Bias-reduced self-normalized importance resampling for deployments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from samplers.common import (
    RewardFunction,
    update_topk_archive,
    validate_sampler_inputs,
)


@dataclass(frozen=True)
class BRSNISConfig:
    rounds: int = 100
    proposal_particles: int = 32
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.rounds <= 0 or self.proposal_particles <= 0:
            raise ValueError("rounds and proposal_particles must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")


def _categorical_indices(
    weights: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    uniform = torch.rand(
        weights.shape[0],
        device=weights.device,
        dtype=weights.dtype,
        generator=generator,
    )
    cumulative = weights.cumsum(dim=1)
    cumulative[:, -1] = 1.0
    return (cumulative < uniform[:, None]).sum(dim=1).clamp_max(weights.shape[1] - 1)


def sample_br_snis(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: BRSNISConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Run parallel ISIR chains and return high-reward concrete particles.

    BR-SNIS is an expectation estimator rather than a proposal generator. For
    deployment it is adapted as iterated sampling-importance resampling (ISIR):
    each chain retains one conditioning deployment and combines it with fresh
    uniform proposals. The selected output is always a real particle; TX
    coordinates are never averaged across permutation-equivalent modes.
    """
    cfg = config or BRSNISConfig()
    validate_sampler_inputs(num_particles, num_tx)
    conditioning = torch.rand(
        (num_particles, num_tx, 2),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    with torch.no_grad():
        conditioning_reward = reward_fn(conditioning).detach()
    archive_points, archive_reward = update_topk_archive(
        None, None, conditioning, conditioning_reward, num_particles
    )
    trace: list[dict[str, float]] = []
    last_selected_weight = torch.ones_like(conditioning_reward)

    for round_index in range(1, cfg.rounds + 1):
        proposals = torch.rand(
            (num_particles, cfg.proposal_particles, num_tx, 2),
            device=device,
            dtype=conditioning.dtype,
            generator=generator,
        )
        pool = torch.cat([conditioning[:, None], proposals], dim=1)
        with torch.no_grad():
            reward = reward_fn(pool.flatten(0, 1)).reshape(
                num_particles, cfg.proposal_particles + 1
            ).detach()
        # The proposal is uniform over [0,1]^(2N), so its log density is zero.
        log_weights = float(cfg.beta) * reward
        weights = torch.softmax(log_weights, dim=1)
        selected = _categorical_indices(weights, generator)
        chain = torch.arange(num_particles, device=device)
        conditioning = pool[chain, selected].detach()
        conditioning_reward = reward[chain, selected].detach()
        last_selected_weight = weights[chain, selected].detach()

        archive_points, archive_reward = update_topk_archive(
            archive_points,
            archive_reward,
            pool.flatten(0, 1),
            reward.flatten(),
            num_particles,
        )
        ess = 1.0 / weights.square().sum(dim=1).clamp_min(1e-12)
        trace.append(
            {
                "round": float(round_index),
                "reward_mean": float(conditioning_reward.mean().item()),
                "reward_max": float(conditioning_reward.max().item()),
                "best_reward_max": float(archive_reward.max().item()),
                "importance_ess_mean": float(ess.mean().item()),
                "selected_weight_mean": float(last_selected_weight.mean().item()),
            }
        )

    return {
        "best_tx_xy_norm01": archive_points,
        "best_reward": archive_reward,
        "final_tx_xy_norm01": conditioning,
        "final_reward": conditioning_reward,
        "final_weights": last_selected_weight,
        "trace": trace,
    }


"""Annealed SMC with Gaussian-RWMH or Langevin-MALA rejuvenation kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from samplers.common import (
    RewardFunction,
    log_target,
    log_target_and_grad,
    systematic_resample,
    unconstrained_to_unit,
    unit_to_unconstrained,
    update_topk_archive,
    validate_sampler_inputs,
)


SMCKernel = Literal["gaussian", "langevin"]


@dataclass(frozen=True)
class AnnealedSMCConfig:
    steps: int = 100
    beta: float = 1.0
    ess_threshold: float = 0.5
    proposal_scale_start: float = 0.45
    proposal_scale_end: float = 0.08
    mutation_steps: int = 1
    tempering_power: float = 1.0
    kernel: SMCKernel = "gaussian"

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.mutation_steps <= 0:
            raise ValueError("steps and mutation_steps must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < self.ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be in (0,1]")
        if self.proposal_scale_start <= 0.0 or self.proposal_scale_end <= 0.0:
            raise ValueError("proposal scales must be positive")
        if self.tempering_power <= 0.0:
            raise ValueError("tempering_power must be positive")
        if self.kernel not in {"gaussian", "langevin"}:
            raise ValueError(f"unsupported SMC kernel: {self.kernel}")


def _scale(config: AnnealedSMCConfig, fraction: float) -> float:
    start = math.log(config.proposal_scale_start)
    end = math.log(config.proposal_scale_end)
    return math.exp(start + fraction * (end - start))


def _effective_sample_size(log_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(log_weights, dim=0)
    ess = 1.0 / weights.square().sum().clamp_min(1e-12)
    return ess, weights


def _gaussian_move(
    reward_fn: RewardFunction,
    state: torch.Tensor,
    reward: torch.Tensor,
    beta: float,
    scale: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_target, _ = log_target(reward_fn, state, beta)
    proposal = state + float(scale) * torch.randn(
        state.shape, device=state.device, dtype=state.dtype, generator=generator
    )
    proposal_target, proposal_reward = log_target(reward_fn, proposal, beta)
    log_acceptance = proposal_target - current_target
    uniform = torch.rand(
        log_acceptance.shape,
        device=state.device,
        dtype=state.dtype,
        generator=generator,
    ).clamp_min(1e-12)
    accept = torch.log(uniform) < torch.minimum(log_acceptance, torch.zeros_like(log_acceptance))
    state = torch.where(accept[:, None, None], proposal, state).detach()
    reward = torch.where(accept, proposal_reward, reward).detach()
    return state, reward, accept


def _langevin_move(
    reward_fn: RewardFunction,
    state: torch.Tensor,
    reward: torch.Tensor,
    beta: float,
    scale: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_target, _, current_gradient = log_target_and_grad(reward_fn, state, beta)
    variance = float(scale) ** 2
    current_mean = state + 0.5 * variance * current_gradient
    proposal = current_mean + float(scale) * torch.randn(
        state.shape, device=state.device, dtype=state.dtype, generator=generator
    )
    proposal_target, proposal_reward, proposal_gradient = log_target_and_grad(
        reward_fn, proposal, beta
    )
    proposal_mean_reverse = proposal + 0.5 * variance * proposal_gradient
    log_q_forward = -0.5 / variance * (proposal - current_mean).flatten(1).square().sum(dim=1)
    log_q_reverse = -0.5 / variance * (state - proposal_mean_reverse).flatten(1).square().sum(dim=1)
    log_acceptance = proposal_target - current_target + log_q_reverse - log_q_forward
    uniform = torch.rand(
        log_acceptance.shape,
        device=state.device,
        dtype=state.dtype,
        generator=generator,
    ).clamp_min(1e-12)
    accept = torch.log(uniform) < torch.minimum(log_acceptance, torch.zeros_like(log_acceptance))
    state = torch.where(accept[:, None, None], proposal, state).detach()
    reward = torch.where(accept, proposal_reward, reward).detach()
    return state, reward, accept


def sample_annealed_smc(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: AnnealedSMCConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Approximate ``exp(beta*reward)`` using annealing and ESS resampling.

    Gaussian mode uses a symmetric random-walk Metropolis kernel. Langevin mode
    uses MALA, including the forward/reverse proposal correction. Both operate
    in unconstrained logistic coordinates, so all returned positions lie in the
    normalized unit square without a non-reversible boundary projection.
    """
    cfg = config or AnnealedSMCConfig()
    validate_sampler_inputs(num_particles, num_tx)
    initial = torch.rand(
        (num_particles, num_tx, 2),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    state = unit_to_unconstrained(initial)
    with torch.no_grad():
        reward = reward_fn(initial).detach()
    archive_points, archive_reward = update_topk_archive(
        None, None, initial, reward, num_particles
    )
    log_weights = torch.zeros(num_particles, device=device, dtype=state.dtype)
    trace: list[dict[str, float]] = []
    previous_beta = 0.0

    for step in range(1, cfg.steps + 1):
        fraction = step / cfg.steps
        current_beta = cfg.beta * fraction**cfg.tempering_power
        log_weights = log_weights + (current_beta - previous_beta) * reward
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        ess_before, weights = _effective_sample_size(log_weights)
        resampled = bool(ess_before < cfg.ess_threshold * num_particles)
        if resampled:
            index = systematic_resample(weights, generator)
            state = state[index].clone()
            reward = reward[index].clone()
            log_weights.zero_()

        scale = _scale(cfg, fraction)
        accepted = []
        for _ in range(cfg.mutation_steps):
            if cfg.kernel == "gaussian":
                state, reward, accept = _gaussian_move(
                    reward_fn, state, reward, current_beta, scale, generator
                )
            else:
                state, reward, accept = _langevin_move(
                    reward_fn, state, reward, current_beta, scale, generator
                )
            accepted.append(accept.to(torch.float32).mean())

        points = unconstrained_to_unit(state)
        archive_points, archive_reward = update_topk_archive(
            archive_points, archive_reward, points, reward, num_particles
        )
        ess_after, _ = _effective_sample_size(log_weights)
        trace.append(
            {
                "step": float(step),
                "beta": float(current_beta),
                "proposal_scale": float(scale),
                "ess_before_resampling": float(ess_before.item()),
                "ess_after_resampling": float(ess_after.item()),
                "resampled": float(resampled),
                "acceptance_rate": float(torch.stack(accepted).mean().item()),
                "reward_mean": float(reward.mean().item()),
                "reward_max": float(reward.max().item()),
                "best_reward_max": float(archive_reward.max().item()),
            }
        )
        previous_beta = current_beta

    _, final_weights = _effective_sample_size(log_weights)
    return {
        "best_tx_xy_norm01": archive_points,
        "best_reward": archive_reward,
        "final_tx_xy_norm01": unconstrained_to_unit(state).detach(),
        "final_reward": reward.detach(),
        "final_weights": final_weights.detach(),
        "trace": trace,
    }


def sample_smc_gaussian(*args, config: AnnealedSMCConfig | None = None, **kwargs):
    cfg = config or AnnealedSMCConfig(kernel="gaussian")
    if cfg.kernel != "gaussian":
        raise ValueError("sample_smc_gaussian requires config.kernel='gaussian'")
    return sample_annealed_smc(*args, config=cfg, **kwargs)


def sample_smc_langevin(*args, config: AnnealedSMCConfig | None = None, **kwargs):
    cfg = config or AnnealedSMCConfig(kernel="langevin")
    if cfg.kernel != "langevin":
        raise ValueError("sample_smc_langevin requires config.kernel='langevin'")
    return sample_annealed_smc(*args, config=cfg, **kwargs)


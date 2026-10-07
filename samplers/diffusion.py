"""Reward-induced variance-exploding diffusion for TX deployment sets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from samplers.common import (
    BoundaryMode,
    RewardFunction,
    apply_boundary,
    validate_sampler_inputs,
)


@dataclass(frozen=True)
class VEDiffusionConfig:
    steps: int = 100
    mc_samples: int = 10
    beta: float = 1.0
    sigma_max: float = 0.35
    sigma_min: float = 0.005
    antithetic: bool = True
    boundary: BoundaryMode = "reflect"
    stochastic_final_step: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.mc_samples <= 0:
            raise ValueError("steps and mc_samples must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.sigma_max <= 0.0 or self.sigma_min <= 0.0:
            raise ValueError("sigmas must be positive")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")


def _normal_samples_like(center: torch.Tensor, count: int, antithetic: bool) -> torch.Tensor:
    if not antithetic:
        return torch.randn(
            (center.shape[0], count, *center.shape[1:]),
            device=center.device,
            dtype=center.dtype,
        )
    half = (count + 1) // 2
    noise = torch.randn(
        (center.shape[0], half, *center.shape[1:]),
        device=center.device,
        dtype=center.dtype,
    )
    return torch.cat([noise, -noise], dim=1)[:, :count]


def estimate_mollified_score(
    reward_fn: RewardFunction,
    center: torch.Tensor,
    sigma: float,
    beta: float,
    mc_samples: int,
    *,
    antithetic: bool = True,
    boundary: BoundaryMode = "reflect",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate the score of the Gaussian-mollified Gibbs density."""
    if center.ndim != 3 or center.shape[-1] != 2:
        raise ValueError("center must have shape [B,N,2]")
    noise = _normal_samples_like(center, mc_samples, antithetic)
    raw = (center[:, None] + float(sigma) * noise).detach().requires_grad_(True)
    samples = apply_boundary(raw, boundary)
    batch, count, num_tx, _ = samples.shape
    reward = reward_fn(samples.reshape(batch * count, num_tx, 2)).reshape(batch, count)
    if reward.shape != (batch, count):
        raise ValueError(f"reward_fn must return [B], got {tuple(reward.shape)}")
    reward_gradient = torch.autograd.grad(reward.sum(), raw, create_graph=False)[0]
    weight = torch.softmax(float(beta) * reward, dim=1)
    score = float(beta) * (weight[:, :, None, None] * reward_gradient).sum(dim=1)
    effective_sample_size = 1.0 / weight.square().sum(dim=1).clamp_min(1e-12)
    return score.detach(), {
        "reward_mean": reward.detach().mean(dim=1),
        "reward_max": reward.detach().amax(dim=1),
        "effective_sample_size": effective_sample_size.detach(),
        "weight_max": weight.detach().amax(dim=1),
    }


def ve_noise_schedule(
    config: VEDiffusionConfig, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.exp(
        torch.linspace(
            math.log(config.sigma_max),
            math.log(config.sigma_min),
            config.steps + 1,
            device=device,
            dtype=dtype,
        )
    )


def sample_reward_guided_ve(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: VEDiffusionConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Sample normalized unordered TX sets with a reward-induced reverse VE SDE."""
    cfg = config or VEDiffusionConfig()
    validate_sampler_inputs(num_particles, num_tx)
    x = 0.5 + cfg.sigma_max * torch.randn(
        (num_particles, num_tx, 2),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    x = apply_boundary(x, cfg.boundary).detach()
    sigmas = ve_noise_schedule(cfg, device, x.dtype)
    trace: list[dict[str, float]] = []

    with torch.no_grad():
        initial_reward = reward_fn(x).detach()
    best_x = x.clone()
    best_reward = initial_reward.clone()

    for step in range(cfg.steps):
        sigma_t = float(sigmas[step].item())
        sigma_next = float(sigmas[step + 1].item())
        delta_variance = max(sigma_t * sigma_t - sigma_next * sigma_next, 1e-12)
        score, score_stats = estimate_mollified_score(
            reward_fn,
            x,
            sigma_t,
            cfg.beta,
            cfg.mc_samples,
            antithetic=cfg.antithetic,
            boundary=cfg.boundary,
        )
        add_noise = step < cfg.steps - 1 or cfg.stochastic_final_step
        noise = (
            torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
            if add_noise
            else 0.0
        )
        x = apply_boundary(
            x + delta_variance * score + math.sqrt(delta_variance) * noise,
            cfg.boundary,
        ).detach()
        with torch.no_grad():
            reward = reward_fn(x).detach()
        improved = reward > best_reward
        best_reward[improved] = reward[improved]
        best_x[improved] = x[improved]
        trace.append(
            {
                "step": float(step + 1),
                "sigma": sigma_t,
                "reward_mean": float(reward.mean().item()),
                "reward_max": float(reward.max().item()),
                "best_reward_max": float(best_reward.max().item()),
                "score_norm_mean": float(
                    torch.linalg.vector_norm(score, dim=-1).mean().item()
                ),
                "mc_ess_mean": float(score_stats["effective_sample_size"].mean().item()),
                "mc_weight_max_mean": float(score_stats["weight_max"].mean().item()),
            }
        )
    return {
        "best_tx_xy_norm01": best_x,
        "best_reward": best_reward,
        "final_tx_xy_norm01": x,
        "final_reward": reward,
        "trace": trace,
    }


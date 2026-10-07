"""Shared contracts and transforms for deployment samplers."""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F


BoundaryMode = Literal["reflect", "clamp", "none"]
RewardFunction = Callable[[torch.Tensor], torch.Tensor]


def reflect_unit_interval(value: torch.Tensor) -> torch.Tensor:
    """Reflect real values onto ``[0, 1]`` with an a.e. gradient."""
    wrapped = torch.remainder(value, 2.0)
    return torch.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def apply_boundary(value: torch.Tensor, mode: BoundaryMode) -> torch.Tensor:
    if mode == "reflect":
        return reflect_unit_interval(value)
    if mode == "clamp":
        return value.clamp(0.0, 1.0)
    if mode == "none":
        return value
    raise ValueError(f"unsupported boundary mode: {mode}")


def validate_sampler_inputs(num_particles: int, num_tx: int) -> None:
    if num_particles <= 0 or num_tx <= 0:
        raise ValueError("num_particles and num_tx must be positive")


def deployment_constraint_penalty(
    tx_xy_norm01: torch.Tensor,
    feasible_mask_hw: torch.Tensor,
    map_size_m: torch.Tensor,
    min_tx_distance_m: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable road-feasibility and minimum-separation penalties."""
    batch, num_tx, _ = tx_xy_norm01.shape
    mask = feasible_mask_hw
    if mask.ndim == 2:
        mask = mask[None, None].expand(batch, 1, -1, -1)
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1)
    grid = (tx_xy_norm01 * 2.0 - 1.0).unsqueeze(2)
    feasibility = F.grid_sample(
        mask.to(tx_xy_norm01.dtype),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).squeeze(-1)
    nondeployable = (1.0 - feasibility).clamp_min(0.0).mean(dim=1)

    min_distance = tx_xy_norm01.new_zeros((batch,))
    if num_tx >= 2 and min_tx_distance_m > 0.0:
        real = (tx_xy_norm01 - 0.5) * map_size_m[:, None, :]
        index = torch.combinations(torch.arange(num_tx, device=tx_xy_norm01.device), r=2)
        distance = torch.linalg.vector_norm(
            real[:, index[:, 0]] - real[:, index[:, 1]], dim=-1
        )
        violation = F.relu(float(min_tx_distance_m) - distance) / float(min_tx_distance_m)
        min_distance = violation.square().mean(dim=1)
    return nondeployable + min_distance, {
        "nondeployable": nondeployable,
        "minimum_distance": min_distance,
    }


def unit_to_unconstrained(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Map an open unit-cube point to an unconstrained logistic coordinate."""
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def unconstrained_to_unit(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(value)


def logistic_log_abs_det_jacobian(value: torch.Tensor) -> torch.Tensor:
    """Log determinant for ``x = sigmoid(value)``, reduced per particle."""
    return (F.logsigmoid(value) + F.logsigmoid(-value)).flatten(1).sum(dim=1)


def log_target(
    reward_fn: RewardFunction,
    unconstrained: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = unconstrained_to_unit(unconstrained)
    reward = reward_fn(points)
    if reward.shape != (unconstrained.shape[0],):
        raise ValueError(f"reward_fn must return [B], got {tuple(reward.shape)}")
    target = float(beta) * reward + logistic_log_abs_det_jacobian(unconstrained)
    return target, reward


def log_target_and_grad(
    reward_fn: RewardFunction,
    unconstrained: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = unconstrained.detach().requires_grad_(True)
    target, reward = log_target(reward_fn, raw, beta)
    gradient = torch.autograd.grad(target.sum(), raw, create_graph=False)[0]
    return target.detach(), reward.detach(), gradient.detach()


def update_topk_archive(
    archive_points: torch.Tensor | None,
    archive_reward: torch.Tensor | None,
    points: torch.Tensor,
    reward: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the highest-reward concrete particles seen so far."""
    points = points.detach()
    reward = reward.detach()
    if archive_points is not None and archive_reward is not None:
        points = torch.cat([archive_points, points], dim=0)
        reward = torch.cat([archive_reward, reward], dim=0)
    count = min(int(capacity), int(reward.numel()))
    values, index = torch.topk(reward, k=count, largest=True, sorted=True)
    return points[index].clone(), values.clone()


def systematic_resample(
    normalized_weights: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Systematic resampling indices for a one-dimensional particle weight vector."""
    count = normalized_weights.numel()
    offset = torch.rand(
        (), device=normalized_weights.device, dtype=normalized_weights.dtype, generator=generator
    ) / count
    positions = offset + torch.arange(
        count, device=normalized_weights.device, dtype=normalized_weights.dtype
    ) / count
    cumulative = normalized_weights.cumsum(dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions).clamp_max(count - 1)


"""No-U-Turn sampling adapted to normalized unordered TX deployment sets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from samplers.common import (
    RewardFunction,
    log_target_and_grad,
    unconstrained_to_unit,
    unit_to_unconstrained,
    update_topk_archive,
    validate_sampler_inputs,
)


@dataclass(frozen=True)
class NUTSConfig:
    warmup_steps: int = 50
    sample_steps: int = 50
    beta: float = 1.0
    initial_step_size: float = 0.15
    target_acceptance: float = 0.8
    max_tree_depth: int = 7
    divergence_threshold: float = 1000.0
    adapt_step_size: bool = True

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.sample_steps <= 0:
            raise ValueError("warmup_steps must be non-negative and sample_steps positive")
        if self.beta <= 0.0 or self.initial_step_size <= 0.0:
            raise ValueError("beta and initial_step_size must be positive")
        if not 0.0 < self.target_acceptance < 1.0:
            raise ValueError("target_acceptance must be in (0,1)")
        if self.max_tree_depth <= 0 or self.divergence_threshold <= 0.0:
            raise ValueError("max_tree_depth and divergence_threshold must be positive")


@dataclass
class _Tree:
    left_z: torch.Tensor
    left_r: torch.Tensor
    left_grad: torch.Tensor
    right_z: torch.Tensor
    right_r: torch.Tensor
    right_grad: torch.Tensor
    proposal_z: torch.Tensor
    proposal_reward: torch.Tensor
    count: int
    continue_tree: bool
    acceptance_sum: float
    acceptance_count: int


def _uniform_scalar(
    device: torch.device, dtype: torch.dtype, generator: torch.Generator | None
) -> torch.Tensor:
    return torch.rand((), device=device, dtype=dtype, generator=generator).clamp_min(1e-12)


def _target_grad_single(
    reward_fn: RewardFunction, state: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target, reward, gradient = log_target_and_grad(reward_fn, state[None], beta)
    return target[0], reward[0], gradient[0]


def _leapfrog(
    reward_fn: RewardFunction,
    state: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    half = momentum + 0.5 * float(step_size) * gradient
    next_state = state + float(step_size) * half
    target, reward, next_gradient = _target_grad_single(reward_fn, next_state, beta)
    next_momentum = half + 0.5 * float(step_size) * next_gradient
    return next_state.detach(), next_momentum.detach(), next_gradient, target, reward


def _is_not_u_turn(
    left_z: torch.Tensor,
    right_z: torch.Tensor,
    left_r: torch.Tensor,
    right_r: torch.Tensor,
) -> bool:
    displacement = right_z - left_z
    return bool(
        (displacement.flatten().dot(left_r.flatten()) >= 0.0)
        and (displacement.flatten().dot(right_r.flatten()) >= 0.0)
    )


def _build_tree(
    reward_fn: RewardFunction,
    state: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    log_slice: torch.Tensor,
    direction: int,
    depth: int,
    step_size: float,
    beta: float,
    initial_joint: torch.Tensor,
    divergence_threshold: float,
    generator: torch.Generator | None,
) -> _Tree:
    if depth == 0:
        next_state, next_momentum, next_gradient, target, reward = _leapfrog(
            reward_fn,
            state,
            momentum,
            gradient,
            direction * step_size,
            beta,
        )
        joint = target - 0.5 * next_momentum.square().sum()
        count = int(bool(log_slice <= joint))
        continue_tree = bool(log_slice < joint + divergence_threshold) and bool(
            torch.isfinite(joint)
        )
        log_acceptance = torch.minimum(joint - initial_joint, joint.new_zeros(()))
        acceptance = float(torch.exp(log_acceptance).item()) if torch.isfinite(joint) else 0.0
        return _Tree(
            next_state,
            next_momentum,
            next_gradient,
            next_state,
            next_momentum,
            next_gradient,
            next_state,
            reward.detach(),
            count,
            continue_tree,
            acceptance,
            1,
        )

    first = _build_tree(
        reward_fn,
        state,
        momentum,
        gradient,
        log_slice,
        direction,
        depth - 1,
        step_size,
        beta,
        initial_joint,
        divergence_threshold,
        generator,
    )
    if not first.continue_tree:
        return first

    if direction < 0:
        second = _build_tree(
            reward_fn,
            first.left_z,
            first.left_r,
            first.left_grad,
            log_slice,
            direction,
            depth - 1,
            step_size,
            beta,
            initial_joint,
            divergence_threshold,
            generator,
        )
        left_z, left_r, left_grad = second.left_z, second.left_r, second.left_grad
        right_z, right_r, right_grad = first.right_z, first.right_r, first.right_grad
    else:
        second = _build_tree(
            reward_fn,
            first.right_z,
            first.right_r,
            first.right_grad,
            log_slice,
            direction,
            depth - 1,
            step_size,
            beta,
            initial_joint,
            divergence_threshold,
            generator,
        )
        left_z, left_r, left_grad = first.left_z, first.left_r, first.left_grad
        right_z, right_r, right_grad = second.right_z, second.right_r, second.right_grad

    proposal_z = first.proposal_z
    proposal_reward = first.proposal_reward
    total = first.count + second.count
    if second.count > 0 and total > 0:
        probability = second.count / total
        if bool(_uniform_scalar(state.device, state.dtype, generator) < probability):
            proposal_z = second.proposal_z
            proposal_reward = second.proposal_reward
    continue_tree = (
        first.continue_tree
        and second.continue_tree
        and _is_not_u_turn(left_z, right_z, left_r, right_r)
    )
    return _Tree(
        left_z,
        left_r,
        left_grad,
        right_z,
        right_r,
        right_grad,
        proposal_z,
        proposal_reward,
        total,
        continue_tree,
        first.acceptance_sum + second.acceptance_sum,
        first.acceptance_count + second.acceptance_count,
    )


def _nuts_transition(
    reward_fn: RewardFunction,
    state: torch.Tensor,
    step_size: float,
    config: NUTSConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    target, current_reward, gradient = _target_grad_single(reward_fn, state, config.beta)
    momentum = torch.randn(
        state.shape, device=state.device, dtype=state.dtype, generator=generator
    )
    initial_joint = target - 0.5 * momentum.square().sum()
    log_slice = initial_joint + torch.log(
        _uniform_scalar(state.device, state.dtype, generator)
    )

    left_z = right_z = state
    left_r = right_r = momentum
    left_grad = right_grad = gradient
    proposal_z = state
    proposal_reward = current_reward.detach()
    count = 1
    continue_tree = True
    acceptance_sum = 0.0
    acceptance_count = 0
    used_depth = 0

    for depth in range(config.max_tree_depth):
        if not continue_tree:
            break
        direction = -1 if bool(_uniform_scalar(state.device, state.dtype, generator) < 0.5) else 1
        if direction < 0:
            tree = _build_tree(
                reward_fn,
                left_z,
                left_r,
                left_grad,
                log_slice,
                direction,
                depth,
                step_size,
                config.beta,
                initial_joint,
                config.divergence_threshold,
                generator,
            )
            left_z, left_r, left_grad = tree.left_z, tree.left_r, tree.left_grad
        else:
            tree = _build_tree(
                reward_fn,
                right_z,
                right_r,
                right_grad,
                log_slice,
                direction,
                depth,
                step_size,
                config.beta,
                initial_joint,
                config.divergence_threshold,
                generator,
            )
            right_z, right_r, right_grad = tree.right_z, tree.right_r, tree.right_grad
        if tree.continue_tree and tree.count > 0:
            probability = min(1.0, tree.count / max(1, count + tree.count))
            if bool(_uniform_scalar(state.device, state.dtype, generator) < probability):
                proposal_z = tree.proposal_z
                proposal_reward = tree.proposal_reward
        count += tree.count
        continue_tree = tree.continue_tree and _is_not_u_turn(
            left_z, right_z, left_r, right_r
        )
        acceptance_sum += tree.acceptance_sum
        acceptance_count += tree.acceptance_count
        used_depth = depth + 1

    acceptance = acceptance_sum / max(1, acceptance_count)
    return proposal_z.detach(), proposal_reward.detach(), acceptance, used_depth


def sample_nuts(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: NUTSConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    """Run parallel NUTS chains and retain the best post-warmup particles."""
    cfg = config or NUTSConfig()
    validate_sampler_inputs(num_particles, num_tx)
    initial = torch.rand(
        (num_particles, num_tx, 2),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    states = unit_to_unconstrained(initial)
    with torch.no_grad():
        rewards = reward_fn(initial).detach()

    step_sizes = [cfg.initial_step_size for _ in range(num_particles)]
    mu = [math.log(10.0 * cfg.initial_step_size) for _ in range(num_particles)]
    h_bar = [0.0 for _ in range(num_particles)]
    log_epsilon_bar = [math.log(cfg.initial_step_size) for _ in range(num_particles)]
    archive_points: torch.Tensor | None = None
    archive_reward: torch.Tensor | None = None
    trace: list[dict[str, float]] = []
    total_steps = cfg.warmup_steps + cfg.sample_steps

    for iteration in range(1, total_steps + 1):
        acceptances = []
        depths = []
        next_states = []
        next_rewards = []
        for chain in range(num_particles):
            next_state, next_reward, acceptance, depth = _nuts_transition(
                reward_fn, states[chain], step_sizes[chain], cfg, generator
            )
            next_states.append(next_state)
            next_rewards.append(next_reward)
            acceptances.append(acceptance)
            depths.append(depth)

            if cfg.adapt_step_size and iteration <= cfg.warmup_steps:
                eta_h = 1.0 / (iteration + 10.0)
                h_bar[chain] = (1.0 - eta_h) * h_bar[chain] + eta_h * (
                    cfg.target_acceptance - acceptance
                )
                log_epsilon = mu[chain] - math.sqrt(iteration) / 0.05 * h_bar[chain]
                eta = iteration ** -0.75
                log_epsilon_bar[chain] = (
                    eta * log_epsilon + (1.0 - eta) * log_epsilon_bar[chain]
                )
                step_sizes[chain] = min(2.0, max(1e-4, math.exp(log_epsilon)))
            elif cfg.adapt_step_size and iteration == cfg.warmup_steps + 1:
                step_sizes[chain] = min(
                    2.0, max(1e-4, math.exp(log_epsilon_bar[chain]))
                )

        states = torch.stack(next_states)
        rewards = torch.stack(next_rewards)
        points = unconstrained_to_unit(states)
        if iteration > cfg.warmup_steps:
            archive_points, archive_reward = update_topk_archive(
                archive_points,
                archive_reward,
                points,
                rewards,
                num_particles,
            )
        trace.append(
            {
                "step": float(iteration),
                "warmup": float(iteration <= cfg.warmup_steps),
                "acceptance_rate_mean": float(sum(acceptances) / len(acceptances)),
                "tree_depth_mean": float(sum(depths) / len(depths)),
                "step_size_mean": float(sum(step_sizes) / len(step_sizes)),
                "reward_mean": float(rewards.mean().item()),
                "reward_max": float(rewards.max().item()),
            }
        )

    if archive_points is None or archive_reward is None:
        raise RuntimeError("NUTS produced no post-warmup samples")
    return {
        "best_tx_xy_norm01": archive_points,
        "best_reward": archive_reward,
        "final_tx_xy_norm01": unconstrained_to_unit(states).detach(),
        "final_reward": rewards.detach(),
        "trace": trace,
    }


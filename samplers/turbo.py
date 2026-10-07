"""Independent TuRBO-1 adaptation for bounded TX deployment optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from samplers.common import RewardFunction, update_topk_archive, validate_sampler_inputs


@dataclass(frozen=True)
class TuRBOConfig:
    evaluation_budget: int = 200
    initial_points: int = 0
    batch_size: int = 4
    candidate_pool: int = 256
    gp_training_steps: int = 25
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 3

    def __post_init__(self) -> None:
        if self.evaluation_budget <= 1 or self.batch_size <= 0 or self.candidate_pool <= 1:
            raise ValueError("invalid TuRBO evaluation, batch, or candidate budget")
        if self.gp_training_steps <= 0 or self.success_tolerance <= 0:
            raise ValueError("GP steps and success tolerance must be positive")
        if not 0 < self.length_min < self.length_init <= self.length_max:
            raise ValueError("TuRBO trust-region lengths are inconsistent")


def _matern52(first: torch.Tensor, second: torch.Tensor, lengthscale: torch.Tensor, scale: torch.Tensor):
    distance = torch.cdist(first / lengthscale, second / lengthscale)
    root5 = math.sqrt(5.0)
    return scale * (1 + root5 * distance + 5 * distance.square() / 3) * torch.exp(-root5 * distance)


def _stable_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    for jitter in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if int(info.max()) == 0:
            return factor
    raise RuntimeError("TuRBO GP covariance was not positive definite")


def _fit_gp(x: torch.Tensor, y: torch.Tensor, steps: int):
    x64 = x.detach().to(torch.float64)
    mean, std = y.mean(), y.std().clamp_min(1e-6)
    target = ((y - mean) / std).detach().to(torch.float64)
    raw_length = torch.full((x.shape[1],), -1.4, device=x.device, dtype=torch.float64, requires_grad=True)
    raw_scale = torch.tensor(-2.0, device=x.device, dtype=torch.float64, requires_grad=True)
    raw_noise = torch.tensor(-5.0, device=x.device, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw_length, raw_scale, raw_noise], lr=0.08)
    for _ in range(steps):
        length = 0.005 + 1.995 * raw_length.sigmoid()
        scale = 0.05 + 19.95 * raw_scale.sigmoid()
        noise = 1e-6 + 0.05 * raw_noise.sigmoid()
        covariance = _matern52(x64, x64, length, scale)
        factor = _stable_cholesky(covariance + noise * torch.eye(len(x64), device=x.device, dtype=torch.float64))
        alpha = torch.cholesky_solve(target[:, None], factor).squeeze(1)
        loss = 0.5 * target.dot(alpha) + factor.diagonal().log().sum() + 0.5 * len(x64) * math.log(2 * math.pi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {
        "x": x64, "target": target, "mean": mean.to(torch.float64), "std": std.to(torch.float64),
        "lengthscale": (0.005 + 1.995 * raw_length.sigmoid()).detach(),
        "scale": (0.05 + 19.95 * raw_scale.sigmoid()).detach(),
        "noise": (1e-6 + 0.05 * raw_noise.sigmoid()).detach(),
    }


def _posterior(gp: dict, candidates: torch.Tensor):
    x, target = gp["x"], gp["target"]
    candidates = candidates.to(torch.float64)
    train_cov = _matern52(x, x, gp["lengthscale"], gp["scale"])
    train_cov = train_cov + gp["noise"] * torch.eye(len(x), device=x.device, dtype=x.dtype)
    factor = _stable_cholesky(train_cov)
    cross = _matern52(x, candidates, gp["lengthscale"], gp["scale"])
    alpha = torch.cholesky_solve(target[:, None], factor)
    posterior_mean = (cross.transpose(0, 1) @ alpha).squeeze(1)
    solved = torch.linalg.solve_triangular(factor, cross, upper=False)
    posterior_cov = _matern52(candidates, candidates, gp["lengthscale"], gp["scale"])
    posterior_cov = posterior_cov - solved.transpose(0, 1) @ solved
    return posterior_mean, posterior_cov


def _sobol(count: int, dimension: int, device: torch.device, generator):
    seed = int(torch.randint(0, 2**31 - 1, (), generator=generator).item())
    return torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed).draw(count).to(device)


def sample_turbo(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: TuRBOConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Maximize a learned or simulator reward with one adaptive local GP."""
    cfg = config or TuRBOConfig()
    validate_sampler_inputs(num_particles, num_tx)
    dimension = 2 * num_tx
    initial_count = cfg.initial_points or max(2 * dimension, 8)
    if initial_count >= cfg.evaluation_budget:
        raise ValueError("TuRBO initial_points must be smaller than evaluation_budget")
    if cfg.evaluation_budget < num_particles:
        raise ValueError("TuRBO evaluation_budget must be at least num_particles")
    failure_tolerance = max(4, math.ceil(dimension / cfg.batch_size))
    global_points = []
    global_rewards = []
    trace = []
    evaluations = 0
    restart = 0
    while evaluations < cfg.evaluation_budget:
        restart += 1
        remaining = cfg.evaluation_budget - evaluations
        count = min(initial_count, remaining)
        local_x = _sobol(count, dimension, device, generator)
        with torch.no_grad():
            local_y = reward_fn(local_x.reshape(count, num_tx, 2)).detach()
        global_points.append(local_x)
        global_rewards.append(local_y)
        evaluations += count
        length = cfg.length_init
        successes = failures = 0
        trace.append({
            "step": float(len(trace) + 1), "restart": float(restart), "evaluations": float(evaluations),
            "trust_region_length": float(length), "reward_mean": float(local_y.mean()),
            "reward_max": float(local_y.max()),
        })
        while evaluations < cfg.evaluation_budget and length >= cfg.length_min:
            gp = _fit_gp(local_x, local_y, cfg.gp_training_steps)
            best_index = local_y.argmax()
            center = local_x[best_index]
            weights = gp["lengthscale"].to(center.dtype)
            weights = weights / weights.mean()
            weights = weights / weights.log().mean().exp()
            lower = (center - 0.5 * length * weights).clamp(0, 1)
            upper = (center + 0.5 * length * weights).clamp(0, 1)
            base = _sobol(cfg.candidate_pool, dimension, device, generator)
            perturb = lower + (upper - lower) * base
            probability = min(20.0 / dimension, 1.0)
            mask = torch.rand((cfg.candidate_pool, dimension), device=device, generator=generator) <= probability
            empty = ~mask.any(1)
            if bool(empty.any()):
                chosen = torch.randint(0, dimension, (int(empty.sum()),), device=device, generator=generator)
                mask[empty, chosen] = True
            candidates = center.expand_as(perturb).clone()
            candidates[mask] = perturb[mask]
            posterior_mean, posterior_cov = _posterior(gp, candidates)
            batch = min(cfg.batch_size, cfg.evaluation_budget - evaluations, cfg.candidate_pool)
            covariance_factor = _stable_cholesky(posterior_cov)
            noise = torch.randn((cfg.candidate_pool, batch), device=device, dtype=torch.float64, generator=generator)
            samples = posterior_mean[:, None] + covariance_factor @ noise
            selected = []
            available = torch.ones(cfg.candidate_pool, dtype=torch.bool, device=device)
            for column in range(batch):
                score = samples[:, column].masked_fill(~available, -torch.inf)
                index = int(score.argmax())
                selected.append(index)
                available[index] = False
            next_x = candidates[torch.tensor(selected, device=device)]
            with torch.no_grad():
                next_y = reward_fn(next_x.reshape(batch, num_tx, 2)).detach()
            previous_best = float(local_y.max())
            improved = float(next_y.max()) > previous_best + 1e-3 * max(abs(previous_best), 1.0)
            if improved:
                successes, failures = successes + 1, 0
            else:
                successes, failures = 0, failures + 1
            if successes >= cfg.success_tolerance:
                length = min(2 * length, cfg.length_max)
                successes = 0
            elif failures >= failure_tolerance:
                length *= 0.5
                failures = 0
            local_x = torch.cat([local_x, next_x], 0)
            local_y = torch.cat([local_y, next_y], 0)
            global_points.append(next_x)
            global_rewards.append(next_y)
            evaluations += batch
            trace.append({
                "step": float(len(trace) + 1), "restart": float(restart), "evaluations": float(evaluations),
                "trust_region_length": float(length), "reward_mean": float(next_y.mean()),
                "reward_max": float(next_y.max()), "best_reward_max": float(local_y.max()),
            })
    all_points = torch.cat(global_points).reshape(-1, num_tx, 2)
    all_rewards = torch.cat(global_rewards)
    archive_points, archive_reward = update_topk_archive(
        None, None, all_points, all_rewards, num_particles
    )
    return {
        "best_tx_xy_norm01": archive_points, "best_reward": archive_reward,
        "final_tx_xy_norm01": local_x.reshape(-1, num_tx, 2).detach(),
        "final_reward": local_y.detach(), "trace": trace,
        "num_reward_evaluations": evaluations,
    }

"""Derivative-free heuristic baselines for continuous TX deployment."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from samplers.common import RewardFunction, update_topk_archive, validate_sampler_inputs


@dataclass(frozen=True)
class GreedyConfig:
    candidate_pool: int = 256
    refinement_rounds: int = 1
    evaluation_batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.candidate_pool <= 0 or self.refinement_rounds < 0:
            raise ValueError("candidate_pool must be positive and refinement_rounds non-negative")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive")


@dataclass(frozen=True)
class PSOConfig:
    steps: int = 100
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive: float = 1.49618
    social: float = 1.49618
    velocity_limit: float = 0.2
    topology: str = "global"

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.velocity_limit <= 0:
            raise ValueError("steps and velocity_limit must be positive")
        if self.cognitive < 0 or self.social < 0:
            raise ValueError("PSO acceleration coefficients must be non-negative")
        if self.topology not in {"global", "ring"}:
            raise ValueError("topology must be global or ring")


@dataclass(frozen=True)
class GAConfig:
    generations: int = 100
    tournament_size: int = 3
    elite_fraction: float = 0.1
    crossover_probability: float = 0.9
    mutation_probability: float = 0.0
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0

    def __post_init__(self) -> None:
        if self.generations <= 0 or self.tournament_size <= 0:
            raise ValueError("generations and tournament_size must be positive")
        if not 0 <= self.elite_fraction < 1:
            raise ValueError("elite_fraction must be in [0,1)")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0,1]")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0,1]")
        if self.crossover_eta <= 0 or self.mutation_eta <= 0:
            raise ValueError("distribution indices must be positive")


def _evaluate_chunks(reward_fn: RewardFunction, points: torch.Tensor, batch_size: int) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            value = reward_fn(points[start : start + batch_size])
            if value.shape != (min(batch_size, len(points) - start),):
                raise ValueError(f"reward_fn must return [B], got {tuple(value.shape)}")
            values.append(value.detach())
    return torch.cat(values)


def sample_greedy(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: GreedyConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Beam greedy addition followed by greedy coordinate replacement.

    This is a deterministic-style marginal-gain heuristic over a finite Sobol
    candidate set. It does not assume or claim that the learned reward is
    submodular.
    """
    cfg = config or GreedyConfig()
    validate_sampler_inputs(num_particles, num_tx)
    seed = int(torch.randint(0, 2**31 - 1, (), generator=generator).item())
    pool = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(cfg.candidate_pool).to(device)
    beam = None
    beam_reward = None
    trace = []
    reward_evaluations = 0
    for count in range(1, num_tx + 1):
        if beam is None:
            proposals = pool[:, None, :]
        else:
            proposals = torch.cat(
                [
                    beam[:, None].expand(-1, cfg.candidate_pool, -1, -1),
                    pool[None, :, None].expand(len(beam), -1, 1, -1),
                ],
                dim=2,
            ).reshape(-1, count, 2)
        values = _evaluate_chunks(reward_fn, proposals, cfg.evaluation_batch_size)
        reward_evaluations += len(proposals)
        keep = min(num_particles, len(values))
        beam_reward, index = torch.topk(values, keep, largest=True, sorted=True)
        beam = proposals[index].clone()
        trace.append({
            "step": float(count), "phase": "addition", "num_tx": float(count),
            "evaluations": float(len(proposals)), "reward_mean": float(beam_reward.mean()),
            "reward_max": float(beam_reward.max()),
        })

    for round_index in range(cfg.refinement_rounds):
        for tx_index in range(num_tx):
            proposals = beam[:, None].expand(-1, cfg.candidate_pool, -1, -1).clone()
            proposals[:, :, tx_index] = pool[None]
            proposals = proposals.reshape(-1, num_tx, 2)
            values = _evaluate_chunks(reward_fn, proposals, cfg.evaluation_batch_size)
            reward_evaluations += len(proposals)
            beam_reward, index = torch.topk(values, min(num_particles, len(values)), largest=True, sorted=True)
            beam = proposals[index].clone()
            trace.append({
                "step": float(len(trace) + 1), "phase": "refinement",
                "round": float(round_index + 1), "tx_index": float(tx_index),
                "evaluations": float(len(proposals)), "reward_mean": float(beam_reward.mean()),
                "reward_max": float(beam_reward.max()),
            })
    return {
        "best_tx_xy_norm01": beam.detach(), "best_reward": beam_reward.detach(),
        "final_tx_xy_norm01": beam.detach(), "final_reward": beam_reward.detach(), "trace": trace,
        "num_reward_evaluations": reward_evaluations,
    }


def sample_pso(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: PSOConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Bounded real-valued PSO with global or ring social topology."""
    cfg = config or PSOConfig()
    validate_sampler_inputs(num_particles, num_tx)
    position = torch.rand((num_particles, num_tx, 2), device=device, generator=generator)
    velocity = (2 * torch.rand(position.shape, device=device, generator=generator) - 1) * cfg.velocity_limit
    with torch.no_grad():
        reward = reward_fn(position).detach()
    personal_position, personal_reward = position.clone(), reward.clone()
    archive_points, archive_reward = update_topk_archive(None, None, position, reward, num_particles)
    trace = []
    for step in range(1, cfg.steps + 1):
        fraction = (step - 1) / max(cfg.steps - 1, 1)
        inertia = cfg.inertia_start + fraction * (cfg.inertia_end - cfg.inertia_start)
        if cfg.topology == "global":
            social_best = personal_position[personal_reward.argmax()].unsqueeze(0).expand_as(position)
        else:
            ids = torch.arange(num_particles, device=device)
            neighbor_ids = torch.stack([(ids - 1) % num_particles, ids, (ids + 1) % num_particles], 1)
            neighbor_rewards = personal_reward[neighbor_ids]
            best_neighbor = neighbor_ids.gather(1, neighbor_rewards.argmax(1, keepdim=True)).squeeze(1)
            social_best = personal_position[best_neighbor]
        random_cognitive = torch.rand(position.shape, device=device, generator=generator)
        random_social = torch.rand(position.shape, device=device, generator=generator)
        velocity = (
            inertia * velocity
            + cfg.cognitive * random_cognitive * (personal_position - position)
            + cfg.social * random_social * (social_best - position)
        ).clamp(-cfg.velocity_limit, cfg.velocity_limit)
        proposed = position + velocity
        out_of_bounds = (proposed < 0) | (proposed > 1)
        position = proposed.clamp(0, 1)
        velocity = torch.where(out_of_bounds, -0.5 * velocity, velocity)
        with torch.no_grad():
            reward = reward_fn(position).detach()
        improved = reward > personal_reward
        personal_reward = torch.where(improved, reward, personal_reward)
        personal_position = torch.where(improved[:, None, None], position, personal_position)
        archive_points, archive_reward = update_topk_archive(
            archive_points, archive_reward, position, reward, num_particles
        )
        trace.append({
            "step": float(step), "inertia": float(inertia),
            "reward_mean": float(reward.mean()), "reward_max": float(reward.max()),
            "best_reward_max": float(archive_reward.max()),
        })
    return {
        "best_tx_xy_norm01": archive_points, "best_reward": archive_reward,
        "final_tx_xy_norm01": position.detach(), "final_reward": reward.detach(), "trace": trace,
        "num_reward_evaluations": num_particles * (cfg.steps + 1),
    }


def _tournament(reward: torch.Tensor, count: int, size: int, generator) -> torch.Tensor:
    contestants = torch.randint(0, len(reward), (count, size), device=reward.device, generator=generator)
    scores = reward[contestants]
    return contestants.gather(1, scores.argmax(1, keepdim=True)).squeeze(1)


def _sbx(first: torch.Tensor, second: torch.Tensor, eta: float, probability: float, generator):
    uniform = torch.rand(first.shape, device=first.device, generator=generator).clamp(1e-7, 1 - 1e-7)
    beta = torch.where(
        uniform <= 0.5,
        (2 * uniform).pow(1 / (eta + 1)),
        (1 / (2 * (1 - uniform))).pow(1 / (eta + 1)),
    )
    child_a = 0.5 * ((1 + beta) * first + (1 - beta) * second)
    child_b = 0.5 * ((1 - beta) * first + (1 + beta) * second)
    choose_a = torch.rand(first.shape, device=first.device, generator=generator) < 0.5
    child = torch.where(choose_a, child_a, child_b)
    cross = torch.rand((len(first), 1, 1), device=first.device, generator=generator) < probability
    return torch.where(cross, child, first).clamp(0, 1)


def _polynomial_mutation(value: torch.Tensor, eta: float, probability: float, generator):
    uniform = torch.rand(value.shape, device=value.device, generator=generator)
    delta_low = value
    delta_high = 1 - value
    left = (2 * uniform + (1 - 2 * uniform) * (1 - delta_low).pow(eta + 1)).pow(1 / (eta + 1)) - 1
    right = 1 - (2 * (1 - uniform) + 2 * (uniform - 0.5) * (1 - delta_high).pow(eta + 1)).pow(1 / (eta + 1))
    delta = torch.where(uniform < 0.5, left, right)
    mutate = torch.rand(value.shape, device=value.device, generator=generator) < probability
    return torch.where(mutate, value + delta, value).clamp(0, 1)


def sample_ga(
    reward_fn: RewardFunction,
    *,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: GAConfig | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Single-objective real-coded GA using SBX and polynomial mutation."""
    cfg = config or GAConfig()
    validate_sampler_inputs(num_particles, num_tx)
    population = torch.rand((num_particles, num_tx, 2), device=device, generator=generator)
    with torch.no_grad():
        reward = reward_fn(population).detach()
    archive_points, archive_reward = update_topk_archive(None, None, population, reward, num_particles)
    trace = []
    mutation_probability = cfg.mutation_probability or 1.0 / (2 * num_tx)
    elite_count = min(num_particles - 1, max(1, round(cfg.elite_fraction * num_particles))) if num_particles > 1 else 0
    for generation in range(1, cfg.generations + 1):
        order = torch.argsort(reward, descending=True)
        elites = population[order[:elite_count]].clone()
        offspring_count = num_particles - elite_count
        first = population[_tournament(reward, offspring_count, cfg.tournament_size, generator)]
        second = population[_tournament(reward, offspring_count, cfg.tournament_size, generator)]
        offspring = _sbx(first, second, cfg.crossover_eta, cfg.crossover_probability, generator)
        offspring = _polynomial_mutation(offspring, cfg.mutation_eta, mutation_probability, generator)
        population = torch.cat([elites, offspring], 0)
        with torch.no_grad():
            reward = reward_fn(population).detach()
        archive_points, archive_reward = update_topk_archive(
            archive_points, archive_reward, population, reward, num_particles
        )
        trace.append({
            "step": float(generation), "reward_mean": float(reward.mean()),
            "reward_max": float(reward.max()), "best_reward_max": float(archive_reward.max()),
        })
    return {
        "best_tx_xy_norm01": archive_points, "best_reward": archive_reward,
        "final_tx_xy_norm01": population.detach(), "final_reward": reward.detach(), "trace": trace,
        "num_reward_evaluations": num_particles * (cfg.generations + 1),
    }

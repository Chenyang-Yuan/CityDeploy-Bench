"""Unified configuration and dispatch for all deployment samplers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

import torch

from samplers.br_snis import BRSNISConfig, sample_br_snis
from samplers.common import RewardFunction
from samplers.diffusion import VEDiffusionConfig, sample_reward_guided_ve
from samplers.heuristics import (
    GAConfig,
    GreedyConfig,
    PSOConfig,
    sample_ga,
    sample_greedy,
    sample_pso,
)
from samplers.nuts import NUTSConfig, sample_nuts
from samplers.smc import AnnealedSMCConfig, sample_annealed_smc
from samplers.turbo import TuRBOConfig, sample_turbo


SAMPLER_NAMES = (
    "diffusion", "smc_gaussian", "smc_langevin", "nuts", "br_snis",
    "greedy", "pso", "ga", "turbo",
)

SAMPLER_CONFIG_TYPES = {
    "diffusion": VEDiffusionConfig,
    "smc_gaussian": AnnealedSMCConfig,
    "smc_langevin": AnnealedSMCConfig,
    "nuts": NUTSConfig,
    "br_snis": BRSNISConfig,
    "greedy": GreedyConfig,
    "pso": PSOConfig,
    "ga": GAConfig,
    "turbo": TuRBOConfig,
}

def sampler_config_from_dict(sampler: str, payload: dict[str, Any]) -> Any:
    if sampler not in SAMPLER_CONFIG_TYPES:
        raise ValueError(f"unsupported sampler: {sampler}")
    values = dict(payload)
    if sampler in {"smc_gaussian", "smc_langevin"}:
        expected = "gaussian" if sampler == "smc_gaussian" else "langevin"
        if "kernel" in values and values["kernel"] != expected:
            raise ValueError(f"{sampler} requires kernel={expected!r}")
        values["kernel"] = expected
    return SAMPLER_CONFIG_TYPES[sampler](**values)


def add_sampler_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampler", choices=SAMPLER_NAMES, default="diffusion")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)

    diffusion = parser.add_argument_group("diffusion")
    diffusion.add_argument("--mc-samples", type=int, default=10)
    diffusion.add_argument("--sigma-max", type=float, default=0.35)
    diffusion.add_argument("--sigma-min", type=float, default=0.005)
    diffusion.add_argument(
        "--boundary", choices=["reflect", "clamp", "none"], default="reflect"
    )

    smc = parser.add_argument_group("SMC")
    smc.add_argument("--ess-threshold", type=float, default=0.5)
    smc.add_argument("--proposal-scale-start", type=float, default=0.45)
    smc.add_argument("--proposal-scale-end", type=float, default=0.08)
    smc.add_argument("--mutation-steps", type=int, default=1)
    smc.add_argument("--tempering-power", type=float, default=1.0)

    nuts = parser.add_argument_group("NUTS")
    nuts.add_argument("--nuts-warmup", type=int, default=50)
    nuts.add_argument("--nuts-samples", type=int, default=50)
    nuts.add_argument("--nuts-step-size", type=float, default=0.15)
    nuts.add_argument("--nuts-target-acceptance", type=float, default=0.8)
    nuts.add_argument("--nuts-max-tree-depth", type=int, default=7)

    br = parser.add_argument_group("BR-SNIS")
    br.add_argument("--br-rounds", type=int, default=100)
    br.add_argument("--br-proposal-particles", type=int, default=32)

    greedy = parser.add_argument_group("greedy")
    greedy.add_argument("--greedy-candidate-pool", type=int, default=256)
    greedy.add_argument("--greedy-refinement-rounds", type=int, default=1)
    greedy.add_argument("--evaluation-batch-size", type=int, default=4096)

    pso = parser.add_argument_group("PSO")
    pso.add_argument("--pso-inertia-start", type=float, default=0.9)
    pso.add_argument("--pso-inertia-end", type=float, default=0.4)
    pso.add_argument("--pso-cognitive", type=float, default=1.49618)
    pso.add_argument("--pso-social", type=float, default=1.49618)
    pso.add_argument("--pso-velocity-limit", type=float, default=0.2)
    pso.add_argument("--pso-topology", choices=["global", "ring"], default="global")

    ga = parser.add_argument_group("GA")
    ga.add_argument("--ga-tournament-size", type=int, default=3)
    ga.add_argument("--ga-elite-fraction", type=float, default=0.1)
    ga.add_argument("--ga-crossover-probability", type=float, default=0.9)
    ga.add_argument("--ga-mutation-probability", type=float, default=0.0)
    ga.add_argument("--ga-crossover-eta", type=float, default=15.0)
    ga.add_argument("--ga-mutation-eta", type=float, default=20.0)

    turbo = parser.add_argument_group("TuRBO")
    turbo.add_argument("--turbo-evaluation-budget", type=int, default=200)
    turbo.add_argument("--turbo-initial-points", type=int, default=0)
    turbo.add_argument("--turbo-batch-size", type=int, default=4)
    turbo.add_argument("--turbo-candidate-pool", type=int, default=256)
    turbo.add_argument("--turbo-gp-training-steps", type=int, default=25)


def sampler_config(args: argparse.Namespace) -> Any:
    if args.sampler == "diffusion":
        return VEDiffusionConfig(
            steps=args.steps,
            mc_samples=args.mc_samples,
            beta=args.beta,
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            boundary=args.boundary,
        )
    if args.sampler in {"smc_gaussian", "smc_langevin"}:
        return AnnealedSMCConfig(
            steps=args.steps,
            beta=args.beta,
            ess_threshold=args.ess_threshold,
            proposal_scale_start=args.proposal_scale_start,
            proposal_scale_end=args.proposal_scale_end,
            mutation_steps=args.mutation_steps,
            tempering_power=args.tempering_power,
            kernel="gaussian" if args.sampler == "smc_gaussian" else "langevin",
        )
    if args.sampler == "nuts":
        return NUTSConfig(
            warmup_steps=args.nuts_warmup,
            sample_steps=args.nuts_samples,
            beta=args.beta,
            initial_step_size=args.nuts_step_size,
            target_acceptance=args.nuts_target_acceptance,
            max_tree_depth=args.nuts_max_tree_depth,
        )
    if args.sampler == "br_snis":
        return BRSNISConfig(
            rounds=args.br_rounds,
            proposal_particles=args.br_proposal_particles,
            beta=args.beta,
        )
    if args.sampler == "greedy":
        return GreedyConfig(
            candidate_pool=args.greedy_candidate_pool,
            refinement_rounds=args.greedy_refinement_rounds,
            evaluation_batch_size=args.evaluation_batch_size,
        )
    if args.sampler == "pso":
        return PSOConfig(
            steps=args.steps,
            inertia_start=args.pso_inertia_start,
            inertia_end=args.pso_inertia_end,
            cognitive=args.pso_cognitive,
            social=args.pso_social,
            velocity_limit=args.pso_velocity_limit,
            topology=args.pso_topology,
        )
    if args.sampler == "ga":
        return GAConfig(
            generations=args.steps,
            tournament_size=args.ga_tournament_size,
            elite_fraction=args.ga_elite_fraction,
            crossover_probability=args.ga_crossover_probability,
            mutation_probability=args.ga_mutation_probability,
            crossover_eta=args.ga_crossover_eta,
            mutation_eta=args.ga_mutation_eta,
        )
    if args.sampler == "turbo":
        return TuRBOConfig(
            evaluation_budget=args.turbo_evaluation_budget,
            initial_points=args.turbo_initial_points,
            batch_size=args.turbo_batch_size,
            candidate_pool=args.turbo_candidate_pool,
            gp_training_steps=args.turbo_gp_training_steps,
        )
    raise ValueError(f"unsupported sampler: {args.sampler}")


def run_sampler(
    reward_fn: RewardFunction,
    *,
    sampler: str,
    num_particles: int,
    num_tx: int,
    device: torch.device,
    config: Any,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    evaluation_count = 0

    def counted_reward(points: torch.Tensor) -> torch.Tensor:
        nonlocal evaluation_count
        evaluation_count += int(points.shape[0])
        return reward_fn(points)

    kwargs = {
        "reward_fn": counted_reward,
        "num_particles": num_particles,
        "num_tx": num_tx,
        "device": device,
        "config": config,
        "generator": generator,
    }
    if sampler == "diffusion":
        result = sample_reward_guided_ve(**kwargs)
    elif sampler in {"smc_gaussian", "smc_langevin"}:
        result = sample_annealed_smc(**kwargs)
    elif sampler == "nuts":
        result = sample_nuts(**kwargs)
    elif sampler == "br_snis":
        result = sample_br_snis(**kwargs)
    elif sampler == "greedy":
        result = sample_greedy(**kwargs)
    elif sampler == "pso":
        result = sample_pso(**kwargs)
    elif sampler == "ga":
        result = sample_ga(**kwargs)
    elif sampler == "turbo":
        result = sample_turbo(**kwargs)
    else:
        raise ValueError(f"unsupported sampler: {sampler}")
    result["num_reward_evaluations"] = evaluation_count
    return result


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)

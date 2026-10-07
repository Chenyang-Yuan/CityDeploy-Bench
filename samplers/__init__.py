"""Reward-guided search and sampling methods for TX deployment."""

import pyarrow  # Load Arrow before PyTorch on Windows to avoid duplicate OpenMP initialization.

from samplers.br_snis import BRSNISConfig, sample_br_snis
from samplers.common import deployment_constraint_penalty
from samplers.diffusion import VEDiffusionConfig, sample_reward_guided_ve
from samplers.heuristics import GAConfig, GreedyConfig, PSOConfig, sample_ga, sample_greedy, sample_pso
from samplers.nuts import NUTSConfig, sample_nuts
from samplers.smc import (
    AnnealedSMCConfig,
    sample_annealed_smc,
    sample_smc_gaussian,
    sample_smc_langevin,
)
from samplers.turbo import TuRBOConfig, sample_turbo

__all__ = [
    "AnnealedSMCConfig",
    "BRSNISConfig",
    "NUTSConfig",
    "VEDiffusionConfig",
    "GreedyConfig",
    "PSOConfig",
    "GAConfig",
    "TuRBOConfig",
    "deployment_constraint_penalty",
    "sample_annealed_smc",
    "sample_br_snis",
    "sample_nuts",
    "sample_reward_guided_ve",
    "sample_smc_gaussian",
    "sample_smc_langevin",
    "sample_greedy",
    "sample_pso",
    "sample_ga",
    "sample_turbo",
]

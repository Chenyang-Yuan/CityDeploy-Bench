import unittest

import torch

from samplers import (
    AnnealedSMCConfig,
    BRSNISConfig,
    GAConfig,
    GreedyConfig,
    NUTSConfig,
    PSOConfig,
    TuRBOConfig,
    sample_br_snis,
    sample_ga,
    sample_greedy,
    sample_nuts,
    sample_pso,
    sample_smc_gaussian,
    sample_smc_langevin,
    sample_turbo,
)


def quadratic_reward(points: torch.Tensor) -> torch.Tensor:
    target = points.new_tensor([0.25, 0.75])
    return -(points - target).square().sum(dim=(1, 2))


class AnnealedSMCTests(unittest.TestCase):
    def _run(self, kernel: str):
        generator = torch.Generator().manual_seed(17)
        config = AnnealedSMCConfig(
            steps=10,
            beta=2.0,
            ess_threshold=0.7,
            proposal_scale_start=0.4,
            proposal_scale_end=0.1,
            mutation_steps=2,
            kernel=kernel,
        )
        function = sample_smc_gaussian if kernel == "gaussian" else sample_smc_langevin
        return function(
            quadratic_reward,
            num_particles=16,
            num_tx=2,
            device=torch.device("cpu"),
            config=config,
            generator=generator,
        )

    def test_gaussian_smc_returns_bounded_ranked_candidates(self):
        result = self._run("gaussian")
        self.assertEqual(len(result["trace"]), 10)
        self.assertTrue(bool((result["best_reward"][:-1] >= result["best_reward"][1:]).all()))
        self.assertTrue(bool(((result["best_tx_xy_norm01"] >= 0) & (result["best_tx_xy_norm01"] <= 1)).all()))

    def test_langevin_smc_returns_bounded_ranked_candidates(self):
        result = self._run("langevin")
        self.assertEqual(len(result["trace"]), 10)
        self.assertTrue(torch.isfinite(result["best_reward"]).all())
        self.assertGreaterEqual(
            float(result["best_reward"].max()), float(result["final_reward"].mean())
        )


class BRSNISTests(unittest.TestCase):
    def test_isir_keeps_concrete_deployment_particles(self):
        result = sample_br_snis(
            quadratic_reward,
            num_particles=8,
            num_tx=2,
            device=torch.device("cpu"),
            config=BRSNISConfig(rounds=8, proposal_particles=8, beta=2.0),
            generator=torch.Generator().manual_seed(23),
        )
        self.assertEqual(tuple(result["best_tx_xy_norm01"].shape), (8, 2, 2))
        self.assertEqual(len(result["trace"]), 8)
        self.assertTrue(torch.isfinite(result["final_weights"]).all())


class NUTSTests(unittest.TestCase):
    def test_nuts_returns_post_warmup_samples_in_unit_square(self):
        result = sample_nuts(
            quadratic_reward,
            num_particles=2,
            num_tx=1,
            device=torch.device("cpu"),
            config=NUTSConfig(
                warmup_steps=3,
                sample_steps=4,
                beta=2.0,
                initial_step_size=0.15,
                max_tree_depth=3,
            ),
            generator=torch.Generator().manual_seed(29),
        )
        self.assertEqual(tuple(result["best_tx_xy_norm01"].shape), (2, 1, 2))
        self.assertEqual(len(result["trace"]), 7)
        self.assertTrue(bool(((result["best_tx_xy_norm01"] > 0) & (result["best_tx_xy_norm01"] < 1)).all()))
        self.assertTrue(torch.isfinite(result["best_reward"]).all())


class HeuristicSearchTests(unittest.TestCase):
    def _assert_result(self, result, particles, num_tx):
        self.assertEqual(tuple(result["best_tx_xy_norm01"].shape), (particles, num_tx, 2))
        self.assertTrue(bool(((result["best_tx_xy_norm01"] >= 0) & (result["best_tx_xy_norm01"] <= 1)).all()))
        self.assertTrue(torch.isfinite(result["best_reward"]).all())
        self.assertTrue(bool((result["best_reward"][:-1] >= result["best_reward"][1:]).all()))

    def test_greedy_builds_variable_cardinality_deployment(self):
        result = sample_greedy(
            quadratic_reward,
            num_particles=4,
            num_tx=2,
            device=torch.device("cpu"),
            config=GreedyConfig(candidate_pool=12, refinement_rounds=1),
            generator=torch.Generator().manual_seed(31),
        )
        self._assert_result(result, 4, 2)
        self.assertTrue(any(item["phase"] == "addition" for item in result["trace"]))

    def test_pso_returns_ranked_archive(self):
        result = sample_pso(
            quadratic_reward,
            num_particles=6,
            num_tx=2,
            device=torch.device("cpu"),
            config=PSOConfig(steps=6, topology="ring"),
            generator=torch.Generator().manual_seed(37),
        )
        self._assert_result(result, 6, 2)

    def test_real_coded_ga_returns_ranked_archive(self):
        result = sample_ga(
            quadratic_reward,
            num_particles=6,
            num_tx=2,
            device=torch.device("cpu"),
            config=GAConfig(generations=6),
            generator=torch.Generator().manual_seed(41),
        )
        self._assert_result(result, 6, 2)

    def test_turbo_uses_bounded_local_gp_search(self):
        result = sample_turbo(
            quadratic_reward,
            num_particles=4,
            num_tx=1,
            device=torch.device("cpu"),
            config=TuRBOConfig(
                evaluation_budget=12,
                initial_points=4,
                batch_size=2,
                candidate_pool=16,
                gp_training_steps=2,
            ),
            generator=torch.Generator().manual_seed(43),
        )
        self._assert_result(result, 4, 1)
        self.assertLessEqual(max(item["evaluations"] for item in result["trace"]), 12)


if __name__ == "__main__":
    unittest.main()

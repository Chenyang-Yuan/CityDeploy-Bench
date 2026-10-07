import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from energy_model.model import HypergraphPotentialConfig, HypergraphPotentialModel
from samplers.diffusion import estimate_mollified_score, sample_reward_guided_ve, VEDiffusionConfig
from energy_model.data import SceneCardinalityBatchSampler, indices_for_scenes, make_scene_split
from energy_model.dataset import SampleRecord
from scene_builder.coordinates import SceneFrame2D, validate_norm01


class CoordinateContractTests(unittest.TestCase):
    def test_metric_and_normalized_coordinates_roundtrip(self):
        frame = SceneFrame2D(512.0, 256.0)
        real = np.asarray([[-256.0, -128.0], [0.0, 0.0], [256.0, 128.0]])
        norm = frame.real_to_norm01(real, check_bounds=True)
        np.testing.assert_allclose(norm, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        np.testing.assert_allclose(frame.norm01_to_real(norm), real)

    def test_normalized_coordinates_reject_out_of_range_values(self):
        with self.assertRaises(ValueError):
            validate_norm01(np.asarray([[1.01, 0.5]]))


class UnifiedHypergraphPotentialTests(unittest.TestCase):
    def _case(self):
        torch.manual_seed(11)
        config = HypergraphPotentialConfig(
            scene_dim=32,
            local_dim=16,
            hidden_dim=32,
            hyper_hidden_dim=24,
            max_num_tx=8,
        )
        model = HypergraphPotentialModel(config).eval()
        x = torch.rand((2, 5, 2), requires_grad=True)
        mask = torch.ones((2, 5), dtype=torch.bool)
        scene = torch.rand((2, config.scene_channels, 32, 32))
        map_size = torch.tensor([[512.0, 512.0], [256.0, 384.0]])
        return model, x, mask, scene, map_size

    def test_orders_one_through_four_share_one_model(self):
        model, x, mask, scene, map_size = self._case()
        out = model(x, mask, scene, map_size)
        self.assertEqual(set(out["order_energies"]), {1, 2, 3, 4})
        self.assertEqual(tuple(out["energy"].shape), (2,))
        out["energy"].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_scalar_energy_is_permutation_invariant(self):
        model, x, mask, scene, map_size = self._case()
        permutation = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            original = model(x, mask, scene, map_size)["energy"]
            permuted = model(x[:, permutation], mask[:, permutation], scene, map_size)["energy"]
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=0.0)


class SceneLevelTrainingProtocolTests(unittest.TestCase):
    @dataclass
    class _DatasetStub:
        records: list[SampleRecord]

    def _dataset(self):
        records = []
        for scene in ("alpha", "beta", "gamma", "delta"):
            for num_tx, count in ((3, 5), (5, 3)):
                records.extend(
                    SampleRecord(Path(f"{scene}_{num_tx}_{index}.npz"), scene, num_tx)
                    for index in range(count)
                )
        return self._DatasetStub(records)

    def test_scene_split_is_disjoint_and_complete(self):
        split = make_scene_split(
            ["alpha", "beta", "gamma", "delta"],
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=19,
        )
        groups = [set(split.train_scenes), set(split.validation_scenes), set(split.test_scenes)]
        self.assertEqual(set.union(*groups), {"alpha", "beta", "gamma", "delta"})
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_batches_never_mix_scene_or_tx_cardinality(self):
        dataset = self._dataset()
        indices = indices_for_scenes(dataset, ["alpha", "beta", "gamma", "delta"])
        sampler = SceneCardinalityBatchSampler(
            dataset,
            indices,
            batch_size=3,
            shuffle=True,
            seed=23,
        )
        visited = []
        for batch in sampler:
            keys = {(dataset.records[index].scene, dataset.records[index].num_tx) for index in batch}
            self.assertEqual(len(keys), 1)
            visited.extend(batch)
        self.assertEqual(sorted(visited), sorted(indices))


class RewardInducedDiffusionTests(unittest.TestCase):
    def test_monte_carlo_score_matches_smoothed_quadratic(self):
        torch.manual_seed(3)
        center = torch.tensor([[[0.62, 0.37]]])
        target = torch.tensor([0.45, 0.55])
        sigma = 0.12
        beta = 1.3

        def reward_fn(x):
            return -0.5 * (x - target).square().sum(dim=(1, 2))

        estimated, _ = estimate_mollified_score(
            reward_fn,
            center,
            sigma,
            beta,
            8192,
            antithetic=True,
            boundary="none",
        )
        expected = -beta * (center - target) / (1.0 + beta * sigma * sigma)
        torch.testing.assert_close(estimated, expected, atol=1.5e-2, rtol=0.0)

    def test_reverse_ve_sampling_improves_reward(self):
        torch.manual_seed(5)
        target = torch.tensor([0.25, 0.75])

        def reward_fn(x):
            return -(x - target).square().sum(dim=(1, 2))

        result = sample_reward_guided_ve(
            reward_fn,
            num_particles=12,
            num_tx=2,
            device=torch.device("cpu"),
            config=VEDiffusionConfig(steps=12, mc_samples=32, sigma_max=0.25, sigma_min=0.01),
        )
        self.assertGreaterEqual(
            float(result["best_reward"].max()),
            float(result["final_reward"].mean()),
        )
        points = result["best_tx_xy_norm01"]
        self.assertTrue(bool(((points >= 0.0) & (points <= 1.0)).all()))


if __name__ == "__main__":
    unittest.main()

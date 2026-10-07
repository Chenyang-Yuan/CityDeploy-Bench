import tempfile
import unittest
from pathlib import Path

import torch

from energy_model.guidance import load_guidance_checkpoint, predict_reward
from energy_model.relational_models import (
    INTERACTION_NETWORK_SCHEMA,
    NRI_SCHEMA,
    InteractionNetworkReward,
    NRIRelationalReward,
    RelationalRewardConfig,
    assert_relational_permutation_invariant,
)


class RelationalRewardModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = RelationalRewardConfig(
            scene_dim=32,
            local_dim=16,
            node_dim=24,
            relation_dim=20,
            hidden_dim=32,
            max_num_tx=9,
            dropout=0.0,
            edge_types=3,
        )
        self.scene = torch.rand(1, self.config.scene_channels, 32, 32)
        self.map_size = torch.tensor([[256.0, 256.0], [256.0, 256.0]])
        self.points = torch.rand(2, 4, 2)
        self.mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    def _check_model(self, model):
        model.eval()
        reward = model(self.points, self.mask, self.scene, self.map_size)
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertTrue(bool(((reward >= 0.0) & (reward <= 1.0)).all()))
        assert_relational_permutation_invariant(
            model, self.points, self.mask, self.scene, self.map_size
        )
        differentiable_points = self.points.clone().requires_grad_(True)
        model(differentiable_points, self.mask, self.scene, self.map_size).sum().backward()
        self.assertIsNotNone(differentiable_points.grad)
        self.assertTrue(torch.isfinite(differentiable_points.grad).all())

    def test_interaction_network_is_bounded_invariant_and_differentiable(self):
        self._check_model(InteractionNetworkReward(self.config))

    def test_nri_is_bounded_invariant_and_differentiable_in_eval_mode(self):
        model = NRIRelationalReward(self.config)
        self._check_model(model)
        reward, auxiliary = model.forward_with_aux(
            self.points, self.mask, self.scene, self.map_size
        )
        self.assertEqual(tuple(auxiliary["edge_probabilities"].shape), (2, 4, 4, 3))
        self.assertEqual(tuple(auxiliary["edge_kl"].shape), (2,))
        self.assertTrue(torch.isfinite(reward).all())

    def test_single_tx_has_no_pairwise_numerical_failure(self):
        points = self.points[:, :1]
        mask = torch.ones(2, 1, dtype=torch.bool)
        for model in (InteractionNetworkReward(self.config), NRIRelationalReward(self.config)):
            model.eval()
            self.assertTrue(torch.isfinite(model(points, mask, self.scene, self.map_size)).all())

    def test_checkpoint_registry_round_trip(self):
        cases = [
            (INTERACTION_NETWORK_SCHEMA, "interaction_network", InteractionNetworkReward(self.config)),
            (NRI_SCHEMA, "nri", NRIRelationalReward(self.config)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (schema, expected_name, model) in enumerate(cases):
                path = Path(directory) / f"model_{index}.pt"
                torch.save(
                    {
                        "model_schema_version": schema,
                        "model_config": self.config.to_dict(),
                        "model_state": model.state_dict(),
                    },
                    path,
                )
                name, loaded, _ = load_guidance_checkpoint(path, torch.device("cpu"))
                self.assertEqual(name, expected_name)
                result = predict_reward(
                    name, loaded, self.points, self.mask, self.scene, self.map_size
                )
                self.assertEqual(tuple(result.shape), (2,))


if __name__ == "__main__":
    unittest.main()

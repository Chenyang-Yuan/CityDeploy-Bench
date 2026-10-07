"""Exercise reward learning and every sampler on a synthetic deployment task."""

from __future__ import annotations

import argparse
import json

import pyarrow
import torch

from energy_model.dataset import SCENE_FEATURE_CHANNELS, set_seed
from energy_model.guidance import assert_guidance_permutation_invariant, predict_reward
from energy_model.model import HypergraphPotentialConfig, HypergraphPotentialModel
from energy_model.reward_model import RewardPredictor, RewardPredictorConfig
from energy_model.relational_models import InteractionNetworkReward, NRIRelationalReward, RelationalRewardConfig
from samplers.runner import run_sampler, sampler_config_from_dict


SMALL_BUDGETS = {
    "diffusion": {"steps": 2, "mc_samples": 2, "sigma_max": 0.2, "sigma_min": 0.02},
    "smc_gaussian": {"steps": 2, "mutation_steps": 1},
    "smc_langevin": {"steps": 2, "mutation_steps": 1},
    "nuts": {"warmup_steps": 1, "sample_steps": 1, "max_tree_depth": 1},
    "br_snis": {"rounds": 2, "proposal_particles": 2},
    "greedy": {"candidate_pool": 4, "refinement_rounds": 0},
    "pso": {"steps": 2},
    "ga": {"generations": 2},
    "turbo": {"evaluation_budget": 8, "initial_points": 4, "batch_size": 2, "candidate_pool": 8, "gp_training_steps": 2},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    scene = torch.zeros(1, len(SCENE_FEATURE_CHANNELS), 16, 16)
    scene[:, 1] = 1
    scene[:, -2:] = 1
    scene[:, 0, 5:11, 5:11] = 1
    map_size = torch.tensor([[128.0, 128.0]])
    points = torch.rand(4, 2, 2)
    mask = torch.ones(4, 2, dtype=torch.bool)
    targets = torch.exp(-((points - 0.5) ** 2).sum(dim=(1, 2)))
    models = {
        "hypergraph_potential": HypergraphPotentialModel(HypergraphPotentialConfig(max_num_tx=9)),
        "reward_predictor": RewardPredictor(RewardPredictorConfig(max_num_tx=9)),
        "interaction_network": InteractionNetworkReward(RelationalRewardConfig(max_num_tx=9)),
        "nri": NRIRelationalReward(RelationalRewardConfig(max_num_tx=9)),
    }
    completed = []
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        optimizer.zero_grad()
        prediction = predict_reward(name, model, points, mask, scene, map_size.expand(4, -1))
        loss = torch.nn.functional.mse_loss(prediction, targets)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
        model.eval()
        assert_guidance_permutation_invariant(name, model, points, mask, scene, map_size.expand(4, -1))

        def reward(candidate):
            active = torch.ones(candidate.shape[:2], dtype=torch.bool)
            return predict_reward(name, model, candidate, active, scene, map_size.expand(len(candidate), -1))

        for method, budget in SMALL_BUDGETS.items():
            set_seed(args.seed)
            result = run_sampler(reward, sampler=method, num_particles=4, num_tx=2, device=device,
                                 config=sampler_config_from_dict(method, budget))
            selected = result["best_tx_xy_norm01"]
            assert selected.shape[1:] == (2, 2)
            assert torch.isfinite(selected).all() and torch.isfinite(result["best_reward"]).all()
            assert ((selected >= 0) & (selected <= 1)).all()
            completed.append({"reward": name, "sampler": method, "passed": True})
    print(json.dumps({"task": "synthetic_interface_test", "passed": True, "combinations": completed}, indent=2))


if __name__ == "__main__":
    main()

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from samplers.evaluate_single import (
    _multi_seed_summary,
    _reward_model_label,
    _run_name_prefix,
    build_parser,
    main,
)


class MultiSeedEvaluationTests(unittest.TestCase):
    def test_target_runner_can_supply_exact_config_and_output(self):
        args = build_parser().parse_args([
            "--checkpoint", "model.pt", "--scene", "paris_07", "--num-tx", "2",
            "--sampler-config", "native.json", "--run-dir", "target/attempt_001",
        ])
        self.assertEqual(args.sampler_config, Path("native.json"))
        self.assertEqual(args.run_dir, Path("target/attempt_001"))

    def test_exact_output_rejects_multiple_seeds_before_model_loading(self):
        argv = ["evaluate_single", "--checkpoint", "missing.pt", "--scene", "paris_07",
                "--num-tx", "2", "--run-dir", "unused", "--num-seeds", "2"]
        with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "exactly one"):
            main()

    def test_exact_output_rejects_multiple_models_before_model_loading(self):
        argv = ["evaluate_single", "--checkpoints", "missing1.pt", "missing2.pt", "--scene", "paris_07",
                "--num-tx", "2", "--run-dir", "unused"]
        with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "exactly one"):
            main()

    def test_reward_model_folder_labels_are_explicit(self):
        self.assertEqual(_reward_model_label("hypergraph_potential"), "hpem")
        self.assertEqual(_reward_model_label("interaction_network"), "interaction_network")
        self.assertEqual(
            _run_name_prefix("london_01", 5, "diffusion", "hypergraph_potential"),
            "london_01_ntx5_diffusion_hpem",
        )

    def test_parser_accepts_seed_count_and_step(self):
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--scene",
                "london_01",
                "--num-tx",
                "5",
                "--num-seeds",
                "3",
                "--seed-step",
                "2",
            ]
        )
        self.assertEqual(args.num_seeds, 3)
        self.assertEqual(args.seed_step, 2)

    def test_parser_accepts_ordered_checkpoint_sweep(self):
        args = build_parser().parse_args(
            [
                "--checkpoints",
                "hpem.pt",
                "reward.pt",
                "in.pt",
                "nri.pt",
                "--scene",
                "london_01",
                "--num-tx",
                "5",
            ]
        )
        self.assertEqual(
            [path.name for path in args.checkpoints],
            ["hpem.pt", "reward.pt", "in.pt", "nri.pt"],
        )

    def test_summary_reports_mean_and_sample_standard_deviation(self):
        args = argparse.Namespace(
            scene="london_01", num_tx=5, sampler="diffusion", num_particles=10
        )
        base = {
            "predicted_reward": 0.5,
            "pathloss_coverage": 0.8,
            "ss_rsrp_coverage": 0.7,
            "sinr_coverage": 0.6,
            "effective_throughput_coverage": 0.9,
            "inference_seconds": 1.0,
            "raytracing_seconds": 2.0,
            "num_reward_evaluations": 100,
        }
        first = {
            **base,
            "seed": 42,
            "joint_4metric_coverage": 0.4,
            "result_json": "seed_42/result.json",
        }
        second = {
            **base,
            "seed": 43,
            "joint_4metric_coverage": 0.6,
            "result_json": "seed_43/result.json",
        }
        summary = _multi_seed_summary(
            args, "hypergraph_potential", Path("model.pt"), [first, second]
        )
        joint = summary["statistics"]["joint_4metric_coverage"]
        self.assertEqual(summary["seeds"], [42, 43])
        self.assertAlmostEqual(joint["mean"], 0.5)
        self.assertAlmostEqual(joint["std"], 2**0.5 / 10.0)


if __name__ == "__main__":
    unittest.main()

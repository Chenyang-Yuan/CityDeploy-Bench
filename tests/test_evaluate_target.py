import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from samplers.evaluate_target import (
    COSTS, COVERAGES, MAP_KEYS, child_command, execute_trial, fingerprint,
    refresh_reports, summarize_chain, summarize_methods, validate_plan,
    validate_result, write_json,
)


class TargetCoverageTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "scene": "paris_07", "min_num_tx": 2, "max_num_tx": 4,
            "target_joint_coverage": 0.7, "seeds": [42, 52],
            "min_tx_distance_m": 20.0, "nondeployable_weight": 10.0,
            "minimum_distance_weight": 10.0,
            "checkpoint": str(Path("model.pt").resolve()),
            "scene_root": str(Path("scenes").resolve()),
            "device": "cuda", "ray_device": "gpu",
        }
        self.method = {"sampler": "diffusion", "num_particles": 100, "config": {"steps": 100}}

    def trial(self, count, coverage):
        return {"num_tx": count, **dict.fromkeys(COVERAGES, coverage),
                **dict.fromkeys(COSTS, 10), "result_json": f"{count}/result.json",
                "predicted_reward": 0.99}

    def chain(self, trials, seed=42, error=False):
        return summarize_chain(self.plan, "diffusion", seed, trials, error)

    def test_stops_at_first_true_success_including_exact_threshold(self):
        result = self.chain([self.trial(2, 0.69), self.trial(3, 0.7), self.trial(4, 0.9)])
        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["first_reached_num_tx"], 3)
        self.assertEqual(result["cumulative_num_reward_evaluations"], 20)

    def test_predicted_reward_never_decides_success(self):
        result = self.chain([self.trial(2, 0.69)])
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["first_reached_num_tx"])

    def test_nonmonotonic_coverage_and_no_success(self):
        result = self.chain([self.trial(2, 0.69), self.trial(3, 0.4), self.trial(4, 0.68)])
        self.assertEqual(result["status"], "not_reached")
        self.assertIsNone(result["first_reached_num_tx"])
        self.assertEqual(result["best_joint_coverage"], 0.69)

    def test_missing_smaller_count_cannot_certify_first_success(self):
        result = self.chain([self.trial(2, 0.5), self.trial(4, 0.8)])
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["tested_counts"], 1)

    def test_error_is_not_failed_coverage(self):
        result = self.chain([self.trial(2, 0.5)], error=True)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["first_reached_num_tx"])

    def test_statistics_are_conditional_not_zero_filled(self):
        success = self.chain([self.trial(2, 0.8)])
        failed = self.chain([self.trial(i, 0.5) for i in (2, 3, 4)], seed=52)
        summary = summarize_methods([self.method], [success, failed])[0]
        self.assertEqual(summary["first_reached_tx_mean_successful"], 2)
        self.assertIsNone(summary["first_reached_tx_std_successful"])
        self.assertEqual(summary["success_rate_completed"], 0.5)
        self.assertEqual(summary["not_reached_seeds"], 1)

    def test_sample_standard_deviation(self):
        chains = [self.chain([self.trial(2, 0.8)]),
                  self.chain([self.trial(2, 0.5), self.trial(3, 0.8)], seed=52)]
        summary = summarize_methods([self.method], chains)[0]
        self.assertEqual(summary["first_reached_tx_mean_successful"], 2.5)
        self.assertAlmostEqual(summary["first_reached_tx_std_successful"], 2**0.5 / 2)

    def test_missing_recovered_timing_stays_unknown(self):
        trial = self.trial(2, 0.8)
        trial["wall_seconds"] = None
        chain = self.chain([trial])
        summary = summarize_methods([self.method], [chain])[0]
        self.assertIsNone(summary["total_wall_seconds"])

    def test_validate_rejects_invalid_target_seed_and_range(self):
        validate_plan(self.plan)
        for key, value in (("target_joint_coverage", 70), ("target_joint_coverage", float("nan")),
                           ("seeds", [42, 42]), ("min_num_tx", 5), ("seeds", [])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_plan({**self.plan, key: value})

    def test_fingerprint_changes_with_protocol(self):
        changed = {**self.plan, "target_joint_coverage": 0.8}
        self.assertNotEqual(fingerprint(self.plan), fingerprint(changed))
        self.assertEqual(fingerprint(self.plan), fingerprint(dict(reversed(list(self.plan.items())))))

    def fixture(self, root):
        manifest = {"plan": self.plan, "methods": [self.method], "fingerprint": "test",
                    "model_kind": "hypergraph_potential", "reward_model": "hpem",
                    "budget_policy": "native", "rf_profile": {"ray_tracing": {"samples_per_tx": 100}}}
        output = root / "diffusion/seed_42/ntx_02/attempt_001"
        output.mkdir(parents=True)
        paths = {}
        for key in MAP_KEYS:
            path = output / key
            path.write_bytes(b"fixture")
            paths[key] = str(path)
        for filename in ("trace.json", "candidates.npz"):
            (output / filename).write_bytes(b"fixture")
        result = {**self.trial(2, 0.8), "scene_id": "paris_07", "seed": 42, "ray_seed": 42,
                  "sampler": "diffusion", "num_particles": 100, "sampler_config": self.method["config"],
                  "rf_profile_snapshot": manifest["rf_profile"], "checkpoint": self.plan["checkpoint"],
                  "model_kind": "hypergraph_potential", "tx_positions_m": [[0, 0, 10], [30, 0, 10]],
                  "tx_xy_norm01": [[0.5, 0.5], [0.6, 0.5]], "radio_maps": paths,
                  "result_json": str(output / "result.json")}
        write_json(output / "result.json", result)
        return manifest, result

    def test_resume_recovers_complete_child_and_never_reruns_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, result = self.fixture(root)
            with patch("samplers.evaluate_target.subprocess.Popen") as child:
                trial = execute_trial(root, manifest, self.method, 42, 2)
                again = execute_trial(root, manifest, self.method, 42, 2)
                child.assert_not_called()
            self.assertEqual(trial, again)
            self.assertIsNone(trial["wall_seconds"])
            summary = refresh_reports(root, manifest)
            self.assertEqual(summary["completed_chains"], 1)
            self.assertEqual(summary["status"], "incomplete")
            self.assertTrue((root / "seed_summary.csv").is_file())

    def test_resume_rejects_wrong_config_or_incomplete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, result = self.fixture(Path(directory))
            wrong = copy.deepcopy(result)
            wrong["sampler_config"]["steps"] = 1
            with self.assertRaises(ValueError):
                validate_result(wrong, manifest, self.method, 42, 2)
            wrong = copy.deepcopy(result)
            wrong["radio_maps"]["joint"] = str(Path(directory) / "missing.png")
            with self.assertRaises(ValueError):
                validate_result(wrong, manifest, self.method, 42, 2)

    def test_child_command_preserves_native_config_seed_and_cardinality(self):
        manifest = {"plan": self.plan}
        command = child_command(manifest, Path("output"), self.method, 52, 3, Path("trial"))
        self.assertEqual(command[command.index("--num-tx") + 1], "3")
        self.assertEqual(command[command.index("--seed") + 1], "52")
        self.assertEqual(command[command.index("--num-particles") + 1], "100")
        self.assertIn("--sampler-config", command)


if __name__ == "__main__":
    unittest.main()

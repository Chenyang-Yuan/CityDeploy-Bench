from __future__ import annotations

import json
import math
import unittest
import tempfile
from pathlib import Path

import numpy as np

from manual_deployment.run import (
    compute_experiment_metrics,
    compute_road_coverages,
    nearest_mask,
    save_deployment_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ManualDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "manual_deployment" / "config.json").read_text(
                encoding="utf-8"
            )
        )

    def test_pathloss_and_ss_rsrp_thresholds_share_one_link_budget(self) -> None:
        radio = self.config["radio"]
        thresholds = self.config["coverage_thresholds"]
        active_subcarriers = int(radio["num_resource_blocks"]) * int(
            radio["subcarriers_per_resource_block"]
        )
        derived_pathloss = (
            float(radio["tx_power_dbm"])
            - float(thresholds["ss_rsrp_dbm_min"])
            - 10.0 * math.log10(active_subcarriers)
            + float(radio["ssb_power_offset_db"])
        )
        self.assertAlmostEqual(derived_pathloss, float(thresholds["pathloss_db_max"]), delta=0.2)

    def test_coverage_denominator_contains_only_road_cells(self) -> None:
        path_gain = np.asarray([[[1e-8, 1e-10], [1e-12, 1e-9]]], dtype=np.float64)
        metrics = compute_experiment_metrics(path_gain, num_tx=1, config=self.config)
        road = np.asarray([[True, False], [True, False]])
        coverage, masks = compute_road_coverages(
            metrics, road, self.config["coverage_thresholds"]
        )
        self.assertEqual(coverage["denominator_road_cells"], 2)
        self.assertFalse(np.any(masks["joint"][~road]))

    def test_nearest_mask_preserves_binary_domain(self) -> None:
        source = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        resized = nearest_mask(source, (6, 8))
        self.assertEqual(resized.shape, (6, 8))
        self.assertEqual(set(np.unique(resized).tolist()), {0.0, 1.0})

    def test_selected_plan_is_saved_without_claiming_simulation_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            points = np.asarray([[51, -83, 10], [120, 35, 10]])
            save_deployment_input(folder, "paris_07", points, [256, 256], self.config)
            payload = json.loads((folder / "deployment_input.json").read_text())
            self.assertEqual(payload["tx_positions_m"], points.tolist())
            self.assertEqual(payload["num_tx"], 2)
            self.assertEqual(payload["configuration"], self.config)
            self.assertFalse((folder / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

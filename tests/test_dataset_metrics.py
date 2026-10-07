from __future__ import annotations

import unittest

import numpy as np

from dataset_builder.metrics import compute_radio_metrics


class DatasetMetricTests(unittest.TestCase):
    def test_metrics_share_strongest_rsrp_association(self) -> None:
        path_gain = np.asarray(
            [
                [[1e-8, 1e-10]],
                [[1e-9, 1e-7]],
            ],
            dtype=np.float64,
        )
        metrics = compute_radio_metrics(path_gain, num_tx=2, tx_power_dbm=23.0)
        np.testing.assert_array_equal(metrics["serving_idx"], [[0, 1]])
        np.testing.assert_allclose(metrics["path_loss_db"], [[80.0, 70.0]], atol=1e-8)
        np.testing.assert_allclose(metrics["rsrp_dbm"], [[-57.0, -47.0]], atol=1e-8)
        self.assertTrue(np.all(np.isfinite(metrics["sinr_db"])))
        self.assertTrue(np.all(metrics["throughput_mbps"] > 0.0))

    def test_metric_fields_are_tx_permutation_invariant(self) -> None:
        rng = np.random.default_rng(7)
        path_gain = rng.uniform(1e-12, 1e-7, size=(3, 4, 5))
        original = compute_radio_metrics(path_gain, num_tx=3)
        permuted = compute_radio_metrics(path_gain[[2, 0, 1]], num_tx=3)
        for key in ("path_loss_db", "rsrp_dbm", "sinr_db", "throughput_mbps"):
            np.testing.assert_allclose(original[key], permuted[key], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

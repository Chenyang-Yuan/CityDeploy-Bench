from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset_builder.metrics import compute_urban_radio_metrics, summarize_urban_metrics
from dataset_builder.generate import (
    _expanded_strategy_schedule,
    _family_subsets,
    _scene_generation_profile,
    _strategy_schedule,
)
from dataset_builder.storage import (
    FamilyMembershipShardWriter,
    ParquetShardWriter,
    build_split_manifest,
    canonical_json_sha256,
    family_membership_id,
    sample_id,
)
from dataset_builder.validate import validate_dataset_v4
from energy_model.dataset import SCENE_FEATURE_CHANNELS, TxDeploymentDataset
from energy_model.reward_model import RewardPredictor, RewardPredictorConfig
from energy_model.train_reward import regression_metrics


def _row() -> dict:
    result = {
        "sample_id": "sample-1",
        "scene_id": "alpha",
        "geographic_group_id": "geo-000",
        "split": "train",
        "contract_id": "contract",
        "family_id": "family",
        "parent_sample_id": None,
        "family_role": "cardinality_subset",
        "subset_indices_in_family": [0, 1],
        "sampling_strategy": "uniform",
        "simulation_seed": 42,
        "num_tx": 2,
        "tx_density_per_km2": 2.0,
        "tx_candidate_indices": [2, 7],
        "tx_xyz_m": [[-25.6, 0.0, 10.0], [25.6, 0.0, 10.0]],
        "tx_xy_norm01": [[0.4, 0.5], [0.6, 0.5]],
        "evaluation_cell_count": 10,
        "pathloss_covered_count": 9,
        "ss_rsrp_covered_count": 8,
        "sinr_covered_count": 7,
        "effective_throughput_covered_count": 6,
        "joint_covered_count": 5,
        "pathloss_coverage": 0.9,
        "ss_rsrp_coverage": 0.8,
        "sinr_coverage": 0.7,
        "effective_throughput_coverage": 0.6,
        "joint_4metric_coverage": 0.5,
        "joint_target_met": False,
        "runtime_seconds": 1.25,
        "backend": "test",
        "rich_map_path": None,
        "valid": True,
        "warning_flags": [],
    }
    for metric in ("pathloss", "ss_rsrp", "sinr", "effective_throughput"):
        for statistic in ("mean", "p05", "p50", "p95"):
            result[f"{metric}_{statistic}"] = 1.0
    return result


class V4MetricTests(unittest.TestCase):
    def test_ss_rsrp_and_effective_throughput_follow_contract(self) -> None:
        path_gain = np.asarray(
            [
                [[1e-8, 1e-9, 1e-10]],
                [[1e-10, 1e-7, 1e-8]],
            ],
            dtype=np.float64,
        )
        metrics = compute_urban_radio_metrics(
            path_gain,
            2,
            tx_power_dbm=24.0,
            bandwidth_hz=20e6,
            noise_figure_db=7.0,
            num_resource_blocks=51,
            subcarriers_per_resource_block=12,
            ssb_power_offset_db=0.0,
            implementation_efficiency=0.65,
            resource_share=0.5,
            max_spectral_efficiency_bps_hz=6.0,
        )
        expected = metrics["rsrp_dbm_best"] - 10.0 * np.log10(51 * 12)
        np.testing.assert_allclose(metrics["ss_rsrp_dbm"], expected)
        self.assertLessEqual(float(np.nanmax(metrics["effective_throughput_mbps"])), 39.0 + 1e-8)

    def test_joint_is_cellwise_intersection(self) -> None:
        metrics = {
            "path_loss_db": np.asarray([[100.0, 120.0]]),
            "ss_rsrp_dbm": np.asarray([[-100.0, -100.0]]),
            "sinr_db": np.asarray([[10.0, 10.0]]),
            "effective_throughput_mbps": np.asarray([[20.0, 20.0]]),
        }
        summary, _ = summarize_urban_metrics(
            metrics,
            np.ones((1, 2), dtype=bool),
            {
                "pathloss_db_max": 106.0,
                "ss_rsrp_dbm_min": -110.0,
                "sinr_db_min": 5.0,
                "effective_throughput_mbps_min": 10.0,
                "joint_coverage_target": 0.9,
            },
        )
        self.assertEqual(summary["pathloss_coverage"], 0.5)
        self.assertEqual(summary["joint_4metric_coverage"], 0.5)

    def test_reward_regression_metrics_are_exact_for_perfect_predictions(self) -> None:
        target = np.asarray([0.1, 0.4, 0.8], dtype=np.float64)
        metrics = regression_metrics(target, target.copy())
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_reward_predictor_is_permutation_invariant(self) -> None:
        model = RewardPredictor(
            RewardPredictorConfig(scene_feature_dim=32, tx_hidden_dim=32, head_hidden_dim=32)
        ).eval()
        scene = torch.zeros((1, len(SCENE_FEATURE_CHANNELS), 32, 32))
        points = torch.tensor([[[0.1, 0.2], [0.7, 0.4], [0.5, 0.9]]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        size = torch.tensor([[256.0, 256.0]])
        with torch.no_grad():
            first = model(points, mask, scene, size)
            second = model(points[:, [2, 0, 1]], mask[:, [2, 0, 1]], scene, size)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=0.0)


class V4SchemaTests(unittest.TestCase):
    def test_strategy_schedule_uses_exact_reproducible_quotas(self) -> None:
        weights = {"uniform": 0.5, "maximin": 0.3, "clustered": 0.2}
        first = _strategy_schedule(weights, 20, seed=42)
        second = _strategy_schedule(weights, 20, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first.count("uniform"), 10)
        self.assertEqual(first.count("maximin"), 6)
        self.assertEqual(first.count("clustered"), 4)
        self.assertEqual(first[:5].count("uniform"), 3)
        self.assertEqual(first[:5].count("maximin"), 1)
        self.assertEqual(first[:5].count("clustered"), 1)

    def test_expanded_strategy_schedule_preserves_original_block(self) -> None:
        weights = {"uniform": 0.5, "maximin": 0.3, "clustered": 0.2}
        original = _strategy_schedule(weights, 20, seed=42)
        expanded = _expanded_strategy_schedule(weights, 20, 3, seed=42)
        self.assertEqual(expanded[:20], original)
        self.assertEqual(len(expanded), 60)
        self.assertEqual(expanded.count("uniform"), 30)
        self.assertEqual(expanded.count("maximin"), 18)
        self.assertEqual(expanded.count("clustered"), 12)

    def test_family_subsets_match_multicity_profiles(self) -> None:
        subsets = _family_subsets(
            9,
            [5, 6, 7, 8, 9],
            4,
            {"1": 2, "2": 6, "3": 4, "4": 1},
            np.random.default_rng(42),
        )
        counts = {num_tx: sum(value == num_tx for value, _ in subsets) for num_tx in range(5, 10)}
        self.assertEqual(counts, {5: 4, 6: 4, 7: 4, 8: 4, 9: 1})
        by_size = {
            num_tx: [set(int(item) for item in subset) for size, subset in subsets if size == num_tx]
            for num_tx in range(5, 10)
        }
        self.assertTrue(all(any(small < large for large in by_size[6]) for small in by_size[5]))

    def test_scene_size_specific_cardinality_profile(self) -> None:
        plan = {
            "master_pools_per_scene": 10,
            "scene_size_profiles": {
                "128x128": {
                    "num_tx_values": [3, 1, 2, 2],
                    "master_pool_size": 3,
                    "master_pools_per_scene": 20,
                }
            },
        }
        profile = _scene_generation_profile(
            plan, {"map_x_m": 128.0, "map_y_m": 128.0}
        )
        self.assertEqual(profile["scene_size_key"], "128x128")
        self.assertEqual(profile["num_tx_values"], [1, 2, 3])
        self.assertEqual(profile["master_pool_size"], 3)
        self.assertEqual(profile["master_pools_per_scene"], 20)

    def test_scene_size_profile_rejects_unconfigured_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "256x256"):
            _scene_generation_profile(
                {
                    "scene_size_profiles": {
                        "128x128": {
                            "num_tx_values": [1, 2, 3],
                            "master_pool_size": 3,
                            "master_pools_per_scene": 20,
                        }
                    }
                },
                {"map_x_m": 256.0, "map_y_m": 256.0},
            )

    def test_sample_id_is_permutation_invariant(self) -> None:
        self.assertEqual(sample_id("c", "s", [7, 2, 5]), sample_id("c", "s", [2, 5, 7]))

    def test_overlapping_scenes_never_cross_splits(self) -> None:
        metadata = {
            "a": {"source_bbox_wgs84": {"min_lon": 0, "min_lat": 0, "max_lon": 2, "max_lat": 2}},
            "b": {"source_bbox_wgs84": {"min_lon": 1, "min_lat": 1, "max_lon": 3, "max_lat": 3}},
            "c": {"source_bbox_wgs84": {"min_lon": 10, "min_lat": 0, "max_lon": 11, "max_lat": 1}},
            "d": {"source_bbox_wgs84": {"min_lon": 20, "min_lat": 0, "max_lon": 21, "max_lat": 1}},
        }
        manifest = build_split_manifest(
            metadata, train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25, seed=4
        )
        self.assertEqual(manifest["scenes"]["a"]["geographic_group_id"], manifest["scenes"]["b"]["geographic_group_id"])
        self.assertEqual(manifest["scenes"]["a"]["split"], manifest["scenes"]["b"]["split"])

    def test_explicit_split_assignments_preserve_geographic_groups(self) -> None:
        metadata = {
            "city_a_01": {"source_bbox_wgs84": {"min_lon": 0, "min_lat": 0, "max_lon": 2, "max_lat": 2}},
            "city_a_02": {"source_bbox_wgs84": {"min_lon": 1, "min_lat": 1, "max_lon": 3, "max_lat": 3}},
            "city_b_01": {"source_bbox_wgs84": {"min_lon": 10, "min_lat": 0, "max_lon": 11, "max_lat": 1}},
            "city_c_01": {"source_bbox_wgs84": {"min_lon": 20, "min_lat": 0, "max_lon": 21, "max_lat": 1}},
        }
        assignments = {
            "city_a_01": "train",
            "city_a_02": "train",
            "city_b_01": "validation",
            "city_c_01": "test",
        }
        manifest = build_split_manifest(
            metadata,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
            seed=4,
            scene_assignments=assignments,
        )
        self.assertEqual(manifest["stratification"], ["city", "map_size_m"])
        self.assertEqual(manifest["scenes"]["city_b_01"]["split"], "validation")
        conflicting = dict(assignments, city_a_02="test")
        with self.assertRaisesRegex(ValueError, "crosses explicit splits"):
            build_split_manifest(
                metadata,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=4,
                scene_assignments=conflicting,
            )

    def test_parquet_rows_load_in_energy_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"schema_version": "citydeploy.dataset.v4"}), encoding="utf-8")
            feature_root = root / "scenes" / "features"
            feature_root.mkdir(parents=True)
            np.savez_compressed(
                feature_root / "alpha.npz",
                features=np.zeros((len(SCENE_FEATURE_CHANNELS), 8, 8), dtype=np.uint8),
                channel_names=np.asarray(SCENE_FEATURE_CHANNELS),
                metadata_json=np.asarray(json.dumps({"map_x_m": 256.0, "map_y_m": 256.0})),
            )
            writer = ParquetShardWriter(root / "data" / "train", 10, 10)
            writer.append(_row())
            writer.flush()
            dataset = TxDeploymentDataset(root, permute_tx_order=False)
            item = dataset[0]
            self.assertEqual(tuple(item["tx_xy_norm01"].shape), (2, 2))
            self.assertAlmostEqual(float(item["joint"]), 0.5)
            self.assertTrue(dataset.has_explicit_splits)

    def test_complete_v4_package_passes_validator(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"schema_version": "citydeploy.dataset-contract.v4", "minimum_tx_distance_m": 0.0}
            contract["contract_id"] = canonical_json_sha256(contract)
            splits = {
                "schema_version": "citydeploy.geographic-split.v1",
                "scenes": {"alpha": {"geographic_group_id": "geo-000", "split": "train"}},
            }
            splits["sha256"] = canonical_json_sha256(splits)
            registry = {"schema_version": "citydeploy.scene-registry.v1", "scenes": {"alpha": {}}}
            registry["sha256"] = canonical_json_sha256(registry)
            (root / "dataset_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "split_manifest.json").write_text(json.dumps(splits), encoding="utf-8")
            (root / "scene_registry.json").write_text(json.dumps(registry), encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": "citydeploy.dataset.v4", "dataset_id": "test", "num_samples_total": 1, "num_family_memberships": 1, "scene_registry_sha256": registry["sha256"]}),
                encoding="utf-8",
            )
            (root / "generation_plan.json").write_text(
                json.dumps({"subsets_per_cardinality": 1, "hypergraph_subsets_per_family": {}}),
                encoding="utf-8",
            )
            contract["scene_generation_profiles"] = {
                "alpha": {"master_pool_size": 2, "num_tx_values": [2]}
            }
            contract["contract_id"] = canonical_json_sha256(
                {key: value for key, value in contract.items() if key != "contract_id"}
            )
            (root / "dataset_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            feature_root = root / "scenes" / "features"
            feature_root.mkdir(parents=True)
            np.savez_compressed(
                feature_root / "alpha.npz",
                features=np.zeros((len(SCENE_FEATURE_CHANNELS), 8, 8), dtype=np.uint8),
                channel_names=np.asarray(SCENE_FEATURE_CHANNELS),
                metadata_json=np.asarray(json.dumps({"map_x_m": 256.0, "map_y_m": 256.0})),
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [{"scene_id": "alpha", "map_x_m": 256.0, "map_y_m": 256.0}]
                ),
                root / "scenes" / "scenes.parquet",
            )
            writer = ParquetShardWriter(root / "data" / "train", 10, 10)
            writer.append(_row())
            writer.flush()
            membership_writer = FamilyMembershipShardWriter(root / "family_memberships" / "train", 10, 10)
            membership_writer.append(
                {
                    "membership_id": family_membership_id("family", "sample-1"),
                    "family_id": "family",
                    "sample_id": "sample-1",
                    "scene_id": "alpha",
                    "split": "train",
                    "contract_id": contract["contract_id"],
                    "pool_index": 0,
                    "master_sample_id": "sample-1",
                    "family_role": "anchor",
                    "subset_indices_in_family": [0, 1],
                    "sampling_strategy": "uniform",
                    "simulation_seed": 42,
                    "num_tx": 2,
                }
            )
            membership_writer.flush()
            report = validate_dataset_v4(root)
            self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()

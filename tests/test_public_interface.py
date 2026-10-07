from __future__ import annotations
import json
import os
import struct
import zlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from citydeploy.paths import config_path, resolve_path
from citydeploy.schema import canonical_schema
from citydeploy.demo_data import create_dataset
from citydeploy.audit import inspect_source
from citydeploy.summarize import aggregate, FIELDS
from dataset_builder.validate import validate_dataset_v4
from energy_model.dataset import TxDeploymentDataset, load_scene_feature_tensor


class PublicInterfaceTests(unittest.TestCase):
    def test_ignore_rules_do_not_hide_bundled_model_and_dataset_configs(self):
        rules = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
        for name in ("datasets", "models", "outputs"):
            self.assertIn(f"/{name}/", rules)
            self.assertNotIn(f"{name}/", rules)

    def test_config_falls_back_to_package_and_workspace_override_wins(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CITYDEPLOY_WORKSPACE": folder}):
            bundled = config_path("models/hpem.json")
            self.assertTrue(bundled.is_file())
            local = Path(folder) / "configs/models/hpem.json"
            local.parent.mkdir(parents=True)
            local.write_text("{}", encoding="utf-8")
            self.assertEqual(config_path("models/hpem.json"), local)
            self.assertEqual(resolve_path("models/example.pt"), Path(folder) / "models/example.pt")
            with self.assertRaises(ValueError):
                config_path("../escape.json")

    def test_legacy_schema_requires_explicit_prefix_and_only_maps_leading_namespace(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(canonical_schema("example.dataset.v4"), "example.dataset.v4")
        with patch.dict(os.environ, {"CITYDEPLOY_LEGACY_SCHEMA_PREFIX": "example"}):
            self.assertEqual(canonical_schema("example.dataset.v4"), "citydeploy.dataset.v4")
            self.assertEqual(canonical_schema("unrelated.example.v4"), "unrelated.example.v4")

    def test_demo_dataset_passes_full_integrity_and_loads_all_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            data, scenes = create_dataset(Path(folder) / "demo", rows_per_split=4)
            result = validate_dataset_v4(data)
            self.assertTrue(result["passed"], json.dumps(result, indent=2))
            for split in ("train", "validation", "test"):
                loaded = TxDeploymentDataset(data, scenes, include_splits=[split])
                self.assertEqual(len(loaded), 4)
            features, _ = load_scene_feature_tensor(scenes / "synthetic_2")
            self.assertEqual(features.shape, (11, 16, 16))
            with self.assertRaises(FileExistsError):
                create_dataset(Path(folder) / "demo")

    def test_source_audit_detects_weights_without_exposing_content(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "checkpoint.pt").write_bytes(b"placeholder")
            report = inspect_source(Path(folder))
            self.assertFalse(report["passed"])
            self.assertEqual(report["findings"][0]["kind"], "generated_or_private_artifact")

    def test_source_audit_limits_figures_to_reviewed_assets_without_metadata(self):
        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)

        header = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        pixels = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        ending = chunk(b"IEND", b"")
        clean = header + pixels + ending
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            figure = root / "docs/assets/benchmark_overview.png"
            figure.parent.mkdir(parents=True)
            figure.write_bytes(clean)
            self.assertTrue(inspect_source(root)["passed"])
            for metadata in (b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"):
                figure.write_bytes(header + chunk(metadata, b"private metadata") + pixels + ending)
                self.assertEqual(inspect_source(root)["findings"][0]["kind"], "figure_metadata_or_unreviewed_chunk")
            for invalid in (b"not an image", clean + b"appended content", clean[:-1]):
                figure.write_bytes(invalid)
                self.assertFalse(inspect_source(root)["passed"])
            figure.write_bytes(clean)
            (root / "unknown.png").write_bytes(clean)
            self.assertEqual(inspect_source(root)["findings"][0]["kind"], "unreviewed_file_type")

    def test_summary_sample_std_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, seed, value, model="hypergraph_potential"):
                path = root / name / "result.json"
                path.parent.mkdir()
                row = {field: value for field in FIELDS}
                row.update(scene_id="synthetic_2", num_tx=2, sampler="diffusion", model_kind=model,
                           seed=seed, tx_positions_m=[[0, 0, 10]], checkpoint=model,
                           sampler_config={"steps": 2}, num_particles=4, rf_profile_snapshot={})
                path.write_text(json.dumps(row), encoding="utf-8")
            write("a", 42, .2)
            write("b", 52, .4)
            result = aggregate(root)
            self.assertEqual(result[0]["n"], 2)
            self.assertAlmostEqual(result[0]["joint_4metric_coverage_mean"], .3)
            self.assertAlmostEqual(result[0]["joint_4metric_coverage_std"], np.std([.2, .4], ddof=1))
            write("c", 42, .5, "nri")
            self.assertEqual(len(aggregate(root)), 2)
            write("d", 42, .2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                aggregate(root)


if __name__ == "__main__":
    unittest.main()

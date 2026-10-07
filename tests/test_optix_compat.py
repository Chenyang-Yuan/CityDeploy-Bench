from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dataset_builder import optix_compat as compat


def fake_dll():
    parts = []
    for count in (6, 12, 1):
        header = (b".func reduce_$s_$s() {\n" if count != 1
                  else b".func (.param .u32 rv) reduce_inc_u32 () {\n")
        parts.append(header + b"    .reg .pred %leader;\n" + b" " * 200
                     + b"\n" + b"    shfl.sync.idx.b32 %q1, %q0, 0, 31, %active;\n" * count + b"}\n")
    return b"MZ-other-sections\0" + b"\0unchanged\0".join(parts) + b"\0exports\0", parts


class OptixCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.state_patch = patch.object(compat, "_state", None)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_backport_changes_only_the_three_text_slots(self):
        source, templates = fake_dll()
        with patch.object(compat, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
            fixed = compat.patch_ptx_templates(source)
        self.assertEqual(len(source), len(fixed))
        self.assertEqual(fixed.count(b"|%unused"), 19)
        self.assertEqual(fixed.count(b".reg .pred %leader, %unused;"), 3)
        inside = set()
        for template in templates:
            begin = source.index(template)
            inside.update(range(begin, begin + len(template)))
        for index, (before, after) in enumerate(zip(source, fixed)):
            if index not in inside:
                self.assertEqual(before, after)
        self.assertEqual([i for i, b in enumerate(source) if b == 0],
                         [i for i, b in enumerate(fixed) if b == 0])

    def test_unknown_binary_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            compat.patch_ptx_templates(b"unknown dll")

    def test_wrong_number_of_templates_is_rejected(self):
        source, _ = fake_dll()
        source = source.replace(b"shfl.sync.idx.b32", b"different.op.b32", 1)
        with patch.object(compat, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
            with self.assertRaises(RuntimeError):
                compat.patch_ptx_templates(source)

    def test_other_platforms_do_not_inspect_driver_or_load_libraries(self):
        with patch.object(compat.sys, "platform", "darwin"), patch.object(compat, "_driver_versions") as probe:
            self.assertEqual(compat.prepare_optix_compatibility()["reason"], "not_windows")
            probe.assert_not_called()

    def test_off_switch_and_old_driver(self):
        with patch.object(compat.sys, "platform", "win32"), patch.dict(compat.os.environ, {"CITYDEPLOY_OPTIX_COMPAT": "off"}):
            self.assertEqual(compat.prepare_optix_compatibility()["reason"], "disabled_by_environment")
        compat._state = None
        with patch.object(compat.sys, "platform", "win32"), patch.object(compat, "_driver_versions", return_value=["595.71"]):
            self.assertEqual(compat.prepare_optix_compatibility()["reason"], "driver_not_affected")

    def test_other_drjit_releases_are_not_patched(self):
        distribution = Mock(version="1.4.0")
        with patch.object(compat.sys, "platform", "win32"), patch.object(compat, "_driver_versions", return_value=["610.60"]), patch.object(compat.importlib.metadata, "distribution", return_value=distribution):
            self.assertEqual(compat.prepare_optix_compatibility()["reason"], "not_the_pinned_drjit_version")
            distribution.locate_file.assert_not_called()

    def test_cached_result_is_not_mutable_by_caller(self):
        compat._state = {"applied": False, "reason": "test", "drivers": ["610.60"]}
        state = compat.prepare_optix_compatibility()
        state["applied"] = True
        state["drivers"].clear()
        self.assertFalse(compat.prepare_optix_compatibility()["applied"])
        self.assertEqual(compat.prepare_optix_compatibility()["drivers"], ["610.60"])

    def test_late_import_refuses_to_load_a_second_core(self):
        source, _ = fake_dll()
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original.dll"
            original.write_bytes(source)
            distribution = Mock(version="1.2.0")
            distribution.locate_file.return_value = original
            with patch.object(compat.sys, "platform", "win32"), patch.object(compat, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()), patch.object(compat, "_driver_versions", return_value=["610.60"]), patch.object(compat.importlib.metadata, "distribution", return_value=distribution), patch.object(compat, "_loaded_core_path", return_value=original):
                with self.assertRaisesRegex(RuntimeError, "already loaded"):
                    compat.prepare_optix_compatibility()
            self.assertEqual(original.read_bytes(), source)

    def test_corrupted_cache_is_not_silently_loaded(self):
        source, _ = fake_dll()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.dll"
            original.write_bytes(source)
            distribution = Mock(version="1.2.0")
            distribution.locate_file.return_value = original
            with patch.object(compat.sys, "platform", "win32"), patch.object(compat, "CACHE_ROOT", root), patch.object(compat, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()), patch.object(compat, "_driver_versions", return_value=["610.60"]), patch.object(compat.importlib.metadata, "distribution", return_value=distribution), patch.object(compat, "_loaded_core_path", return_value=None):
                digest = hashlib.sha256(compat.patch_ptx_templates(source)).hexdigest()
                cache = root / f"drjit-1.2.0-{digest[:16]}" / "drjit-core.dll"
                cache.parent.mkdir()
                cache.write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "Corrupt"):
                    compat.prepare_optix_compatibility()
            self.assertEqual(original.read_bytes(), source)

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cache" / "manifest.json"
            compat._atomic_write(output, b'{"passed":true}')
            self.assertTrue(json.loads(output.read_text())["passed"])
            self.assertEqual(len(list(output.parent.iterdir())), 1)

    def test_concurrent_identical_write_to_loaded_dll_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "core.dll"
            output.write_bytes(b"identical")
            with patch.object(compat.os, "replace", side_effect=PermissionError):
                compat._atomic_write(output, b"identical")
                with self.assertRaises(PermissionError):
                    compat._atomic_write(output, b"different")
            self.assertEqual(output.read_bytes(), b"identical")


if __name__ == "__main__":
    unittest.main()

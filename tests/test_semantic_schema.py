from __future__ import annotations

import unittest

import numpy as np

from scene_builder.semantics import (
    SCHEMA_VERSION,
    highway_half_width_m,
    primary_semantic_class,
    semantic_flags,
)
from scene_builder.geo import parse_osm_length_m, random_building_height
from scene_builder.scene_inputs import compose_tx_feasible_mask


class SemanticSchemaTests(unittest.TestCase):
    def test_landuse_is_context_not_a_building(self) -> None:
        flags = semantic_flags({"landuse": "residential"})
        self.assertFalse(flags.building)
        self.assertTrue(flags.built_up)
        self.assertEqual(primary_semantic_class({"landuse": "residential"}), "built_up")

    def test_explicit_building_wins_over_landuse(self) -> None:
        tags = {"building": "apartments", "landuse": "residential"}
        self.assertEqual(primary_semantic_class(tags), "building")

    def test_explicit_road_width_and_lanes_precede_class_default(self) -> None:
        self.assertAlmostEqual(highway_half_width_m({"highway": "service", "width": "8 m"}), 4.0)
        self.assertAlmostEqual(highway_half_width_m({"highway": "service", "lanes": "2"}), 3.2)
        self.assertAlmostEqual(highway_half_width_m({"highway": "residential"}), 4.0)

    def test_schema_is_versioned(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "citydeploy.osm-semantics.v3")


class BuildingHeightTests(unittest.TestCase):
    def test_osm_length_units(self) -> None:
        self.assertAlmostEqual(parse_osm_length_m("12 m"), 12.0)
        self.assertAlmostEqual(parse_osm_length_m("30 ft"), 9.144)

    def test_levels_are_used_without_unrelated_level_tag(self) -> None:
        self.assertAlmostEqual(random_building_height({"building:levels": "4"}, None), 12.8)

    def test_fallback_is_deterministic(self) -> None:
        building = {"building": "house"}
        self.assertEqual(random_building_height(building, None), random_building_height(building, None))
        self.assertAlmostEqual(random_building_height(building, None), 6.4)


class TxFeasibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.masks = {
            "road_mask": np.array([[1, 0, 0]], dtype=np.float32),
            "explicit_open_mask": np.array([[0, 1, 0]], dtype=np.float32),
            "background_mask": np.array([[0, 0, 1]], dtype=np.float32),
        }

    def test_default_does_not_treat_background_as_safe(self) -> None:
        feasible, _ = compose_tx_feasible_mask(self.masks)
        np.testing.assert_array_equal(feasible, [[1, 0, 0]])

    def test_explicit_open_requires_opt_in(self) -> None:
        feasible, _ = compose_tx_feasible_mask(self.masks, "road_and_explicit_open")
        np.testing.assert_array_equal(feasible, [[1, 1, 0]])


if __name__ == "__main__":
    unittest.main()

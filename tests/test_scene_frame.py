from __future__ import annotations

import unittest

from pyproj import Transformer

from scene_builder.geo import get_utm_epsg_code_from_gps, metric_bbox_from_center


class SceneFrameTests(unittest.TestCase):
    def test_query_bbox_contains_all_projected_scene_corners(self) -> None:
        lon, lat = -0.138, 51.523
        width, height = 512.0, 512.0
        min_lon, min_lat, max_lon, max_lat = metric_bbox_from_center(lon, lat, width, height)
        epsg = get_utm_epsg_code_from_gps(lon, lat)
        to_utm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
        to_gps = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        center_x, center_y = to_utm.transform(lon, lat)
        for x in (center_x - width / 2, center_x + width / 2):
            for y in (center_y - height / 2, center_y + height / 2):
                corner_lon, corner_lat = to_gps.transform(x, y)
                self.assertLessEqual(min_lon, corner_lon)
                self.assertLessEqual(corner_lon, max_lon)
                self.assertLessEqual(min_lat, corner_lat)
                self.assertLessEqual(corner_lat, max_lat)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from Geo2SigMap; see THIRD_PARTY_NOTICES.md for attribution and changes.
import logging
import math
import os
import json

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape, Polygon
from shapely import affinity
from shapely.ops import unary_union
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import osmnx as ox


from tqdm import tqdm
from triangle import triangulate
from PIL import Image, ImageDraw
from pyproj import Transformer

from .geo import *
from .itu_materials import ITU_MATERIALS
from .semantics import (
    OSM_QUERY_TAGS,
    SCHEMA_VERSION as SEMANTIC_SCHEMA_VERSION,
    highway_half_width_m,
    primary_semantic_class,
)
import open3d.core as o3c

import datetime

import pyvista as pv
from pathlib import Path

from .terrain.dem import generate_terrain_mesh_dem
# Create a module-level logger
logger = logging.getLogger(__name__)

def _to_clean_lower_str(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def _has_tag_value(v) -> bool:
    return _to_clean_lower_str(v) != ""


def _surface_mask_from_semantic_gdf(gdf: "pd.DataFrame"):
    primary = gdf.apply(primary_semantic_class, axis=1)
    road_mask = primary == "transport"
    green_mask = primary == "green"
    water_mask = primary == "water"
    obstacle_mask = primary == "obstacle"
    return road_mask, green_mask, water_mask, obstacle_mask


def _geoms_from_masked_features(gdf: "pd.DataFrame", mask, kind: str):
    g = gdf[mask].copy()
    if g.empty:
        return []

    geoms = []
    geom_type = g.geometry.geom_type
    is_poly = geom_type.isin(["Polygon", "MultiPolygon"])
    is_line = geom_type.isin(["LineString", "MultiLineString"])
    is_point = geom_type.isin(["Point", "MultiPoint"])

    geoms.extend(list(g[is_poly].geometry.values))

    if kind == "road":
        for _, row in g[is_line].iterrows():
            w_buf = 2.0 if _has_tag_value(row.get("railway", np.nan)) else highway_half_width_m(row)
            geoms.append(row.geometry.buffer(w_buf))
        for _, row in g[is_point].iterrows():
            geoms.append(row.geometry.buffer(1.5))
    elif kind == "water":
        for _, row in g[is_line].iterrows():
            geoms.append(row.geometry.buffer(1.5))
        for _, row in g[is_point].iterrows():
            geoms.append(row.geometry.buffer(2.0))
    elif kind == "green":
        for _, row in g[is_line].iterrows():
            geoms.append(row.geometry.buffer(0.9))
        for _, row in g[is_point].iterrows():
            geoms.append(row.geometry.buffer(0.8))
    elif kind == "obstacle":
        for _, row in g[is_line].iterrows():
            geoms.append(row.geometry.buffer(0.8))
        for _, row in g[is_point].iterrows():
            geoms.append(row.geometry.buffer(0.8))
    return [x for x in geoms if x is not None and (not x.is_empty)]


def _polygon_list_from_geoms(geoms):
    if not geoms:
        return []
    u = unary_union(geoms)
    if u.is_empty:
        return []
    if u.geom_type == "Polygon":
        return [u]
    if u.geom_type == "MultiPolygon":
        return list(u.geoms)
    return []


class Scene:
    """
    Construct a metric urban scene from OpenStreetMap geometry and planar ground.


    Usage:
        scene_instance = Scene()
        scene_instance(
            points=[(lon1, lat1), (lon2, lat2), ...],
            data_dir="path/to/output",
            hag_tiff_path="path/to/hag.tiff",
            osm_server_addr="https://overpass-api.de/api/interpreter",
            lidar_calibration=False,
            generate_building_map=True
        )
    """

    def __call__(
        self,
        points,
        data_dir,
        hag_tiff_path,
        osm_server_addr=None,
        osm_requests_timeout: int = 90,
        osm_use_cache: bool = True,
        lidar_calibration: bool = False,
        generate_building_map: bool = True,
        write_ply_ascii: bool = False,
        ground_scale: float = 1.5,
        ground_material_type="mat-itu_wet_ground",
        empty_land_material_type=None,
        road_material_type="mat-itu_wet_ground",
        water_material_type="mat-itu_fresh_water",
        green_material_type="mat-itu_wood",
        obstacle_material_type="mat-itu_concrete",
        rooftop_material_type="mat-itu_metal",
        wall_material_type="mat-itu_concrete",
        lidar_terrain:bool = False,
        dem_terrain:bool = False,
        gen_lidar_terrain_only:bool = False
    ):
        """
        Generate a ground mesh from the given polygon (defined by `points`),
        query OSM for building footprints, extrude them into 3D meshes,
        and optionally produce a 2D building-height map.

        Parameters
        ----------
        points : list of (float, float)
            Coordinates defining the polygon (in WGS84 lon/lat).
        data_dir : str
            Directory where output files (XML, meshes, etc.) will be saved.
        osm_server_addr : str, optieonal
            Custom Overpass API endpoint. If None, osmnx's default is used.
        lidar_calibration : bool, optional
            If True, attempt to derive building heights from the HAG file; else use random fallback.
        generate_building_map : bool, optional
            If True, generate a 2D building map image (and save as a NumPy file).
        write_ply_ascii : bool, optional
            If True, write the ply file in ascii format, otherwise binary format will be used.
        ground_scale : float, optional
            The ratio to scale the ground polygon. TODO:Add examples to show why need scale. OSMNX query intersection.

        Returns
        -------
        np.ndarray
            If generate_building_map is True, returns a 2D NumPy array of building heights.
            Otherwise, returns None.
        """

        if lidar_terrain or lidar_calibration or gen_lidar_terrain_only or dem_terrain:
            raise ValueError("This urban scene interface uses planar ground and OSM-derived building heights.")

        if empty_land_material_type is None:
            empty_land_material_type = ground_material_type

        if ground_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid ground material type: {ground_material_type}")
        if empty_land_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid empty land material type: {empty_land_material_type}")
        if road_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid road material type: {road_material_type}")
        if water_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid water material type: {water_material_type}")
        if green_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid green material type: {green_material_type}")
        if obstacle_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid obstacle material type: {obstacle_material_type}")
        if rooftop_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid rooftop material type: {rooftop_material_type}")
        if wall_material_type not in ITU_MATERIALS:
            raise ValueError(f"Invalid wall material type: {wall_material_type}")
        
        # ---------------------------------------------------------------------
        # 1) Setup OSM server and transforms
        # ---------------------------------------------------------------------
        try:
            str(Path(data_dir).resolve()).encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "Open3D mesh output path must contain ASCII characters only on Windows: "
                f"{Path(data_dir).resolve()}"
            ) from exc
        if osm_server_addr:
            ox.settings.overpass_url = osm_server_addr
        # Respect public Overpass capacity signals and retain successful
        # responses so a later pipeline retry does not repeat the same query.
        ox.settings.overpass_rate_limit = True
        ox.settings.requests_timeout = int(osm_requests_timeout)
        ox.settings.use_cache = bool(osm_use_cache)
        # Determine the UTM projection from the first point
        projection_UTM_EPSG_code = get_utm_epsg_code_from_gps(
            points[0][0], points[0][1]
        )
        logger.info(f"Using UTM Zone: {projection_UTM_EPSG_code}")

        # Create transformations between WGS84 (EPSG:4326) and UTM
        to_projection = Transformer.from_crs(
            "EPSG:4326", projection_UTM_EPSG_code, always_xy=True
        )
        to_4326 = Transformer.from_crs(
            projection_UTM_EPSG_code, "EPSG:4326", always_xy=True
        )

        # ---------------------------------------------------------------------
        # 2) Prepare output directories and camera / material settings
        # ---------------------------------------------------------------------
        mesh_data_dir = os.path.join(data_dir, "mesh")
        os.makedirs(os.path.join(mesh_data_dir), exist_ok=True)

        def print_material_info(surface_name, material_type):
            if isinstance(ITU_MATERIALS[material_type]["lower_freq_limit"], list):
                logger.info(
                    "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz) | {:^5} - {:^5} (GHz)".format(
                        "{} Material Type:".format(surface_name),
                        ITU_MATERIALS[material_type]["name"],
                        print_if_int(
                            ITU_MATERIALS[material_type]["lower_freq_limit"][0] / 1e9
                        ),
                        print_if_int(
                            ITU_MATERIALS[material_type]["upper_freq_limit"][0] / 1e9
                        ),
                        print_if_int(
                            ITU_MATERIALS[material_type]["lower_freq_limit"][1] / 1e9
                        ),
                        print_if_int(
                            ITU_MATERIALS[material_type]["upper_freq_limit"][1] / 1e9
                        ),
                    )
                )
            else:
                logger.info(
                    "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz)".format(
                        "{} Material Type:".format(surface_name),
                        ITU_MATERIALS[material_type]["name"],
                        print_if_int(
                            ITU_MATERIALS[material_type]["lower_freq_limit"] / 1e9
                        ),
                        print_if_int(
                            ITU_MATERIALS[material_type]["upper_freq_limit"] / 1e9
                        ),
                    )
                )

        logger.info("")
        print_material_info("Ground (legacy)", ground_material_type)
        print_material_info("Empty Land", empty_land_material_type)
        print_material_info("Road", road_material_type)
        print_material_info("Water", water_material_type)
        print_material_info("Green", green_material_type)
        print_material_info("Obstacle", obstacle_material_type)
        print_material_info("Building Rooftop", rooftop_material_type)
        print_material_info("Building Wall", wall_material_type)
        logger.info("")


        camera_settings = {
            "rotation": (0, 0, -90),  # Assuming Z-up orientation
            "fov": 42.854885,
        }

        # ---------------------------------------------------------------------
        # 3) Build the XML scene root
        # ---------------------------------------------------------------------


        # Default Mitsuba rendering parameters
        spp_default = 4096
        resx_default = 1024
        resy_default = 1024

        scene = ET.Element("scene", version="2.1.0")
        # Default integrator / film settings
        ET.SubElement(scene, "default", name="spp", value=str(spp_default))
        ET.SubElement(scene, "default", name="resx", value=str(resx_default))
        ET.SubElement(scene, "default", name="resy", value=str(resy_default))

        ET.SubElement(scene, "default", name="scenegen_version", value=str(get_package_version()))
        ET.SubElement(scene, "default", name="scenegen_create_time", value=str(datetime.datetime.now()))

        ET.SubElement(scene, "default", name="scenegen_min_lat", value=str(points[0][1]))
        ET.SubElement(scene, "default", name="scenegen_max_lat", value=str(points[1][1]))
        ET.SubElement(scene, "default", name="scenegen_min_lon", value=str(points[0][0]))
        ET.SubElement(scene, "default", name="scenegen_max_lon", value=str(points[2][0]))
        


        ET.SubElement(scene, "default", name="scenegen_ground_material", value=str(ground_material_type))
        ET.SubElement(scene, "default", name="scenegen_empty_land_material", value=str(empty_land_material_type))
        ET.SubElement(scene, "default", name="scenegen_road_material", value=str(road_material_type))
        ET.SubElement(scene, "default", name="scenegen_water_material", value=str(water_material_type))
        ET.SubElement(scene, "default", name="scenegen_green_material", value=str(green_material_type))
        ET.SubElement(scene, "default", name="scenegen_obstacle_material", value=str(obstacle_material_type))
        ET.SubElement(scene, "default", name="scenegen_rooftop_material", value=str(rooftop_material_type))
        ET.SubElement(scene, "default", name="scenegen_wall_material", value=str(wall_material_type))

        ET.SubElement(scene, "default", name="scenegen_UTM_zone", value=str(projection_UTM_EPSG_code))
        
       

        integrator = ET.SubElement(scene, "integrator", type="path")
        ET.SubElement(integrator, "integer", name="max_depth", value="12")

        # Define materials
        # Sionna's ITU parser in common releases does not accept
        # fresh_water / sea_water ITU types, so skip them in scene.xml.
        unsupported_sionna_ids = {
            "mat-itu_fresh_water",
            "mat-itu_sea_water",
            "mat-fresh_water_P.527",
            "mat-sea_water_P.527",
        }
        for material_id, material_content in ITU_MATERIALS.items():
            
            # Skip unsupported ids and vacuum placeholder.
            if material_id in unsupported_sionna_ids:
                continue
            if "vacuum" in material_id:
                continue

            if "P.527" not in material_id:
                bsdf_twosided = ET.SubElement(
                    scene, "bsdf", type="twosided", id=material_id
                )
                bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
                rgb = material_content["mitsuba_color"]
                ET.SubElement(
                    bsdf_diffuse,
                    "rgb",
                    value=f"{rgb[0]} {rgb[1]} {rgb[2]}",
                    name="reflectance",
                )
            else:
                bsdf_twosided = ET.SubElement(
                    scene, "bsdf", type="radio-material", id=material_id
                )
                

        # Add emitter (constant environment light)
        emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
        ET.SubElement(
            emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance"
        )

        # Add camera (sensor)
        sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
        ET.SubElement(sensor, "string", name="fov_axis", value="x")
        ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
        ET.SubElement(
            sensor, "float", name="principal_point_offset_x", value="0.000000"
        )
        ET.SubElement(
            sensor, "float", name="principal_point_offset_y", value="-0.000000"
        )
        ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
        ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
        sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
        ET.SubElement(
            sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0])
        )
        ET.SubElement(
            sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1])
        )
        ET.SubElement(
            sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2])
        )
        camera_position = np.array([0, 0, 100])  # Adjust camera height
        ET.SubElement(
            sionna_transform, "translate", value=" ".join(map(str, camera_position))
        )
        sampler = ET.SubElement(sensor, "sampler", type="independent")
        ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
        film = ET.SubElement(sensor, "film", type="hdrfilm")
        ET.SubElement(film, "integer", name="width", value="$resx")
        ET.SubElement(film, "integer", name="height", value="$resy")

        # ---------------------------------------------------------------------
        # 4) Create ground polygon (in UTM) and ground mesh
        # ---------------------------------------------------------------------

        # # Define the points in counter-clockwise order to create the polygon
        # points = [top_left, top_right, bottom_right, bottom_left]
        ground_polygon_4326 = shapely.geometry.Polygon(points)
        ground_polygon_4326_bbox = ground_polygon_4326.bounds

        # Transform each WGS84 coordinate into UTM
        coords = [to_projection.transform(x, y) for x, y in points]
        ground_polygon = shapely.geometry.Polygon(coords)
        ground_polygon_bbox = ground_polygon.bounds

        self._ground_polygon_envelope_UTM = ground_polygon.envelope

        center_x = ground_polygon.envelope.centroid.x
        center_y = ground_polygon.envelope.centroid.y

        ET.SubElement(scene, "default", name="scenegen_center_lat", value=f"{ground_polygon_4326.envelope.centroid.y:.6f}")
        ET.SubElement(scene, "default", name="scenegen_center_lon", value=f"{ground_polygon_4326.envelope.centroid.x:.6f}")


        # ---------------------------------------------------------------------
        # Construct the planar ground mesh in the projected local frame.
        #######Open3D#######
        outer_xy = unique_coords(
            reorder_localize_coords(ground_polygon.exterior, center_x, center_y)
        )
        holes_xy = []

        def edge_idxs(nv):
            i = np.append(np.arange(nv), 0)
            return np.stack([i[:-1], i[1:]], axis=1)

        nv = 0
        verts, edges = [], []
        for loop in (outer_xy, *holes_xy):
            logger.debug(f"Loop: {loop}")
            verts.append(loop)
            edges.append(nv + edge_idxs(len(loop)))
            nv += len(loop)

        verts, edges = np.concatenate(verts), np.concatenate(edges)

        logger.debug(f"Verts: {verts}, Edges: {edges}")

        # Triangulate needs to know a single interior point for each hole
        # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
        # e.g. shapely's `polylabel`
        holes = np.array([np.mean(h, axis=0) for h in holes_xy])

        # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
        d = triangulate(dict(vertices=verts, segments=edges), opts="p")

        # Convert back to pyvista
        v, f = d["vertices"], d["triangles"]
        nv, nf = len(v), len(f)
        points = np.concatenate([v, np.zeros((nv, 1))], axis=1)

        logger.debug(f"points from triangulate: {points}")
        # print("faces from triangulate", faces)

        # Build Open3D TriangleMesh
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(points)
        mesh_o3d.triangle.indices = o3d.core.Tensor(f)

        # logger.debug(f"mesh_o3d.get_center():{mesh_o3d.scale(1.2, mesh_o3d.get_center())}" )

        mesh_o3d.scale(ground_scale, mesh_o3d.get_center())
        ground_path = os.path.join(mesh_data_dir, "ground.ply")
        if not o3d.t.io.write_triangle_mesh(
            ground_path, mesh_o3d, write_ascii=write_ply_ascii
        ):
            raise OSError(f"Open3D failed to write mesh: {ground_path}")

        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
        if lidar_terrain:
            ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/lidar_terrain.ply")
        else:
            ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/ground.ply")
        bsdf_ref = ET.SubElement(
            sionna_shape, "ref", id=empty_land_material_type, name="bsdf"
        )
        ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

        # ---------------------------------------------------------------------
        # 4.5) Query OSM semantic surface features and create overlays:
        #      road, water, green, obstacle.
        #      Empty land remains the base ground material.
        # ---------------------------------------------------------------------
        surface_tags = OSM_QUERY_TAGS
        road_feature_count = 0
        green_feature_count = 0
        water_feature_count = 0
        obstacle_feature_count = 0
        road_mesh_count = 0
        green_mesh_count = 0
        water_mesh_count = 0
        obstacle_mesh_count = 0

        def _add_surface_meshes(poly_list, name_prefix, z_offset, material_type):
            mesh_idx = 0
            for poly in poly_list:
                if poly is None or poly.is_empty:
                    continue
                outer_xy = unique_coords(
                    reorder_localize_coords(poly.exterior, center_x, center_y)
                )
                if len(outer_xy) < 3:
                    continue

                holes_xy = []
                for inner_hole in list(poly.interiors):
                    valid_coords = reorder_localize_coords(inner_hole, center_x, center_y)
                    hole_xy = unique_coords(valid_coords)
                    if len(hole_xy) >= 3:
                        holes_xy.append(hole_xy)

                def _edge_idxs(nv):
                    i = np.append(np.arange(nv), 0)
                    return np.stack([i[:-1], i[1:]], axis=1)

                nv = 0
                verts, edges = [], []
                for loop in (outer_xy, *holes_xy):
                    verts.append(loop)
                    edges.append(nv + _edge_idxs(len(loop)))
                    nv += len(loop)
                if not verts:
                    continue

                verts = np.concatenate(verts)
                edges = np.concatenate(edges)
                holes = np.array([np.mean(h, axis=0) for h in holes_xy]) if holes_xy else np.array([])
                try:
                    if len(holes) != 0:
                        d = triangulate(dict(vertices=verts, segments=edges, holes=holes), opts="p")
                    else:
                        d = triangulate(dict(vertices=verts, segments=edges), opts="p")
                    v, f = d["vertices"], d["triangles"]
                except Exception:
                    continue
                if len(v) < 3 or len(f) < 1:
                    continue

                points_s = np.concatenate([v, np.full((len(v), 1), z_offset)], axis=1)
                surface_mesh = o3d.t.geometry.TriangleMesh()
                surface_mesh.vertex.positions = o3d.core.Tensor(points_s)
                surface_mesh.triangle.indices = o3d.core.Tensor(f)
                ply_name = f"{name_prefix}_{mesh_idx}.ply"
                surface_path = os.path.join(mesh_data_dir, ply_name)
                if not o3d.t.io.write_triangle_mesh(
                    surface_path, surface_mesh, write_ascii=write_ply_ascii
                ):
                    raise OSError(f"Open3D failed to write mesh: {surface_path}")

                shape_el = ET.SubElement(scene, "shape", type="ply", id=f"mesh-{name_prefix}_{mesh_idx}")
                ET.SubElement(shape_el, "string", name="filename", value=f"mesh/{ply_name}")
                ET.SubElement(shape_el, "ref", id=material_type, name="bsdf")
                ET.SubElement(shape_el, "boolean", name="face_normals", value="true")
                mesh_idx += 1
            return mesh_idx

        canonical_gdf = None
        try:
            # One OSM response is the authoritative vector source for both the
            # semantic overlays and the building meshes.  This prevents the 2-D
            # and 3-D products from silently observing different OSM states.
            canonical_wgs84 = ox.features.features_from_bbox(
                bbox=ground_polygon_4326_bbox,
                tags=surface_tags,
            )
            if not canonical_wgs84.empty:
                canonical_wgs84 = canonical_wgs84.to_crs("EPSG:4326")
                canonical_gdf = canonical_wgs84.to_crs(projection_UTM_EPSG_code)
                canonical_gdf = canonical_gdf[canonical_gdf.geometry.intersects(ground_polygon)].copy()
                canonical_gdf["geometry"] = canonical_gdf.geometry.intersection(ground_polygon)
                canonical_gdf = canonical_gdf[~canonical_gdf.geometry.is_empty].copy()

                if not canonical_gdf.empty:
                    source_dir = Path(data_dir) / "source"
                    source_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_wgs84 = canonical_gdf.to_crs("EPSG:4326")
                    for column in snapshot_wgs84.columns:
                        if column == snapshot_wgs84.geometry.name:
                            continue
                        snapshot_wgs84[column] = snapshot_wgs84[column].map(
                            lambda value: json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, tuple, dict, set))
                            else value
                        )
                    snapshot_wgs84.to_file(source_dir / "osm_features.geojson", driver="GeoJSON")
                    (source_dir / "query.json").write_text(
                        json.dumps(
                            {
                                "bbox_wgs84": {
                                    "min_lon": float(ground_polygon_4326_bbox[0]),
                                    "min_lat": float(ground_polygon_4326_bbox[1]),
                                    "max_lon": float(ground_polygon_4326_bbox[2]),
                                    "max_lat": float(ground_polygon_4326_bbox[3]),
                                },
                                "tags": surface_tags,
                                "overpass_url": str(ox.settings.overpass_url),
                                "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    surface_gdf = canonical_gdf
                    road_mask, green_mask, water_mask, obstacle_mask = _surface_mask_from_semantic_gdf(surface_gdf)
                    road_feature_count = int(road_mask.sum())
                    green_feature_count = int(green_mask.sum())
                    water_feature_count = int(water_mask.sum())
                    obstacle_feature_count = int(obstacle_mask.sum())

                    road_geoms = _geoms_from_masked_features(surface_gdf, road_mask, kind="road")
                    green_geoms = _geoms_from_masked_features(surface_gdf, green_mask, kind="green")
                    water_geoms = _geoms_from_masked_features(surface_gdf, water_mask, kind="water")
                    obstacle_geoms = _geoms_from_masked_features(surface_gdf, obstacle_mask, kind="obstacle")

                    building_rows = canonical_gdf.apply(primary_semantic_class, axis=1) == "building"
                    building_union = unary_union(list(canonical_gdf[building_rows].geometry.values))
                    if not building_union.is_empty:
                        def _outside_buildings(geoms):
                            return [
                                clipped
                                for geom in geoms
                                for clipped in [geom.difference(building_union)]
                                if clipped is not None and not clipped.is_empty
                            ]
                        road_geoms = _outside_buildings(road_geoms)
                        green_geoms = _outside_buildings(green_geoms)
                        water_geoms = _outside_buildings(water_geoms)
                        obstacle_geoms = _outside_buildings(obstacle_geoms)

                    road_mesh_count = _add_surface_meshes(
                        _polygon_list_from_geoms(road_geoms), "road", 0.06, road_material_type
                    )
                    green_mesh_count = _add_surface_meshes(
                        _polygon_list_from_geoms(green_geoms), "green", 0.08, green_material_type
                    )
                    water_mesh_count = _add_surface_meshes(
                        _polygon_list_from_geoms(water_geoms), "water", 0.10, water_material_type
                    )
                    obstacle_mesh_count = _add_surface_meshes(
                        _polygon_list_from_geoms(obstacle_geoms), "obstacle", 0.12, obstacle_material_type
                    )
        except Exception as e:
            raise RuntimeError(f"Authoritative OSM scene query failed: {e}") from e

        ET.SubElement(scene, "default", name="scenegen_road_feature_count", value=str(road_feature_count))
        ET.SubElement(scene, "default", name="scenegen_green_feature_count", value=str(green_feature_count))
        ET.SubElement(scene, "default", name="scenegen_water_feature_count", value=str(water_feature_count))
        ET.SubElement(scene, "default", name="scenegen_obstacle_feature_count", value=str(obstacle_feature_count))
        ET.SubElement(scene, "default", name="scenegen_road_mesh_count", value=str(road_mesh_count))
        ET.SubElement(scene, "default", name="scenegen_green_mesh_count", value=str(green_mesh_count))
        ET.SubElement(scene, "default", name="scenegen_water_mesh_count", value=str(water_mesh_count))
        ET.SubElement(scene, "default", name="scenegen_obstacle_mesh_count", value=str(obstacle_mesh_count))
        logger.info(
            f"Semantic surface features => road:{road_feature_count}, green:{green_feature_count}, "
            f"water:{water_feature_count}, obstacle:{obstacle_feature_count}"
        )
        logger.info(
            f"Generated semantic meshes => road:{road_mesh_count}, green:{green_mesh_count}, "
            f"water:{water_mesh_count}, obstacle:{obstacle_mesh_count}"
        )

        # ---------------------------------------------------------------------
        # 5) Query OSM for buildings within the bounding box
        # ---------------------------------------------------------------------

        # ground_polygon_4326_bbox => (west, south, east, north)
        west = ground_polygon_4326_bbox[0]  # minx
        south = ground_polygon_4326_bbox[1]  # miny
        east = ground_polygon_4326_bbox[2]  # maxx
        north = ground_polygon_4326_bbox[3]  # maxy
        # Calculate width/height in UTM
        raw_width = ground_polygon_bbox[2] - ground_polygon_bbox[0]
        raw_height = ground_polygon_bbox[3] - ground_polygon_bbox[1]

        def stable_ceil_dimension(value: float, tolerance: float = 1e-6) -> int:
            nearest = round(value)
            return int(nearest) if abs(value - nearest) <= tolerance else int(math.ceil(value))

        width = stable_ceil_dimension(raw_width)
        height = stable_ceil_dimension(raw_height)
        logger.info(f"Estimated ground polygon size: width={width}m, height={height}m")

        ET.SubElement(scene, "default", name="scenegen_bbox_width", value=str(width))
        ET.SubElement(scene, "default", name="scenegen_bbox_length", value=str(height))

        # if width > 5000 or height > 5000:
        #     logger.warning(f"Too large!")
        #     exit(-1)

        # OSMnx features API uses bounding box in the form (north, south, east, west)
        logger.debug(
            f"OSM bounding box: (north={north}, south={south}, east={east}, west={west})"
        )
        if canonical_gdf is None:
            raise RuntimeError("OSM query produced no canonical feature table")
        building_rows = canonical_gdf.apply(primary_semantic_class, axis=1) == "building"
        filtered_buildings = canonical_gdf[building_rows].copy()
        filtered_buildings = filtered_buildings.explode(index_parts=False).reset_index(drop=True)
        filtered_buildings = filtered_buildings[
            filtered_buildings.geometry.geom_type == "Polygon"
        ].copy()
        buildings_list = filtered_buildings.to_dict("records")

        # ---------------------------------------------------------------------
        # 6) If generating building map, prepare an empty grayscale image
        # ---------------------------------------------------------------------
        # Create a new empty Image, mode 'L' means 8bit grayscale image.
        self._building_map = Image.new("L", (width, height), 0)

        # ---------------------------------------------------------------------
        # 7) Init the building height handler. (osm or lidar)
        # ---------------------------------------------------------------------
        if lidar_calibration:
            try:
                hag_handler = GeoTIFFHandler(hag_tiff_path)
            except Exception as e:
                hag_handler = None
        else:
            hag_handler = None

        # ---------------------------------------------------------------------
        # 8) Process each building to create a 3D mesh (extrude by building height)
        # ---------------------------------------------------------------------

        height_source_counts = {
            "lidar_hag": 0,
            "building:height": 0,
            "height": 0,
            "building:levels": 0,
            "deterministic_type_prior": 0,
        }
        for idx, building in tqdm(
            enumerate(buildings_list),
            total=len(buildings_list),
            desc="Parsing buildings",
        ):
            # Debug the inner hole buildings
            # if building['type'] != "multipolygon":
            #     continue
            # Convert building geometry to a shapely polygon
            building_polygon = shape(building["geometry"])

            if building_polygon.geom_type != "Polygon":
                logger.debug(
                    f"building_polygon.geom_type: {building_polygon.geom_type}"
                )
                continue

            # First try to get building height from LiDAR
            if hag_handler:
                random_points = generate_random_points(building_polygon, 30)
                abs_height = []
                for point in random_points:
                    res = hag_handler.query(to_4326.transform(point.x, point.y), False)
                    abs_height.append(res)


                print("Building height list: ", abs_height)
                print()
                filtered_list = [
                    x for x in abs_height if x.size > 0 and x != -9999 and x > 2
                ]
                print("Building height list: ", abs_height)
                print()
                try:
                    building_height = np.mean(filtered_list)
                    print("Avg Building Height: ", building_height)
                    if math.isnan(building_height):
                        raise ValueError("The value is NaN")
                    height_source = "lidar_hag"
                except Exception as e:
                    print("Random Building Height: ", building_height)
                    building_height, height_source = resolve_building_height(building)
            else:
                building_height, height_source = resolve_building_height(building)

            # Skip buildings with height <= 0
            if building_height <=0:
                continue
            height_source_counts[height_source] += 1
            # building_height = NYC_LiDAR_building_height(building, building_polygon)

            outer_xy = unique_coords(
                reorder_localize_coords(building_polygon.exterior, center_x, center_y)
            )
            
            
            if lidar_terrain:
                mesh = surface_mesh
                # Z bounds of the mesh
                bottom, top = mesh.bounds[-2:]
                # buffer = 1.0
                # res_z = []
                # for points in outer_xy:
                #     # Define two points that form a line that interesects the mesh
                #     x = points[0]
                #     y = points[1]
                #     start = [x, y, bottom - buffer]
                #     stop = [x, y, top + buffer]
                    
                #     # Perform ray trace
                #     points, ind = mesh.ray_trace(start, stop)
                    
                #     # Create geometry to represent ray trace
                #     ray = pv.Line(start, stop)
                #     intersection = pv.PolyData(points)
                #     res_z.append(intersection.bounds[-1])

                # res_z = np.array(res_z)
                # res_z[res_z == -1e+299] = 1e+299
                # #print(res_z)
                # building_z_value = int(np.floor(np.min(res_z)))

                
                # if building_z_value > 1e+20:
                #     building_z_value = 0

                from concurrent.futures import ThreadPoolExecutor

                def ray_trace_z(x, y):
                    start = [x, y, bottom - 1.0]
                    stop = [x, y, top + 1.0]
                    points, ind = mesh.ray_trace(start, stop)
                    if points.shape[0] == 0:
                        return 1e+299
                    return np.max(points[:, 2])

                with ThreadPoolExecutor() as executor:
                    res_z_parallel = list(executor.map(lambda pt: ray_trace_z(pt[0], pt[1]), outer_xy))
                building_z_value = int(np.floor(np.min(res_z_parallel)))
                #building_z_value = int(np.floor(np.min(res_z)))


                if building_z_value > 1e+20:
                    building_z_value = 0
            else:
                building_z_value = 0
                
        
            #print("Building's Z-value: ", building_z_value)
            
            

            holes_xy = []
            if len(list(building_polygon.interiors)) != 0:
                for inner_hole in list(building_polygon.interiors):
                    valid_coords = reorder_localize_coords(
                        inner_hole, center_x, center_y
                    )
                    holes_xy.append(unique_coords(valid_coords))

            def edge_idxs(nv):
                i = np.append(np.arange(nv), 0)
                return np.stack([i[:-1], i[1:]], axis=1)

            nv = 0
            verts, edges = [], []
            for loop in (outer_xy, *holes_xy):
                verts.append(loop)
                edges.append(nv + edge_idxs(len(loop)))
                nv += len(loop)

            verts, edges = np.concatenate(verts), np.concatenate(edges)

            # Triangulate needs to know a single interior point for each hole
            # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
            # e.g. shapely's `polylabel`
            holes = np.array([np.mean(h, axis=0) for h in holes_xy])

            # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
            if len(holes) != 0:
                d = triangulate(
                    dict(vertices=verts, segments=edges, holes=holes), opts="p"
                )
            else:
                d = triangulate(dict(vertices=verts, segments=edges), opts="p")

            # Convert back to pyvista
            v, f = d["vertices"], d["triangles"]
            nv, nf = len(v), len(f)

            # print(v)
            # print(f)

            #points = np.concatenate([v, np.zeros((nv, 1))], axis=1)
            points = np.concatenate([v, np.full((nv, 1), fill_value=building_z_value)], axis=1)
    
            mesh_o3d = o3d.t.geometry.TriangleMesh()
            mesh_o3d.vertex.positions = o3d.core.Tensor(points)
            mesh_o3d.triangle.indices = o3d.core.Tensor(f)

            wedge_t = mesh_o3d.extrude_linear([0, 0, building_height])
            # Get vertices and faces
            vertices_tensor = wedge_t.vertex["positions"]
            faces_tensor = wedge_t.triangle["indices"]

            # Convert to NumPy for calculations
            vertices_np = vertices_tensor.numpy()
            faces_np = faces_tensor.numpy()

            # Compute face centroids
            face_centroids = np.mean(vertices_np[faces_np], axis=1)

            z_values = vertices_np[:, 2]
            top_vertex_indices = np.where(z_values == building_height + building_z_value)[
                0
            ].tolist()  # Indices of top vertices
            #print("top vertex indices: ", top_vertex_indices)
            


            # Extract the top surface
            top_surface = wedge_t.select_by_index(top_vertex_indices)

            other_faces_np = faces_np[face_centroids[:, 2] < building_height+building_z_value]
            if len(other_faces_np) == 0:
                print("All vertices: ", vertices_np)
                print("top vertex indices: ", top_vertex_indices)
                print("max height of meshes: ", np.max(z_values))
                print("min height of meshes: ", np.min(z_values))
                print("building height: ", building_height)
                print("building z value: ", building_z_value)
                print("building height + building z value: ", building_height + building_z_value)
                
            
                print("other faces np: ", other_faces_np)
                print("max height of meshes: ", np.max(z_values))
                print("building height: ", building_height)
                print("building z value: ", building_z_value)
                print("building height + building z value: ", building_height + building_z_value)
            # Convert to Open3D Tensor API
            other_faces_o3c = o3c.Tensor(other_faces_np, dtype=o3c.int32)

            wall_mesh = o3d.t.geometry.TriangleMesh()
            wall_mesh.vertex["positions"] = vertices_tensor  # Same vertices
            wall_mesh.triangle["indices"] = other_faces_o3c

            # open3d<=0.18 does not provide this API on tensor mesh.
            if hasattr(wall_mesh, "remove_unreferenced_vertices"):
                wall_mesh.remove_unreferenced_vertices()
            else:
                wall_mesh_legacy = wall_mesh.to_legacy()
                wall_mesh_legacy.remove_unreferenced_vertices()
                wall_mesh = o3d.t.geometry.TriangleMesh.from_legacy(wall_mesh_legacy)

            rooftop_path = os.path.join(mesh_data_dir, f"building_{idx}_rooftop.ply")
            if not o3d.t.io.write_triangle_mesh(
                rooftop_path, top_surface, write_ascii=write_ply_ascii
            ):
                raise OSError(f"Open3D failed to write mesh: {rooftop_path}")
            wall_path = os.path.join(mesh_data_dir, f"building_{idx}_wall.ply")
            if not o3d.t.io.write_triangle_mesh(
                wall_path, wall_mesh, write_ascii=write_ply_ascii
            ):
                raise OSError(f"Open3D failed to write mesh: {wall_path}")

            # o3d.t.io.write_triangle_mesh(os.path.join(mesh_data_dir, f"building_{idx}.ply"), wedge, write_ascii=write_ply_ascii)

            # Add shape elements for PLY files in the folder
            sionna_shape = ET.SubElement(
                scene, "shape", type="ply", id=f"mesh-building_{idx}_rooftop"
            )
            ET.SubElement(
                sionna_shape,
                "string",
                name="filename",
                value=f"mesh/building_{idx}_rooftop.ply",
            )
            ET.SubElement(sionna_shape, "ref", id=rooftop_material_type, name="bsdf")
            ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            sionna_shape = ET.SubElement(
                scene, "shape", type="ply", id=f"mesh-building_{idx}_wall"
            )
            ET.SubElement(
                sionna_shape,
                "string",
                name="filename",
                value=f"mesh/building_{idx}_wall.ply",
            )
            ET.SubElement(sionna_shape, "ref", id=wall_material_type, name="bsdf")
            ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            if generate_building_map:
                self._draw_building(building_polygon, building_height)

        for source_name, source_count in height_source_counts.items():
            ET.SubElement(
                scene,
                "default",
                name=f"scenegen_height_source_{source_name.replace(':', '_')}",
                value=str(source_count),
            )
        del hag_handler
        xml_string = ET.tostring(scene, encoding="utf-8")
        xml_pretty = minidom.parseString(xml_string).toprettyxml(
            indent="    "
        )  # Adjust the infdent as needed

        with open(
            os.path.join(data_dir, "scene.xml"), "w", encoding="utf-8"
        ) as xml_file:
            xml_file.write(xml_pretty)

        if generate_building_map:
            np.save(
                os.path.join(data_dir, "2D_Building_Height_Map.npy"),
                np.array(self._building_map),
            )

        return np.array(self._building_map)

    def _draw_building(self, building_polygon, building_height):
        local_exterior = reorder_localize_coords(
            building_polygon.exterior,
            self._ground_polygon_envelope_UTM.bounds[0],
            self._ground_polygon_envelope_UTM.bounds[3],
        )
        ImageDraw.Draw(self._building_map).polygon(
            [(x, -y) for x, y in list(local_exterior)],
            outline=int(building_height),
            fill=int(building_height),
        )
        # local_coor_building_polygon = affinity.translate(building_polygon, xoff=-1 * tmpres[0], yoff=-1 * tmpres[1])
        # # print("local_coor_building_polygon:",local_coor_building_polygon,'\n\n\n\n')

        # local_coor_building_polygon = round_polygon_coordinates(local_coor_building_polygon)

        # ImageDraw.Draw(img).polygon([(x, -y) for x, y in list(local_coor_building_polygon.exterior.coords)],
        #                             outline=int(building_height), fill=int(building_height))

        # # Handle the "holes" inside the polygon
        # if len(list(local_coor_building_polygon.interiors)) != 0:
        #     for inner_hole in list(local_coor_building_polygon.interiors):
        #         ImageDraw.Draw(img).polygon([(x, -y) for x, y in list(inner_hole.coords)], outline=int(0),
        #                                     fill=int(0))
        # Create and write the XML file

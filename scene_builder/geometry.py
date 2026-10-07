import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


def _parse_ply_header(fp):
    header_lines = []
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        s = line.decode("ascii", errors="strict").strip()
        header_lines.append(s)
        if s == "end_header":
            break

    fmt = None
    vertex_count = None
    vertex_props = []
    in_vertex_element = False

    for s in header_lines:
        parts = s.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex_element = parts[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex_element:
            if parts[1] == "list":
                raise ValueError("List properties in vertex are not supported")
            vertex_props.append((parts[1], parts[2]))

    if fmt is None or vertex_count is None:
        raise ValueError("Invalid PLY header: missing format or vertex count")
    return fmt, vertex_count, vertex_props


def read_ply_vertices(ply_path: Path) -> np.ndarray:
    type_map = {
        "char": ("b", np.int8),
        "uchar": ("B", np.uint8),
        "short": ("h", np.int16),
        "ushort": ("H", np.uint16),
        "int": ("i", np.int32),
        "uint": ("I", np.uint32),
        "float": ("f", np.float32),
        "double": ("d", np.float64),
    }

    with ply_path.open("rb") as fp:
        fmt, vertex_count, vertex_props = _parse_ply_header(fp)

        prop_names = [n for _, n in vertex_props]
        for key in ("x", "y", "z"):
            if key not in prop_names:
                raise ValueError(f"PLY missing '{key}' property: {ply_path}")

        if fmt == "ascii":
            data = np.empty((vertex_count, 3), dtype=np.float64)
            x_idx, y_idx, z_idx = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
            for i in range(vertex_count):
                vals = fp.readline().decode("ascii", errors="strict").strip().split()
                data[i, 0] = float(vals[x_idx])
                data[i, 1] = float(vals[y_idx])
                data[i, 2] = float(vals[z_idx])
            return data

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt} in {ply_path}")

        dtype_fields = []
        for t, n in vertex_props:
            if t not in type_map:
                raise ValueError(f"Unsupported PLY property type '{t}' in {ply_path}")
            _, np_type = type_map[t]
            dtype_fields.append((n, np_type))

        arr = np.fromfile(fp, dtype=np.dtype(dtype_fields), count=vertex_count)
        xyz = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float64, copy=False)
        return xyz


def get_building_mesh_paths(scene_path: Path):
    tree = ET.parse(scene_path)
    root = tree.getroot()
    scene_dir = scene_path.parent

    mesh_paths = []
    for shape in root.findall("shape"):
        if shape.get("type") != "ply":
            continue
        string_node = shape.find("string")
        if string_node is None or string_node.get("name") != "filename":
            continue
        rel = string_node.get("value", "")
        rel_l = rel.lower()
        if not rel_l.startswith("mesh/building_"):
            continue
        if not (rel_l.endswith("_wall.ply") or rel_l.endswith("_rooftop.ply")):
            continue
        mesh_paths.append(scene_dir / rel)
    return mesh_paths


def get_scene_map_size(scene_path: Path):
    """Read scene bbox width/length from scene.xml defaults."""
    tree = ET.parse(scene_path)
    root = tree.getroot()

    width = None
    length = None
    for default in root.findall("default"):
        name = default.get("name", "")
        value = default.get("value", "")
        if name == "scenegen_bbox_width":
            width = float(value)
        elif name == "scenegen_bbox_length":
            length = float(value)

    return width, length


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotonic chain convex hull. Returns hull vertices in CCW order."""
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def get_rooftop_mesh_paths(scene_path: Path):
    return [p for p in get_building_mesh_paths(scene_path) if p.name.endswith("_rooftop.ply")]


def build_rooftop_polygons(scene_path: Path):
    rooftop_paths = get_rooftop_mesh_paths(scene_path)
    polygons = []
    for mesh_path in rooftop_paths:
        if not mesh_path.exists():
            continue
        vertices = read_ply_vertices(mesh_path)
        hull = convex_hull_2d(vertices[:, :2])
        if len(hull) >= 3:
            polygons.append(hull)
    return polygons, len(rooftop_paths)


def get_building_vertex_stats(scene_path: Path):
    mesh_paths = get_building_mesh_paths(scene_path)
    if not mesh_paths:
        raise RuntimeError("No building mesh files found in scene.xml")

    vertices_all = []
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            continue
        vertices_all.append(read_ply_vertices(mesh_path))

    if not vertices_all:
        raise RuntimeError("Building mesh files listed in scene.xml were not readable")

    vertices_all = np.vstack(vertices_all)
    return len(mesh_paths), vertices_all.shape[0]


def build_deployable_tx_points(polygons, map_x, map_y, cell_size):
    """Build full-grid TX candidate points and return deployable subset."""
    x_step, y_step = float(cell_size[0]), float(cell_size[1])
    if x_step <= 0 or y_step <= 0:
        raise ValueError("cell-size must be > 0 in both x and y")

    x_half = map_x / 2.0
    y_half = map_y / 2.0
    xs = np.arange(-x_half, x_half + 1e-9, x_step, dtype=np.float64)
    ys = np.arange(-y_half, y_half + 1e-9, y_step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    all_points = np.column_stack((gx.ravel(), gy.ravel()))

    blocked = np.zeros(all_points.shape[0], dtype=bool)
    for poly in polygons:
        blocked |= MplPath(poly).contains_points(all_points, radius=1e-9)

    deployable_points = all_points[~blocked]
    return all_points, deployable_points, blocked


def count_deployable_tx_points(polygons, map_x, map_y, cell_size):
    """Count total/deployable TX candidate points on a regular ground grid."""
    all_points, deployable_points, blocked = build_deployable_tx_points(
        polygons, map_x, map_y, cell_size
    )
    total_points = int(all_points.shape[0])
    blocked_points = int(np.count_nonzero(blocked))
    deployable_points = total_points - blocked_points

    return total_points, deployable_points, blocked_points


def is_point_deployable(polygons, xy):
    x, y = float(xy[0]), float(xy[1])
    for poly in polygons:
        if MplPath(poly).contains_point((x, y), radius=1e-9):
            return False
    return True


def parse_tx_points_from_args(args):
    points = []
    if args.tx_points is not None:
        if len(args.tx_points) % 2 != 0:
            raise ValueError("--tx-points must contain even number of values: x1 y1 x2 y2 ...")
        for i in range(0, len(args.tx_points), 2):
            points.append([float(args.tx_points[i]), float(args.tx_points[i + 1])])

    if args.tx_json is not None:
        data = json.loads(Path(args.tx_json).read_text(encoding="utf-8"))
        matched = None
        for item in data:
            if int(item.get("case_id", -1)) == int(args.case_id):
                matched = item
                break
        if matched is None:
            raise ValueError(f"case_id={args.case_id} not found in {args.tx_json}")
        points = [[float(p[0]), float(p[1])] for p in matched.get("tx_positions", [])]

    return points


def main():
    from citydeploy.paths import workspace_root
    project_root = workspace_root()

    parser = argparse.ArgumentParser(
        description="Load a scene and visualize building-only 2D layout."
    )
    parser.add_argument("--scene", default="sheffield_01", help="Scene package name under scene root")
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help="Root directory containing scene packages (default: PROJECT_ROOT/datasets/scenes)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("X", "Y"),
        help="Cell size in meters [x y] for TX deployable-point counting (default: 1.0 1.0)",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Output image size in pixels")
    parser.add_argument(
        "--map-size",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Scene extent in meters [x y]. Default: read from scene.xml (scenegen_bbox_width/length).",
    )
    parser.add_argument(
        "--tx-points",
        type=float,
        nargs="+",
        default=None,
        metavar=("X", "Y"),
        help="TX points to overlay: x1 y1 x2 y2 ...",
    )
    parser.add_argument(
        "--tx-json",
        type=str,
        default=None,
        help="Path to random_two_tx results JSON. If set, reads TX points from --case-id.",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=1,
        help="Case id to load from --tx-json (default: 1)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional output image path. If set, save figure instead of only showing.",
    )
    args = parser.parse_args()

    scene_root = args.scene_root or (project_root / "datasets" / "scenes")
    scene_path = scene_root / args.scene / "scene.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    print(f"Loading scene: {scene_path}")
    shape_count, vertex_count = get_building_vertex_stats(scene_path)
    print(f"Building meshes used: {shape_count}")
    print(f"Total building vertices: {vertex_count}")

    # Draw rooftop polygons for a clear building layout view.
    polygons, rooftop_count = build_rooftop_polygons(scene_path)
    print(f"Rooftop polygons drawn: {len(polygons)} / {rooftop_count}")

    fig = plt.figure(figsize=(args.img_size / 100.0, args.img_size / 100.0), dpi=100)
    ax = plt.gca()
    for poly in polygons:
        ax.fill(poly[:, 0], poly[:, 1], color="black", alpha=1.0, linewidth=0)

    if args.map_size is not None:
        map_x, map_y = args.map_size
    else:
        map_x, map_y = get_scene_map_size(scene_path)
        if map_x is None or map_y is None:
            map_x, map_y = 512.0, 512.0

    x_half = map_x / 2.0
    y_half = map_y / 2.0
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-y_half, y_half)
    print(f"Display extent: x=[{-x_half:.1f}, {x_half:.1f}], y=[{-y_half:.1f}, {y_half:.1f}]")

    total_points, deployable_points, blocked_points = count_deployable_tx_points(
        polygons, map_x, map_y, args.cell_size
    )
    deploy_ratio = 100.0 * deployable_points / total_points
    print(
        f"TX candidate points (cell-size={args.cell_size[0]:g}m x {args.cell_size[1]:g}m): total={total_points}, "
        f"deployable={deployable_points}, blocked_by_building={blocked_points}, "
        f"deployable_ratio={deploy_ratio:.2f}%"
    )

    tx_points = parse_tx_points_from_args(args)
    if tx_points:
        print("Overlay TX points:")
        for i, p in enumerate(tx_points):
            ok = is_point_deployable(polygons, p)
            status = "DEPLOYABLE" if ok else "BLOCKED_BY_BUILDING"
            print(f"  TX{i}: ({p[0]:.3f}, {p[1]:.3f}) -> {status}")
            if i == 0:
                ax.scatter(p[0], p[1], marker="*", s=280, c="red", edgecolors="black", linewidths=0.8, zorder=10, label="TX0")
            else:
                ax.scatter(p[0], p[1], marker="x", s=170, c="#E69F00", linewidths=2.6, zorder=10, label=f"TX{i}")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Building Layout Only ({args.scene}) - {args.img_size}x{args.img_size}, map={int(map_x)}x{int(map_y)}m"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_facecolor("white")
    fig.set_size_inches(args.img_size / 100.0, args.img_size / 100.0, forward=True)
    plt.tight_layout()
    if args.save_path:
        out = Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()

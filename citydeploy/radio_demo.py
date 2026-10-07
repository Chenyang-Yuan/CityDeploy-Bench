"""Trace two isotropic TXs above a procedural ground plane and save five radio maps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from citydeploy.paths import config_path
from citydeploy.demo_data import write_json


GROUND = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
-16 -16 0
16 -16 0
16 16 0
-16 16 0
3 0 1 2
3 0 2 3
"""
SCENE = """<scene version="3.0.0">
<bsdf type="itu-radio-material" id="mat-ground"><string name="type" value="concrete"/><float name="thickness" value="0.1"/></bsdf>
<shape type="ply" id="mesh-ground"><string name="filename" value="ground.ply"/><ref id="mat-ground" name="bsdf"/></shape>
</scene>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/radio_demo"))
    parser.add_argument("--ray-device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--samples-per-tx", type=int, default=10000)
    args = parser.parse_args()
    if args.samples_per_tx < 1:
        parser.error("samples-per-tx must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    scene = args.output / "scene"
    scene.mkdir(parents=True)
    (scene / "ground.ply").write_text(GROUND, encoding="ascii")
    (scene / "scene.xml").write_text(SCENE, encoding="ascii")
    # The procedural scene also exposes the standard model/evaluation inputs.
    inputs = scene / "inputs"
    (inputs / "tx_deployable").mkdir(parents=True)
    road = np.ones((16, 16), dtype=np.uint8)
    zeros = np.zeros_like(road)
    np.savez_compressed(inputs / "scene_inputs.npz", building_occupancy=zeros,
                        deployable_mask=road, tx_feasible_mask=road, evaluation_mask=road)
    np.savez_compressed(inputs / "semantic_masks.npz", building_mask=zeros, road_mask=road)
    yy, xx = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    np.save(inputs / "tx_deployable" / "points_local_xy.npy", np.column_stack((xx.ravel(), yy.ravel())))
    write_json(inputs / "metadata.json", {"map_x_m": 16., "map_y_m": 16.,
                                          "scene_id": "scene", "provenance": "procedural_ground_plane"})
    rf = json.loads(config_path("rf/urban_3p65ghz.json").read_text(encoding="utf-8"))
    rf["ray_tracing"]["samples_per_tx"] = args.samples_per_tx
    write_json(args.output / "rf_profile.json", rf)
    from dataset_builder.raytracing import SceneRayTracer
    from dataset_builder.generate import _metric_kwargs
    from dataset_builder.metrics import compute_urban_radio_metrics, summarize_urban_metrics, tx_axis_last
    from visualization.radio_maps import save_radio_map_outputs
    tracer = SceneRayTracer(scene, rf, args.ray_device)
    tx = np.asarray([[-4., 0., 10.], [4., 0., 10.]])
    raw, seconds = tracer.trace(tx, (16., 16.), 42)
    gain = tx_axis_last(raw, len(tx))
    if not np.isfinite(gain).all() or not (gain > 0).any():
        raise RuntimeError("Radio demo produced no finite nonzero path gains.")
    metrics = compute_urban_radio_metrics(gain, len(tx), **_metric_kwargs(rf))
    road = np.ones(gain.shape[:2], dtype=bool)
    summary, masks = summarize_urban_metrics(metrics, road, rf["coverage_thresholds"])
    save_radio_map_outputs(args.output, metrics, masks, road, np.zeros_like(road), tx, 16., 16., dpi=120)
    result = {"scene": "procedural_ground_plane", "physical_verification": True,
              "note": "Backend smoke test, not an urban benchmark result.", "seconds": seconds,
              "rf_profile": rf, "runtime": tracer.runtime_info, "coverage": summary}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

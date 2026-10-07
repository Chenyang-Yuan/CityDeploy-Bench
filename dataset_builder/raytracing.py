"""One-scene ray-tracing runner used by the V4 family generator."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _remove_transmitters(scene) -> None:
    for name in list(scene.transmitters):
        try:
            scene.remove(name)
        except Exception:
            try:
                scene.remove(scene.transmitters[name])
            except Exception:
                pass


class SceneRayTracer:
    """Cache one Sionna scene and return independent per-TX path-gain maps."""

    def __init__(self, scene_dir: Path, rf_profile: dict, device: str) -> None:
        from dataset_builder.runtime import require_supported_runtime

        self.runtime_info = require_supported_runtime(requested_device=device)
        from radio_backend.sionna_adapter import load_and_preview_scene

        self.scene_dir = Path(scene_dir)
        self.rf = rf_profile
        self.device = device
        started = time.perf_counter()
        self.scene = load_and_preview_scene(str(self.scene_dir), preview=False)
        self.scene_load_seconds = time.perf_counter() - started
        self.applied_scattering = self._apply_scattering()

    def _apply_scattering(self) -> dict[str, float]:
        applied: dict[str, float] = {}
        coefficients = self.rf.get("diffuse_scattering_coefficients", {})
        for name, value in coefficients.items():
            token = str(name).lower().replace("_", "")
            for material_name, material in self.scene.radio_materials.items():
                if token not in str(material_name).lower().replace("_", ""):
                    continue
                try:
                    material.scattering_coefficient = float(value)
                    applied[str(material_name)] = float(value)
                except Exception:
                    pass
        return applied

    def trace(
        self,
        tx_xyz_m: np.ndarray,
        map_size_m: tuple[float, float],
        simulation_seed: int,
    ) -> tuple[np.ndarray, float]:
        from radio_backend.sionna_adapter import (
            configure_transmitters_receivers,
            generate_coverage_map,
        )

        _remove_transmitters(self.scene)
        tx_array = self.rf["tx_array"]
        rx_array = self.rf["rx_array"]
        configure_transmitters_receivers(
            self.scene,
            tx_positions=np.asarray(tx_xyz_m, dtype=float).tolist(),
            tx_azimuths=[0.0] * len(tx_xyz_m),
            frequency=float(self.rf["frequency_hz"]),
            tx_power_dbm=float(self.rf["tx_power_dbm"]),
            tx_pattern=str(tx_array["pattern"]),
            tx_polarization=str(tx_array["polarization"]),
            rx_pattern=str(rx_array["pattern"]),
            rx_polarization=str(rx_array["polarization"]),
        )
        ray = self.rf["ray_tracing"]
        started = time.perf_counter()
        radio_map = generate_coverage_map(
            self.scene,
            max_depth=int(ray["max_depth"]),
            los=bool(ray["line_of_sight"]),
            cell_size=tuple(float(value) for value in ray["cell_size_m"]),
            size=[float(map_size_m[0]), float(map_size_m[1])],
            samples_per_tx=int(ray["samples_per_tx"]),
            use_planar=bool(ray["use_planar"]),
            device=self.device,
            tx_position=np.asarray(tx_xyz_m[0], dtype=float).tolist(),
            specular_reflection=bool(ray["specular_reflection"]),
            diffuse_reflection=bool(ray["diffuse_reflection"]),
            refraction=bool(ray["refraction"]),
            diffraction=bool(ray["diffraction"]),
            edge_diffraction=bool(ray["edge_diffraction"]),
            seed=int(simulation_seed),
        )
        # Materializing the host array waits for asynchronous GPU work as well.
        path_gain = np.asarray(radio_map.path_gain)
        elapsed = time.perf_counter() - started
        return path_gain, elapsed

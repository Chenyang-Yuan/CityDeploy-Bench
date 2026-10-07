# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from Geo2SigMap; see THIRD_PARTY_NOTICES.md for attribution and changes.
"""Sionna scene loading, antenna configuration, and radio-map evaluation."""
from __future__ import annotations
import os
import time
import numpy as np
from dataset_builder.optix_compat import prepare_optix_compatibility
prepare_optix_compatibility()
import mitsuba as mi
from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver, transform_mesh

def load_and_preview_scene(scene_folder, preview=False):
    """
    Load a self-contained scene package and optionally preview its geometry.
    
    Args:
        scene_folder: Path to the scene folder containing scene.xml
        preview: If True, show interactive 3D visualization
        
    Returns:
        Loaded scene object
    """
    # Normalize path to handle both Windows and Unix-style paths
    scene_folder = os.path.normpath(os.path.expanduser(scene_folder))
    scene_path = os.path.join(scene_folder, "scene.xml")
    scene_path = os.path.normpath(scene_path)
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}\n"
                              f"Please check that the scene folder exists and contains scene.xml")
    
    print(f"Loading scene from: {scene_path}")
    scene = load_scene(scene_path)
    
    if preview:
        print("Displaying interactive 3D visualization...")
        print("Controls: Mouse left=Rotate, Scroll wheel=Zoom, Mouse right=Move")
        scene.preview()
    
    return scene


def configure_transmitters_receivers(
    scene,
    tx_positions=None,
    tx_azimuths=None,
    frequency=3.65e9,
    tx_power_dbm=23.0,
    tx_pattern="iso",
    tx_polarization="V",
    rx_pattern="dipole",
    rx_polarization="cross",
):
    """
    Module 2: Configure the transmitters and receivers.
    
    We specify:
    - The transmit (TX) antenna array is a planar array with a single element and isotropic pattern;
    - The receive (RX) antenna array is a planar array with a single element but with a dipole pattern.
    
    Then, we create Transmitter objects with specific positions and orientations, 
    and add them to the scene.
    
    Note: Scene geographical coordinate system
    - Coordinate system: The center of the scene is at (0,0,0), the z-axis for the flat ground polygon is 0.
    - Unit: Meters (m).
    
    Note: Sionna Azimuth/Elevation Format
    - Sionna defines the antenna azimuth in the range [-π, π) [rad]. 
      The following code converts the commonly used north-origin, clockwise-degree 
      azimuth to the format suitable for Sionna.
    
    Args:
        scene: Sionna scene object
        tx_positions: List of transmitter positions [[x, y, z], ...] in meters. 
                     If single position provided, will be converted to list.
                     Default: [[0, 0, 100]]
        tx_azimuths: List of transmitter azimuth angles in degrees. 
                    If single value provided, will be used for all TXs.
                    Default: [210] (or 210 for all if single value)
        frequency: Operating frequency in Hz. Default: 3.65e9 (CBRS band)
    
    Returns:
        List of TX names added to the scene
    """
    # Handle backward compatibility: single position/azimuth
    if tx_positions is None:
        tx_positions = [[0, 0, 100]]
    elif not isinstance(tx_positions[0], (list, tuple, np.ndarray)):
        # Single position provided, convert to list
        tx_positions = [tx_positions]
    
    if tx_azimuths is None:
        tx_azimuths = [210]
    elif not isinstance(tx_azimuths, (list, tuple, np.ndarray)):
        # Single azimuth provided, use for all TXs
        tx_azimuths = [tx_azimuths] * len(tx_positions)
    
    # Ensure we have enough azimuths (repeat last one if needed)
    while len(tx_azimuths) < len(tx_positions):
        tx_azimuths.append(tx_azimuths[-1] if tx_azimuths else 210)
    
    # Transmit array (single iso element)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=tx_pattern,
        polarization=tx_polarization,
    )

    # Receive array (single dipole element)
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=rx_pattern,
        polarization=rx_polarization,
    )

    # Create multiple transmitters
    tx_names = []
    for i, (tx_pos, tx_az) in enumerate(zip(tx_positions, tx_azimuths)):
        tx_name = f"tx{i}"
        tx = Transmitter(
            name=tx_name,
            position=list(tx_pos),
            orientation=[-1 * (tx_az - 90) / 180 * np.pi, 0, 0],
            power_dbm=float(tx_power_dbm),
        )
        scene.add(tx)
        tx_names.append(tx_name)
        print(f"Transmitter {i} configured: position={tx_pos}, azimuth={tx_az}°")

    # Set the operating frequency in Hz
    scene.frequency = frequency
    
    print(f"Total {len(tx_names)} transmitter(s) configured, frequency={frequency/1e9:.2f} GHz")
    return tx_names


def generate_coverage_map(scene, max_depth=5, los=True, cell_size=(2., 2.),
                          size=[512, 512], samples_per_tx=int(1e8), use_planar=False,
                          device=None, tx_position=None, *,
                          specular_reflection=True,
                          diffuse_reflection=False,
                          refraction=False,
                          diffraction=False,
                          edge_diffraction=False,
                          seed=42):
    """
    Module 3: Generate the coverage map (or path gain map).
    
    The RadioMap method creates a grid of RX locations and performs ray tracing 
    to calculate path gain between the TX and each grid point.
    
    Parameters:
    - max_depth: Maximum number of reflections/refractions (default: 5)
    - los: Enable LOS paths (default: True)
    - cell_size: Grid resolution in meters, e.g., (2., 2.) means 2×2 meter per pixel
    - size: Radio map size in meters, e.g., [512, 512] covers 512m × 512m
    - samples_per_tx: Number of samples per source (default: 1e8)
    - use_planar: If True, generate PlanarRadioMap (2D grid). If False, generate MeshRadioMap (requires PLY file)
    - tx_position: Transmitter position [x, y, z] in meters. If None, uses [0, 0, 1.5]
    
    Args:
        scene: Sionna scene object
        max_depth: Maximum ray depth
        los: Enable line-of-sight paths
        cell_size: Cell size in (x, y) meters
        size: Radio map size in [x, y] meters
        samples_per_tx: Number of samples per transmitter
        use_planar: Whether to use PlanarRadioMap (True) or MeshRadioMap (False)
        tx_position: Transmitter position [x, y, z]. The measurement plane remains centered on the scene.
        
    Returns:
        RadioMap object
    """
    print("\nGenerating coverage map...")
    print(f"  Max depth: {max_depth}")
    print(f"  LOS enabled: {los}")
    print(f"  Specular reflection: {specular_reflection}")
    print(f"  Diffuse reflection: {diffuse_reflection}")
    print(f"  Refraction: {refraction}")
    print(f"  Diffraction: {diffraction}")
    print(f"  Edge diffraction: {edge_diffraction}")
    print(f"  Cell size: {cell_size} m")
    print(f"  Map size: {size} m")
    print(f"  Samples per TX: {samples_per_tx}")
    print(f"  RadioMap type: {'PlanarRadioMap' if use_planar else 'MeshRadioMap'}")
    
    variant = mi.variant() or "uninitialized"
    requested = (device or "auto").lower()
    if requested == "gpu" and "cuda" not in variant:
        raise RuntimeError(f"GPU requested but Sionna RT is using Mitsuba variant {variant!r}")
    if requested == "cpu" and "llvm" not in variant:
        raise RuntimeError(f"CPU requested but Sionna RT is using Mitsuba variant {variant!r}")
    print(f"  Sionna RT backend: {variant}")
    
    rm_solver = RadioMapSolver()

    start_time = time.time()
    
    # RadioMap always centers at scene center (0, 0, 1.5) to cover the entire scene
    # TX position can vary within the scene, and RadioMap will show coverage from that TX position
    map_center = [0, 0, 1.5]  # Scene center (fixed)
    print(f"  RadioMap center: {map_center} (scene center, fixed)")
    if tx_position is not None:
        print(f"  TX position: {tx_position} (can vary within scene)")
    
    if use_planar:
        # Generate PlanarRadioMap - 2D grid, can be directly visualized
        rm = rm_solver(
            scene,
            max_depth=max_depth,
            los=los,
            refraction=refraction,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            cell_size=cell_size,
            size=size,
            center=map_center,  # Fixed scene-centered measurement plane
            orientation=[0, 0, 0],  # Orientation of the radio map plane
            precoding_vec=None,
            samples_per_tx=samples_per_tx,
            seed=int(seed),
        )
    else:
        # Generate MeshRadioMap - uses terrain mesh, requires PLY file for visualization
        measurement_surface = scene.objects["ground"].clone(as_mesh=True)
        # Measurement surface always centered at scene center (0, 0, 1.5)
        transform_mesh(measurement_surface, translation=[0, 0, 1.5])
        
        rm = rm_solver(
            scene,
            max_depth=max_depth,
            los=los,
            refraction=refraction,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            cell_size=cell_size,
            size=size,
            measurement_surface=measurement_surface,
            precoding_vec=None,
            samples_per_tx=samples_per_tx,
            seed=int(seed),
        )
    
    elapsed_time = time.time() - start_time
    print(f"Coverage map generated in {elapsed_time:.2f} seconds")
    
    return rm


# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from Geo2SigMap; see THIRD_PARTY_NOTICES.md for attribution and changes.
"""
ITU Material Properties Data

This module provides predefined material properties based on
ITU-R Recommendation P.2040-2: "[Effects of building materials and structures
on radiowave propagation above about 100 MHz](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2040-3-202308-I!!PDF-E.pdf)".

TODO: Verify the exact material names using by Mitsuba and Sionna. Especially how they named the low/high frequency glass and ceiling board materials.
TODO: Check what is the exact colors using by Mitsuba/Sionna/Blender/Blosm for the materials.
"""

ITU_MATERIALS = {
    "mat-vacuum": {
        "name": "Vacuum (\u2248Air)",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.1216, 0.4667, 0.7059),
    },
    "mat-itu_concrete": {
        "name": "Concrete",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.539479, 0.539479, 0.539480),
    },
    "mat-itu_brick": {
        "name": "Brick",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(1.0000, 0.4980, 0.0549),
    },
    "mat-itu_plasterboard": {
        "name": "Plasterboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(1.0000, 0.7333, 0.4706),
    },
    "mat-itu_wood": {
        "name": "Wood",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.043, 0.58, 0.184)
    },
    "mat-itu_glass": {
        "name": "Glass",
        "lower_freq_limit": [0.1e9,220e9],
        "upper_freq_limit": [100e9,450e9],
        "mitsuba_color":(0.5961, 0.8745, 0.5412),
    },
    "mat-itu_ceiling_board": {
        "name": "Ceiling Board",
        "lower_freq_limit": [1e9, 220e9],
        "upper_freq_limit": [100e9, 450e9],
        "mitsuba_color":(1.0000, 0.5961, 0.5882),
    },
    "mat-itu_chipboard": {
        "name": "Chipboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.7725, 0.6902, 0.8353),
    },
    "mat-itu_plywood": {
        "name": "Plywood",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(0.5490, 0.3373, 0.2941),
    },
    "mat-itu_marble": {
        "name": "Marble",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 60e9,
        "mitsuba_color": (0.701101, 0.644479, 0.485150),
    },
    "mat-itu_floorboard": {
        "name": "Floorboard",
        "lower_freq_limit": 50e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.8902, 0.4667, 0.7608),
    },
    "mat-itu_metal": {
        "name": "Metal",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.219526, 0.219526, 0.254152)
    },
    "mat-itu_very_dry_ground": {
        "name": "Very Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-itu_medium_dry_ground": {
        "name": "Medium Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-itu_wet_ground": {
        "name": "Wet Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
    "mat-itu_fresh_water": {
        "name": "Fresh Water",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.0, 0.46, 0.7),
    },
    "mat-itu_sea_water": {
        "name": "Sea Water",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.05, 0.2, 0.5),
    },
    "mat-very_dry_ground_P.527": {
        "name": "Very Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-medium_dry_ground_P.527": {
        "name": "Medium Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-wet_ground_P.527": {
        "name": "Wet Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
    "mat-fresh_water_P.527": {
        "name": "Fresh Water (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.0, 0.46, 0.7),
    },
    "mat-sea_water_P.527": {
        "name": "Sea Water (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.05, 0.2, 0.5),
    },
}
import pandas as pd

try:
    from importlib.resources import files
except ImportError:  # Python < 3.9 compatibility
    from importlib_resources import files
from scipy.interpolate import make_smoothing_spline


def _read_material_curve_csv(filename: str) -> pd.DataFrame:
    """
    Read material curve CSV from packaged data if available.

    This module should stay importable even when `scene_builder.data` is
    not installed in the current environment (common for script-mode runs).
    We only require the CSV files when a P.527 callback is actually invoked.
    """
    try:
        with files("scene_builder.data").joinpath(filename).open("r") as f:
            df = pd.read_csv(f, header=None, names=["x", "y"])
    except Exception as e:
        raise RuntimeError(
            f"Missing material curve data file '{filename}' under package "
            "'scene_builder.data'. Install/copy the data package to use "
            "P.527 callback materials."
        ) from e
    return df.sort_values(by="x")

def get_material(material_name):
    from sionna.rt import RadioMaterial
    material_dict = {
        "mat-wet_ground_P.527":RadioMaterial("wet_ground_P.527_impl",
                                                 frequency_update_callback=wet_ground_ITU_P527_callback),
        "mat-medium_dry_ground_P.527":RadioMaterial("medium_dry_ground_P.527_impl",
                                                        frequency_update_callback=medium_dry_ground_ITU_P527_callback),
        "mat-very_dry_ground_P.527":RadioMaterial("very_dry_ground_P.527_impl",
                                                        frequency_update_callback=very_dry_ground_ITU_P527_callback),
        # NOTE:
        # P.527 water curves are not yet included in scene_builder/data.
        # Approximate these materials using the P.527 wet-ground model.
        # Replace callback with dedicated fresh/sea-water curves when available.
        "mat-fresh_water_P.527":RadioMaterial("fresh_water_P.527_impl",
                                                frequency_update_callback=fresh_water_ITU_P527_callback),
        "mat-sea_water_P.527":RadioMaterial("sea_water_P.527_impl",
                                              frequency_update_callback=sea_water_ITU_P527_callback),
    }
    if material_name not in material_dict.keys():
        return None
    else:
        return material_dict[material_name]

def fresh_water_ITU_P527_callback(f_hz):
    return wet_ground_ITU_P527_callback(f_hz)

def sea_water_ITU_P527_callback(f_hz):
    return wet_ground_ITU_P527_callback(f_hz)

def wet_ground_ITU_P527_callback(f_hz):

    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

 

    
    # Read and process conductivity/permittivity data.
    df_con = _read_material_curve_csv("B_wet_ground_con.csv")
    df_per = _read_material_curve_csv("B_wet_ground_per.csv")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    
    return (relative_permittivity.item(), conductivity.item())

def medium_dry_ground_ITU_P527_callback(f_hz):

    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

 

    
    # Read and process conductivity/permittivity data.
    df_con = _read_material_curve_csv("D_medium_dry_ground_con.csv")
    df_per = _read_material_curve_csv("D_medium_dry_ground_per.csv")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    return (relative_permittivity.item(), conductivity.item())

def very_dry_ground_ITU_P527_callback(f_hz):
    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)


    
    # Read and process conductivity/permittivity data.
    df_con = _read_material_curve_csv("E_very_dry_ground_con.csv")
    df_per = _read_material_curve_csv("E_very_dry_ground_per.csv")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    
    return (relative_permittivity.item(), conductivity.item())

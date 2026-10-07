# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from Geo2SigMap; see THIRD_PARTY_NOTICES.md for attribution and changes.
import os
import pooch
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


URLS = {
    "10_km_cell_grid.gpkg":
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/FullExtentSpatialMetadata/10_km_cell_grid.gpkg",
    "FESM_1m.gpkg":
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/FullExtentSpatialMetadata/FESM_1m.gpkg",
}


def get_app_info():
    """Return (app_name, app_version) based on installed package metadata."""
    package = __name__.split('.')[0]
    try:
        return package, version(package)
    except PackageNotFoundError:
        # editable mode or execution outside installed package
        return package, "0.0.0"


def _cache_dir():
    """Return the cache directory for this app."""
    app_name, _ = get_app_info()
    return Path(pooch.os_cache(app_name))


def download_fesm_files():
    """
    Download all FESM data files and return list of local file paths.
    Always downloads if not cached yet.
    """
    cache_path = _cache_dir()
    cache_path.mkdir(parents=True, exist_ok=True)

    local_paths = []

    for fname, url in URLS.items():
        print(f"Downloading {fname} ...")
        path = pooch.retrieve(
            url=url,
            fname=fname,
            path=cache_path,
            known_hash=None,     # can replace with real checksum later
            progressbar=True,
        )
        print(f" → saved/cached: {path}")
        local_paths.append(Path(path))

    return local_paths


def get_fesm_paths():
    """
    Return local paths to FESM files.
    - If files already exist in cache → return immediately.
    - If missing → download automatically.
    """
    cache_path = _cache_dir()

    paths = [cache_path / fname for fname in URLS.keys()]
    if all(p.exists() for p in paths):
        print("FESM files already cached.")
        return paths

    print("FESM files missing — downloading now...")
    return download_fesm_files()

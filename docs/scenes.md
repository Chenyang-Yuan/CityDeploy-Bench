# Custom scenes and simulated deployments

## Construct a geographic scene

Install the `scene` and `radio` extras, then provide a WGS84 bounding box in **longitude, latitude** order:

```console
citydeploy build-scene --bbox -0.141 51.521 -0.135 51.525 --place-name custom_london --save-root datasets/custom_scenes
```

This is an illustrative region, not a published benchmark scene. The generator allocates a scene ID and reports the final metric dimensions. Read the resulting metadata; do not assume an arbitrary geographic box is exactly 128, 256 or 512 meters. For a fixed metric window, use the paired `--requested-size-m WIDTH HEIGHT` and `--local-frame-center LON LAT` options with a containing query bbox.

The package contains scene XML, referenced meshes, semantic rasters, TX feasibility inputs, deployment candidate coordinates, metadata, source snapshots and QA reports. Roads define the default deployment/evaluation domain. Inspect the source masks and QA before simulation, especially map completeness and estimated building heights. Queries use OpenStreetMap through Overpass; this is a data access service, not a different underlying map dataset.

Construction depends on public network services and optional geospatial libraries. This source-release verification does not claim fresh OSM scene generation was tested on every supported OS. The procedural radio demo tests the backend without those network dependencies.

## Configure deployment data

```console
citydeploy init
```

Copy `configs/datasets/urban_multicity.json` to a new plan and edit:

- `scenes_root`, `scenes` and `output_root` to select your packages and a new dataset destination.
- The map-size generation profiles, allowed TX counts and number of families to control scope.
- `split.scene_assignments` so every supplied scene is assigned to one split. Overlapping geographic areas must stay together. At least three non-overlapping groups are needed for a three-way geographic split.
- The RF profile, minimum separation, strategy mixture and map-storage fractions.

The preset was designed for square 128/256/512 m scenes. The generator validates configured sizes; add a matching generation profile for other dimensions rather than silently substituting a benchmark size. Begin with a small family count and check storage cost before increasing it. Saving all per-TX path gains and full-resolution maps can be expensive.

```console
citydeploy build-data --plan configs/datasets/custom.json --dry-run
citydeploy build-data --plan configs/datasets/custom.json
citydeploy validate-data datasets/raytracing/custom
```

The output contract binds simulation settings to labels. Resume only with a compatible contract; use another output directory when changing geometry, RF settings or coverage definitions. Model training then accepts this dataset through `--dataset-root` and its scenes through `--scene-root`.

Geographic data, including derived scene packages, need appropriate source attribution and licensing. See `THIRD_PARTY_NOTICES.md`; the code license does not replace a data license.

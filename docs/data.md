# Scene and deployment data

## Dataset availability

The companion dataset is hosted at
[Chenyang-Yuan/CityDeploy-Data on Hugging Face](https://huggingface.co/datasets/Chenyang-Yuan/CityDeploy-Data).
It is currently a **private preview**, containing 18 example deployment rows and a
60-scene catalog. Authorized users can inspect the Dataset Card and its loading
examples. Visitors without access may see a not-found page; the link does not
imply public availability.

The preview does not include the complete training splits, cached scene tensors,
ray-tracing scene packages, Radio Maps or pretrained weights. It is not a drop-in
input for `citydeploy train`, `citydeploy evaluate`, or the complete data validator.
Use `citydeploy demo --all-samplers` for the self-contained tutorial while the
full data release is being prepared.

The reference corpus has 54,193 unique deployments with scalar labels; 36,599 have
archived Radio Maps. Missing spatial outputs do not mean zero coverage. Full-release
download instructions and integrity metadata will accompany the final corpus.
Dataset licensing is separate from the code license.

## Scene inputs

Each scene directory supplies `scene.xml` and the geometry files it references.
The `inputs/` directory contains `scene_inputs.npz`, `semantic_masks.npz`, and
`metadata.json`. All metric coordinates use a scene-centered local frame;
geographic coordinates are projected to a consistent metric reference system.

The 11 feature channels, in order, are `building_mask`, `road_mask`, `green_mask`,
`water_mask`, `obstacle_mask`, `built_up_mask`, `explicit_open_mask`,
`unknown_feature_mask`, `background_mask`, `tx_feasible_mask`, and
`evaluation_mask`. Raster row zero corresponds to the minimum y coordinate.

The TX feasibility mask constrains deployment. The cached feature channel named
`evaluation_mask` is scene context and must not automatically be treated as the
metric denominator: some scene packages define it over all non-water cells.
Coverage labels use road evaluation cells; archived Radio Maps expose their
denominator explicitly as `road_evaluation_mask`. Preserve the trained feature
channel order when loading data rather than replacing this context channel.

## Deployment storage

The generator writes a manifest, RF contract, geographic split manifest, scene
registry, cached scene tensors, split-specific Parquet shards, and optional
radio-map arrays. `dataset_builder.storage` defines the exact typed schema.

| Field group | Content |
| --- | --- |
| Identity | Sample ID, scene ID, contract ID and deployment-family ID |
| Deployment | TX count, candidate indices, metric XYZ coordinates and normalized XY coordinates |
| Targets | Path-loss, SS-RSRP, SINR, throughput and joint coverage |
| Distribution | Metric means and percentiles |
| Reproducibility | Sampling strategy, random seed, simulation settings and contract hashes |
| Spatial outputs | Optional aggregate maps, serving association and per-TX path gains |

Sample IDs are invariant to TX permutation. Multiple subsets can reuse the
same master-pool ray traces. Family memberships are stored separately so that
one physical deployment can participate in several learning comparisons without
duplicating its radio simulation.

## Splits and validation

Overlapping geographic scenes belong to one split group. The default plan
provides explicit scene assignments. The validator checks coordinate bounds and
round trips, cardinalities, finite metrics, family membership, geographic split
consistency, contract hashes, and referenced spatial outputs. Review non-finite
metric values and invalid-row flags when importing external datasets; the
validator is not a complete physical plausibility proof.

The source distribution uses the `citydeploy` schema namespace. New datasets
should be built with this distribution. Earlier namespaces can be read with
the explicit global option `--legacy-schema-prefix <namespace>`; the reader
maps only that leading schema namespace in memory, without modifying files or
skipping integrity hashes. Model tensor layouts and data fields must still
match. Renaming identifiers inside a dataset alone is insufficient because
manifests contain integrity hashes. Do not apply this option to arbitrary
incompatible schemas.

The validator writes its QA report under the dataset's `qa/` directory. For a
strictly read-only inventory, use `citydeploy inspect-data <dataset>`.

No geographic data are needed for the synthetic smoke test. Full experiments
require real scene assets and simulated deployment records. Scene construction
can retrieve current public map data, but changing map snapshots can change
geometry and radio results. Use identical scene assets and RF contracts for a
controlled method comparison.

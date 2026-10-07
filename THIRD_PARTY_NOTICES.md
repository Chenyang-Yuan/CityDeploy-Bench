# Third-party notices and license scope

## First-party code

The MIT license in `LICENSE` covers first-party CityDeploy-Bench contributions, excluding the derived components identified below and separately installed dependencies. It does not grant rights to geographic data, imagery or model/data artifacts obtained elsewhere.

## Geo2SigMap-derived components — CC BY-NC 4.0

The geographic mesh/terrain utilities and Sionna coverage-map adapter incorporate and adapt functionality from **Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases**, by **Yiming Li, Zeyu Li, Zhihui Gao and Tingjun Chen**.

- Source: <https://github.com/functions-lab/geo2sigmap>
- Upstream license declaration: the source repository README, “License” section.
- License: Creative Commons Attribution–NonCommercial 4.0 International, <https://creativecommons.org/licenses/by-nc/4.0/>.
- Legal terms: <https://creativecommons.org/licenses/by-nc/4.0/legalcode.en>.

The following files are conservatively excluded from the first-party MIT grant and distributed under those retained terms, including modifications in this distribution:

```text
radio_backend/sionna_adapter.py
scene_builder/geo.py
scene_builder/mesh_builder.py
scene_builder/itu_materials.py
scene_builder/terrain/__init__.py
scene_builder/terrain/dem.py
scene_builder/terrain/dem_gpkg_data.py
scene_builder/terrain/dem_utils.py
scene_builder/terrain/usgs_locator.py
```

Modifications include scene-centered metric frames, semantic scene packaging, radio-profile integration, runtime compatibility preparation, and deployment-oriented interfaces. No upstream endorsement is implied. Retain this attribution, the license link and the modification notice when redistributing these derived components. The full distribution must not be represented as unrestricted commercial-use MIT software.

## Separately installed software

Sionna RT, Mitsuba, Dr.Jit, PyTorch, NumPy, SciPy, PyArrow, Matplotlib, and geographic/mesh libraries are installed as dependencies, not redistributed in this source archive. Their own licenses and notices remain applicable. The optional Windows compatibility helper creates a local Dr.Jit cache copy at runtime; it does not ship a Dr.Jit binary or supersede its license.

## Geographic data and imagery

OpenStreetMap data require attribution to OpenStreetMap contributors and compliance with the applicable Open Database License: <https://www.openstreetmap.org/copyright>. USGS elevation products and other data sources retain their respective terms. No urban scene snapshot or geographic dataset is included. Documentation contains selected manuscript figures, including embedded geographic imagery; these figures are excluded from the MIT code license. Attribution and visual-asset scope are documented in [docs/assets/README.md](docs/assets/README.md). A future data release must contain its own provenance and license documentation.

## Scientific algorithms

The search methods and relational models are task-specific implementations/adaptations described in `docs/methods.md`. Scientific references are not claims of exact upstream code equivalence. In particular, scalar GA is not NSGA-III, the BR-SNIS planner uses ISIR transitions, and static NRI rewards differ from trajectory inference.

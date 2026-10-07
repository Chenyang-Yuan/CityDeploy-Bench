# Development and release checks

## Test the core

```console
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m examples.smoke_test
citydeploy demo --all-samplers --output outputs/demo_check
```

The complete test suite also imports a small geographic subset: PyProj, Shapely and Rasterio. Install these packages, or install the full `scene` extra, before running all tests. Unit tests cover coordinate contracts, set invariance, hyperedge orders, physical metrics, dataset integrity, search algorithms, target-run resume logic and the isolated OptiX compatibility helper.

## Extend a component

For a new utility model, provide a permutation-invariant scalar prediction with coordinate gradients, add a schema/configuration to `energy_model/guidance.py`, and test serialization and TX-order invariance. Neural utility prediction must remain separate from physical evaluation.

For a new planner, implement the result contract used by `samplers/runner.py`: ranked candidate TX tensors, reward values, trace and counted reward evaluations. Register its configuration, CLI options and smoke budget. Do not compare equal iteration counts as equal evaluation budgets.

For an RF change, copy and modify the profile, generate a new dataset contract, and retain its provenance. Do not relabel existing data under a new threshold/power configuration without recomputing the affected quantities.

## Packaging

```console
python -m build
citydeploy audit-release .
```

`dist/`, build intermediates, data, model weights and runtime outputs are ignored by version control. Audit a clean source tree before packaging; the audit deliberately rejects generated artifacts. Validate a wheel in a separate workspace so source-tree imports cannot hide missing modules or configuration resources.

The included CI definition targets Windows and Linux core tests and wheel construction. A workflow definition is not evidence that hosted CI has run. Radio and geographic network tests are separate because they require hardware, optional binary dependencies and external services.

## Local verification scope

Release validation exercises the Windows Python 3.10 reference environment, source tests, full one-epoch synthetic training for four models, all 36 model/search combinations, and real CUDA/OptiX radio tracing on a procedural scene. Relocated wheel tests reuse installed dependency packages and are not proof of a fresh installation on every operating system. No full urban benchmark has been rerun as part of source packaging.

Before a public release, inspect the artifact inventory and archive metadata, supply final citation metadata, and publish dataset/model artifacts with their own licenses and hashes. Do not publish local validation workspaces or historical research outputs as part of the source repository.

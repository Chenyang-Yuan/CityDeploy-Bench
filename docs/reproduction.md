# Reproduction scope

## What can be reproduced immediately

The source tests, synthetic training/search example, and procedural radio demo can run without the benchmark dataset. These establish interface and runtime behavior only. Exact urban results require the original scene snapshots, RF contracts, deployment splits, trained weights or matched training settings, and complete per-method experiment configurations.

No urban dataset, pretrained model or historical result is bundled. Public dataset hosting and model artifact publication are separate release steps. Do not substitute current OSM geometry and call the result an exact reproduction of a past scene snapshot.

## Reference recipes

Run `citydeploy init` first to obtain editable configuration copies.

| Recipe | Configuration | Status and scope |
| --- | --- | --- |
| Base dataset | `configs/datasets/urban_multicity.json` | Six cities, 60 scenes; needs geographic assets and RT |
| Four utility models | `configs/models/*.json` | Supported training entry points; needs labeled data |
| Fixed-TX seen scenes | `configs/experiments/fixed_tx_seen.json` | 48 conditions, four utilities, nine methods, five repeats; method-specific budgets |
| Target-coverage pilot | `configs/experiments/paris_07_hpem_target70.json` | Paris 256 m, target 70%, counts 2–9; not all paper target experiments |
| Adaptation dataset | `configs/datasets/reward_adaptation_targets.json` | Three target scene/cardinality pairs |
| Few/full adaptation | `configs/energy_model/reward_adaptation_three_targets.json` | Separate held-out target deployments, sequential 30%-pool/full-pool training |

The base generation recipe uses TX counts 1/2 for 128 m, 2/3/4 for 256 m, and 5/6/7/8/9 for 512 m. The fixed-TX evaluation recipe uses 1/2, 2/3/4 and 5/7/9 respectively. Generation and evaluation cardinalities are different concepts.

```console
citydeploy build-data --plan configs/datasets/urban_multicity.json --dry-run
citydeploy benchmark --plan configs/experiments/fixed_tx_seen.json --dry-run
citydeploy benchmark --plan configs/experiments/fixed_tx_seen.json --max-trials 1 --device cuda --ray-device gpu
```

Inspect the pilot before removing `--max-trials`. Dry runs validate required inputs; they cannot run successfully when the specified scene packages or model weights are missing. The benchmark runner writes its own trial/summary records. Its historical trial format differs from `evaluate`'s per-seed `result.json`; do not feed an unrelated record schema into `summarize`.

## Few-shot and full-pool adaptation

The targets are `paris_11` (128 m, 2 TX), `new_york_12` (256 m, 4 TX), and `tokyo_01` (512 m, 8 TX).

```console
citydeploy build-data --plan configs/datasets/reward_adaptation_targets.json --dry-run
citydeploy adapt --plan configs/energy_model/reward_adaptation_three_targets.json --dry-run
citydeploy adapt --plan configs/energy_model/reward_adaptation_three_targets.json --device cuda
```

The preset reserves 200 evaluation deployments per target and takes a 990-deployment adaptation pool. The few-shot stage uses 297 deployments (30%) for 30 epochs. Full-pool adaptation continues for 70 epochs from that state using all 990 adaptation deployments, **not the held-out evaluation labels**. Four base model kinds can be filtered with `--model`; targets with `--target`. Adapted outputs live under a separate output root and do not overwrite the base weights. Use saved stage checkpoint paths in `citydeploy evaluate --checkpoint ...` for matched deployment tests.

Zero-shot, few-shot and full-pool comparisons must hold scene, TX count, RF contract, method budget and evaluation seeds fixed. The manuscript's “In-distribution” stage denotes full target-pool adaptation, not training on its held-out evaluation data. “Unseen scene” does not necessarily mean “unseen city.”

## Outstanding paper-to-artifact alignment

These recipes do **not** cover every setting in the manuscript. In particular, expanded-capacity runs above nine TXs, all target-rate sweeps with larger seed counts, and all ablations require their exact configurations and compatible model artifacts before claiming table-level reproduction. Do not extrapolate from the included Paris pilot or silently reuse a nine-TX model for larger sets.

Record hashes of supplied datasets and model files with a release. Check scene IDs, geographic split assignments, counts per condition, seed sets, search budgets and aggregation rules against the relevant table. The current source contains no prefilled paper scores.

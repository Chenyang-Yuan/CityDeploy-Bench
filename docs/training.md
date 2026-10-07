# Training utility models

## Inputs and splits

Training requires typed deployment rows, normalized TX coordinates, metric scene size, coverage labels and scene features. Cached tensors in the dataset allow training without importing the geographic construction stack. HPEM also uses deployment-family memberships for its ranking objective.

```console
citydeploy init
citydeploy validate-data datasets/raytracing/urban_multicity
citydeploy train --model hpem --device cuda --dry-run
citydeploy train --model hpem --device cuda
```

The trainer honors explicit dataset partitions. Do not move test deployments into training or randomly split overlapping geographic scenes. `--max-samples` is useful for an interface pilot, not a replacement for the published training split. Selection of the best model uses validation data; test metrics are reported separately.

## Four presets

| Command model | Representation | Learning rate | Batch | Epoch limit | Weight decay |
| --- | --- | --- | --- | --- | --- |
| `hpem` | Joint orders 1, 2, 3, 4 | 3e-4 | 32 | 100 | 1e-4 |
| `regressor` | Invariant pooled TX embeddings | 1e-4 | 32 | 100 | 1e-5 |
| `in` | Pairwise interaction network | 1e-4 | 32 | 100 | 1e-5 |
| `nri` | Latent categorical relations | 1e-4 | 32 | 100 | 1e-5 |

These are the shipped reference presets, not a claim that every paper table used the same training invocation. Regressor/IN/NRI have dropout 0.2 and early-stopping patience 15. NRI uses four edge types, Gumbel temperature 0.5, and KL weight 1e-4. HPEM energy, family-ranking, and auxiliary loss weights are 1.0, 0.25, 0.25. All presets default to at most nine TXs. See JSON and `docs/methods.md` for the full configuration.

## Changing settings

`citydeploy init` creates editable `configs/models/*.json`. Change values in the **training** object, or supply `--config`. The **architecture_reference** object documents fixed layer defaults; it is not a second source of training overrides. Architecture changes beyond the trainer's exposed options require editing the model configuration and trainer construction together. The effective architecture is saved inside each checkpoint.

Common overrides are directly supported:

```console
citydeploy train --model nri --epochs 50 --batch-size 16 --max-num-tx 9 --seed 123 --device cuda --dataset-root datasets/raytracing/urban_multicity --scene-root datasets/scenes --output-root outputs/training_seed123 --model-root models_seed123
```

`reward` is a compatibility alias for `regressor`. The public stable filename is `models/regressor/checkpoint_best.pt`. The internal model kind remains `reward_predictor`.

The wrapper preserves per-run training files and exports a stable best-model copy. Choose a different `--model-root` before retraining; the wrapper refuses to overwrite existing weights. Multiple concurrent writers must use different output/model roots.

Only load trusted weights. The shared inference loader uses PyTorch's restricted `weights_only=True` loader; this is not a blanket safety guarantee for arbitrary third-party files.

## Capacity and normalization

All supported scene sizes share a model. XY is normalized as `(xy_m / map_size_m) + 0.5`; physical map dimensions remain model inputs. Fixed TX height comes from the RF profile. A model's capacity is **not evidence of training coverage** at all cardinalities. Evaluate only within the configured capacity, and disclose extrapolation beyond the training TX counts.

Fine-tuning recipes and held-out target pools are described in [reproduction](reproduction.md).

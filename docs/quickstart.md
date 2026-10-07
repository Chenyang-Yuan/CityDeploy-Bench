# Synthetic quickstart

This tutorial checks the learning and planning interfaces without downloading urban data or using a radio backend.

```console
citydeploy demo --all-samplers
```

The command creates three synthetic scenes with separate train/validation/test labels, invokes the actual trainers for all four utility models, and searches with all nine methods. It writes:

```text
outputs/demo/
  dataset/       Typed Parquet rows, contracts, splits, cached features
  scenes/        Synthetic semantic inputs for the tutorial
  training/      Per-model training outputs
  models/        Four best-model copies
  preview.png    Four illustrative Diffusion-selected point sets
  summary.json   Coordinates and predictions for each model/planner pair
```

Labels count the fraction of cells within an analytic service radius. All four marginal labels intentionally repeat that analytic fraction. They are **not** path loss, SS-RSRP, SINR or throughput measurements. One-epoch training and tiny search budgets are smoke tests, not performance comparisons. Synthetic scene inputs do not represent geographic scene packages and must not be used as urban radio assets.

To run individual steps instead:

```console
citydeploy demo-data --output outputs/tutorial
citydeploy validate-data outputs/tutorial/dataset
citydeploy inspect-data outputs/tutorial/dataset
citydeploy train --model hpem --dataset-root outputs/tutorial/dataset --scene-root outputs/tutorial/scenes --epochs 1 --batch-size 8 --device cpu --model-root outputs/tutorial/models --output-root outputs/tutorial/training
```

Use the same training command with `regressor`, `in`, or `nri`. Existing dataset/model destinations are protected against accidental overwrite.

For a separate **real ray-tracing** smoke test, install the radio extra and run `citydeploy radio-demo --ray-device gpu`. This constructs a procedural ground plane and saves physical metric arrays and five maps. Its small ray budget and simple geometry do not reproduce urban benchmark results.

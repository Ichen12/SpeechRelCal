# Training and evaluation

Prepare calibration pairs and query files according to [the input specification](input_format.md). Data acquisition, speech-feature extraction, and Ridge readouts are described in [the reproduction notes](reproduction.md). The CLI starts from their raw relation scores.

```bash
# Fit both interfaces using only the independent calibration split.
speechrelcal calibrate --pairs local_data/calibration.csv --output outputs/calibration.json

# Prepare the common calibrated inputs for Product, Log-linear, and DeepSets.
speechrelcal prepare --queries local_data/evaluation_raw.json \
  --calibration outputs/calibration.json --interface full_state \
  --output outputs/evaluation

# Product has no composition training step.
speechrelcal evaluate --queries outputs/evaluation/queries.json --output outputs/product
```

Use `--interface sigmoid` for Std. inputs; `slope_only`, `intercept_only`, and `full_affine` run the Figure 2 controls. Add `--relation-aware` during preparation when using DeepSets+R.

Prepare speaker-disjoint source fit and validation queries with the same frozen calibration. Select hyperparameters on source data:

```bash
speechrelcal select --queries local_data/source_fit/queries.json \
  --validation local_data/source_validation/queries.json \
  --model deepsets --seed 2026090411 --output outputs/selection
```

The result is `selection.json`. Refit on **all budgeted source queries** using its selected width, learning rate, and epoch count. For example, if selection returns width 16, learning rate 0.001, and 8 epochs:

```bash
speechrelcal train --queries local_data/source_all/queries.json \
  --model deepsets --width 16 --lr 0.001 --epochs 8 --seed 2026090411 \
  --output outputs/refit
speechrelcal evaluate --queries outputs/evaluation/queries.json \
  --checkpoint outputs/refit/model.pt --output outputs/deepsets
```

The other model names are `log_linear` and `deepsets_r`. Repeat the refit for the dataset's three initialization seeds. Checkpoints save model, optimizer, query order, cursor, and sampler state every 100 updates and at completion. Repeating `train` with the **same configuration and output directory** resumes it; use a new output directory for a different configuration. Use a fresh selection directory for each selection run.

[configs/paper.yaml](../configs/paper.yaml) records dataset-specific seeds, thresholds, partitions, and optimization settings. In particular, Figure 3(b) uses weight decay `0.0001` and epoch candidates `0 8 16 32`, whereas the main comparisons use `0.01` and `8 16 32`.

# SpeechRelCal

Code and final results for **Beyond Rankings: Score Scales Shape Multi-Attribute Speech Retrieval**.

Chen Li, Jiale Cao, Zhihao Tang, Peiji Yang, Zhisheng Wang, Jianxing Yu, and Jian Yin.

Given a reference utterance, a query can ask for the same speaker, higher pitch, and slower speaking rate. SpeechRelCal examines how the numerical scales of individual relation scores affect retrieval when those conditions are combined.

With frozen speech experts, rank-preserving calibration improves Product retrieval by **5.03–15.29 nAP percentage points**. Intercept-only calibration recovers **84.6%–98.5%** of that gain. After scale alignment, learned composers provide smaller, condition-dependent gains. Product uses no joint-query supervision; attribute readouts and calibration still require single-attribute supervision.

## Release scope

This repository provides the calibration and composition code, a portable training/evaluation interface for user-prepared scores, final aggregate results, and the settings needed to interpret and reproduce the reported comparisons. It does not distribute datasets, audio, embeddings, pretrained speech models, or the full experiment orchestration history.

There are two separately fitted interfaces:

- **Figure 2:** positive affine event-sigmoid calibration, with fixed-sigmoid, slope-only, intercept-only, and full-affine controls. These preserve within-relation rankings.
- **Table 1 and Figure 3:** full-state calibration, followed by Product, positive Log-linear, DeepSets, or relation-aware DeepSets (**DeepSets+R**). No general rank-preservation guarantee is asserted for the full-state interface.

## Installation

Use Python 3.12 and a separate environment. Install a PyTorch 2.7 build appropriate for your hardware, then:

```bash
python -m pip install -e .
```

The core operates on scores and needs no audio-model dependencies. The versions used for release verification are recorded in [requirements-tested.txt](requirements-tested.txt). They describe this release check, not a complete lockfile of the historical audio-extraction environments.

## Check the paper results

No dataset is needed:

```bash
python scripts/check_paper_results.py
```

This checks all 40 Table 1 values, all 50 Figure 3(a) values and significance marks, the released figure estimates and confidence intervals against their original aggregates, and key numerical statements in the text. The final-PDF fingerprint and rounded references are in [paper_reference.json](results/paper_reference.json); the check report is [paper_check.json](results/paper_check.json).

| Result | Released file |
|---|---|
| Table 1: input scale × composer, nAP (%) | [table1.csv](results/table1.csv) |
| Figure 2: gains over fixed sigmoid, pp | [fig2.csv](results/fig2.csv) |
| Figure 3(a): supervision budgets, pp | [fig3a.csv](results/fig3a.csv) |
| Figure 3(b): query-matched training, pp | [fig3b.csv](results/fig3b.csv) |
| Calibration quality, prior sensitivity, AUROC, query counts | [source/](results/source/) |
| Original experiment/result mapping | [provenance.json](results/provenance.json) |

For example, the calibrated Product nAP values in Table 1 are:

| MSP: 2 attributes | MSP: 3 | MSP: 4 | LibriTTS-P: 2 | LibriTTS-P: 3 |
|---:|---:|---:|---:|---:|
| 26.07 | 16.90 | 11.76 | 42.51 | 24.86 |

Checking these aggregates does **not** rerun the audio experiments. Exact end-to-end reproduction also requires the original corpus versions, panel membership and ordering, extracted features, and fitted readouts. These assets are not bundled.

## Run on prepared scores

Prepare calibration pairs and query files according to [the input specification](docs/input_format.md). Data acquisition, speech-feature extraction, and Ridge readouts are described in [the reproduction notes](docs/reproduction.md). The CLI starts from their raw relation scores.

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

[configs/paper.yaml](configs/paper.yaml) records dataset-specific seeds, thresholds, partitions, and optimization settings. It is a reference configuration, not an automatic experiment launcher. In particular, Figure 3(b) uses weight decay `0.0001` and epoch candidates `0 8 16 32`, whereas the main comparisons use `0.01` and `8 16 32`.

## Reproduction details

See [docs/reproduction.md](docs/reproduction.md) for candidate pools, query filtering, supervision budgets, statistical aggregation, and implementation details beyond the paper's compact description. These include inclusive threshold boundaries in the original experiments and the regularization used by the full-state ordinal calibrator.

Core verification:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The tests exercise calibration/ranking behavior, the multi-positive objective, checkpoint recovery, and the portable CLI. They do not train speech encoders or recreate the full corpus experiments.

## Citation

```bibtex
@misc{li_speechrelcal,
  title = {Beyond Rankings: Score Scales Shape Multi-Attribute Speech Retrieval},
  author = {Li, Chen and Cao, Jiale and Tang, Zhihao and Yang, Peiji and Wang, Zhisheng and Yu, Jianxing and Yin, Jian},
  howpublished = {Manuscript; code repository},
  url = {https://github.com/Ichen12/SpeechRelCal}
}
```

This citation intentionally omits an unverified publication year, DOI, and acceptance status. Dataset and pretrained-model terms remain those of their respective providers.

## License

The code and accompanying documentation are released under the [MIT License](LICENSE). Datasets and pretrained models are not included and remain subject to their original terms.

# SpeechRelCal

Official implementation of **Beyond Rankings: Score Scales Shape Multi-Attribute Speech Retrieval**.

**Chen Li, Jiale Cao, Zhihao Tang, Peiji Yang, Zhisheng Wang, Jianxing Yu, and Jian Yin**

## Motivation

Given a reference utterance, a multi-attribute query may ask for the **same speaker, higher pitch, and slower speaking rate**. Each condition is scored by a speech attribute model, and retrieval requires a candidate to satisfy all conditions simultaneously.

Strong rankings for individual attributes do not necessarily produce a strong combined ranking. Models trained for different attributes can assign scores on different numerical scales. A learned composer may therefore improve retrieval partly by adapting to these scale differences. Our work separates **relation-score scale alignment** from **composition learning** to examine their respective contributions.

## Method

We keep speech experts fixed, calibrate their relation scores using single-attribute labels, and combine the resulting probabilities with **Product**:

$$
S_{\mathrm{Product}}(r,x,q)=\sum_{e\in q}\log\max\{p_e(r,x),10^{-7}\}.
$$

Here, $p_e(r,x)$ is the probability that candidate $x$ satisfies relation $e$ relative to reference $r$. Product requires no joint-query supervision at the composition stage.

The study has three parts:

1. **Isolate score-scale effects.** Compare fixed sigmoid, slope-only, intercept-only, and full-affine event calibration. Positive affine mappings preserve single-relation rankings, allowing us to measure the effect of score scales on compositional retrieval.
2. **Examine slope and intercept contributions.** Measure how much of the full-affine retrieval gain can be recovered by adjusting relation-specific intercepts alone.
3. **Re-evaluate composition learning.** Feed the same multi-state calibrated probabilities to Product, positive Log-linear, DeepSets, and relation-aware DeepSets (**DeepSets+R**), then compare their performance across supervision budgets and query settings.

The event-level calibration used for rank-preserving analysis and the multi-state calibration used for composer comparisons are fitted separately.

## Experiments and findings

We evaluate two speech relation retrieval settings with speaker-disjoint fitting, calibration, and evaluation data:

| Dataset | Attributes | Frozen speech models |
|---|---|---|
| MSP-Podcast | Speaker, arousal, valence, dominance | ECAPA-TDNN; ExHuBERT with Ridge attribute readouts |
| LibriTTS-P | Speaker, pitch, speaking rate | ECAPA-TDNN; WavLM Base+ with Ridge attribute readouts |

The main composer experiments train on seen two-attribute relation combinations and evaluate held-out combinations and queries with more attributes. The primary metric is normalized average precision (**nAP**), aggregated with equal weight per reference speaker.

### Score scales affect retrieval even when individual rankings stay fixed

Full-affine calibration improves Product by **5.03–15.29 nAP percentage points** across both datasets and all evaluated query sizes. Single-relation rankings remain unchanged. Mean ECE decreases from **0.254 to 0.020** on MSP-Podcast and from **0.278 to 0.006** on LibriTTS-P.

### Intercept adjustment recovers most of the calibration gain

Intercept-only calibration recovers **84.6%–98.5%** of the full-affine gain and outperforms slope-only calibration in every setting. Under positive-to-negative calibration weights of 0.25, 1, and 4, the recovered fraction remains **82.1%–98.5%**.

### Scale alignment narrows the advantage of learned composers

The following results reproduce Table 1 of the paper. **Std.** denotes fixed-sigmoid inputs; **Cal.** denotes multi-state calibrated probabilities. Values are nAP (%), and column numbers indicate query attribute counts.

| Input | Composer | MSP-2 | MSP-3 | MSP-4 | Libri-2 | Libri-3 |
|---|---|---:|---:|---:|---:|---:|
| Std. | Product | 17.70 | 9.85 | 6.77 | 27.25 | 17.72 |
| Std. | Log-linear | 15.48 | 9.59 | 6.35 | 28.64 | 19.08 |
| Std. | DeepSets | 23.29 | 15.25 | 10.29 | 37.76 | 22.38 |
| Std. | DeepSets+R | 21.22 | 14.57 | 10.05 | 41.42 | 23.94 |
| Cal. | Product | 26.07 | 16.90 | 11.76 | 42.51 | 24.86 |
| Cal. | Log-linear | 25.83 | 16.62 | 11.71 | 42.83 | 24.99 |
| Cal. | DeepSets | 25.86 | 16.70 | 11.83 | 42.49 | 24.85 |
| Cal. | DeepSets+R | 25.87 | 16.67 | 11.75 | 42.86 | 24.90 |

DeepSets substantially outperforms Product with Std. inputs. After calibration, their absolute performance differences shrink to approximately **0.21 pp or less**. Calibrated Product also exceeds standardized DeepSets by **1.47–4.75 pp**.

Additional composition learning provides small, setting-dependent gains. Across joint-query supervision budgets of **1%, 5%, 10%, 25%, and 100%**, Log-linear's gain on two-attribute LibriTTS-P queries reaches approximately **0.32 pp**. With query-matched training, DeepSets+R gains approximately **0.55 pp** on both two- and three-attribute LibriTTS-P queries, while MSP-Podcast shows no positive gain. These findings support controlling relation-score scales when assessing the added value of learned composition.

Full-precision results and confidence intervals are available in [Table 1](results/table1.csv), [Figure 2](results/fig2.csv), [Figure 3(a)](results/fig3a.csv), and [Figure 3(b)](results/fig3b.csv).

## Getting started

Use Python 3.12 and PyTorch 2.7, then install the package:

```bash
python -m pip install -e .
```

Fit relation calibrators and evaluate Product on prepared scores:

```bash
speechrelcal calibrate --pairs local_data/calibration.csv --output outputs/calibration.json

speechrelcal prepare --queries local_data/evaluation_raw.json \
  --calibration outputs/calibration.json --interface full_state \
  --output outputs/evaluation

speechrelcal evaluate --queries outputs/evaluation/queries.json --output outputs/product
```

- [Input format](docs/input_format.md): calibration pairs, raw scores, and query files.
- [Training and evaluation](docs/usage.md): composer selection, training, and evaluation commands.
- [Reproduction notes](docs/reproduction.md): data preparation, frozen experts, splits, and statistical procedures.
- [Paper configuration](configs/paper.yaml): thresholds, seeds, supervision budgets, and optimization settings.
- [Tested dependencies](requirements-tested.txt): package versions used for verification.

To check the published results against the paper:

```bash
python scripts/check_paper_results.py
```

## Citation

```bibtex
@misc{li_speechrelcal,
  title = {Beyond Rankings: Score Scales Shape Multi-Attribute Speech Retrieval},
  author = {Li, Chen and Cao, Jiale and Tang, Zhihao and Yang, Peiji and Wang, Zhisheng and Yu, Jianxing and Yin, Jian},
  howpublished = {Manuscript; code repository},
  url = {https://github.com/Ichen12/SpeechRelCal}
}
```

## License

[MIT License](LICENSE).

# Reproduction notes

This release starts from raw attribute-relation scores and user-prepared query labels. It contains the core algorithms and final aggregate evidence, rather than a complete corpus-download, feature-extraction, and historical job-scheduling pipeline. Checking published numbers is separate from rerunning experiments.

## Data and frozen experts

Obtain MSP-Podcast from its corpus provider and LibriTTS-P metadata from [LINE's LibriTTS-P repository](https://github.com/line/LibriTTS-P), with the corresponding LibriTTS-R audio. Follow the providers' data access terms. No utterance-level data is included here.

| Dataset | Speaker | Continuous attributes |
|---|---|---|
| MSP-Podcast | ECAPA-TDNN cosine | Frozen ExHuBERT + separate Ridge readouts for arousal, valence, dominance |
| LibriTTS-P | ECAPA-TDNN cosine | Frozen WavLM Base+ + separate Ridge readouts for log-F0 and speaking rate |

The model identifiers in the experiment configuration are `speechbrain/spkrec-ecapa-voxceleb`, `amiriparian/ExHuBERT`, and `microsoft/wavlm-base-plus`. The ExHuBERT extraction record requested revision `49891e2a17d926eae8ce54219589ccf71f342750` (resolved revision was not recorded). WavLM and ECAPA extraction records did not record a revision. The release does not redistribute upstream checkpoint binaries. Matching a model name alone is insufficient to guarantee identical embeddings.

Use 16 kHz mono audio. ECAPA embeddings are L2-normalized before cosine scoring. ExHuBERT uses its final HuBERT hidden layer, before the classification head, averaged over valid frames. Its original extraction wrapper normalizes the waveform and pads/truncates to **48,000 samples (3 seconds)**, using an attention mask and FP16 inference; outputs are FP32. WavLM uses valid-frame mean pooling of the final layer, FP16 inference, and FP32 stored embeddings. ECAPA extraction uses FP32. These extraction details matter; the compact paper does not enumerate them.

Ridge has `alpha=10.0`, with separate intercept-bearing readouts fitted on FP32 embeddings. For MSP, normalize each annotation by `(value - 1) / 6`; evaluation applies the saved FP32 readout weights using FP64 embedding arrays. LibriTTS-P predictions use the FP32 embedding/weight arrays, with metadata targets `raw_lf0_mean` and `raw_speaking_rate`. Continuous pair scores are predicted candidate minus predicted reference values. Ground-truth relation labels come from the corresponding annotations.

## Panels and splits

**MSP.** The base panel has 256 source speakers from the official train partition and 256 evaluation speakers from validation, each with 16 utterances. The official test partition is not used for this main panel. Three source speaker re-splits use NumPy `default_rng` with seeds `2026090902/03/04`: shuffle sorted source speaker IDs; the first 128 speakers fit readouts and train composers, and the remaining 128 calibrate relations. Evaluation has 4,096 utterances and 4,095 nonself candidates per query. Re-splits share the same source panel and are not independent corpora.

Base panel selection uses seed `2026090305` and SHA-256 of NUL-separated strings, used only as a deterministic ranking function. Among speakers with at least 16 usable utterances, select the first 256 by hash of `(seed, role, "speaker", speaker_id)` for roles `atomic_train` and `development`. Within each selected speaker, select 16 utterances by hash of `(seed, role, "uid", uid)`. Sort the resulting panel by role, speaker, and UID. This requires the original metadata's canonical UID and speaker-ID normalization; this release does not redistribute that inventory. The four-attribute experiment uses the first **one** utterance per physical speaker as its reference, preserving panel order.

**LibriTTS-P.** Each of readout/composer fitting, calibration, and evaluation contains 128 speakers per gender and 16 utterances per speaker (256 speakers and 4,096 utterances per role). Candidate pools contain only the reference's gender: 2,047 nonself candidates. Use the first two utterances per speaker as references.

The panel selection seed is `2026090404`. Retain metadata rows with `invalid=0`, finite positive `raw_lf0_mean`, speaking rate in `[1,12]`, and speakers outside the upstream exclusion list, from `train-clean-100`, `train-clean-360`, `dev-clean`, and `test-clean`. Require at least 16 eligible utterances per speaker. In role order `atomic_train, calibration, evaluation` and gender order `F,M`, select 128 previously unused speakers by SHA-256 rank of `(seed, role, gender, speaker_id)`. Select each speaker's 16 utterances by hash of `(seed, role, speaker_id, uid)`; `uid` is `librittsp:` plus metadata `item_name`. Sort by role, gender, speaker, UID. Fields are NUL-separated before hashing. The experiment used metadata file `metadata_w_style_prompt_tags_v230922.csv` (SHA-256 `ff118bdb8cdac08cdb103409920626afff0315b9449d89a644a1282889dbe064`).

Readout fitting, independent calibration, and final evaluation must have disjoint physical speakers and utterances. Composer validation is a held-out subset of the readout-fitting side, so it is not independent of Ridge fitting.

## Relations and calibration

Thresholds are MSP A/V/D `0.075`, Libri pitch `0.10` log-F0 units, and rate `0.50` syllables/second. Main retrieval queries request same/different or higher/lower (faster/slower); similar-state pairs remain in calibration and candidate pools.

**Boundary detail:** historical experiment code labels `delta >= threshold` as higher and `delta <= -threshold` as lower. The paper writes strict inequalities, assigning equality to similar. Preserve the historical inclusive convention when reconstructing its results. The release does not claim that switching conventions has been measured to be immaterial.

For MSP, sample up to 2,000,000 nonself ordered calibration pairs uniformly with replacement, seed `2026090581`. Draw `left` uniformly from `[0,N)`, `right` from `[0,N-1)`, then add one wherever `right >= left`. For small pools use `min(cap,N*(N-1))` draws. Standardize scores using their calibration-set population mean and standard deviation. Do not standardize separately within each query.

For Libri ordinal full-state and Figure 2 fits, sample up to one million pairs per gender with seed `2026090602`, concatenate the two groups, and keep nonself pairs. The inherited binary speaker calibrator and speaker standardization instead use up to two million pairs sampled **without replacement** from all gender-matched nonself ordered calibration pairs, seed `2026090410`. This historical distinction should be retained for exact reproduction. Figure 2 speaker event fits reuse that standardization with the Figure 2 pair sample.

Event controls fit unweighted, unregularized binary cross-entropy. Positive slopes are parameterized as `exp(theta)` with `theta` in `[-6,6]`. Fixed sigmoid sets slope 1 and intercept 0. The intercept-only and full-affine fits use independent event labels, retaining similar pairs as negatives for higher/lower. The prior-weight sensitivity control changes positive-to-negative weights to `0.25, 1, 4` while keeping pairs and standardization fixed; its results are provided, not its historical launcher.

Full-state continuous calibration uses `sklearn.linear_model.LogisticRegression(max_iter=1000)` on the standardized scalar score and classes lower/similar/higher. **The original implementation retains sklearn's default L2 regularization (`C=1`)**; the paper describes the cross-entropy component without this implementation detail. Binary full-state calibration is represented as a nonnegative-slope sigmoid for same and its complement for different (slope bounded to `[0,100]`). This differs from the strictly positive event-affine fit in Figure 2.

`calibrate` uses one supplied pair set for all fits of a factor. It is suitable for new prepared inputs; reconstructing the historical Libri speaker sampling/standardization requires calling the calibration functions separately with the appropriate samples. The CLI does not silently substitute one historical sample for another.

Product sums `log(max(p,1e-7))` over active attributes. Relation probabilities are cast to FP32 before composer evaluation. It is a ranking score, not a claimed calibrated joint probability.

## Query combinations and filtering

MSP pair families, in order: speaker–arousal, speaker–valence, speaker–dominance, arousal–valence, arousal–dominance, valence–dominance. Libri pair families: speaker–pitch and speaker–rate. Positive signs denote same/higher/faster; negative signs denote different/lower/slower. The exact three seen/held-out pair partitions are in `configs/paper.yaml`; note that the direction-shift partition differs between datasets.

Main training uses only seen two-attribute combinations. Evaluation uses held-out pairs plus every sign combination of each three-attribute family, and MSP's four-attribute family. No higher-arity target labels train the main composers. Figure 3(b) instead matches source query templates to evaluation in attribute counts and relation combinations, still using only source utterances and source-side selection.

Each retained query must have at least one positive and at least one candidate violating exactly one constraint. Self is never positive or a single-violation candidate. Do not remove similar-state or other negative candidates. Filter once from annotations and use the same query set for all methods. The zero-positive audit is in `results/source/query_statistics.csv`.

## Composer training and budgets

DeepSets already receives attribute identity; DeepSets+R adds relation-state identity. Log-linear has positive softplus weights initialized to one, hence initially matches Product. The loss is

`logsumexp(candidate_scores) - mean(scores_of_all_positive_candidates)`.

This is uniform-positive cross-entropy, not negative log total positive probability. Exclude self by setting its score to negative infinity. Average four query losses per optimizer update. AdamW uses its default betas and epsilon, with weight decay `0.01` in the main comparisons. Query sampling permutes the ordered query list, consumes four entries per update, and reshuffles only when exhausted. An epoch budget is converted to `epochs * ceil(number_of_queries / 4)` updates.

For source validation, shuffle sorted source speaker IDs with seed `2026090708` and take the first `floor(N/5)` (at least one). Restrict both reference and candidate speakers to the same fit/validation group. Drop resulting queries with no positive. Select by physical-speaker-equal validation nAP using the first initialization seed and the width/learning-rate/epoch grid in the config. Refit on all budgeted source queries with each of the three initialization seeds, retaining full source candidate pools. Never select a model or hyperparameter on final evaluation.

Figure 3(a) uses budgets `1%,5%,10%,25%,100%`. For fractions below one, take the first `ceil(fraction * query_count)` queries from a permutation with seed `2026090804`; 100% retains the original order. Use one common permutation for nested reduced budgets. Subsample **before** the fit/validation split; validation labels count toward the budget. Re-select separately at every budget. These are label budgets, not matched optimizer-update budgets.

Figure 3(b) additionally uses weight decay `0.0001` and includes epoch zero in model selection. Its gains cannot be attributed solely to template matching because optimization settings also differ.

## Metrics, statistics, and result sources

nAP is `(AP - positive_prevalence) / (1 - positive_prevalence)`, excluding self from prevalence. AP sorts descending with stable tie handling. Average queries within physical reference speaker first, then weight speakers equally. Do not replace this by an unweighted mean of all queries. ECE uses 15 equal-width bins. NLL/Brier/ECE clip probabilities to `[1e-7,1-1e-7]` for their evaluation; Product uses the lower clip and an upper limit of 1.

Main paired bootstrap intervals use 10,000 draws. MSP independently resamples reference speakers, source re-splits, and model initializations, keeping relation partitions fixed. Sensitivity additionally resamples partitions. Event-only controls have no model initialization variation. Libri has one calibration setup with three composition initializations. Intervals are pointwise, not multiplicity adjusted; an interval crossing zero is not evidence of statistical equivalence.

The source mapping is in `results/provenance.json`. Table 1 Libri results use the consolidated relation-identity comparison, which includes the original Product, Log-linear, and DeepSets values. Published CSVs retain original precision; percentages and percentage-point differences are generated from those values. All displayed Figure 3(a) stars were checked against the supplied final PDF visually as well as against the saved main intervals.

The final results check does not establish a new independent evaluation or rerun feature extraction. The test suite verifies mathematical and runtime behavior on synthetic inputs. The original panels, feature caches, and readouts are needed to reproduce the reported aggregate numbers exactly.

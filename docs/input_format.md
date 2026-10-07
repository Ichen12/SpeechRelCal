# Input format

The portable entry points consume scores prepared outside this repository. Paths in a query manifest are relative to that manifest. Keep source fitting, calibration, validation, and final evaluation separated as described in [reproduction.md](reproduction.md).

## Calibration CSV

Columns: `factor,kind,score,state`.

- `factor`: `speaker`, `arousal`, `valence`, `dominance`, `pitch`, or `rate`.
- `kind`: `binary` for speaker; `ordinal` for continuous attributes.
- `score`: unoriented speaker cosine, or predicted candidate value minus predicted reference value.
- `state`: speaker `different=0, same=1`; ordinal `lower=0, similar=1, higher=2`.

Rows are sampled **nonself calibration pairs**, not query relevance labels. Do not balance the natural-prevalence main calibration set. Retain similar-state ordinal pairs. For rate, higher means faster and lower means slower. Each attribute must include all its states. Parameters are fitted independently for each attribute.

The generic `calibrate` command uses the supplied pairs for both interfaces. The historical LibriTTS-P speaker calibration/standardization inherited a separate pair sample. To reproduce those exact parameters, follow the sampling distinction in the reproduction notes; do not assume arbitrary resampling reproduces the paper.

## Raw query manifest

A JSON list of entries with this structure (the names below are illustrative):

```json
[
  {
    "query_id": "reference-id:speaker=same;pitch=higher",
    "anchor_speaker": "reference-speaker-id",
    "regime": "heldout_pair_tuple",
    "arrays": "query_0.npz",
    "self_index": 0,
    "factors": ["speaker", "pitch", "rate"],
    "states": ["same", "higher", ""]
  }
]
```

The corresponding NPZ contains:

| Key | Shape | Meaning |
|---|---|---|
| `raw` | candidates × all attributes | Unoriented pair scores; keep the same factor order in every query |
| `positive` | candidates | Boolean conjunction relevance labels; the self entry is false |
| `candidate_uids` | candidates | Unicode string utterance IDs, used to check fit/validation isolation |
| `candidate_speakers` | candidates | Unicode string physical speaker IDs |

Include the reference itself at `self_index`. Its score is suppressed during training and its entry is removed when calculating evaluation prevalence. Thus an MSP array has 4,096 rows but 4,095 evaluated candidates. LibriTTS-P arrays contain 2,048 same-gender utterances, with 2,047 evaluated candidates.

Use an empty state for an inactive attribute. Active states are `same/different`, `higher/lower`, or `faster/slower` for rate. Prepare labels from **annotations**, not predicted scores. The original experiments keep queries with at least one positive and at least one nonself candidate violating exactly one active relation. Apply that filter before producing manifests.

`prepare` creates `inputs` and `mask` arrays. For Product, Log-linear, and DeepSets, `inputs` has shape `[candidates, factors]`; inactive probabilities are one and the mask is zero. For DeepSets+R, `inputs` has shape `[candidates, factors, 3]`: probability followed by two state-indicator channels. State order is speaker `(same,different)`, rate `(slower,faster)`, and other continuous attributes `(lower,higher)`.

## Prepared queries and training

`train`, `select`, and `evaluate` consume the prepared manifest and arrays. The original ordering of queries and candidates affects seeded minibatches and stable AP tie breaking; preserve it when comparing to the paper. Keep the same mask and candidate pool across all methods in a comparison.

Selection uses separate fit/validation manifests; both reference and candidate speakers must be disjoint. The CLI checks this overlap once at selection entry. The final refit manifest combines the budgeted source queries after hyperparameter selection, and remains disjoint from calibration and final evaluation. Correct calibration/evaluation isolation is a prerequisite of external data preparation.

`evaluate` writes per-query nAP and equal-reference-speaker aggregates by regime. To aggregate multiple source splits, partitions, and seeds, concatenate the per-query records with those run identifiers, average within physical reference speaker, then average across speakers. `speechrelcal.statistics.paired` accepts a DataFrame with `method, partition, source_split_seed, seed, query_id, anchor_speaker` and metric columns. Use the same query keys for both methods. Released confidence intervals retain the original bootstrap draws.

Full query arrays can be large. The portable loader keeps prepared inputs in memory; CPU RAM depends on query count × candidate count × attribute count. This compact release does not reproduce the original shared-cache experiment scheduler.

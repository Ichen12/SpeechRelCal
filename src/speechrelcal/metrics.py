"""Paper metrics; AP uses stable candidate-order tie breaking."""
import numpy as np

def average_precision(scores: np.ndarray, valid: np.ndarray) -> float:
    """Compute binary average precision with stable tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != valid.shape or not valid.any():
        raise ValueError("Average precision requires aligned scores and positives.")
    order = np.argsort(-scores, kind="stable")
    ranked = valid[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())

def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    ece_bins: int = 15,
) -> dict[str, float]:
    """Return proper scoring rules and equal-width expected calibration error."""
    values = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    nll = -np.mean(targets * np.log(values) + (1.0 - targets) * np.log1p(-values))
    brier = np.mean((values - targets) ** 2)
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    bins = np.minimum(np.searchsorted(edges, values, side="right") - 1, ece_bins - 1)
    ece = 0.0
    for index in range(ece_bins):
        selected = bins == index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(values[selected].mean()) - float(targets[selected].mean())
            )
    return {"nll": float(nll), "brier": float(brier), "ece": float(ece)}


def normalized_average_precision(scores, positive):
    prevalence = np.asarray(positive).mean()
    return (average_precision(scores, positive)-prevalence)/(1-prevalence)

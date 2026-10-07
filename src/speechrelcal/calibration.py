"""Separately fitted event-sigmoid and full-state calibration interfaces."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression

def fit_event(z, y, method):
    learn_slope = method in ('slope_only', 'full_affine')
    learn_intercept = method in ('intercept_only', 'full_affine')

    def unpack(theta):
        return (np.exp(theta[0]) if learn_slope else 1.,
                theta[-1] if learn_intercept else 0.)

    def objective(theta):
        slope, intercept = unpack(theta)
        logits = slope*z + intercept
        error = expit(logits)-y
        grad = ([np.mean(error*slope*z)] if learn_slope else [])
        grad += [error.mean()] if learn_intercept else []
        return np.mean(np.logaddexp(0, logits)-y*logits), np.asarray(grad)

    if method == 'sigmoid':
        return dict(slope=1., intercept=0., nll=float(objective([])[0]))
    result = minimize(objective, np.zeros(int(learn_slope)+int(learn_intercept)), jac=True,
                      method='L-BFGS-B',
                      bounds=([(-6, 6)] if learn_slope else []) + ([(None, None)] if learn_intercept else []),
                      options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 500})
    # A failed atomic fit invalidates the comparison of calibration objectives.
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f'Event BCE fit failed: {result.message}')
    slope, intercept = unpack(result.x)
    return dict(slope=float(slope), intercept=float(intercept), nll=float(result.fun),
                log_slope_bound_distance=float(6-abs(np.log(slope))) if learn_slope else None)

def fit_monotonic_logit(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | str]:
    """Fit logit = nonnegative_slope * standardized_score + intercept."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(scores) != len(labels) or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Monotone calibration requires aligned binary classes.")
    mean = float(scores.mean())
    scale = float(scores.std())
    if scale < 1e-12:
        return {"slope": 0.0, "intercept": 0.0, "mean": mean, "scale": 1.0, "status": "constant"}
    standardized = (scores - mean) / scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        logits = slope * standardized + intercept
        loss = float(np.logaddexp(0.0, logits).mean() - (labels * logits).mean())
        residual = expit(logits) - labels
        gradient = np.asarray(
            [(residual * standardized).mean(), residual.mean()], dtype=np.float64
        )
        return loss, gradient

    result = minimize(
        lambda value: objective(value)[0],
        np.asarray([1.0, 0.0]),
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
        bounds=((0.0, 100.0), (None, None)),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Monotone calibration failed: {result.message}")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "mean": mean,
        "scale": scale,
        "status": "fit",
    }


def fit_calibration(scores, states, kind, seed=2026090581):
    """Fit on calibration pairs only. Binary states: different=0/same=1;
    ordinal states: lower=0/similar=1/higher=2. Scores are unoriented.
    """
    raw = np.asarray(scores, dtype=np.float64)
    states = np.asarray(states, dtype=int)
    expected = {0, 1} if kind == 'binary' else {0, 1, 2}
    if set(np.unique(states)) != expected or not np.isfinite(raw).all():
        raise ValueError('Calibration needs finite scores and all relation states.')
    mean, scale = float(raw.mean()), float(raw.std())
    if scale < 1e-12:
        raise ValueError('Constant scores cannot support the rank-preserving analysis.')
    z = (raw - mean) / scale
    names = ('same', 'different') if kind == 'binary' else ('higher', 'lower')
    positive = 1 if kind == 'binary' else 2
    events = {}
    for name, direction, label in zip(names, (1, -1), (positive, 0)):
        events[name] = {method: fit_event(direction*z, (states == label).astype(float), method)
                        for method in ('sigmoid', 'slope_only', 'intercept_only', 'full_affine')}
    if kind == 'binary':
        full_state = fit_monotonic_logit(raw, states)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed).fit(z[:, None], states)
        full_state = dict(coef=model.coef_.ravel().tolist(), intercept=model.intercept_.tolist(),
                          classes=model.classes_.tolist())
    return dict(kind=kind, mean=mean, scale=scale, events=events, full_state=full_state)


def probability(raw, calibration, state, interface='full_state'):
    state = {'faster': 'higher', 'slower': 'lower'}.get(state, state)
    z = (np.asarray(raw, dtype=np.float64)-calibration['mean'])/calibration['scale']
    if interface != 'full_state':
        params = calibration['events'][state][interface]
        direction = -1 if state in ('different', 'lower') else 1
        return expit(params['slope']*direction*z+params['intercept'])
    params = calibration['full_state']
    if calibration['kind'] == 'binary':
        # Binary softmax is represented as a sigmoid and its complement.
        z = (np.asarray(raw)-params['mean'])/params['scale']
        p = expit(params['slope']*z+params['intercept'])
        return 1-p if state == 'different' else p
    logits = z[..., None]*np.asarray(params['coef'])+np.asarray(params['intercept'])
    label = {'lower': 0, 'similar': 1, 'higher': 2}[state]
    return softmax(logits, axis=-1)[..., params['classes'].index(label)]


def product(probabilities, mask=None):
    values = np.log(np.clip(probabilities, 1e-7, 1.0))
    return (values if mask is None else values*mask).sum(axis=-1)

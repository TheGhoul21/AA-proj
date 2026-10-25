"""Finite-domain function analysis, independent of the training framework."""

from itertools import product

import numpy as np

from patterns import ALPHABET, CHAR_TO_IDX, SIGMA


def enumerate_windows(m: int) -> np.ndarray:
    """All windows, lexicographic order. Deliberately cap exponential work."""
    if not 1 <= m <= 8:
        raise ValueError("Exhaustive analysis supports lengths 1..8")
    return np.array(list(product(range(SIGMA), repeat=m)), dtype=np.int32)


def additive_projection(values: np.ndarray, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Orthogonal projection under the uniform product distribution.

    Input order must match enumerate_windows(m). Each centered row is
    E[f | W_i=c] - E[f]. Independence makes these row effects orthogonal;
    their sum plus the mean is the least-squares additive approximation.
    This is a decomposition of the supplied function, not a held-out fit.
    """
    values = np.asarray(values, dtype=np.float64)
    if m < 1 or values.shape != (SIGMA ** m,):
        raise ValueError("Expected one scalar per lexicographically ordered window")
    cube = values.reshape((SIGMA,) * m)
    mean = float(values.mean())
    effects = np.empty((m, SIGMA))
    approximation = np.full(cube.shape, mean)
    for i in range(m):
        axes = tuple(j for j in range(m) if j != i)
        effects[i] = cube.mean(axis=axes) - mean
        shape = [1] * m
        shape[i] = SIGMA
        approximation += effects[i].reshape(shape)
    return mean, effects, approximation.ravel()


def describe_function(values: np.ndarray, pattern: str, windows: np.ndarray) -> dict:
    """Separate score geometry, threshold decisions, and exact separability."""
    values = np.asarray(values, dtype=np.float64)
    mean, effects, additive = additive_projection(values, len(pattern))
    residual = values - additive
    target = np.array([CHAR_TO_IDX[c] for c in pattern])
    distances = np.sum(windows != target, axis=1)
    positive = distances == 0
    decisions = values > 0  # Same convention as analysis.metrics_at_threshold.
    accepted = np.flatnonzero(decisions)
    worst = np.argsort(-np.abs(residual), kind="stable")[:5]
    variance = float(np.var(values))
    gap = float(values[positive][0] - values[~positive].max())
    additive_gap = float(additive[positive][0] - additive[~positive].max())
    def word(index):
        return "".join(ALPHABET[c] for c in windows[index])
    return {
        "false_positives": int(np.sum(decisions & ~positive)),
        "false_negatives": int(np.sum(~decisions & positive)),
        "accepted_windows": [word(i) for i in accepted],
        "false_positives_by_hamming": {
            str(d): int(np.sum(decisions & (distances == d)))
            for d in range(1, len(pattern) + 1)
        },
        "pattern_logit": float(values[positive][0]),
        "max_nonmatch_logit": float(values[~positive].max()),
        "separation_gap": gap,
        # Existential diagnostic only: no threshold selected on this domain.
        "some_threshold_is_exact": gap > 0,
        "additive_separation_gap": additive_gap,
        "additive_some_threshold_is_exact": additive_gap > 0,
        "interaction_variance_fraction": float(np.mean(residual ** 2) / variance)
        if variance > 0 else None,
        "max_additive_residual": float(np.max(np.abs(residual))),
        "additive_decision_disagreements": int(np.sum(decisions != (additive > 0))),
        "additive_false_positives": int(np.sum((additive > 0) & ~positive)),
        "additive_false_negatives": int(np.sum((additive <= 0) & positive)),
        "additive_mean": mean,
        "additive_effects": effects.tolist(),
        "worst_residual_windows": [
            {"window": word(i), "logit": float(values[i]),
             "additive_logit": float(additive[i]), "residual": float(residual[i])}
            for i in worst
        ],
    }

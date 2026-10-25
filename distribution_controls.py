"""Matched interventions on one-mismatch negatives and exact lookup costs.

Training examples are isolated windows, not overlapping windows from texts.
All conditions share labels, positive/background examples, and negative
support. Only the frequency of hard-negative mismatch positions changes.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from exhaustive_analysis import enumerate_windows
from patterns import ALPHABET, CHAR_TO_IDX, SIGMA


CONDITIONS = ("balanced", "focus_first", "focus_last")


def target_indices(pattern: str) -> np.ndarray:
    if len(pattern) != 5 or set(pattern) - set(ALPHABET):
        raise ValueError("This matched protocol requires five symbols from ABCD")
    return np.array([CHAR_TO_IDX[c] for c in pattern], dtype=np.int32)


def position_probabilities(condition: str) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown intervention: {condition}")
    if condition == "balanced":
        return np.full(5, .2)
    probabilities = np.full(5, .05)
    probabilities[0 if condition == "focus_first" else 4] = .8
    return probabilities


@dataclass
class InterventionDataset:
    windows: np.ndarray
    labels: np.ndarray
    source: np.ndarray  # 0: positive, 1: shared random negative, 2: hard negative

    @property
    def x(self):
        return np.eye(SIGMA, dtype=np.float32)[self.windows]


def build_intervention_dataset(pattern: str, condition: str, seed: int = 0) -> InterventionDataset:
    """2,400 examples: 120 positives, 1,140 background, 1,140 one-mismatch.

    Background offsets are uniform over all nonzero base-four codes; adding
    them modulo four to P makes the random draw coupled across patterns.
    Hard-negative counts per substitute are exact: 76 per position when
    balanced; 304 at the focused position and 19 at each other position.
    """
    target = target_indices(pattern)
    probabilities = position_probabilities(condition)
    rng = np.random.default_rng(seed)
    codes = rng.integers(1, SIGMA ** 5, size=1140)
    offsets = (codes[:, None] // SIGMA ** np.arange(4, -1, -1)) % SIGMA
    background = (target + offsets) % SIGMA
    hard = []
    for i, probability in enumerate(probabilities):
        count = int(round(1140 * probability / (SIGMA - 1)))
        for offset in range(1, SIGMA):
            window = target.copy()
            window[i] = (window[i] + offset) % SIGMA
            hard.extend([window] * count)
    hard = np.array(hard, dtype=np.int32)[rng.permutation(1140)]
    windows = np.concatenate([np.tile(target, (120, 1)), background, hard]).astype(np.int32)
    labels = np.zeros((2400, 1), dtype=np.float32)
    labels[:120] = 1
    source = np.repeat([0, 1, 2], [120, 1140, 1140]).astype(np.int8)
    # Same permutation in every condition: class/source membership per batch
    # is coupled even though the hard-negative content changes.
    order = rng.permutation(2400)
    return InterventionDataset(windows[order], labels[order], source[order])


def evaluation_probabilities(pattern: str, condition: str, windows: np.ndarray) -> np.ndarray:
    """Exact population mixture, independent of the finite training sample.

    5% P, 47.5% uniform nonmatches, 47.5% one-mismatch negatives. Also supports
    a fully uniform window distribution (positive mass 1/1024, not 5%).
    """
    target = target_indices(pattern)
    if condition == "uniform":
        return np.full(len(windows), 1 / len(windows))
    probabilities = position_probabilities(condition)
    mismatches = windows != target
    distances = mismatches.sum(axis=1)
    weights = np.where(distances == 0, .05, .475 / (SIGMA ** 5 - 1))
    for i in range(5):
        weights[(distances == 1) & mismatches[:, i]] += .475 * probabilities[i] / (SIGMA - 1)
    return weights


def single_substitution_drops(values: np.ndarray, pattern: str) -> np.ndarray:
    """f(P)-f(P with one substituted symbol), for either network architecture.

    Only for a single additive filter does this equal a context-independent
    penalty table. For a nonlinear network this measures effects around P.
    """
    target = target_indices(pattern)
    powers = SIGMA ** np.arange(4, -1, -1)
    target_id = int(target @ powers)
    drops = np.empty((5, SIGMA))
    for i in range(5):
        ids = target_id + (np.arange(SIGMA) - target[i]) * powers[i]
        drops[i] = values[target_id] - values[ids]
    return drops


def window_costs(windows: np.ndarray, pattern: str, order: tuple[int, ...]) -> np.ndarray:
    """Exact matching, stride one, early exit in the specified comparison order.

    No BM shifts: the earlier shift rule cannot simply be combined with an
    arbitrary comparison permutation. Cost counts only equality tests.
    """
    if sorted(order) != list(range(5)):
        raise ValueError("Comparison order must permute positions 0..4")
    mismatches = (windows != target_indices(pattern))[:, order]
    return np.where(mismatches.any(axis=1), mismatches.argmax(axis=1) + 1, 5)


def comparison_order_analysis(
    values: np.ndarray, pattern: str, dataset: InterventionDataset,
    windows: np.ndarray | None = None,
) -> dict:
    """Use neural drops only to choose order; equality still decides matches.

    Compare to an empirical mismatch-frequency heuristic and the best of all
    120 fixed orders under each evaluation distribution. The oracle uses
    evaluation probabilities; it is a reference bound, not a learned result.
    """
    if windows is None:
        windows = enumerate_windows(5)
    target = target_indices(pattern)
    mask = np.arange(SIGMA)[None, :] != target[:, None]
    drops = single_substitution_drops(values, pattern)
    mean_drops = drops[mask].reshape(5, SIGMA - 1).mean(axis=1)
    observed_mismatch_frequency = (dataset.windows != target).mean(axis=0)
    orders = {
        "left_to_right": tuple(range(5)),
        "right_to_left": tuple(reversed(range(5))),
        "neural_drop": tuple(np.argsort(-mean_drops, kind="stable").tolist()),
        "empirical_mismatch_frequency": tuple(np.argsort(-observed_mismatch_frequency, kind="stable").tolist()),
    }
    all_orders = list(permutations(range(5)))
    all_costs = np.array([window_costs(windows, pattern, order) for order in all_orders])
    distributions = {}
    for condition in (*CONDITIONS, "uniform"):
        weights = evaluation_probabilities(pattern, condition, windows)
        expected = all_costs @ weights
        oracle_id = int(np.argmin(expected))
        distributions[condition] = {
            "costs": {label: float(window_costs(windows, pattern, order) @ weights)
                      for label, order in orders.items()},
            "best_fixed_order_cost": float(expected[oracle_id]),
            "best_fixed_order": all_orders[oracle_id],
        }
    return {"orders": orders, "mean_substitution_drop": mean_drops.tolist(),
            "observed_mismatch_frequency": observed_mismatch_frequency.tolist(),
            "evaluation_distributions": distributions}

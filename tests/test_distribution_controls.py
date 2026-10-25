"""Check the experimental intervention and its exact cost model."""

from itertools import permutations

import numpy as np
import pytest

from distribution_controls import (
    CONDITIONS, build_intervention_dataset, comparison_order_analysis,
    evaluation_probabilities, single_substitution_drops, target_indices,
    window_costs,
)
from exhaustive_analysis import enumerate_windows


@pytest.mark.parametrize("pattern", ["ABABC", "ABABA"])
def test_interventions_only_change_hard_negative_frequencies(pattern):
    target = target_indices(pattern)
    datasets = [build_intervention_dataset(pattern, c) for c in CONDITIONS]
    baseline = datasets[0]
    expected_counts = ([228]*5, [912,57,57,57,57], [57,57,57,57,912])
    for ds, counts in zip(datasets, expected_counts):
        np.testing.assert_array_equal(ds.labels, baseline.labels)
        np.testing.assert_array_equal(ds.source, baseline.source)
        np.testing.assert_array_equal(ds.windows[ds.source != 2], baseline.windows[baseline.source != 2])
        np.testing.assert_array_equal(np.unique(ds.windows, axis=0), np.unique(baseline.windows, axis=0))
        np.testing.assert_array_equal(ds.labels[:,0], np.all(ds.windows == target, axis=1))
        assert ds.labels.sum() == 120
        hard = ds.windows[ds.source == 2] != target
        np.testing.assert_array_equal(hard.sum(axis=1), 1)
        np.testing.assert_array_equal(hard.sum(axis=0), counts)
        np.testing.assert_array_equal(ds.x.argmax(axis=2), ds.windows)


def test_population_distributions_and_uniform_cost_invariance():
    windows = enumerate_windows(5)
    target = target_indices("ABABC")
    positive = np.all(windows == target, axis=1)
    for condition in (*CONDITIONS, "uniform"):
        weights = evaluation_probabilities("ABABC", condition, windows)
        assert np.all(weights > 0)
        assert weights.sum() == pytest.approx(1)
        assert weights[positive].sum() == pytest.approx(1/1024 if condition == "uniform" else .05)
    weights = evaluation_probabilities("ABABC", "uniform", windows)
    for order in permutations(range(5)):
        assert window_costs(windows, "ABABC", order) @ weights == pytest.approx(sum(4**(-k) for k in range(5)))


def test_lookup_costs_against_independent_scalar_search():
    windows = enumerate_windows(5)
    target = target_indices("ABABA")
    for order in ((0,1,2,3,4), (4,3,2,1,0), (2,0,4,1,3)):
        expected = []
        for window in windows:
            comparisons = 0
            for position in order:
                comparisons += 1
                if window[position] != target[position]:
                    break
            expected.append(comparisons)
        np.testing.assert_array_equal(window_costs(windows, "ABABA", order), expected)
    with pytest.raises(ValueError):
        window_costs(windows, "ABABA", (0,0,0,0,0))


def test_substitution_drops_and_optimal_order_control():
    windows = enumerate_windows(5)
    target = target_indices("ABABA")
    penalty = np.array([2., 3., 4., 5., 9.])
    scores = .5 - ((windows != target) * penalty).sum(axis=1)
    drops = single_substitution_drops(scores, "ABABA")
    np.testing.assert_array_equal(drops[np.arange(5), target], 0)
    for position in range(5):
        np.testing.assert_array_equal(np.delete(drops[position], target[position]), penalty[position])
    ds = build_intervention_dataset("ABABA", "focus_last")
    analysis = comparison_order_analysis(scores, "ABABA", ds, windows)
    assert analysis["orders"]["neural_drop"] == (4,3,2,1,0)
    matched = analysis["evaluation_distributions"]["focus_last"]
    assert matched["costs"]["neural_drop"] == pytest.approx(matched["best_fixed_order_cost"])
    # Moving the frequent hard mismatch from last to first saves four tests
    # on 80% of hard negatives, offset by the other positions' 5% masses.
    assert matched["costs"]["left_to_right"]-matched["best_fixed_order_cost"] == pytest.approx(.475*3.)

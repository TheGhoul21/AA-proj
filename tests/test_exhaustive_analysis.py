"""Independent mathematical and exhaustive correctness checks for experiment 7."""

from itertools import product

import numpy as np
import pytest

from exact_search import bad_character_search, kmp_counted, naive_search
from exhaustive_analysis import additive_projection, describe_function, enumerate_windows
from patterns import SIGMA


def test_projection_agrees_with_independent_least_squares():
    windows = enumerate_windows(3)
    values = np.random.default_rng(12).normal(size=len(windows))
    design = np.column_stack([np.ones(len(windows)), np.eye(SIGMA)[windows].reshape(len(windows), -1)])
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    mean, effects, projected = additive_projection(values, 3)
    np.testing.assert_allclose(projected, fitted, atol=1e-14)
    np.testing.assert_allclose(effects.mean(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(design.T @ (values - projected), 0, atol=1e-13)
    assert mean == pytest.approx(values.mean())


def test_projection_detects_pairwise_interaction_and_preserves_additive_scores():
    windows = enumerate_windows(2)
    additive = 2.5 + windows[:, 0] - 3 * windows[:, 1]
    np.testing.assert_allclose(additive_projection(additive, 2)[2], additive)
    interaction = (windows[:, 0] - 1.5) * (windows[:, 1] - 1.5)
    np.testing.assert_allclose(additive_projection(additive + interaction, 2)[2], additive)
    np.testing.assert_allclose(additive + interaction - additive_projection(additive + interaction, 2)[2], interaction)


def test_exact_template_and_constant_function():
    windows = enumerate_windows(2)
    scores = .5 - np.sum(windows != np.array([0, 1]), axis=1)
    report = describe_function(scores, "AB", windows)
    assert report["accepted_windows"] == ["AB"]
    assert report["false_positives"] == report["false_negatives"] == 0
    assert report["some_threshold_is_exact"]
    assert report["additive_some_threshold_is_exact"]
    assert report["interaction_variance_fraction"] == 0
    assert report["additive_decision_disagreements"] == 0
    constant = describe_function(np.zeros(len(windows)), "AB", windows)
    assert constant["interaction_variance_fraction"] is None
    assert not constant["some_threshold_is_exact"]


def test_score_separation_is_not_the_same_as_correct_threshold():
    windows = enumerate_windows(2)
    # Pattern is uniquely highest, but every score is negative at threshold 0.
    scores = -1. - np.sum(windows != np.array([0, 1]), axis=1)
    report = describe_function(scores, "AB", windows)
    assert report["false_negatives"] == 1
    assert report["additive_false_negatives"] == 1
    assert report["some_threshold_is_exact"]
    assert report["additive_some_threshold_is_exact"]


@pytest.mark.parametrize("search", [naive_search, kmp_counted, bad_character_search])
def test_scanners_exhaustively_against_string_slices(search):
    # Includes short/empty texts, overlaps, and full matches at both boundaries.
    for m in range(1, 5):
        for letters in product("AB", repeat=m):
            pattern = "".join(letters)
            for n in range(8):
                for symbols in product("AB", repeat=n):
                    text = "".join(symbols)
                    expected = [i for i in range(n - m + 1) if text[i:i + m] == pattern]
                    found, count = search(text, pattern)
                    assert found == expected, (search.__name__, text, pattern)
                    assert count >= 0
                    if search is kmp_counted:
                        assert count <= 2 * n


def test_known_comparison_counts():
    assert naive_search("AAAAAA", "AAAA") == ([0, 1, 2], 12)
    assert kmp_counted("AAAAAA", "AAAA") == ([0, 1, 2], 6)
    assert bad_character_search("AAAAAA", "AAAA") == ([0, 1, 2], 12)
    assert bad_character_search("DDDDDD", "ABC") == ([], 2)


def test_observer_does_not_change_training():
    import jax
    from data import build_dataset
    from model import init_params
    from train import train

    ds = build_dataset("ABABC", 4, 30, 1, 1, seed=0)
    params = init_params(jax.random.PRNGKey(3), 5, SIGMA)
    snapshots = []
    kwargs = dict(n_epochs=3, batch_size=2, lr=.01, pos_weight=2., log_every=3)
    plain, losses = train(params, ds.x, ds.y, **kwargs)
    observed, observed_losses = train(
        params, ds.x, ds.y, **kwargs,
        on_epoch=lambda epoch, p: snapshots.append((epoch, {k: np.array(v) for k, v in p.items()})),
    )
    assert [epoch for epoch, _ in snapshots] == [0, 1, 2, 3]
    np.testing.assert_array_equal(losses, observed_losses)
    for key in params:
        np.testing.assert_array_equal(plain[key], observed[key])
        np.testing.assert_array_equal(snapshots[0][1][key], params[key])
        np.testing.assert_array_equal(snapshots[-1][1][key], observed[key])

"""Unit tests for experiment5.py matchers:
- matcher_naive (Algorithm A)
- matcher_early_exit (Algorithm B)
- matcher_early_exit_bm (Algorithm C)

Verifies:
1. All three matchers return identical accept positions at tau=0 (exact matching).
2. Accept positions match ground-truth exact pattern occurrences (kmp_search).
3. Early-exit and Boyer-Moore optimizations reduce operation counts (ops_A >= ops_B >= ops_C).
"""

from __future__ import annotations

import os
# Ensure matplotlib does not attempt to write to non-writable dirs
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pytest

from experiment5 import (
    PATTERN,
    build_penalty_table,
    build_shift_table,
    encode_text,
    generate_long_text,
    matcher_early_exit,
    matcher_early_exit_bm,
    matcher_naive,
    train_filter,
)
from patterns import CHAR_TO_IDX, SIGMA, kmp_search


def make_synthetic_penalty_and_shift(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct an exact penalty table and shift table for a given pattern."""
    m = len(pattern)
    w = np.zeros((m, SIGMA), dtype=np.float32)
    for i, c in enumerate(pattern):
        w[i, CHAR_TO_IDX[c]] = 10.0
    pen, p_rec = build_penalty_table(w)
    assert p_rec == pattern
    shift = build_shift_table(pen, eta=0.0)
    return pen, shift


@pytest.fixture(scope="session")
def trained_filter_setup():
    """Train filter once for PATTERN to test with real learned weights."""
    w, b = train_filter(PATTERN)
    pen, p_rec = build_penalty_table(w)
    assert p_rec == PATTERN
    shift = build_shift_table(pen, eta=0.0)
    return pen, shift


@pytest.mark.parametrize("seed", [42, 101, 2026, 777, 9999])
def test_matchers_identical_accepts_synthetic(seed: int) -> None:
    """Verify naive, early_exit, and early_exit_bm yield identical accept positions at tau=0."""
    pen, shift = make_synthetic_penalty_and_shift(PATTERN)
    text, _ = generate_long_text(
        pattern=PATTERN,
        length=2000,
        n_complete=10,
        n_partial=30,
        seed=seed,
    )
    tidx = encode_text(text)

    pa, oa = matcher_naive(tidx, pen, tau=0.0)
    pb, ob = matcher_early_exit(tidx, pen, tau=0.0)
    pc, oc = matcher_early_exit_bm(tidx, pen, shift, tau=0.0)

    # All three must produce identical accept lists
    assert pa == pb, f"Naive and early-exit differed at seed {seed}"
    assert pa == pc, f"Naive and early-exit+BM differed at seed {seed}"

    # Also check against reference kmp_search
    expected = kmp_search(text, PATTERN)
    assert pa == expected, f"Accept positions do not match kmp_search at seed {seed}"

    # Operations hierarchy: naive >= early_exit >= early_exit_bm
    assert oa >= ob, f"Expected oa ({oa}) >= ob ({ob})"
    assert ob >= oc, f"Expected ob ({ob}) >= oc ({oc})"


def test_matchers_identical_accepts_trained(trained_filter_setup) -> None:
    """Verify equivalence on multiple seeds using the real trained filter."""
    pen, shift = trained_filter_setup

    for seed in [123, 456, 789]:
        text, _ = generate_long_text(
            pattern=PATTERN,
            length=3000,
            n_complete=15,
            n_partial=40,
            seed=seed,
        )
        tidx = encode_text(text)

        pa, oa = matcher_naive(tidx, pen, tau=0.0)
        pb, ob = matcher_early_exit(tidx, pen, tau=0.0)
        pc, oc = matcher_early_exit_bm(tidx, pen, shift, tau=0.0)

        assert pa == pb == pc
        expected = kmp_search(text, PATTERN)
        assert pa == expected
        assert oa >= ob >= oc


@pytest.mark.parametrize("pattern", ["AAAA", "ABABA", "ABCD", "A"])
def test_matchers_on_different_patterns(pattern: str) -> None:
    """Test matcher equivalence on other patterns."""
    pen, shift = make_synthetic_penalty_and_shift(pattern)
    rng = np.random.default_rng(42)
    # Generate random text with some planted patterns
    alphabet = list(CHAR_TO_IDX.keys())
    chunks = []
    for _ in range(50):
        chunks.append("".join(rng.choice(alphabet, size=rng.integers(1, 10))))
        chunks.append(pattern)
    text = "".join(chunks)
    tidx = encode_text(text)

    pa, oa = matcher_naive(tidx, pen, tau=0.0)
    pb, ob = matcher_early_exit(tidx, pen, tau=0.0)
    pc, oc = matcher_early_exit_bm(tidx, pen, shift, tau=0.0)

    assert pa == pb == pc
    expected = kmp_search(text, pattern)
    assert pa == expected


def test_matchers_edge_cases() -> None:
    """Test text shorter than pattern, text equal to pattern, and text with zero occurrences."""
    pen, shift = make_synthetic_penalty_and_shift("ABABC")

    # Short text
    tidx_short = encode_text("ABA")
    pa, _ = matcher_naive(tidx_short, pen, tau=0.0)
    pb, _ = matcher_early_exit(tidx_short, pen, tau=0.0)
    pc, _ = matcher_early_exit_bm(tidx_short, pen, shift, tau=0.0)
    assert pa == pb == pc == []

    # Exactly matching text
    tidx_exact = encode_text("ABABC")
    pa, _ = matcher_naive(tidx_exact, pen, tau=0.0)
    pb, _ = matcher_early_exit(tidx_exact, pen, tau=0.0)
    pc, _ = matcher_early_exit_bm(tidx_exact, pen, shift, tau=0.0)
    assert pa == pb == pc == [0]

    # No matches text
    tidx_none = encode_text("DDDDDDDDDD")
    pa, _ = matcher_naive(tidx_none, pen, tau=0.0)
    pb, _ = matcher_early_exit(tidx_none, pen, tau=0.0)
    pc, _ = matcher_early_exit_bm(tidx_none, pen, shift, tau=0.0)
    assert pa == pb == pc == []

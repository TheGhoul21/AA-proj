"""Unit tests for the penalty table algebraic identity:
    logit(W) = pi - sum_i pen[i, W[i]]

Properties to verify:
1. pen[i, c] >= 0 everywhere.
2. pen[i, P_i] == 0 for all positions i (where P_i = argmax_c w[i, c]).
3. logit(W) == pi - sum_i pen[i, W[i]] within error < 1e-5 on random filters and windows.
4. Compatibility with JAX model forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model import forward
from patterns import ALPHABET, CHAR_TO_IDX, SIGMA


def compute_penalty_and_pi(w: np.ndarray, b: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute the penalty table and pattern logit pi from filter w and bias b."""
    P_idx = w.argmax(axis=1)
    pen = np.stack([w[i, P_idx[i]] - w[i, :] for i in range(w.shape[0])])
    pi = float(b + sum(w[i, P_idx[i]] for i in range(w.shape[0])))
    return pen, pi, P_idx


def test_penalty_nonnegative_and_zero_at_pattern() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        m = rng.integers(1, 30)
        sigma = rng.integers(2, 10)
        w = rng.standard_normal(size=(m, sigma))
        b = float(rng.standard_normal())

        pen, pi, P_idx = compute_penalty_and_pi(w, b)

        # pen[i, c] >= 0 everywhere
        assert np.all(pen >= -1e-12), f"Found negative penalty: {pen.min()}"

        # pen[i, P_i] == 0 for all i
        for i in range(m):
            assert abs(pen[i, P_idx[i]]) < 1e-7, (
                f"pen[{i}, {P_idx[i]}] = {pen[i, P_idx[i]]} != 0"
            )


@pytest.mark.parametrize("m,sigma", [(1, 4), (5, 4), (8, 4), (10, 6), (20, 2)])
def test_algebraic_identity_random_windows(m: int, sigma: int) -> None:
    rng = np.random.default_rng(123)
    for _ in range(10):
        w = rng.standard_normal(size=(m, sigma))
        b = float(rng.standard_normal())
        pen, pi, P_idx = compute_penalty_and_pi(w, b)

        # Sample 50 random windows
        for _ in range(50):
            chars = rng.integers(0, sigma, size=m)
            # One-hot window
            window_onehot = np.zeros((m, sigma), dtype=np.float32)
            for i, c in enumerate(chars):
                window_onehot[i, c] = 1.0

            # Direct linear filter score: sum(w * window) + b
            direct_logit = float(np.sum(w * window_onehot) + b)

            # Reconstructed from penalty table: pi - sum_i pen[i, chars[i]]
            recon_logit = float(pi - sum(pen[i, chars[i]] for i in range(m)))

            diff = abs(direct_logit - recon_logit)
            assert diff < 1e-5, f"Identity error {diff:.2e} >= 1e-5 for direct={direct_logit}, recon={recon_logit}"


def test_identity_with_jax_model_forward() -> None:
    """Verify that JAX model.forward matches the penalty table reconstruction."""
    rng = np.random.default_rng(2026)
    m = 5
    sigma = SIGMA
    batch_size = 4
    seq_len = 30

    w = rng.standard_normal(size=(m, sigma)).astype(np.float32)
    b = float(rng.standard_normal())
    pen, pi, P_idx = compute_penalty_and_pi(w, b)

    # Generate random sequence of character indices
    char_indices = rng.integers(0, sigma, size=(batch_size, seq_len))
    # One-hot encode
    x = np.eye(sigma, dtype=np.float32)[char_indices]  # [B, seq_len, sigma]

    # Model forward pass
    params = {"w": jnp.asarray(w), "b": jnp.array(b)}
    logits = np.array(forward(params, jnp.asarray(x)))  # [B, seq_len - m + 1]

    n_pos = seq_len - m + 1
    for b_idx in range(batch_size):
        for j in range(n_pos):
            recon = pi - sum(pen[i, char_indices[b_idx, j + i]] for i in range(m))
            err = abs(logits[b_idx, j] - recon)
            assert err < 1e-5, f"Mismatch at batch {b_idx}, pos {j}: error={err:.2e}"


def test_identity_tied_argmax() -> None:
    """Ties in argmax should still satisfy pen >= 0 and logit identity."""
    # Construct a filter with equal values across two characters
    w = np.array([[2.0, 2.0, 1.0, 0.0], [0.5, 1.5, 1.5, 0.0]], dtype=np.float32)
    b = 0.5
    pen, pi, P_idx = compute_penalty_and_pi(w, b)

    assert np.all(pen >= -1e-12)
    assert abs(pen[0, P_idx[0]]) < 1e-7
    assert abs(pen[1, P_idx[1]]) < 1e-7

    # Test all possible 2-char sequences over 4 characters
    for c0 in range(4):
        for c1 in range(4):
            direct = float(w[0, c0] + w[1, c1] + b)
            recon = float(pi - (pen[0, c0] + pen[1, c1]))
            assert abs(direct - recon) < 1e-5

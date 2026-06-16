"""Experiment 4: extract algorithmically-computable structures from a trained
single-filter CNN, and compare them with KMP's sp and the Z-array.

For a trained filter w of shape [m, sigma] and bias b, the logit on a
size-m window is exactly

    logit(window) = sum_{i=0..m-1} w[i, window[i]] + b

so the CNN's behaviour on every possible window is fully determined by w
and b. From these we can derive several deterministic structures:

  * P_recovered    = argmax(w, axis=1)                         shape [m]
  * diag           = w[i, P[i]]                                shape [m]
  * margin         = w[i, P[i]] - max_{c != P[i]} w[i, c]      shape [m]
  * cum_ratio[j]   = sum_{i<=j} diag[i] / sum_i diag[i]
  * near_miss[i,c] = logit if we replace P[i] with c           shape [m, sigma]
  * one_off_drop[i] = pattern_logit - max_{c != P[i]} near_miss[i, c]

We then check, for both ABABC and ABABA, whether any of these structures
has a regular (algorithmic) relationship with sp = KMP failure or Z-array.

To make the analysis comparable across patterns we re-train identical
single-filter models on both, with identical hyper-parameters and seeds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from analysis import filter_argmax_string, show_filter
from data import CHAR_TO_IDX, SIGMA, build_dataset
from model import forward, init_params
from patterns import ALPHABET, kmp_failure, z_array
from train import train


TARGET_LENGTH = 50
N_TRAIN = 2000
N_COMPLETE = 2
N_PARTIAL = 4
N_EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-2
POS_WEIGHT = 20.0
SEED = 42


def train_single_filter(pattern: str) -> tuple[np.ndarray, float]:
    train_ds = build_dataset(
        pattern, n_samples=N_TRAIN, target_length=TARGET_LENGTH,
        n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=0,
    )
    key = jax.random.PRNGKey(SEED)
    params = init_params(key, m=len(pattern), sigma=SIGMA)
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, pos_weight=POS_WEIGHT, seed=0, log_every=80,
    )
    return np.array(params["w"]), float(params["b"])


def derive_structures(pattern: str, w: np.ndarray, b: float) -> dict:
    m = len(pattern)
    sigma = w.shape[1]
    P_idx = np.array([CHAR_TO_IDX[c] for c in pattern])

    # Reconstructed pattern
    P_recovered = filter_argmax_string(w)

    # Diagonal: score of the "right" char at each position
    diag = np.array([w[i, P_idx[i]] for i in range(m)])

    # Margin per position
    margin = np.empty(m)
    for i in range(m):
        right = w[i, P_idx[i]]
        others = np.delete(w[i], P_idx[i])
        margin[i] = right - others.max()

    # Cumulative ratio
    cum = np.cumsum(diag)
    pattern_score = float(cum[-1])
    cum_ratio = cum / max(1e-9, pattern_score)
    pattern_logit = pattern_score + b

    # Near-miss table: logit if we replace P[i] with c
    near_miss = np.empty((m, sigma))
    for i in range(m):
        for c_idx in range(sigma):
            # Replacing position i changes only that term.
            near_miss[i, c_idx] = pattern_logit - w[i, P_idx[i]] + w[i, c_idx]

    # Drop relative to TP (always non-negative for c != P[i] if w[i, P[i]] is argmax)
    drop = pattern_logit - near_miss

    # One-off drop: best 1-mismatch alternative per position
    one_off_drop = np.empty(m)
    for i in range(m):
        mask = np.ones(sigma, dtype=bool)
        mask[P_idx[i]] = False
        one_off_drop[i] = drop[i, mask].min()  # smallest drop = closest to TP

    return {
        "pattern": pattern,
        "P_recovered": P_recovered,
        "sp": kmp_failure(pattern),
        "Z": z_array(pattern),
        "diag": diag,
        "margin": margin,
        "cum_ratio": cum_ratio,
        "pattern_logit": pattern_logit,
        "near_miss": near_miss,
        "drop": drop,
        "one_off_drop": one_off_drop,
        "w": w,
        "b": b,
    }


def report(s: dict) -> None:
    p = s["pattern"]
    m = len(p)
    print(f"\n{'=' * 64}")
    print(f"Pattern: {p}  (recovered: {s['P_recovered']})  ", end="")
    print("OK" if s["P_recovered"] == p else "MISMATCH")
    print(f"  sp:  {s['sp']}")
    print(f"  Z:   {s['Z']}")
    print(f"  bias b = {s['b']:+.3f}    pattern_logit (TP score) = {s['pattern_logit']:+.3f}")
    print()
    print(f"  {'pos':>3}  {'P[i]':>4}  {'sp[i]':>5}  {'Z[i]':>4}  "
          f"{'diag':>6}  {'margin':>6}  {'cum_ratio':>9}  {'one_off_drop':>12}")
    for i in range(m):
        print(f"  {i:>3}  {p[i]:>4}  {s['sp'][i]:>5}  {s['Z'][i]:>4}  "
              f"{s['diag'][i]:+6.2f}  {s['margin'][i]:+6.2f}  "
              f"{s['cum_ratio'][i]:>9.3f}  {s['one_off_drop'][i]:+12.2f}")

    # Near-miss table
    print(f"\n  Near-miss logit table (replace P[i] with column char):")
    print("       " + "    ".join(ALPHABET))
    for i in range(m):
        row = "  ".join(f"{v:+5.2f}" for v in s["near_miss"][i])
        print(f"  pos {i} ({p[i]}): {row}")

    # Drop table
    print(f"\n  Drop from TP logit (= TP - near_miss):")
    print("       " + "    ".join(ALPHABET))
    for i in range(m):
        row = "  ".join(f"{v:+5.2f}" for v in s["drop"][i])
        print(f"  pos {i} ({p[i]}): {row}")


def cross_pattern_diagnostic(s_list: list[dict]) -> None:
    print(f"\n\n{'=' * 64}")
    print("  Cross-pattern: relation between sp[i] and the learned margin[i]")
    print(f"{'=' * 64}")
    print(f"  {'pattern':>8}  {'pos':>3}  {'sp[i]':>5}  {'Z[i]':>4}  "
          f"{'margin':>7}  {'one_off_drop':>12}")
    for s in s_list:
        for i, _ in enumerate(s["pattern"]):
            print(f"  {s['pattern']:>8}  {i:>3}  {s['sp'][i]:>5}  {s['Z'][i]:>4}  "
                  f"{s['margin'][i]:+7.2f}  {s['one_off_drop'][i]:+12.2f}")
        print()

    # Pearson correlations across all (pattern, pos) tuples
    sp_all, z_all, margin_all, drop_all = [], [], [], []
    for s in s_list:
        sp_all.extend(s["sp"])
        z_all.extend(s["Z"])
        margin_all.extend(s["margin"])
        drop_all.extend(s["one_off_drop"])
    sp_all = np.array(sp_all)
    z_all = np.array(z_all)
    margin_all = np.array(margin_all)
    drop_all = np.array(drop_all)

    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float((a * b).sum() / denom)

    print(f"  Pearson correlations across all (pattern, pos):")
    print(f"    sp vs margin:        {pearson(sp_all, margin_all):+.3f}")
    print(f"    sp vs one_off_drop:  {pearson(sp_all, drop_all):+.3f}")
    print(f"    Z  vs margin:        {pearson(z_all, margin_all):+.3f}")
    print(f"    Z  vs one_off_drop:  {pearson(z_all, drop_all):+.3f}")


def verify_decomposition(s: dict, val_x: np.ndarray) -> None:
    """Numerically verify the additive decomposition of the CNN logit.

    For a single-filter CNN on one-hot input, the logit of any window W is
        logit(W) = b + sum_i w[i, W[i]]
    which can be rewritten via the *penalty table*  pen[i, c] = w[i, P[i]] - w[i, c]
    (>= 0 with equality at c = P[i] when P is argmax) as
        logit(W) = pattern_logit - sum_i pen[i, W[i]].

    pen is a [m, sigma] real-valued matrix; it generalises the Boyer-Moore
    bad-character rule by giving a *graded* mismatch cost per (position, char).
    """
    p = s["pattern"]
    m = len(p)
    w = s["w"]
    b = s["b"]
    P_idx = np.array([CHAR_TO_IDX[c] for c in p])
    pen = np.empty_like(w)
    for i in range(m):
        pen[i] = w[i, P_idx[i]] - w[i, :]

    print(f"\n  Penalty table  pen[i, c] = w[i, P[i]] - w[i, c]   (the structure):")
    print("       " + "    ".join(ALPHABET))
    for i in range(m):
        row = "  ".join(f"{v:+5.2f}" for v in pen[i])
        print(f"  pos {i} ({p[i]}): {row}")

    # Numerical check on a few real windows from the validation set
    print(f"\n  Numerical verification:  logit(W) = pattern_logit - sum_i pen[i, W[i]]")
    pattern_logit = float(s["pattern_logit"])
    # Run the actual CNN forward on val_x
    params = {"w": jnp.asarray(w), "b": jnp.array(b)}
    cnn_logits = np.array(forward(params, jnp.asarray(val_x)))  # [N, T-m+1]
    # Reconstruct from the penalty decomposition
    N, n_pos = cnn_logits.shape
    char_idx = val_x.argmax(axis=2)  # [N, T]
    recon = np.empty_like(cnn_logits)
    for n in range(N):
        for j in range(n_pos):
            window_idx = char_idx[n, j:j + m]
            penalty_sum = sum(pen[i, window_idx[i]] for i in range(m))
            recon[n, j] = pattern_logit - penalty_sum
    err = np.abs(cnn_logits - recon).max()
    print(f"    checked on {N} samples x {n_pos} positions = "
          f"{N * n_pos} windows.  max |CNN - decomposition| = {err:.2e}")


def multi_filter_decomposition_check(pattern: str) -> None:
    """Counter-example: train the K=4 multi-filter model from Experiment 2 and
    show that the same per-position-per-char additive decomposition does NOT
    hold (because of the ReLU non-linearity)."""
    from model import (
        forward_multi,
        forward_multi_intermediates,
        init_params_multi,
        loss_fn_multi,
    )

    print(f"\n\n{'=' * 64}")
    print(f"  Counter-example: does the additive decomposition hold for K>1+ReLU?")
    print(f"{'=' * 64}")
    train_ds = build_dataset(
        pattern, n_samples=4000, target_length=TARGET_LENGTH,
        n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=0,
    )
    val_ds = build_dataset(
        pattern, n_samples=500, target_length=TARGET_LENGTH,
        n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=1,
    )
    key = jax.random.PRNGKey(7)
    params = init_params_multi(key, K=4, m=len(pattern), sigma=SIGMA)
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=150, batch_size=64, lr=5e-3, pos_weight=POS_WEIGHT,
        seed=0, log_every=150, loss_fn=loss_fn_multi,
    )

    # Try to fit a single best per-position-per-char penalty table by least
    # squares over the validation logits, and see how big the residual is.
    val_logits = np.array(forward_multi(params, jnp.asarray(val_ds.x)))  # [N, n_pos]
    char_idx = val_ds.x.argmax(axis=2)  # [N, T]
    m = len(pattern)
    sigma = SIGMA
    N, n_pos = val_logits.shape

    # Linear system: each window contributes 1[w_i = c] indicator features.
    # Variables: a "score" S[i, c] for each (pos in kernel, char) plus a global
    # constant c0, fit so that logit(W) ~= c0 + sum_i S[i, W[i]].
    # Total features: m * sigma + 1.
    A_rows = []
    b_rows = []
    for n in range(N):
        for j in range(n_pos):
            row = np.zeros(m * sigma + 1, dtype=np.float32)
            row[-1] = 1.0
            for i in range(m):
                row[i * sigma + char_idx[n, j + i]] = 1.0
            A_rows.append(row)
            b_rows.append(val_logits[n, j])
    A = np.stack(A_rows)
    bvec = np.array(b_rows)
    sol, residuals, rank, sv = np.linalg.lstsq(A, bvec, rcond=None)
    pred = A @ sol
    err = bvec - pred
    print(f"  Best additive (PSSM-style) approximation of the multi-filter logit:")
    print(f"    R^2 of fit:                 {1 - err.var() / bvec.var():.4f}")
    print(f"    residual std:               {err.std():.2f}")
    print(f"    logit std (target):         {bvec.std():.2f}")
    print(f"    max |residual|:             {np.abs(err).max():.2f}")
    print(f"  -> if R^2 == 1 the multi-filter model would be reducible to a "
          f"PSSM scorer.")
    print(f"  -> here R^2 < 1 means the ReLU nonlinearity creates true "
          f"interactions between positions that no per-(i,c) lookup table "
          f"can capture.")


def main() -> None:
    patterns = ["ABABC", "ABABA"]
    structures = []
    val_xs = []
    for p in patterns:
        print(f"\n>>> Training single filter on pattern '{p}' ...")
        w, b = train_single_filter(p)
        structures.append(derive_structures(p, w, b))
        val_xs.append(build_dataset(
            p, n_samples=200, target_length=TARGET_LENGTH,
            n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=99,
        ).x)

    for s in structures:
        report(s)

    for s, val_x in zip(structures, val_xs):
        verify_decomposition(s, val_x)

    cross_pattern_diagnostic(structures)

    multi_filter_decomposition_check("ABABC")


if __name__ == "__main__":
    main()

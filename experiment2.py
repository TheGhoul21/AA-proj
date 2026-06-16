"""Experiment 2: same task as Experiment 1 but with K > 1 filters.

Architecture:
  K conv filters of kernel size m  ->  ReLU  ->  1x1 mixing  ->  per-position logit.

Question: does any filter specialise on a *proper prefix* of the pattern
(AB, ABA, ABAB), which would be the first trace of failure-function-like
structure distributed across the learned weights?

If a filter has learned a prefix of length j < m, we expect:
  - its row-norms drop off after position j-1 (the "tail" carries no signal)
  - its argmax string starts with pattern[:j] but the trailing rows look noisy
  - it strongly activates on partial-prefix windows in the data, not only on
    complete pattern matches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from analysis import (
    analyze_false_positives,
    filter_argmax_string,
    metrics_at_threshold,
    show_filter,
)
from data import SIGMA, build_dataset
from model import (
    forward_multi,
    forward_multi_intermediates,
    init_params_multi,
    loss_fn_multi,
)
from patterns import ALPHABET, kmp_failure, z_array
from train import train


PATTERN = "ABABC"
TARGET_LENGTH = 50
N_TRAIN = 4000
N_VAL = 500
N_COMPLETE = 2
N_PARTIAL = 4
N_EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-3
POS_WEIGHT = 20.0
K_FILTERS = 4  # one per non-trivial prefix of ABABC: AB, ABA, ABAB, ABABC


def per_row_norms(w: np.ndarray) -> np.ndarray:
    return np.linalg.norm(w, axis=1)


def top_activating_windows(
    activations_k: np.ndarray,  # [N, n_pos] for filter k
    texts: list[str],
    m: int,
    top: int = 8,
):
    """Return the top-activating windows for one filter, with their pre-ReLU score."""
    flat = activations_k.ravel()
    n_pos = activations_k.shape[1]
    order = np.argsort(-flat)[:top]
    out = []
    for idx in order:
        s = idx // n_pos
        p = idx % n_pos
        out.append((float(flat[idx]), texts[int(s)][int(p):int(p) + m], int(s), int(p)))
    return out


def main() -> None:
    print(f"Pattern: {PATTERN}    K = {K_FILTERS} filters")
    print(f"  KMP failure (sp): {kmp_failure(PATTERN)}")
    print(f"  Z-array:          {z_array(PATTERN)}")
    print(f"  Non-trivial prefixes (length 2..{len(PATTERN)}): "
          f"{[PATTERN[:j] for j in range(2, len(PATTERN) + 1)]}")

    print("\nBuilding datasets...")
    train_ds = build_dataset(
        PATTERN, n_samples=N_TRAIN, target_length=TARGET_LENGTH,
        n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=0,
    )
    val_ds = build_dataset(
        PATTERN, n_samples=N_VAL, target_length=TARGET_LENGTH,
        n_complete=N_COMPLETE, n_partial=N_PARTIAL, seed=1,
    )
    print(f"  train: x {train_ds.x.shape}, y {train_ds.y.shape}, "
          f"positive_rate={train_ds.y.mean():.4f}")

    print("\nInitializing multi-filter model...")
    key = jax.random.PRNGKey(7)
    params = init_params_multi(key, K=K_FILTERS, m=len(PATTERN), sigma=SIGMA)

    print("\nTraining...")
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, pos_weight=POS_WEIGHT, seed=0, log_every=10,
        loss_fn=loss_fn_multi,
    )

    logits_val = np.array(forward_multi(params, jnp.asarray(val_ds.x)))
    print("\nValidation metrics (threshold=0):")
    raw = metrics_at_threshold(logits_val, val_ds.y, t=0.0)
    for k, v in raw.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    W = np.array(params["W"])           # [K, m, sigma]
    b_conv = np.array(params["b_conv"]) # [K]
    W_out = np.array(params["W_out"])   # [K]
    b_out = float(params["b_out"])

    print(f"\n--- Per-filter inspection ---")
    print(f"  W_out (mixing weights): {W_out}")
    print(f"  b_out: {b_out:+.3f}")
    print(f"  b_conv: {b_conv}")

    for k in range(K_FILTERS):
        wk = W[k]
        norms = per_row_norms(wk)
        argmax_str = filter_argmax_string(wk)
        print(f"\n  Filter #{k}  W_out[{k}]={W_out[k]:+.3f}  b_conv[{k}]={b_conv[k]:+.3f}")
        print(f"    row L2-norms: {[f'{n:.2f}' for n in norms]}")
        print(f"    argmax string: {argmax_str}    (pattern: {PATTERN})")
        show_filter(wk, f"    weights of filter #{k}:")

    # --- Specialisation probe: top-activating windows per filter ---
    print("\n--- Top-activating windows per filter (val set, pre-ReLU) ---")
    _logits_jax, conv_pre_relu = forward_multi_intermediates(
        params, jnp.asarray(val_ds.x)
    )
    conv_pre_relu = np.array(conv_pre_relu)  # [N, n_pos, K]
    m = len(PATTERN)
    for k in range(K_FILTERS):
        act_k = conv_pre_relu[:, :, k]
        tops = top_activating_windows(act_k, val_ds.texts, m, top=8)
        print(f"\n  Filter #{k}:")
        for score, win, s, p in tops:
            note = ""
            if win == PATTERN:
                note = "  <-- exact pattern"
            else:
                # Note any prefix overlap
                for j in range(m, 0, -1):
                    if win.startswith(PATTERN[:j]):
                        note = f"  <-- starts with prefix '{PATTERN[:j]}' (len {j})"
                        break
            print(f"    score={score:+5.2f}  window={win!r}  "
                  f"(sample {s:4d} pos {p:2d}){note}")

    # --- Activation profile on complete match vs each partial prefix ---
    # For each pattern[:j] + 'X' built window, we ask which filter fires the most.
    print("\n--- Activation pattern on diagnostic windows ---")
    print("(Built by hand: a complete match, and each partial prefix followed")
    print(" by a non-extending char. We pad to length m by appending random")
    print(" letters so we can run a single conv at one position.)")

    def build_diag(prefix: str, mismatch_char: str) -> str:
        # Use 'D' as filler since 'D' rarely participates in the pattern (ABABC).
        s = prefix + mismatch_char
        return s + "D" * (m - len(s))

    diag = [("complete  ", PATTERN)]
    for j in range(1, m):
        forbidden = PATTERN[j]
        bad = next(c for c in ALPHABET if c != forbidden)
        win = build_diag(PATTERN[:j], bad)
        diag.append((f"prefix[{j}]+{bad} ", win))

    # Compute per-filter pre-ReLU activation on each diagnostic window
    print(f"\n  {'window':>16}  " + "  ".join(f"f#{k}" for k in range(K_FILTERS))
          + "    final logit")
    for label, win in diag:
        x_one = np.zeros((1, m, SIGMA), dtype=np.float32)
        for i, c in enumerate(win):
            from data import CHAR_TO_IDX
            x_one[0, i, CHAR_TO_IDX[c]] = 1.0
        # Pad x to length m so forward_multi works with n_pos=1
        logit, conv = forward_multi_intermediates(params, jnp.asarray(x_one))
        conv = np.array(conv)[0, 0]  # [K]
        logit = float(np.array(logit)[0, 0])
        print(f"  {label:>16}={win!r:<10}  "
              + "  ".join(f"{v:+5.2f}" for v in conv) + f"    {logit:+6.2f}")

    analyze_false_positives(logits_val, val_ds.texts, val_ds.y, PATTERN, threshold=0.0)


if __name__ == "__main__":
    main()

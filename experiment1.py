"""Experiment 1: train a single-filter 1D CNN to detect a fixed pattern,
then inspect the learned weights and compare against:
  - the pattern itself (the "ground-truth PWM")
  - the KMP failure function (sp) and the Z-array of the pattern.

Pattern: ABABC  (sp = [0,0,1,2,0],  Z = [5,0,2,0,0])
The training set deliberately injects partial prefixes (ABA, ABAB, ...)
that are NOT followed by the matching final character, forcing the network
to discriminate complete matches from auto-overlapping near-misses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from analysis import (
    analyze_false_positives,
    best_f1_threshold,
    filter_argmax_string,
    metrics_at_threshold,
    normalize_minmax_global,
    normalize_minmax_per_row,
    normalize_relu_then_max,
    show_filter,
)
from data import SIGMA, build_dataset, one_hot_encode
from model import forward, init_params
from patterns import ALPHABET, kmp_failure, z_array
from train import train


PATTERN = "ABABC"
TARGET_LENGTH = 50
N_TRAIN = 2000
N_VAL = 500
N_COMPLETE = 2
N_PARTIAL = 4
N_EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-2
POS_WEIGHT = 20.0


def evaluate_with_filter(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[dict, dict]:
    """Run the single-filter model with a custom w and bias=0.
    Returns (metrics @ threshold=0, metrics @ best F1 threshold)."""
    params = {"w": jnp.asarray(w), "b": jnp.array(0.0)}
    logits = np.array(forward(params, jnp.asarray(x)))
    return metrics_at_threshold(logits, y, t=0.0), best_f1_threshold(logits, y)


def main() -> None:
    print(f"Pattern: {PATTERN}")
    print(f"  KMP failure function (sp): {kmp_failure(PATTERN)}")
    print(f"  Z-array:                   {z_array(PATTERN)}")

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

    print("\nInitializing model...")
    key = jax.random.PRNGKey(42)
    params = init_params(key, m=len(PATTERN), sigma=SIGMA)

    print("\nTraining...")
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, pos_weight=POS_WEIGHT, seed=0, log_every=10,
    )

    logits_val = np.array(forward(params, jnp.asarray(val_ds.x)))
    print("\nValidation metrics (raw, threshold=0):")
    raw = metrics_at_threshold(logits_val, val_ds.y, t=0.0)
    for k, v in raw.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    w = np.array(params["w"])
    b = float(params["b"])
    show_filter(one_hot_encode(PATTERN), "Ground-truth PWM (pattern one-hot):")
    show_filter(w, f"Learned filter w (b={b:+.3f}):")

    learned_chars = filter_argmax_string(w)
    print(f"\nLearned per-position argmax: {learned_chars}")
    print(f"Pattern:                     {PATTERN}")
    print(f"Match: {learned_chars == PATTERN}")

    pwm = one_hot_encode(PATTERN)
    cos_per_pos = []
    for i in range(len(PATTERN)):
        a, bv = w[i], pwm[i]
        denom = (np.linalg.norm(a) * np.linalg.norm(bv)) + 1e-9
        cos_per_pos.append(float(np.dot(a, bv) / denom))
    print(f"Per-position cosine sim. with pattern PWM: "
          f"{[f'{c:.3f}' for c in cos_per_pos]}")

    print("\n=== Effect of normalizing the learned filter to [0, 1] ===")
    print("(For each variant the bias is set to 0 and we report metrics at "
          "threshold=0 AND at the best F1 threshold on validation.)")
    variants = {
        "min-max per row":   normalize_minmax_per_row(w),
        "min-max global":    normalize_minmax_global(w),
        "ReLU + row-max=1":  normalize_relu_then_max(w),
        "ground-truth PWM":  one_hot_encode(PATTERN),
    }
    for name, w_norm in variants.items():
        m0, mb = evaluate_with_filter(w_norm, val_ds.x, val_ds.y)
        show_filter(w_norm, f"[{name}]")
        print(f"  @ threshold=0          F1={m0['f1']:.3f}  "
              f"prec={m0['precision']:.3f}  rec={m0['recall']:.3f}  "
              f"tp={int(m0['tp'])} fp={int(m0['fp'])} fn={int(m0['fn'])}")
        print(f"  @ best-F1 threshold    F1={mb['f1']:.3f}  "
              f"prec={mb['precision']:.3f}  rec={mb['recall']:.3f}  "
              f"tp={int(mb['tp'])} fp={int(mb['fp'])} fn={int(mb['fn'])}  "
              f"(t={mb['threshold']:+.2f})")

    analyze_false_positives(logits_val, val_ds.texts, val_ds.y, PATTERN, threshold=0.0)


if __name__ == "__main__":
    main()

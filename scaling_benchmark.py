"""Scaling Benchmark and Soft Admissibility Evaluation.

This script implements two systematic experiments evaluating the algorithmic
properties of the CNN-derived penalty and shift tables:

1. Experiment 1: Scaling with Pattern Length m in [4, 6, 8, 12, 16, 20].
   - Evaluates Naive, Early-Exit, and Early-Exit + BM shift matchers on
     long texts (T = 20,000 chars) across multiple random patterns.
   - Measures operations per character and speedup factor over Naive.
   - Demonstrates that speedup scales linearly with m, mirroring Boyer-Moore's
     sublinear O(n/m) character-examination efficiency.

2. Experiment 2: Soft Matching with eta > 0 on Noisy Text.
   - Trains a single-filter CNN on a target pattern (m = 8) and extracts
     the continuous learned penalty matrix.
   - Evaluates performance under controlled substitution noise (0, 1, 2 mismatches).
   - Sweeps the soft threshold eta in [0.0, 0.5, 1.0, 1.5, 2.0] for the BM shift rule
     and cumulative penalty acceptance.
   - Demonstrates how eta > 0 turns the continuous penalty matrix into an
     approximate/k-mismatch matcher with a tunable speed vs. recall trade-off.

Outputs publication-quality 2-panel figure: figures/fig9_scaling_and_noise.png
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import random
import time
from typing import Dict, List, Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np

from data import CHAR_TO_IDX, SIGMA, build_dataset
from experiment5 import (
    build_penalty_table,
    build_shift_table,
    encode_text,
    f1_from_predictions,
    generate_long_text,
    matcher_early_exit,
    matcher_early_exit_bm,
    matcher_naive,
)
from model import init_params
from patterns import ALPHABET
from train import train


# ---------------------------------------------------------------------------
# Helpers for Experiment 1
# ---------------------------------------------------------------------------

def make_canonical_penalty_table(pattern: str, delta: float = 6.0) -> np.ndarray:
    """Canonical bad-character penalty table matching the CNN's converged PSSM.
    
    pen[i, P[i]] = 0.0; pen[i, c] = delta for c != P[i].
    """
    m = len(pattern)
    pen = np.full((m, SIGMA), delta, dtype=np.float32)
    for i, c in enumerate(pattern):
        pen[i, CHAR_TO_IDX[c]] = 0.0
    return pen


def run_scaling_benchmark(
    m_list: list[int] = [4, 6, 8, 12, 16, 20],
    n_trials: int = 5,
    text_length: int = 20000,
    n_complete: int = 80,
    n_partial: int = 200,
    tau: float = 4.0,
) -> dict:
    """Run Experiment 1: sweep pattern length m and record ops and speedups."""
    print("=" * 78)
    print("EXPERIMENT 1: SCALING WITH PATTERN LENGTH m")
    print("=" * 78)

    results_by_m = {}

    for m in m_list:
        trial_speedups_ee = []
        trial_speedups_bm = []
        trial_ops_naive = []
        trial_ops_ee = []
        trial_ops_bm = []

        for trial in range(n_trials):
            seed = 1000 * m + trial
            rng = random.Random(seed)
            pattern = "".join(rng.choice(ALPHABET) for _ in range(m))

            # Build canonical penalty table and shift table at eta = 0.0
            pen = make_canonical_penalty_table(pattern, delta=6.0)
            shift = build_shift_table(pen, eta=0.0)

            # Generate long text with planted occurrences and partial prefixes
            text, labels = generate_long_text(
                pattern, text_length, n_complete, n_partial, seed=seed,
            )
            text_idx = encode_text(text)
            n_pos = len(text) - m + 1

            _, ops_n = matcher_naive(text_idx, pen, tau)
            _, ops_ee = matcher_early_exit(text_idx, pen, tau)
            _, ops_bm = matcher_early_exit_bm(text_idx, pen, shift, tau)

            trial_ops_naive.append(ops_n / n_pos)
            trial_ops_ee.append(ops_ee / n_pos)
            trial_ops_bm.append(ops_bm / n_pos)

            trial_speedups_ee.append(ops_n / max(1, ops_ee))
            trial_speedups_bm.append(ops_n / max(1, ops_bm))

        results_by_m[m] = {
            "ops_naive_mean": float(np.mean(trial_ops_naive)),
            "ops_naive_std": float(np.std(trial_ops_naive)),
            "ops_ee_mean": float(np.mean(trial_ops_ee)),
            "ops_ee_std": float(np.std(trial_ops_ee)),
            "ops_bm_mean": float(np.mean(trial_ops_bm)),
            "ops_bm_std": float(np.std(trial_ops_bm)),
            "speedup_ee_mean": float(np.mean(trial_speedups_ee)),
            "speedup_ee_std": float(np.std(trial_speedups_ee)),
            "speedup_bm_mean": float(np.mean(trial_speedups_bm)),
            "speedup_bm_std": float(np.std(trial_speedups_bm)),
        }

        r = results_by_m[m]
        print(f"m={m:2d} | Naive: {r['ops_naive_mean']:.2f} ops/pos | "
              f"Early-Exit: {r['ops_ee_mean']:.2f} ops/pos ({r['speedup_ee_mean']:.2f}x) | "
              f"Early-Exit+BM: {r['ops_bm_mean']:.2f} ops/pos ({r['speedup_bm_mean']:.2f}x)")

    return results_by_m


# ---------------------------------------------------------------------------
# Helpers for Experiment 2
# ---------------------------------------------------------------------------

def generate_noisy_long_text(
    pattern: str,
    length: int,
    n_planted: int,
    k_mismatch: int,
    seed: int,
) -> tuple[str, np.ndarray]:
    """Generate long text with planted occurrences having controlled substitution noise."""
    rng = random.Random(seed)
    m = len(pattern)
    labels = np.zeros(length - m + 1, dtype=np.int8)
    text_chars = [rng.choice(ALPHABET) for _ in range(length)]

    step = length // (n_planted + 2)
    for idx in range(n_planted):
        pos = (idx + 1) * step + rng.randint(-3, 3)
        pat_list = list(pattern)
        if k_mismatch > 0:
            mismatch_indices = rng.sample(range(m), min(k_mismatch, m))
            for mi in mismatch_indices:
                choices = [c for c in ALPHABET if c != pat_list[mi]]
                pat_list[mi] = rng.choice(choices)
        for j in range(m):
            text_chars[pos + j] = pat_list[j]
        labels[pos] = 1

    return "".join(text_chars), labels


def train_single_filter_target(
    pattern: str,
    n_samples: int = 2500,
    n_epochs: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Train single-filter CNN on the target pattern."""
    m = len(pattern)
    target_length = max(50, m * 7)
    train_ds = build_dataset(
        pattern, n_samples=n_samples, target_length=target_length,
        n_complete=2, n_partial=4, seed=0,
    )
    key = jax.random.PRNGKey(seed)
    params = init_params(key, m=m, sigma=SIGMA)
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=n_epochs, batch_size=64, lr=1e-2,
        pos_weight=20.0, seed=0, log_every=n_epochs,
    )
    w = np.array(params["w"])
    b = float(params["b"])
    return w, b


def run_soft_matching_benchmark(
    pattern: str = "ABABCABD",
    eta_list: list[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    text_length: int = 20000,
    n_planted: int = 80,
) -> dict:
    """Run Experiment 2: soft matching on noisy texts."""
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 2: SOFT MATCHING WITH eta > 0 ON NOISY TEXT (Pattern: {pattern!r})")
    print("=" * 78)

    print(f">>> Training single filter on {pattern!r} ...")
    w, b = train_single_filter_target(pattern)
    pen_raw, p_rec = build_penalty_table(w)
    print(f"    Recovered pattern: {p_rec!r} (match={p_rec == pattern})")

    # Normalize penalty matrix by the mean non-zero mismatch penalty so that 1 mismatch ~ 1.0
    scale = float(np.mean(pen_raw[pen_raw > 0]))
    pen_norm = pen_raw / scale
    print(f"    Raw penalty range: [{pen_raw.min():.2f}, {pen_raw.max():.2f}] -> scaled by {scale:.2f}")

    results_by_noise = {}

    for k in [0, 1, 2]:
        print(f"\n--- Noise level: {k} substitution mismatch(es) ---")
        text, labels = generate_noisy_long_text(
            pattern, text_length, n_planted, k_mismatch=k, seed=123 + k,
        )
        text_idx = encode_text(text)
        n_pos = len(text) - len(pattern) + 1

        results_by_eta = []
        for eta in eta_list:
            shift = build_shift_table(pen_norm, eta=eta)
            preds, ops = matcher_early_exit_bm(text_idx, pen_norm, shift, tau=eta)
            metrics = f1_from_predictions(preds, labels)
            entry = {
                "eta": eta,
                "ops": ops,
                "ops_per_pos": ops / n_pos,
                **metrics,
            }
            results_by_eta.append(entry)
            print(f"  eta={eta:3.1f} | ops/pos={entry['ops_per_pos']:5.2f} | "
                  f"prec={entry['precision']:5.3f} | rec={entry['recall']:5.3f} | "
                  f"F1={entry['f1']:5.3f} | (TP={entry['tp']}, FP={entry['fp']}, FN={entry['fn']})")

        results_by_noise[k] = results_by_eta

    return {"results_by_noise": results_by_noise, "pen_norm": pen_norm, "pattern": pattern}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_figure(
    scaling_results: dict,
    soft_results: dict,
    save_path: str = "figures/fig9_scaling_and_noise.png",
) -> None:
    """Generate publication-quality 2-panel figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---------------- Panel A: Scaling with m ----------------
    m_vals = sorted(scaling_results.keys())
    naive_speedups = [1.0 for _ in m_vals]
    ee_means = [scaling_results[m]["speedup_ee_mean"] for m in m_vals]
    ee_stds = [scaling_results[m]["speedup_ee_std"] for m in m_vals]
    bm_means = [scaling_results[m]["speedup_bm_mean"] for m in m_vals]
    bm_stds = [scaling_results[m]["speedup_bm_std"] for m in m_vals]

    # Baseline Naive
    ax1.plot(m_vals, naive_speedups, "--", lw=1.2, label="Naive (1.0× baseline)", color="#888888")

    # Early-Exit
    ax1.plot(m_vals, ee_means, "o-", color="#2c7a2c", lw=2.0, ms=6, label="Early-Exit (stride 1)")
    ax1.fill_between(
        m_vals,
        np.array(ee_means) - np.array(ee_stds),
        np.array(ee_means) + np.array(ee_stds),
        color="#2c7a2c", alpha=0.15,
    )

    # Early-Exit + BM
    ax1.plot(m_vals, bm_means, "s-", color="#2c5fb0", lw=2.2, ms=6, label="Early-Exit + BM shift")
    ax1.fill_between(
        m_vals,
        np.array(bm_means) - np.array(bm_stds),
        np.array(bm_means) + np.array(bm_stds),
        color="#2c5fb0", alpha=0.18,
    )

    ax1.set_xlabel("Pattern length $m$")
    ax1.set_ylabel("Speedup factor over Naive (ops reduction)")
    ax1.set_title("(A) Speedup vs Pattern Length $m$ (Text $T=20$k)", fontweight="bold")
    ax1.set_xticks(m_vals)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax1.set_ylim(0, max(bm_means) * 1.18)

    # Annotation highlighting linear scaling (placed above curve to prevent overlap)
    ax1.annotate(
        f"BM: {bm_means[-1]:.1f}× at $m=20$\n(sublinear $O(n/m)$ scan cost)",
        xy=(m_vals[-1], bm_means[-1]),
        xytext=(m_vals[-1] - 8.2, bm_means[-1] + 0.5),
        arrowprops=dict(arrowstyle="->", color="#2c5fb0", lw=1.2),
        fontsize=8.5,
        color="#2c5fb0",
        fontweight="semibold",
    )

    # ---------------- Panel B: Soft Admissibility ----------------
    noise_1_res = soft_results["results_by_noise"][1]
    etas = [r["eta"] for r in noise_1_res]
    f1s = [r["f1"] for r in noise_1_res]
    recs = [r["recall"] for r in noise_1_res]
    precs = [r["precision"] if (r["tp"] + r["fp"]) > 0 else np.nan for r in noise_1_res]
    ops_pos = [r["ops_per_pos"] for r in noise_1_res]

    color_f1 = "#117733"
    color_rec = "#2c5fb0"
    color_prec = "#994455"
    color_ops = "#d95f02"

    # Left axis: Detection metrics
    l1 = ax2.plot(etas, f1s, "o-", color=color_f1, lw=2.4, ms=6, label="F1 Score")
    l2 = ax2.plot(etas, recs, "^--", color=color_rec, lw=1.5, ms=5, alpha=0.85, label="Recall")
    l3 = ax2.plot(etas, precs, "v:", color=color_prec, lw=1.5, ms=5, alpha=0.85, label="Precision")

    ax2.set_xlabel("Soft admissibility threshold $\\eta$")
    ax2.set_ylabel("Detection Score (Precision / Recall / F1)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xticks(etas)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Right axis: Operations per text character
    ax2_right = ax2.twinx()
    l4 = ax2_right.plot(etas, ops_pos, "d-.", color=color_ops, lw=2.0, ms=6, label="Ops / character")
    ax2_right.set_ylabel("Operations per position (lookups / char)", color=color_ops)
    ax2_right.tick_params(axis="y", labelcolor=color_ops)
    ax2_right.set_ylim(0.5, max(ops_pos) * 1.25)

    ax2.set_title(f"(B) Soft Admissibility on Noisy Text ($m=8$, 1 mismatch)", fontweight="bold")

    # Combine legends from both axes (place in upper left)
    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\n>>> Saved publication figure to {save_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    # Run Experiment 1
    scaling_results = run_scaling_benchmark(
        m_list=[4, 6, 8, 12, 16, 20],
        n_trials=5,
        text_length=20000,
    )

    # Run Experiment 2
    soft_results = run_soft_matching_benchmark(
        pattern="ABABCABD",
        eta_list=[0.0, 0.5, 1.0, 1.5, 2.0],
        text_length=20000,
        n_planted=80,
    )

    # Generate Figure 9
    generate_figure(scaling_results, soft_results, "figures/fig9_scaling_and_noise.png")

    t_elapsed = time.perf_counter() - t_start
    print(f"\nAll experiments complete in {t_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()

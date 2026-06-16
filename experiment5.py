"""Experiment 5: prototype for Direction 2 — penalty-driven dynamic stride.

The penalty table  pen[i, c] = w[i, P[i]] - w[i, c]  derived from the trained
single-filter CNN of Experiment 1 is, by Section 4 of the paper, an exact
real-valued generalisation of the Boyer-Moore bad-character rule. It is a
*data structure precomputed from the pattern* (more precisely from the
filter, which has converged to the pattern). We use it as the preprocessing
step of an exact / approximate matching algorithm and measure its effect
on the per-text cost.

We compare three matchers on the same long text:

  A. naive            for every text position, do all m penalty lookups,
                      decide match if cumulative penalty <= tau.
  B. early-exit       same as A but stop accumulating as soon as the running
                      sum exceeds tau. Stride still +1, but the per-position
                      cost is data-dependent.
  C. early-exit + BM  same as B, plus a Boyer-Moore-style shift derived from
                      the penalty table: when we reject a window at kernel
                      position i because of character c, we shift by
                      max(1, i - max{k <= i : pen[k, c] <= eta})
                      i.e. we slide the kernel just enough to bring c onto
                      a position where it is admissible (cost <= eta).
                      With eta = 0 this is exactly the classical BM
                      bad-character rule applied to the recovered pattern.

Cost is reported in penalty-table lookups; that is the elementary operation
common to all three. We also report wall-clock time and F1 on the long text.
"""

from __future__ import annotations

import os
import random
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

from data import CHAR_TO_IDX, SIGMA, build_dataset, random_partial_token
from model import init_params
from patterns import ALPHABET, kmp_failure, kmp_search
from train import train


PATTERN = "ABABC"
LONG_LENGTH = 20000
N_COMPLETE_LONG = 80
N_PARTIAL_LONG = 200
SEED_LONG = 123
TAU = 4.96      # max cumulative penalty admitted (= pattern_logit at b=0
                # -> threshold 0 in the trained model). All true matches have
                # cum_penalty == 0; near-miss windows can survive only if
                # their cum_penalty <= TAU.
ETA = 0.0       # admissibility threshold for the BM-style shift table


def generate_long_text(
    pattern: str, length: int, n_complete: int, n_partial: int, seed: int,
) -> tuple[str, np.ndarray]:
    """Same logic as data.generate_sample but at a much larger scale."""
    rng = random.Random(seed)
    m = len(pattern)
    tokens: list[str] = [pattern] * n_complete
    for _ in range(n_partial):
        k = rng.randint(1, m - 1)
        tokens.append(random_partial_token(pattern, k, rng))
    base_len = sum(len(t) for t in tokens)
    if base_len > length:
        raise ValueError("token block exceeds requested length")
    tokens.extend(rng.choice(ALPHABET) for _ in range(length - base_len))
    rng.shuffle(tokens)
    text = "".join(tokens)
    matches = kmp_search(text, pattern)
    n_pos = length - m + 1
    labels = np.zeros(n_pos, dtype=np.int8)
    for p in matches:
        if p < n_pos:
            labels[p] = 1
    return text, labels


def train_filter(pattern: str) -> tuple[np.ndarray, float]:
    train_ds = build_dataset(
        pattern, n_samples=2000, target_length=50,
        n_complete=2, n_partial=4, seed=0,
    )
    key = jax.random.PRNGKey(42)
    params = init_params(key, m=len(pattern), sigma=SIGMA)
    params, _ = train(
        params, train_ds.x, train_ds.y,
        n_epochs=80, batch_size=64, lr=1e-2,
        pos_weight=20.0, seed=0, log_every=80,
    )
    w = np.array(params["w"])
    b = float(params["b"])
    return w, b


def build_penalty_table(w: np.ndarray) -> tuple[np.ndarray, str]:
    """pen[i, c] = w[i, argmax] - w[i, c].  Also return the recovered pattern."""
    m, sigma = w.shape
    pen = np.empty_like(w)
    P_idx = w.argmax(axis=1)
    for i in range(m):
        pen[i] = w[i, P_idx[i]] - w[i, :]
    P_recovered = "".join(ALPHABET[i] for i in P_idx)
    return pen, P_recovered


def build_shift_table(pen: np.ndarray, eta: float) -> np.ndarray:
    """Generalised Boyer-Moore bad-character table derived from the penalty
    table.  shift[i, c] = how far to slide the kernel forward when we reject
    at kernel position i because of character c.

    Rule: find the largest k <= i with pen[k, c] <= eta (c is admissible in
    pattern position k). If found, shift by i - k. Otherwise shift by i + 1
    (move c past the kernel entirely).
    """
    m, sigma = pen.shape
    shift = np.zeros((m, sigma), dtype=np.int32)
    for i in range(m):
        for c in range(sigma):
            best_k = -1
            for k in range(i, -1, -1):
                if pen[k, c] <= eta:
                    best_k = k
                    break
            shift[i, c] = (i - best_k) if best_k >= 0 else (i + 1)
            if shift[i, c] < 1:
                shift[i, c] = 1
    return shift


def encode_text(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text], dtype=np.int32)


# --------- the three matchers ---------

def matcher_naive(
    text_idx: np.ndarray, pen: np.ndarray, tau: float,
) -> tuple[list[int], int]:
    """A: full m lookups per position."""
    m, _ = pen.shape
    n = len(text_idx) - m + 1
    matches: list[int] = []
    ops = 0
    for s in range(n):
        cum = 0.0
        for i in range(m):
            cum += pen[i, text_idx[s + i]]
            ops += 1
        if cum <= tau:
            matches.append(s)
    return matches, ops


def matcher_early_exit(
    text_idx: np.ndarray, pen: np.ndarray, tau: float,
) -> tuple[list[int], int]:
    """B: stop accumulating as soon as cum > tau. Stride 1."""
    m, _ = pen.shape
    n = len(text_idx) - m + 1
    matches: list[int] = []
    ops = 0
    for s in range(n):
        cum = 0.0
        for i in range(m):
            cum += pen[i, text_idx[s + i]]
            ops += 1
            if cum > tau:
                break
        else:
            matches.append(s)
    return matches, ops


def matcher_early_exit_bm(
    text_idx: np.ndarray, pen: np.ndarray, shift: np.ndarray, tau: float,
) -> tuple[list[int], int]:
    """C: early exit + BM-style shift derived from the penalty table."""
    m, _ = pen.shape
    n = len(text_idx)
    matches: list[int] = []
    ops = 0
    s = 0
    while s + m <= n:
        cum = 0.0
        rejected_at = -1
        for i in range(m):
            c = text_idx[s + i]
            cum += pen[i, c]
            ops += 1
            if cum > tau:
                rejected_at = i
                break
        if rejected_at == -1:
            matches.append(s)
            s += 1
        else:
            c = text_idx[s + rejected_at]
            s += int(shift[rejected_at, c])
    return matches, ops


def f1_from_predictions(preds: list[int], labels: np.ndarray) -> dict:
    pred_set = set(preds)
    n_pos = int(labels.sum())
    tp = sum(1 for p in preds if p < len(labels) and labels[p] == 1)
    fp = len(preds) - tp
    fn = n_pos - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    print(f">>> Training single filter on {PATTERN!r} ...")
    w, b = train_filter(PATTERN)
    pen, P_recovered = build_penalty_table(w)
    print(f"  recovered pattern: {P_recovered!r}  "
          f"(matches given pattern: {P_recovered == PATTERN})")
    print(f"  pen min={pen.min():.2f}  pen max={pen.max():.2f}")
    shift = build_shift_table(pen, eta=ETA)
    print(f"\nShift table (BM bad-character derived from pen, eta={ETA}):")
    print("       " + "    ".join(ALPHABET))
    for i in range(len(PATTERN)):
        print(f"  pos {i} ({PATTERN[i]}): " + "  ".join(f"{int(v):4d}"
                                                       for v in shift[i]))

    print(f"\nGenerating long text  (length {LONG_LENGTH}, "
          f"{N_COMPLETE_LONG} planted matches, {N_PARTIAL_LONG} near-misses)...")
    text, labels = generate_long_text(
        PATTERN, LONG_LENGTH, N_COMPLETE_LONG, N_PARTIAL_LONG, SEED_LONG,
    )
    text_idx = encode_text(text)
    n_pos = len(text) - len(PATTERN) + 1
    print(f"  total positions: {n_pos}    true matches (KMP labels): "
          f"{int(labels.sum())}")

    matchers = [
        ("A naive",            lambda: matcher_naive(text_idx, pen, TAU)),
        ("B early-exit",       lambda: matcher_early_exit(text_idx, pen, TAU)),
        ("C early-exit + BM",  lambda: matcher_early_exit_bm(
                                        text_idx, pen, shift, TAU)),
    ]

    results = []
    for name, fn in matchers:
        t0 = time.perf_counter()
        preds, ops = fn()
        dt = time.perf_counter() - t0
        metrics = f1_from_predictions(preds, labels)
        results.append({"name": name, "ops": ops, "wall_s": dt, **metrics})

    # --- report ---
    print(f"\n{'matcher':<22} {'ops':>10} {'ops/pos':>8} "
          f"{'wall (s)':>9} {'tp':>5} {'fp':>5} {'fn':>5} "
          f"{'prec':>6} {'recall':>7} {'F1':>6}")
    base_ops = results[0]["ops"]
    for r in results:
        rel = r["ops"] / base_ops
        print(f"  {r['name']:<20} {r['ops']:>10d} {r['ops']/n_pos:>8.2f} "
              f"{r['wall_s']:>9.4f} {r['tp']:>5d} {r['fp']:>5d} {r['fn']:>5d} "
              f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f}  "
              f"({rel:.2f}x ops vs naive)")

    # --- figure ---
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    names = [r["name"] for r in results]
    ops = [r["ops"] for r in results]
    colors = ["#888888", "#2c7a2c", "#2c5fb0"]

    ax = axes[0]
    bars = ax.bar(names, ops, color=colors)
    ax.set_ylabel("penalty-table lookups (smaller = better)")
    ax.set_title(f"Cost on a {LONG_LENGTH}-char text "
                 f"({n_pos} candidate positions, m={len(PATTERN)})")
    for bar, v, r in zip(bars, ops, results):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v}\n({v/n_pos:.2f}/pos)\n{v/base_ops:.2f}x",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(ops) * 1.25)

    ax = axes[1]
    f1s = [r["f1"] for r in results]
    recs = [r["recall"] for r in results]
    precs = [r["precision"] for r in results]
    x = np.arange(len(names))
    width = 0.27
    ax.bar(x - width, precs, width, color="#cc6677", label="precision")
    ax.bar(x,         recs,  width, color="#88ccee", label="recall")
    ax.bar(x + width, f1s,   width, color="#117733", label="F1")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Detection quality (tau={TAU})")
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("Penalty-table preprocessing as a Boyer-Moore-style "
                 "skip table", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig8_penalty_skip_prototype.png", dpi=150)
    plt.close(fig)
    print("\nSaved figure to figures/fig8_penalty_skip_prototype.png")


if __name__ == "__main__":
    main()

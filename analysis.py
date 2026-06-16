"""Reusable analysis utilities: metrics, threshold sweep, weight normalization,
filter pretty-printing, FP forensics. Operate directly on numpy arrays so they
are independent of the specific model architecture.
"""

from __future__ import annotations

import numpy as np

from patterns import ALPHABET


# ---------- pretty-print ----------

def show_filter(w: np.ndarray, label: str) -> None:
    print(f"\n{label}")
    print("        " + "    ".join(ALPHABET))
    for i, row in enumerate(w):
        chars = "  ".join(f"{v:+5.2f}" for v in row)
        print(f"  pos {i}: {chars}")


def filter_argmax_string(w: np.ndarray) -> str:
    """Per-row argmax over the alphabet, as a readable string."""
    return "".join(ALPHABET[i] for i in w.argmax(axis=1))


# ---------- metrics ----------

def metrics_at_threshold(logits: np.ndarray, y: np.ndarray, t: float) -> dict:
    preds = (logits > t).astype(np.float32)
    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def best_f1_threshold(logits: np.ndarray, y: np.ndarray, n_grid: int = 400) -> dict:
    flat = logits.ravel()
    lo, hi = float(flat.min()), float(flat.max())
    cands = np.linspace(lo, hi, n_grid)
    best = None
    for t in cands:
        m = metrics_at_threshold(logits, y, t=float(t))
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


# ---------- weight normalization variants ----------

def normalize_minmax_per_row(w: np.ndarray) -> np.ndarray:
    out = np.empty_like(w)
    for i in range(w.shape[0]):
        row = w[i]
        lo, hi = row.min(), row.max()
        out[i] = (row - lo) / max(1e-9, hi - lo)
    return out


def normalize_minmax_global(w: np.ndarray) -> np.ndarray:
    lo, hi = w.min(), w.max()
    return (w - lo) / max(1e-9, hi - lo)


def normalize_relu_then_max(w: np.ndarray) -> np.ndarray:
    pos = np.maximum(w, 0.0)
    out = np.empty_like(pos)
    for i in range(pos.shape[0]):
        m = pos[i].max()
        out[i] = pos[i] / max(1e-9, m)
    return out


# ---------- false-positive forensics ----------

def analyze_false_positives(
    logits: np.ndarray,
    texts: list[str],
    y: np.ndarray,
    pattern: str,
    threshold: float = 0.0,
    max_examples: int = 10,
) -> None:
    preds = (logits > threshold).astype(np.float32)
    fp_mask = (preds == 1) & (y == 0)
    tp_mask = (preds == 1) & (y == 1)
    neg_mask = y == 0

    fp_logits = logits[fp_mask]
    tp_logits = logits[tp_mask]
    neg_logits = logits[neg_mask]

    print(f"\n--- False-positive forensics (threshold={threshold:+.2f}) ---")
    if tp_mask.any():
        print(f"  TP logits: n={tp_mask.sum():>4}  "
              f"min={tp_logits.min():+.2f}  median={np.median(tp_logits):+.2f}  "
              f"max={tp_logits.max():+.2f}")
    else:
        print("  TP logits: none")
    if fp_mask.any():
        print(f"  FP logits: n={fp_mask.sum():>4}  "
              f"min={fp_logits.min():+.2f}  median={np.median(fp_logits):+.2f}  "
              f"max={fp_logits.max():+.2f}")
        if tp_mask.any() and float(np.median(tp_logits)) > 0:
            ratio = fp_logits / float(np.median(tp_logits))
            print(f"  FP/TP-median ratio: median={np.median(ratio):.2f}  "
                  f"max={ratio.max():.2f}  "
                  f"(<<1 = barely above threshold; ~1 = as confident as TPs)")
    else:
        print("  FP logits: none")
    print(f"  All negatives: n={neg_mask.sum()}  mean logit "
          f"{neg_logits.mean():+.2f} (std {neg_logits.std():.2f})")

    if not fp_mask.any():
        return

    m = len(pattern)
    sample_idx, pos_idx = np.where(fp_mask)
    by_overlap = {k: 0 for k in range(m + 1)}
    examples: list[tuple[float, str, int, int, int]] = []
    for s, p in zip(sample_idx, pos_idx):
        window = texts[int(s)][int(p):int(p) + m]
        overlap = sum(1 for a, b in zip(window, pattern) if a == b)
        by_overlap[overlap] += 1
        if len(examples) < max_examples:
            examples.append((float(logits[s, p]), window, overlap, int(s), int(p)))

    print(f"\n  FP windows by # matching chars vs pattern={pattern!r}:")
    for k in range(m + 1):
        print(f"    {k}/{m} matches: {by_overlap[k]:4d}")

    print(f"\n  First {len(examples)} FP examples:")
    for logit, window, overlap, s, p in examples:
        print(f"    sample {s:4d} pos {p:2d}  logit={logit:+.2f}  "
              f"window={window!r}  overlap={overlap}/{m}")

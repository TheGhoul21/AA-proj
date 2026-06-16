# CNN vs Exact Pattern Matching

**Convolutional pattern matching, the failure function, and a real-valued bad-character rule**

*Luca Simonetti — Advanced Algorithms project (April 2026)*

---

## Overview

This repository contains the code, experiments, and figures for an empirical and algorithmic study of the analogy between **convolutional neural networks (CNNs)** and **exact pattern-matching algorithms** (KMP, Boyer–Moore).

The project explores two directions:

- **D1 — Do learned CNN weights resemble preprocessing tables?**  
  We show that every single-filter CNN, and even a multi-filter ReLU network, converges to a **Position-Specific Scoring Matrix (PSSM)**.  The per-window logit satisfies an exact additive identity that positions the learned penalty matrix as a real-valued, position-aware generalisation of the Boyer–Moore *bad-character* table.

- **D2 — Can learned filters be used as actual algorithmic preprocessing?**  
  Using the recovered penalty table as a generalised bad-character shift table delivers a **4.06× reduction in elementary operations** over a naive scan on a 20 000-character text, with no loss of recall and identical F1.

See [`paper.md`](paper.md) or [`paper.pdf`](paper.pdf) for the full write-up.

---

## Repository Structure

```
.
├── paper.md / paper.tex / paper.pdf   # Write-up
├── LITERATURE_REVIEW.md               # SOTA survey on the CNN ↔ pattern-matching analogy
├── main.py                            # Entry point
├── model.py                           # CNN model definition (JAX / Flax)
├── data.py                            # Data generation utilities
├── patterns.py                        # Pattern definitions
├── train.py                           # Training loop
├── analysis.py                        # Weight analysis and PSSM recovery
├── make_figures.py                    # All figure generation
├── adversarial_check.py               # Adversarial / robustness checks
├── experiment1.py  …  experiment5.py  # Individual experiment scripts
└── figures/                           # Generated figures (PDF/PNG)
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Run all experiments
uv run python main.py
```

### Requirements

- Python ≥ 3.13
- JAX ≥ 0.10
- Optax ≥ 0.2.8
- NumPy ≥ 2.4
- Matplotlib ≥ 3.10

---

## Experiments

| Script | Description |
|--------|-------------|
| `experiment1.py` | Single-filter CNN training and weight analysis |
| `experiment2.py` | Multi-filter (K=4) ReLU network, PSSM verification |
| `experiment3.py` | Position-penalty table extraction |
| `experiment4.py` | Generalised bad-character shift table construction |
| `experiment5.py` | Throughput benchmark (operations saved vs naive scan) |

---

## Results

- Single-filter and multi-filter CNNs both converge to exact PSSM structure (R² ≥ 0.987 across seeds)
- Residual std 0.01 vs target std 23 for the multi-filter model
- **4.06× operation reduction** on a 20 000-character text with Σ = {A, B, C, D}, m = 5

---

## License

MIT

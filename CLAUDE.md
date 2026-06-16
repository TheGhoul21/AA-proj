# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Advanced Algorithms exam project (single author, Luca Simonetti) investigating whether 1D CNNs have anything algorithmic in common with exact string matching, in three directions:

- **D1**: Do CNN-learned weights resemble a KMP failure function (`sp`) or Z-array?
- **D2**: Can the trained filter be preprocessed into a structure that yields a non-constant stride (à la KMP / Boyer–Moore)?
- **D3 (Level B)**: Can a small model be *trained* to produce the Z-array (or `sp`, or other preprocessing structures) given only the pattern as input? If so, does the predicted structure remain operationally correct when plugged into a classical algorithm?

The deliverable is the paper (`paper.md` → `paper.tex` → `paper.pdf`). The central empirical results:

**D1/D2:** A single-filter CNN (and even a K=4 ReLU multi-filter net) converges to a PSSM whose per-window logit obeys an exact additive identity `logit(W) = π − Σ pen[i, W[i]]`, where `pen[i,c] = w[i,P_i] − w[i,c] ≥ 0`. This `pen` table is a real-valued, position-aware generalisation of the Boyer–Moore bad-character rule — not a failure function (CNNs are stateless; `sp` requires sequential state, cf. Merrill 2019). Using `pen` as actual preprocessing yields a **4.06× operation reduction** with no recall loss.

**D3 (Level B):** An MLP trained to predict Z(P) achieves ~76% per-position accuracy in-distribution but 0% exact-match accuracy and 0% OOD. It learns the statistical prior (most Z[i]=0 for random 4-char alphabet) but fails on the non-zero positions that require global recursion. When the predicted sp is plugged into KMP and tested on adversarially overlapping texts, it fails 3/5 hard patterns. This negative result is *informative*: it is predicted by Merrill 2019's expressiveness argument and matches the neural algorithmic reasoning literature.

## Commands

```bash
uv run python experiment1.py    # single filter, ABABC (D1 baseline)
uv run python experiment2.py    # K=4 multi-filter, ABABC (does it specialise on prefixes?)
uv run python experiment3.py    # single filter, ABABA (auto-overlapping pattern)
uv run python experiment4.py    # penalty-table identity + multi-filter PSSM-collapse check
uv run python experiment5.py    # D2 prototype: penalty table -> BM-style shift, on 20k-char text
uv run python experiment6.py    # D3/Level B: MLP learning the Z-array from patterns
uv run python make_figures.py   # regenerate all figures/ PNGs (retrains models)
```

Each `model.py`, `data.py`, `patterns.py` has a `__main__` smoke test runnable the same way. The whole suite runs in well under a minute on CPU. There is no test framework, linter, or `main.py` entry point (the latter is unused boilerplate). JAX runs CPU-only here.

## Architecture

The code is a flat set of single-purpose modules; experiments compose them. Read in this order:

- **`patterns.py`** — classical reference algorithms (`kmp_failure`, `z_array`, `kmp_search`) and the global alphabet config: `ALPHABET = "ABCD"`, `SIGMA = 4`, `CHAR_TO_IDX`. These constants are imported everywhere; changing the alphabet ripples through all modules.
- **`data.py`** — dataset generation. The key design choice: every sample deliberately plants both complete matches AND *partial-prefix near-misses* (`pattern[:k] + c` where `c ≠ pattern[k]`) — exactly the windows where naive matching wastes work and KMP's `sp` earns its keep. Labels are produced by running `kmp_search` on the final shuffled text (so accidental boundary matches are correct). One-hot encoding is `[T, SIGMA]`. `Dataset` carries `x [N,T,SIGMA]`, `y [N,T-m+1]`, and raw `texts`.
- **`model.py`** — two JAX models, no framework. (1) Single filter: `w [m,SIGMA]` + bias, logit = dot product over a window. (2) Multi-filter: K filters → ReLU → 1×1 mix. Convolution is written explicitly as `vmap` over positions (via `jax.lax.dynamic_slice`) so weights are readable position-by-position — this transparency is the whole point, do not replace it with `lax.conv`. `forward`/`forward_multi` are `vmap`-batched versions of the `_single` functions. `bce_loss` uses `pos_weight` to counter the ~5% positive rate.
- **`train.py`** — generic `train()` loop (optax Adam, jitted update). Accepts a `loss_fn` argument so the same loop trains both the single-filter (`loss_fn`, the default) and multi-filter (`loss_fn_multi`) models.
- **`analysis.py`** — architecture-independent numpy utilities: threshold sweep / F1 metrics, weight normalisation variants, filter pretty-printing, and false-positive forensics (bins FPs by character overlap with the pattern).

The two reference patterns recur throughout: `ABABC` (disambiguating final char, no auto-overlap) vs `ABABA` (strongly auto-overlapping). Comparing them is how the paper probes whether `sp`-structure shows up.

### Figures

`make_figures.py` retrains the models and emits `figures/fig1..fig7`. `experiment5.py` emits `fig8`. Figures are referenced by filename in both `paper.md` and `paper.tex`, so keep names stable. Hyperparameters for figure training are duplicated inside `make_figures.py` (`train_single`, `train_multi`) — they must stay in sync with the experiment scripts for the paper's numbers to match.

## Paper workflow

`paper.md` is the source of truth for prose; `paper.tex` is the LaTeX version that compiles to `paper.pdf` (the submission). When changing results or claims, keep `paper.md`, `paper.tex`, and the regenerated figures consistent. Numerical claims in the paper (e.g. the 4.06× op reduction, R²=1.0000 multi-filter collapse) come directly from `experiment4.py` / `experiment5.py` output — re-run those if you touch the models, data generation, or hyperparameters.

`LITERATURE_REVIEW.md` is the standalone SOTA survey backing both directions (Q1 = failure-function analogy, Q2 = content-aware stride / spatial gating). The paper's References section is a curated subset of it.

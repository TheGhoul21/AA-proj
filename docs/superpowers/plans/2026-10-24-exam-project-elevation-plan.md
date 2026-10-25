# Exam Project Elevation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:dispatching-parallel-agents to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate `cnn-vs-exact-pattern-matching` to a comprehensive academic exam project on par with `weight-trajectories` by adding GRU stateful neural algorithmic reasoning, scaling & soft-matching benchmarks, statistical hypothesis testing, unified CLI & test suite, and full LaTeX paper synchronization.

**Architecture:** 
- Subsystem A: Pure JAX GRU in `experiment6.py` for stateful border recursion.
- Subsystem B: `scaling_benchmark.py` testing $m \in [4, 20]$ and $\eta > 0$ soft matching, generating `figures/fig9_scaling_and_noise.png`.
- Subsystem C: `statistical_analysis.py` with multi-seed hypothesis testing ($N=10$, p-values, 95% CI).
- Subsystem D: Unified CLI in `main.py` and automated `pytest` suite in `tests/`.
- Subsystem E: Update `paper.md`, `paper.tex` with Lemma/Theorem/Algorithm environments, D3/scaling content, and compile `paper.pdf`.

**Tech Stack:** Python 3.13, JAX, Optax, NumPy, Matplotlib, SciPy (for stats), pytest, pdflatex.

---

### Task 1: Subsystem A — GRU Stateful Model for Direction 3 in `experiment6.py`

**Files:**
- Modify: `experiment6.py`

- [ ] **Step 1: Implement pure JAX GRU cell and forward pass with `jax.lax.scan`**
  - Implement `gru_cell(h, x_t, params)` and `gru_forward(params, x_seq)`.
  - Add `make_gru_model(hidden_size=64)` returning `(init_fn, forward_fn)`.
- [ ] **Step 2: Add GRU to evaluation loop alongside MLP-small and MLP-large**
  - Train GRU on Z-array and `sp` targets.
  - Evaluate on In-Distribution, Out-of-Distribution, and the 5 adversarial patterns.
- [ ] **Step 3: Run `experiment6.py` and verify GRU passes the adversarial patterns**
  - Verify that GRU achieves high accuracy on adversarial overlapping patterns (`ABAB`, `ABABA`, `AAAA`, `ABCABC`, `ABAAB`).

---

### Task 2: Subsystem B — Scaling Benchmark & Soft Matching

**Files:**
- Create: `scaling_benchmark.py`
- Output: `figures/fig9_scaling_and_noise.png`

- [ ] **Step 1: Implement `scaling_benchmark.py`**
  - Sweep $m \in [4, 6, 8, 12, 16, 20]$: train filters or construct patterns, run Naive, Early-Exit, and BM-shift matchers.
  - Measure ops/character and speedup as a function of $m$.
  - Sweep $\eta \in [0.0, 0.5, 1.0, 1.5, 2.0]$ on text with injected substitution noise ($k$-mismatches), recording precision, recall, and op count.
- [ ] **Step 2: Generate Figure 9**
  - Plot Panel A (Speedup vs pattern length $m$, showing sublinear scaling).
  - Plot Panel B (Precision-Recall & operations vs $\eta$ under noise).
  - Save to `figures/fig9_scaling_and_noise.png`.
- [ ] **Step 3: Run and verify execution**
  - Run `uv run python scaling_benchmark.py`.

---

### Task 3: Subsystem C — Statistical Analysis & Hypothesis Testing

**Files:**
- Create: `statistical_analysis.py`

- [ ] **Step 1: Implement multi-seed evaluation**
  - Run $N=10$ independent seeds for:
    1. Single-filter pattern recovery and identity error.
    2. Multi-filter PSSM collapse ($R^2$, max residual, residual std).
    3. Matcher operation counts on long texts ($T=20\,000$).
- [ ] **Step 2: Compute statistical tests**
  - Compute Mean $\pm$ Std and 95% Confidence Intervals.
  - Paired t-test and Wilcoxon signed-rank test comparing Naive vs Early-Exit and Early-Exit vs BM-Shift.
- [ ] **Step 3: Print LaTeX-formatted tables and summaries**
  - Output summary tables ready for direct inclusion into `paper.tex`.

---

### Task 4: Subsystem D — Unified CLI & Automated Test Suite

**Files:**
- Create: `tests/test_algorithms.py`
- Create: `tests/test_identity.py`
- Create: `tests/test_matchers.py`
- Modify: `main.py`

- [ ] **Step 1: Write unit tests with `pytest`**
  - Test KMP failure function, Z-array, and KMP search correctness against edge cases.
  - Test the algebraic penalty-table identity $\operatorname{logit}(W) = \pi - \sum_i \operatorname{pen}[i, W[i]]$.
  - Test exact match equivalence between Naive, Early-Exit, and BM-Shift.
- [ ] **Step 2: Refactor `main.py` into a CLI**
  - Add subcommands: `run-all`, `experiments`, `scaling`, `stats`, `adversarial`, `test`.
- [ ] **Step 3: Run pytest and CLI tests**
  - Verify all tests pass with `uv run pytest`.

---

### Task 5: Subsystem E — Update `paper.md`, `paper.tex` and Recompile `paper.pdf`

**Files:**
- Modify: `paper.md`
- Modify: `paper.tex`
- Output: `paper.pdf`

- [ ] **Step 1: Update `paper.md`**
  - Update Section 10 with the GRU results vs MLP.
  - Add Section 6.1 on the $m$-scaling curve and soft matching ($\eta$).
  - Add the statistical hypothesis testing results.
- [ ] **Step 2: Synchronize and enrich `paper.tex`**
  - Integrate Section 10 into LaTeX.
  - Add formal `\newtheorem{lemma}{Lemma}`, `\newtheorem{theorem}{Theorem}`.
  - Add formal `\begin{algorithm}` pseudocodes for table construction and dynamic-stride scan.
  - Insert Figure 9 and update the bibliography.
- [ ] **Step 3: Compile `paper.pdf`**
  - Run `pdflatex paper.tex` (twice) and ensure zero fatal errors.
  - Confirm the resulting `paper.pdf` includes all sections and figures cleanly.

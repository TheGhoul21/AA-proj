# Exam Project Elevation: CNN vs Exact Pattern Matching
**Design Document — October 24, 2026**
**Author:** Luca Simonetti

---

## 1. Goal and Vision

Elevate the `cnn-vs-exact-pattern-matching` project from a compact empirical note into a comprehensive, publication-grade academic exam project on par with `../weight-trajectories`.

The elevated project will demonstrate:
1. **Direction 1 (Representational Analysis)**: Algebraic PSSM penalty identity and empirical collapse across seeds with full statistical hypothesis testing ($N=10$, $p < 10^{-4}$, 95% CI).
2. **Direction 2 (Algorithmic Utility & Scaling)**:
   - Operation reduction curve as a function of pattern length $m \in \{4, 6, 8, 12, 16, 20\}$, showing classical sublinear $O(n/m)$ Boyer-Moore scaling behavior.
   - Soft-threshold admissibility ($\eta > 0$) for approximate / $k$-mismatch string matching, demonstrating where real-valued penalty matrices provide an advantage over discrete integer tables.
3. **Direction 3 (Neural Algorithmic Reasoning: Stateless vs Stateful)**:
   - Prove why stateless MLPs fail on overlapping patterns (zero-prior trap, 2/5 adversarial pass rate).
   - Implement a pure JAX recurrent model (GRU via `jax.lax.scan`) that maintains sequential state and solves border propagation, reaching near-perfect operational exportability (5/5 adversarial pass rate).
   - Firmly ground this contrast in Merrill (2019) and CLRS (Veličković et al., 2022).
4. **Deliverables & Engineering Polish**:
   - Comprehensive LaTeX document (`paper.tex` synchronized with `paper.md`, compiled to `paper.pdf`), formatted with formal Lemma, Theorem, and Algorithm environments (`algpseudocode`).
   - Unified CLI tool in `main.py` for running experiments, benchmarks, and verification.
   - Automated test suite in `tests/` using `pytest`.

---

## 2. Architecture and Subsystems

### Subsystem A: Stateful Algorithmic Reasoning (GRU in Direction 3)
- **Module**: `experiment6.py` (enhanced with GRU baseline)
- **Architecture**:
  - Pure JAX GRU cell using `jax.lax.scan` over one-hot sequence $X \in \mathbb{R}^{L \times |\Sigma|}$.
  - State dimension $H = 64$.
  - Linear projection $W_{\text{out}} h_t + b_{\text{out}}$ followed by sigmoid/clipping to $[0, 1]$ for normalized target.
- **Evaluation**:
  - In-distribution (lengths 8–32) and Out-of-distribution (lengths 33–64).
  - Exact-match accuracy, per-position accuracy, mean L1.
  - Operational correctness on held-out text.
  - Adversarial pattern suite (the 5 overlapping patterns: `ABAB`, `ABABA`, `AAAA`, `ABCABC`, `ABAAB`).

### Subsystem B: Scaling Benchmark & Soft Matching (Direction 2)
- **Module**: `scaling_benchmark.py`
- **Experiments**:
  1. **Scaling with $m$**: For $m \in [4, 6, 8, 12, 16, 20]$, evaluate Naive, Early-Exit, and Early-Exit + BM shift.
     - Metric: operations per text character and overall speedup over Naive.
     - Output: `figures/fig9_scaling_and_noise.png` (Panel A: speedup vs $m$).
  2. **Soft matching ($\eta > 0$)**: Evaluate matchers on texts with injected substitution noise ($k$-mismatches). Compare precision-recall trade-offs and operation counts between discrete BM and continuous penalty-guided shift.
     - Output: `figures/fig9_scaling_and_noise.png` (Panel B: precision-recall vs $\eta$).

### Subsystem C: Statistical Analysis & Hypothesis Testing
- **Module**: `statistical_analysis.py`
- **Methodology**:
  - 10 independent random seeds for training single-filter and multi-filter models.
  - Compute Mean $\pm$ Std and 95% Confidence Intervals for:
    - Single-filter pattern recovery and identity error.
    - Multi-filter PSSM fit ($R^2$, max residual, residual std).
    - Speedup across multiple long text realizations ($T=20\,000$).
  - Paired Student's t-test and Wilcoxon signed-rank test comparing Naive vs Early-Exit and Early-Exit vs BM-Shift ($H_0$: no difference in operations).

### Subsystem D: Software Engineering & Tooling
- **`main.py`**: Rich CLI using `argparse` with subcommands:
  - `run-all`: runs all experiments in sequence.
  - `scaling`: runs the $m$-scaling and noise benchmarks.
  - `stats`: computes statistical significance tests.
  - `adversarial`: runs adversarial checks.
  - `figures`: regenerates all figures (fig1 through fig9).
- **`tests/`**:
  - `test_algorithms.py`: unit tests for `kmp_failure`, `z_array`, `kmp_search`.
  - `test_identity.py`: verifies the algebraic penalty-table identity for random and trained filters.
  - `test_matchers.py`: verifies exact match equivalence between Naive, Early-Exit, and BM-shift matchers.

### Subsystem E: LaTeX Paper Synchronization & Formalization
- **`paper.tex`**:
  - Add Section 10 (Direction 3: MLP vs GRU, Neural Algorithmic Reasoning, Level B/C operational exportability).
  - Add Section 6 extension (Scaling with $m$ and Soft-Admissibility with $\eta$).
  - Formal Lemma 1: Algebraic PSSM Identity.
  - Formal Theorem 2: Soundness of Early-Exit with threshold $\tau$.
  - Formal Algorithm 1: Construction of `ShiftTable` from $\operatorname{pen}$.
  - Formal Algorithm 2: Dynamic-Stride Scan with Early-Exit.
  - Compile clean PDF via `pdflatex` to `paper.pdf`.

---

## 3. Plan for Execution with Subagents

We will execute this work using focused subagents:
1. **Subagent 1**: Implement the GRU stateful model in `experiment6.py` and run the comparison against MLP (Subsystem A).
2. **Subagent 2**: Implement `scaling_benchmark.py` and generate `figures/fig9_scaling_and_noise.png` (Subsystem B).
3. **Subagent 3**: Implement `statistical_analysis.py` and compute formal p-values and confidence intervals (Subsystem C).
4. **Subagent 4**: Implement `tests/` and refactor `main.py` into a unified CLI (Subsystem D).
5. **Subagent 5**: Update `paper.md`, `paper.tex`, include formal Lemma/Algorithm environments, sync numbers, and compile `paper.pdf` (Subsystem E).

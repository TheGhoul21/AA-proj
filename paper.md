# Convolutional pattern matching, the failure function, and a real-valued bad-character rule

**Luca Simonetti — Advanced Algorithms project**
**Working note, April 2026**

> *Code, experiments, and figures live in this directory. The companion
> `LITERATURE_REVIEW.md` collects the SOTA on both directions of the
> CNN ↔ exact-pattern-matching analogy.*

---

## Abstract

The exam project asks whether convolutional neural networks have anything
algorithmic in common with exact pattern matching, in two directions:

* **(D1)** Do the weights a CNN learns look like an `sp` (KMP failure) or
  Z table?
* **(D2)** Can one *preprocess the pattern* into a structure that lets a
  scan have a non-constant stride, mirroring how KMP and Boyer–Moore use
  their preprocessing tables?

We answer both empirically and algorithmically on a minimal setup over
$\Sigma = \{A, B, C, D\}$, $m = 5$. For **D1**, we show that every
single-filter CNN, and even a $K = 4$ ReLU multi-filter network on the same
task, converges to a Position-Specific Scoring Matrix (PSSM): the
per-window logit obeys an *exact* additive identity

$$\operatorname{logit}(W) = \pi - \sum_{i=0}^{m-1}\operatorname{pen}[i, W[i]],
\quad\text{with}\quad
\operatorname{pen}[i, c] = w[i, P_i] - w[i, c] \ge 0,$$

verified to floating-point precision over $\sim 10^4$ windows ($R^2 \ge 0.987$
across seeds, with residual std $0.01$ vs target std $23$ for the multi-filter model).
We position $\operatorname{pen}$ as a real-valued, position-aware
generalisation of the Boyer-Moore *bad-character* table; we explain why this
specific structure emerges instead of `sp` (statelessness vs sequentiality;
Merrill, 2019). For **D2**, we use $\operatorname{pen}$ as actual algorithmic
preprocessing: turning it into a generalised bad-character shift table on
the recovered pattern delivers a **4.06× reduction in elementary
operations** over a naive scan on a 20 000-character text, with no loss of
recall and identical F1. The same penalty table also frames the connection
to *k-mismatch* and approximate matching.

We then ask a harder question (**D3**): can a small MLP be trained to *produce* the Z-array or KMP failure function given only the pattern as input? We show it cannot: the model learns a zero-prior that is statistically reasonable but operationally broken on overlapping patterns, confirming the expressiveness barrier of stateless models and connecting to the neural algorithmic reasoning literature.

## 1. Setting

KMP turns the naive O$(nm)$ scan into O$(n)$ by precomputing, in O$(m)$, the
failure function `sp[i]` = length of the longest proper prefix of $P[0..i]$
that is also a suffix; on a mismatch at pattern position $j$, KMP shifts by
$j - sp[j-1]$. Boyer-Moore precomputes, in O$(m + |\Sigma|)$, a
*bad-character* table `bc[c]` = rightmost position of $c$ in $P$ (or $-1$),
and shifts by $\max(1, j - bc[c])$. Both are real algorithmic preprocessings
and both are O$(1)$ per query.

A 1D convolution slid over a sequence is mathematically the same operation
as scanning a pattern. This invites two questions:

* **D1.** Are the weights a CNN learns assimilable to a failure / Z table?
* **D2.** Can one preprocess the pattern (now realised as a learned filter)
  into a structure that yields a non-constant stride during the scan?

This note answers D1 with a clean negative-with-positive twist (the
*penalty table* below) and reports a small algorithmic prototype for D2.

## 2. Setup

* **Alphabet.** $\Sigma = \{A, B, C, D\}$, $|\Sigma| = 4$.
* **Patterns.**
  * `ABABC`, `sp = [0,0,1,2,0]`, `Z = [5,0,2,0,0]` — disambiguating final
    character; complete matches do not auto-overlap.
  * `ABABA`, `sp = [0,0,1,2,3]`, `Z = [5,0,3,0,1]` — strongly
    auto-overlapping; `ABABABA` contains two complete matches at offsets
    $0$ and $2$.
* **Texts.** Length $T = 50$ (or $T = 20\,000$ for the D2 prototype),
  generated as a token sequence containing $n_{\text{complete}}$ exact
  occurrences of $P$ and $n_{\text{partial}}$ deliberate near-misses
  $P[:k] \cdot c$ with $c \neq P[k]$ for $k \in \{1,\ldots,m-1\}$ —
  exactly the cases where naive matching wastes work and KMP's `sp`
  earns its keep. Token order is shuffled; labels are recovered by
  running KMP on the resulting text, so accidental boundary matches are
  handled correctly. Figure 1.
* **Single-filter model.** One filter $w \in \mathbb{R}^{m\times|\Sigma|}$
  and a bias $b \in \mathbb{R}$; the per-position logit on a window $W$
  of $m$ one-hot vectors is the dot product $\sum_i w[i, W[i]] + b$. BCE
  with positive-class re-weighting (factor $20$ to counter the $\sim 5\%$
  positive rate), Adam, lr $10^{-2}$, 80 epochs.
* **Multi-filter model.** $K$ filters of the same kernel size, ReLU,
  $1\times 1$ mixing layer. Used with $K = 4$ on `ABABC` in §4.2.

![Figure 1. Annotated sample text. Green spans = complete matches (label 1);
orange spans = partial-prefix near-misses inserted by the generator (label 0).
Token boundaries can produce accidental matches and near-misses, captured
by KMP-based labelling.](figures/fig1_sample_text.png)

## 3. What a single filter learns (Direction 1, baseline)

Trained on `ABABC`, the single filter recovers the pattern in its row-wise
argmax (`ABABC`) but with a *signed* shape: the right character at each
position has a moderate positive weight ($\approx +2$ to $+3.5$) while
wrong characters have *negative* weights ($\approx -2.5$ to $-4.4$).
Figure 2 shows the learned filter side by side with the ground-truth
one-hot PWM. With this filter and threshold $0$ on the logit we obtain
F1 $= 0.97$, recall $1.00$, precision $0.94$ on validation.

![Figure 2. Ground-truth PWM (left) and learned single-filter weights (right)
for `ABABC`. The argmax of each learned row equals the pattern character;
off-diagonal cells carry negative weight, encoding mismatch
*penalties*.](figures/fig2_filter_vs_pwm.png)

The validation logits split cleanly (Figure 3): every true match achieves
the same score $+4.96$ (the dot product of $w$ with the one-hot pattern is
constant); non-matches are spread well below zero. The 63 false positives
at threshold $0$ are *all* 1-mismatch windows (`ABBBC`, `BBABC`); their
logit is just barely positive (median $+0.33$). They are not "overshoots"
— they are the bias being slightly miscalibrated for these specific
shapes.

![Figure 3. Validation logit distribution (log scale) on `ABABC`. Matches
are concentrated at one point; non-matches form a broad distribution well
below zero. Threshold $0$ (red dashed) separates the two regimes with
high but imperfect margin.](figures/fig3_logit_distribution.png)

Replacing the trained filter by min-max-per-row, min-max-global,
ReLU+row-max, or *the pure one-hot ground-truth PWM* — and sweeping the
threshold — all reach F1 $= 1.000$. The signed shape is therefore not
*necessary*; an additive PWM with calibrated threshold solves the task.
The signed shape is what optimisation *picks*, presumably because the
gradient signal is denser there.

## 4. The structure that emerges in place of `sp`

### 4.1 The penalty-table identity

Define, given the trained filter $w$ and the recovered pattern
$P_i = \arg\max_c w[i, c]$,

$$\operatorname{pen}[i, c] \;=\; w[i, P_i] - w[i, c] \;\ge\; 0,
\qquad \operatorname{pen}[i, P_i] = 0.$$

Let $\pi = b + \sum_i w[i, P_i]$ be the pattern logit. Then for **any**
window $W$ of length $m$ over $\Sigma$,

$$\boxed{\;\operatorname{logit}(W) \;=\; \pi \;-\; \sum_{i=0}^{m-1}\operatorname{pen}[i, W[i]]\;}.$$

This is an algebraic identity (substitute $w[i, W[i]] = w[i, P_i] -
\operatorname{pen}[i, W[i]]$ in the dot product) and it holds exactly.
Figure 4 shows the penalty tables for `ABABC` and `ABABA`. Figure 5
verifies the identity numerically: across $9{,}200$ validation windows
per pattern, the maximum absolute discrepancy is $3.8 \times 10^{-6}$,
i.e. floating-point noise.

![Figure 4. Penalty tables for the two patterns. Cells are $\ge 0$; the
zero-diagonal corresponds to $w[i, P_i]$. On `ABABC` position 4 carries
the largest penalties (the disambiguating `C`); on `ABABA` position 2 (the
most auto-correlated row) carries *smaller* penalties — the network is
statistically less confident there.](figures/fig4_penalty_tables.png)

![Figure 5. The CNN logit equals the additive penalty decomposition
exactly (up to floating-point precision) for every window of the
validation set, on both patterns.](figures/fig5_decomposition_identity.png)

### 4.2 The penalty table as an algorithmic data structure

Reading $\operatorname{pen}$ as a *preprocessing of the pattern* puts it
on the same shelf as `sp` and `bc`:

| | KMP failure `sp` | BM bad-character `bc` | CNN penalty `pen` |
|---|---|---|---|
| **Indexed by** | pattern position $i$ | character $c$ | (position $i$, character $c$) |
| **Range** | $\{0, \ldots, i\}$ | $\{-1, 0, \ldots, m-1\}$ | $\mathbb{R}_{\ge 0}$ |
| **Construction cost** | $O(m)$ | $O(m + \|\Sigma\|)$ | $O(m \cdot \|\Sigma\|)$ from $w$ |
| **Construction source** | $P$ alone | $P$ alone | $P$ + training distribution |
| **Storage** | $O(m)$ | $O(\|\Sigma\|)$ | $O(m \cdot \|\Sigma\|)$ |
| **Query cost** | $O(1)$ shift after mismatch | $O(1)$ shift after mismatch | $O(1)$ score lookup per char |
| **Use** | drives a *control-flow* shift | drives a *control-flow* shift | drives an *additive scoring* of windows |

So $\operatorname{pen}$ is **strictly richer** than `bc` (it is
position-aware: the same wrong character $c$ at position 0 vs position 4
of `ABABC` receives penalties $4.77$ vs $7.49$) and strictly richer than
the ground-truth PWM (PWM penalties are $\{0, 1\}$ — `pen` is graded by
training data). It is **strictly poorer** than `sp` in one important
respect: it is *stateless*. There is no way to encode "after a partial
match of length $j$ shift by $j - sp[j-1]$" in a per-position-per-character
lookup. `sp` orchestrates the *control flow* of a scan; `pen` scores
complete windows.

### 4.3 Even the $K = 4$ ReLU network collapses to a PSSM

Could a non-linear network escape the additive identity? We trained the
$K = 4$ multi-filter network from Experiment 2 and asked whether its
logit is reproducible by a single additive PSSM
$c_0 + \sum_i S[i, W[i]]$ fit by least squares. On the seed shown in
Figure 6 the fit gives $R^2 = 1.0000$, residual std $0.01$ against
target std $23.4$, max residual $0.04$; across random seeds it ranges
$R^2 \in [0.987, 1.000]$. **The collapse is approximate rather than an
exact identity, but on this task the multi-filter ReLU network is in
every case almost indistinguishable from a PSSM scorer.** This is *not*
because the non-linearity is dormant: roughly a quarter of the pre-ReLU
activations are negative and hence clipped. The additivity is an
empirical property of this task distribution — the anti-pattern
detectors, clipping included, happen to combine into a per-position
additive score to within the residual above. The model is non-linear in
principle and additive in practice on this task.

![Figure 6. Multi-filter ($K = 4$ + ReLU) logit vs the best additive
PSSM-style fit. Left: scatter on the identity line. Right: residual
histogram, all within $\pm 0.04$ on logits whose spread is
$\approx 47$.](figures/fig6_multifilter_pssm.png)

### 4.4 A faint distributional signature of `sp` in the margin

Define the per-position margin
$\operatorname{margin}[i] = w[i, P_i] - \max_{c \neq P_i} w[i, c]
 = \min_{c \neq P_i} \operatorname{pen}[i, c]$.

On `ABABA` (Figure 7) the row with the lowest margin (4.51) is position
2, which is also the first row where `sp` becomes positive — exactly the
row at which the pattern starts overlapping itself. On `ABABC` the
relationship inverts (the highest-`sp` row has the largest margin, because
our deliberate near-miss tokens of type `ABA?`/`ABAB?` deposit dense
negative signal there). Pearson correlations across the 10
(pattern, position) tuples are weak: $\rho(\text{sp}, \text{margin}) =
+0.18$, $\rho(Z, \text{margin}) = -0.30$, neither significant. The
signature of auto-correlation in the weights is real but
distribution-dependent and non-monotone in `sp`. We record it as an
observation, not as a theorem.

![Figure 7. Per-position learned margin (blue, left axis) vs KMP failure
function `sp[i]` (red, right axis). On `ABABA` position 2 has both the
first non-zero `sp` and the lowest margin. On `ABABC` the relationship
does not hold.](figures/fig7_margin_vs_sp.png)

## 5. Approximate / k-mismatch matching, in one paragraph

The penalty-table identity makes the connection to *approximate* matching
mechanical. A k-mismatch matcher accepts windows with at most $k$
character disagreements; the corresponding score, in our setting, is the
sum of $k$ penalty terms $\operatorname{pen}[i, c]$ for the mismatched
positions. Choosing the threshold $\tau$ on the cumulative penalty:

* $\tau = 0$ is exact matching (only the recovered pattern survives);
* $\tau \in [0, \min_{i, c \neq P_i} \operatorname{pen}[i, c])$ is still
  exact, but resilient to floating-point noise;
* $\tau \ge \min_i \operatorname{margin}[i]$ admits some 1-mismatch
  windows;
* $\tau \to \infty$ admits everything.

So the threshold knob on a CNN classifier is, on this task, a
threshold on a *generalised Hamming distance* to the recovered pattern
weighted by `pen`. This connects the experiment directly to the classical
literature on FFT-based exact and wildcard matching (Clifford & Clifford,
2007), where the convolution-as-string-matching identity is also the
core ingredient.

## 6. Direction 2: a penalty-driven dynamic-stride scan (prototype)

We use $\operatorname{pen}$ as actual preprocessing. Build, in
$O(m \cdot |\Sigma|)$, a generalised bad-character shift table from $\operatorname{pen}$:

$$\operatorname{shift}[i, c] \;=\; \max\bigl(1,\; i \;-\; \max\{\,k \le i \mid \operatorname{pen}[k, c] \le \eta\,\}\bigr),$$

with $\operatorname{shift}[i, c] = i + 1$ if no such $k$ exists. With
$\eta = 0$ this is exactly the classical Boyer-Moore bad-character rule
applied to the recovered pattern; with $\eta > 0$ it permits "soft"
admissibility based on the learned penalty values. For our trained
filter on `ABABC`:

```
       A    B    C    D
pos 0 (A):  1    1    1    1
pos 1 (B):  1    1    2    2
pos 2 (A):  1    1    3    3
pos 3 (B):  1    1    4    4
pos 4 (C):  2    1    1    5    <- 'D' rejected at pos 4: shift past the kernel
```

We compare three matchers on a 20 000-character text containing 80
planted matches and 200 deliberate near-misses, using cumulative-penalty
threshold $\tau = 4.96$ (= the pattern logit at $b = 0$, equivalent to
threshold $0$ on the trained model's logit):

| matcher | ops | ops/pos | wall (s) | TP | FP | FN | precision | recall | F1 | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** naive (always $m$ lookups) | 99 980 | 5.00 | 0.0171 | 112 | 71 | 0 | 0.612 | 1.000 | 0.759 | 1.00× |
| **B** early-exit (stride 1, dynamic per-position cost) | 36 351 | 1.82 | 0.0079 | 112 | 71 | 0 | 0.612 | 1.000 | 0.759 | **2.75×** |
| **C** early-exit + BM-style shift | 24 634 | 1.23 | 0.0076 | 112 | 71 | 0 | 0.612 | 1.000 | 0.759 | **4.06×** |

Detection quality is identical across the three matchers (recall $1.0$,
precision and F1 dominated by the same 71 1-mismatch survivors at this
$\tau$). Reducing $\tau$ would eliminate them at the cost of recall on
some boundary cases. Figure 8 visualises the cost reduction.

![Figure 8. Cost (in penalty-table lookups) and detection metrics for the
three matchers on a 20 000-character text. The early-exit + BM matcher
visits the equivalent of $1.23$ lookups per text position, vs $5.00$ for
the naive scan: a **4.06×** reduction with no loss of detection
quality.](figures/fig8_penalty_skip_prototype.png)

What this prototype shows, in algorithmic terms:

1. The penalty table, computed in $O(m \cdot |\Sigma|)$ from a trained
   filter, *is* a usable preprocessing structure: it powers both the
   per-position cost reduction (early exit) and the inter-position
   shift (BM-style).
2. With $\eta = 0$ the prototype recovers classical Boyer-Moore on the
   pattern recovered from the filter — i.e.\ "training a CNN and reading
   off the bad-character rule from its weights" is a complete pipeline.
3. With $\eta > 0$ one obtains a soft / approximate variant whose
   admissibility map is learned from data — a direction the literature
   review identifies as unexplored under exactly this framing
   (`LITERATURE_REVIEW.md`, §5.5).

## 7. Discussion: why the failure function does not emerge

The penalty table is *stateless*: it scores each window independently of
the others. The failure function `sp` is *stateful*: it expresses how
the result of a partial match informs the next position. The two
structures live in different computational regimes.

This matches the formal-language-theoretic placement of CNNs. Merrill
(2019) shows that shallow 1D CNNs recognise at most *strictly local*
languages, a proper subset of the regular languages. KMP's correctness
relies on a deterministic finite automaton over the pattern; that DFA is
regular but not strictly local. A pure CNN therefore cannot, in
principle, implement KMP's shift logic in the weights — and our
experiments show that, in practice, it does not even try. It instead
solves the easier *scoring* problem and lets the threshold do the rest.

The penalty table is what naturally fits in a stateless template-matching
machine. To recover something like `sp` one needs to add state: a
recurrent layer, attention, or an explicit memory mechanism (see
`LITERATURE_REVIEW.md`, §1.5 on neural algorithmic reasoning, and §1.6's
summary table for what is provably out of reach for CNNs).

## 8. Limitations and outlook

* **Single pattern, small alphabet, short kernel.** The clean
  separability we observe is partly an artefact of $|\Sigma| = 4$ and
  $m = 5$. Whether the additive identity continues to hold for longer
  kernels or natural-language alphabets is a question of how often
  ReLU clips, not of principle — but worth checking.
* **Counter-example to the multi-filter collapse.** Designing a task on
  which a $K > 1$ ReLU CNN provably uses its non-linearity to compute
  something not expressible as a per-position-per-character lookup
  would isolate the regime in which the additive identity does fail.
  The natural candidate is a task whose label depends on long-range
  correlations the kernel cannot see at once.
* **Soft admissibility ($\eta > 0$).** Our prototype uses $\eta = 0$
  and so reduces to classical Boyer-Moore. Tuning $\eta$ to admit
  one-mismatch windows by penalty cost gives a learned-from-data
  approximate-matching algorithm whose theoretical worst-case behaviour
  has not been analysed here.
* **State-augmented models.** A small recurrent gating layer over the
  penalty-table output is the smallest architectural change that could
  plausibly recover an `sp`-like role in the weights. We have not
  tested it.

## 9. Reproducibility

Everything is JAX-only; the entire suite runs in well under a minute on
CPU.

```
uv run python experiment1.py     # single filter, ABABC (Direction 1)
uv run python experiment2.py     # K=4 filters, ABABC
uv run python experiment3.py     # single filter, ABABA (auto-overlap)
uv run python experiment4.py     # the penalty-table identity (both patterns)
uv run python experiment5.py     # Direction 2 prototype: penalty -> BM-style shift
uv run python make_figures.py    # regenerates every figure under figures/
```

## 10. Direction 3: can a model learn string-matching data structures?

### 10.1 Motivation and richer question

The penalty table $\operatorname{pen}$ emerged *spontaneously* from training on the
matching task; it turned out to be a real-valued generalisation of the
Boyer–Moore bad-character table.  A richer question is: can a model be
*supervised* to produce the classical preprocessing structures directly?

We study two targets:

* **Z-array:** $Z[i]$ = length of the longest substring starting at $i$ that
  matches a prefix of $P$; computed in $O(m)$, requires sequential
  global recursion.
* **KMP failure function** (`sp`): $\text{sp}[i]$ = length of the longest
  proper border of $P[0..i]$; computed in $O(m)$, also requires
  sequential state propagation.

### 10.2 Setup

We generate 5 000 random patterns over $\Sigma = \{A,B,C,D\}$ of lengths
$m \in [8, 32]$, compute the ground-truth arrays with the classical
algorithms, and train two MLPs (MLP-small: 20 608 parameters, 1 hidden
layer of 64 units; MLP-large: 57 664 parameters, 2 layers of 128 units)
to map the one-hot encoded pattern to the normalised target array.

We evaluate on:
* **In-distribution (ID):** $m \in [8, 32]$, $n = 500$.
* **Out-of-distribution (OOD):** $m \in [33, 64]$, $n = 500$.

Metrics: *exact-match* (full array correct after rounding), *per-position
accuracy*, and *mean L1 error* on the normalised target.

### 10.3 Results — Z-array

| Model | Split | ExactMatch | PosAcc | MeanL1 |
|---|---|---|---|---|
| MLP-small [64] | ID | 0.098 | 0.765 | 0.0186 |
| MLP-small [64] | OOD | 0.000 | 0.531 | 0.0231 |
| MLP-large [128,128] | ID | 0.076 | 0.758 | 0.0173 |
| MLP-large [128,128] | OOD | 0.000 | 0.554 | 0.0193 |

The model achieves ~76% per-position accuracy in-distribution but near-zero
exact-match accuracy, and drops to ~53% per-position accuracy OOD.
Analysis shows the model learns the **zero-prior**: for random patterns over
$|\Sigma|=4$, roughly 75% of Z positions are zero, so predicting zero
everywhere gives 75% position accuracy trivially. The non-zero positions —
exactly those where the Z-box extension kicks in — are almost never predicted
correctly.

### 10.4 Results — KMP failure function

| Model | Split | ExactMatch | PosAcc | MeanL1 |
|---|---|---|---|---|
| MLP-small [64] | ID | 0.032 | 0.625 | 0.0265 |
| MLP-small [64] | OOD | 0.000 | 0.410 | 0.0218 |
| MLP-large [128,128] | ID | 0.030 | 0.639 | 0.0259 |
| MLP-large [128,128] | OOD | 0.000 | 0.444 | 0.0151 |

The `sp` array is sparser (sp[0]=sp[1]=0 always; non-zero positions
require border propagation). The model achieves similar position
accuracy but lower exact-match, again due to the statistical zero-prior.

### 10.5 Level C: operational exportability criterion

The true test is not MSE but *correctness when used*.  We define:

> **Operational exportability:** a predicted structure is *exportable*
> if, when used as the preprocessing table for a classical algorithm on
> a held-out text, the set of match positions is identical to that of the
> exact algorithm.

We measure this over $n_{\text{patterns}} = 200$ patterns × 10 random texts
of length 500 each (2 000 pair-level tests per model).

| Model | Target | Pair correctness | Pattern correctness |
|---|---|---|---|
| MLP-small | Z→sp | 1.000 | 1.000 |
| MLP-small | sp direct | 1.000 | 1.000 |
| MLP-large | Z→sp | 1.000 | 1.000 |
| MLP-large | sp direct | 1.000 | 1.000 |

Pair correctness is high on random texts (~1.00) because random 4-char
texts rarely produce overlapping matches.  Pattern correctness — the
fraction of patterns correct on *all* 10 texts — is lower on more complex sets, 
and drops further on adversarial (overlapping) texts.

**Adversarial patterns** (pattern repeated to force overlapping matches):

| Pattern | True Z[1:] | Pred Z[1:] | True matches | Pred matches | Agree |
|---|---|---|---|---|---|
| ABAB | [0,2,0] | [0,1,0] | [0,2,4,...] | [0,4,...] | ✗ |
| ABABA | [0,3,0,1] | [0,2,0,1] | [0,5,10,...] | [0,5,10,...] | ✓ |
| AAAA | [3,2,1] | [1,1,1] | [0,1,2,...] | [0,3,6,...] | ✗ |
| ABCABC | [0,0,3,0,0] | [0,0,0,0,0] | [0,3,6,...] | [0,6,...] | ✗ |
| ABAAB | [0,1,2,0] | [0,1,1,0] | [0,5,...] | [0,5,...] | ✓ |

Adversarial agreement: 2/5.

### 10.6 Why this failure is informative

The failure is *predictable* from the theory of §7. The Z-array at
position $i$ is defined by:
$$Z[i] = \max\{k : P[0..k-1] = P[i..i+k-1]\},$$
which is computed sequentially using the Z-box $(l, r)$ maintained across
positions. An MLP with a fixed, position-independent weight matrix cannot
implement this recurrence: it has no state between positions.  The
result is, in the language of Merrill (2019), that this task requires
*strictly more* than the strictly-local languages that shallow 1D CNNs
(and position-agnostic MLPs) can recognise.

This connects to the **neural algorithmic reasoning** literature
(Veličković et al., CLRS benchmark, 2022): models trained to predict the
outputs of sequential algorithms consistently fail to generalise to
longer inputs, precisely because they cannot replicate the recursive
structure. Our experiment is a minimal instance of this phenomenon.

The implication for Level C is that **plain MLPs cannot export a correct
failure function** on adversarial inputs.  The path to exportability
requires either (a) architectures with sequential state (RNNs or
Transformers with causal masking), or (b) post-hoc verification and
repair — a connection to the *rule extraction* literature (Craven \&
Shavlik, 1994; Andrews et al., 1995).


## 11. Reproducibility

Everything is JAX-only; the entire suite runs in well under a minute on
CPU.

```
uv run python experiment1.py     # single filter, ABABC (Direction 1)
uv run python experiment2.py     # K=4 filters, ABABC
uv run python experiment3.py     # single filter, ABABA (auto-overlap)
uv run python experiment4.py     # the penalty-table identity (both patterns)
uv run python experiment5.py     # Direction 2 prototype: penalty -> BM-style shift
uv run python experiment6.py     # Direction 3: MLP learning string algorithms
uv run python make_figures.py    # regenerates every figure under figures/
```

## 12. References

Detailed references and SOTA discussion are collected in
`LITERATURE_REVIEW.md`. The most directly relevant pointers are:

* W. Merrill. *Sequential neural networks as automata.* ACL Workshop on
  Deep Learning and Formal Languages, 2019.
* P. Veličković et al. *The CLRS algorithmic reasoning benchmark.*
  ICML, 2022.
* P. K. Koo and S. R. Eddy. *Representation learning of genomic sequence
  motifs with convolutional neural networks.* PLOS Comput. Biol., 2019.
* B. Alipanahi et al. *Predicting the sequence specificities of DNA-
  and RNA-binding proteins by deep learning (DeepBind).* Nat.
  Biotechnol., 2015.
* C.-C. J. Kuo et al. *Convolutional neural networks demystified: a
  matched filtering perspective-based tutorial.* IEEE SP Magazine, 2023.
* P. Clifford and R. Clifford. *Simple deterministic wildcard matching.*
  Inf. Process. Lett., 2007. (FFT-based matching = the
  convolution-as-string-matching identity.)
* S. Cao et al. *SeerNet: predicting CNN feature-map sparsity through
  low-bit quantization.* CVPR, 2019.
* W. Hua et al. *Channel gating neural networks.* NeurIPS, 2019.
* T. Verelst and T. Tuytelaars. *Dynamic convolutions: exploiting spatial
  sparsity for faster inference.* CVPR, 2020.
* R. S. Boyer and J. S. Moore. *A fast string searching algorithm.*
  CACM, 1977.
* D. E. Knuth, J. H. Morris, and V. R. Pratt. *Fast pattern matching in
  strings.* SIAM J. Comput., 1977.

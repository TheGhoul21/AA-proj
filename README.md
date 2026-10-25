# CNN vs Exact Pattern Matching

Luca Simonetti — Advanced Algorithms exam project.

The project compares classical string matching and neural pattern recognition:
what information each solution represents, how it prepares the search, and
which computations it can safely avoid. Interpretability is used to explain
the relationship between the learned function and the search algorithm.

## Current research

Start with the [first algorithmic comparison and results](docs/algorithmic-comparison-first-results.md)
(Italian), the [controlled distribution intervention](docs/distribution-intervention-results.md),
then the [research program](docs/interpretability-research-program.md).

Experiment 7 evaluates **all 1,024 windows** for each of two five-character
patterns (`ABABC`, `ABABA`). It compares a constructed exact CNN, trained
single-filter and multi-filter CNNs, additive approximations, and exact-search
controls. The default run uses three initialization seeds and saves 90
checkpoints across 12 training runs.

Experiment 8 changes hard-negative frequencies while keeping the pattern,
class balance, negative support, initialization and background examples fixed.
Across 36 training runs, all final classifiers are exact on the 1,024 windows,
but their substitution penalties change with the data. Ordering comparisons
by neural score drops matches a simple frequency heuristic on this controlled
mixture; both reach the best fixed-order expected cost. This is a window-level
cost analysis, not an end-to-end text-search speed benchmark.

Key distinctions:

- A single filter is additive by construction; learning determines its weights.
- A high-quality additive approximation need not preserve decisions at the
  same threshold. Threshold recalibration is a separate question.
- Recovering a pattern from weights allows classical preprocessing, but does
  not show that the network learned the classical search procedure.
- At zero penalty threshold, learned and binary penalties with the same
  unique zero positions induce the same decisions and shifts in our scanner.
- Lookup/comparison counts describe online work, not measured hardware speed.

The existing [paper](paper.md) and [PDF](paper.pdf) document the earlier
experiments. Their broad interpretations of PSSM collapse, expressiveness,
and scaling require revision in light of the current audit; they have not
yet been rewritten around the new results.

## Run

Python ≥ 3.13; dependencies are recorded in `pyproject.toml` and `uv.lock`.

```bash
uv sync
uv run python experiment7.py
uv run python experiment8.py
uv run python -m pytest -q
```

With an existing environment, use `.venv/bin/python` instead of `uv run python`.
The experiment supports `--seeds`, `--patterns` (length five), and `--output`.
For example:

```bash
uv run python experiment7.py --seeds 7 --output results/exhaustive-seed7
```

Artifacts in [`results/exhaustive/`](results/exhaustive/):

- `results.json`: protocol, source hashes, environment, checkpoint diagnostics,
  and exact-search operation counts.
- `functions.npz`: every evaluated function, window ordering, and loss histories.
- `checkpoints/`: parameter snapshots, including initialization. These are for
  analysis, not optimizer-resume checkpoints.
- `trajectories.png` / `.pdf`: errors, interactions and separation during training.
- `function_comparison.png` / `.pdf`: network versus additive approximation.

The exhaustive domain is used to describe the entire learned function. It is
not a held-out generalization benchmark or a threshold-selection set. Training
data and shuffle seeds are fixed; the three runs vary initialization only.

Experiment 8 writes to [`results/distribution-intervention/`](results/distribution-intervention/):
`results.json`, `functions.npz`, six controlled datasets, 36 final parameter
sets, and two figures in PNG/PDF. Its 180 saved function evaluations describe
training on isolated windows, a separate protocol from experiment 7.

## Code map

| File | Role |
|---|---|
| `patterns.py` | Reference KMP, failure function and Z-array |
| `data.py` | Planted matches and partial-prefix negatives |
| `model.py` | Explicit single-filter and multi-filter JAX models |
| `train.py` | Adam training with optional checkpoint observer |
| `exhaustive_analysis.py` | Exact finite-domain additive projection and diagnostics |
| `exact_search.py` | Counted naive, KMP and bad-character-only searches |
| `experiment7.py` | Exhaustive analysis and controlled exact-search comparison |
| `distribution_controls.py` | Matched negative distributions and exact comparison-order cost model |
| `experiment8.py` | Distribution intervention, paired effects and order-selection controls |
| `experiment1.py`–`experiment4.py` | Initial weight and representation analyses |
| `experiment5.py` | Penalty scan, early exit and left-to-right bad-character shifts |
| `experiment6.py` | Neural Z/sp prediction; causal target protocol needs revision |
| `scaling_benchmark.py` | Earlier scaling and noisy-matching experiments |
| `statistical_analysis.py` | Earlier repeated-run statistics |
| `main.py` | CLI for experiments 1–6 and other original workflows |
| `tests/` | Algorithm, identity, projection and training-observer checks |

Run `uv run python main.py --help` for the original CLI. Experiments 7 and 8
currently have their own entry points. The simplified bad-character baseline is explicitly
not full Boyer–Moore: it omits the good-suffix rule.

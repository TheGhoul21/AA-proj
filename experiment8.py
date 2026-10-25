"""Experiment 8: shift hard negatives, measure representation and search order.

Run .venv/bin/python experiment8.py. Conditions have identical class balance,
background examples, hard-negative support, initialization, and batch order.
This window-based intervention is distinct from experiment 7's text training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from distribution_controls import (
    CONDITIONS, build_intervention_dataset, comparison_order_analysis,
    single_substitution_drops, target_indices,
)
from exhaustive_analysis import describe_function, enumerate_windows
from model import forward, forward_multi, init_params, init_params_multi, loss_fn, loss_fn_multi
from patterns import SIGMA, kmp_failure, z_array
from train import train


def paired_effects(final: list[dict]) -> list[dict]:
    effects = []
    for pattern, model, seed in dict.fromkeys((r["pattern"], r["model"], r["seed"]) for r in final):
        conditions = {r["condition"]: r for r in final
                      if (r["pattern"], r["model"], r["seed"]) == (pattern, model, seed)}
        if any(r["relative_mean_drops"] is None for r in conditions.values()):
            effects.append({"pattern": pattern, "model": model, "seed": seed,
                            "valid": False, "reason": "nonpositive mean drop at some position"})
            continue
        def contrast(condition):
            margins = conditions[condition]["relative_mean_drops"]
            return margins[0] - margins[4]
        first, balanced, last = (contrast(c) for c in ("focus_first", "balanced", "focus_last"))
        effects.append({"pattern": pattern, "model": model, "seed": seed, "valid": True,
                        "first_minus_last_position": {"focus_first": first, "balanced": balanced, "focus_last": last},
                        "focus_first_minus_balanced": first - balanced,
                        "focus_last_minus_balanced": last - balanced,
                        "focus_first_minus_focus_last": first - last})
    return effects


def make_figures(final: list[dict], output: Path) -> None:
    patterns = list(dict.fromkeys(r["pattern"] for r in final))
    models = ("single", "multi")
    matrices = {}
    for pattern in patterns:
        for model in models:
            rows = []
            for condition in CONDITIONS:
                values = [r["relative_mean_drops"] for r in final
                          if (r["pattern"], r["model"], r["condition"]) == (pattern, model, condition)
                          and r["relative_mean_drops"] is not None]
                rows.append(np.mean(values, axis=0) if values else np.full(5, np.nan))
            matrices[pattern, model] = np.array(rows)
    limit = max(float(np.nanmax(v)) for v in matrices.values())
    fig, axes = plt.subplots(len(patterns), 2, figsize=(10, 3.3 * len(patterns)), squeeze=False,
                             layout="constrained")
    for row, pattern in enumerate(patterns):
        for col, model in enumerate(models):
            ax = axes[row, col]
            matrix = matrices[pattern, model]
            shown = ax.imshow(matrix, vmin=0, vmax=limit, cmap="Blues", aspect="auto")
            ax.set(xticks=range(5), xticklabels=[f"{i}: {c}" for i, c in enumerate(pattern)],
                   yticks=range(3), yticklabels=CONDITIONS, title=f"{pattern} — {model}", xlabel="Position: symbol")
            for i in range(3):
                for j in range(5):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                            color="white" if matrix[i,j] > .6 * limit else "black")
    fig.colorbar(shown, ax=axes, label="Mean substitution drop / average across positions")
    fig.suptitle("Same pattern; only the hard-negative distribution changes\nMean across initialization seeds")
    fig.savefig(output / "distribution_effect.png", dpi=160)
    fig.savefig(output / "distribution_effect.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(patterns), 2, figsize=(11, 3.6 * len(patterns)), squeeze=False,
                             layout="constrained")
    labels = ("left_to_right", "right_to_left", "neural_drop", "empirical_mismatch_frequency")
    colors = ("#8b8b8b", "#b0b0b0", "#2769a0", "#b44b36")
    for row, pattern in enumerate(patterns):
        for col, model in enumerate(models):
            ax = axes[row, col]
            group = [r for r in final if (r["pattern"],r["model"]) == (pattern,model)]
            for j, (label, color) in enumerate(zip(labels, colors)):
                vals = [[r["comparison_order"]["evaluation_distributions"][c]["costs"][label]
                         for r in group if r["condition"] == c] for c in CONDITIONS]
                means = [np.mean(v) for v in vals]
                ax.bar(np.arange(3) + (j - 1.5) * .2, means, width=.2, color=color, label=label)
            ax.set(xticks=range(3), xticklabels=CONDITIONS, title=f"{pattern} — {model}",
                   ylabel="Expected comparisons per window", ylim=(0, 3.5))
            ax.grid(axis="y", alpha=.2)
    handles, legend_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Exact stride-one search; evaluation mixture matches training condition\nNetwork chooses comparison order, equality decides matches")
    fig.savefig(output / "comparison_order.png", dpi=160)
    fig.savefig(output / "comparison_order.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42])
    parser.add_argument("--patterns", nargs="+", default=["ABABC", "ABABA"])
    parser.add_argument("--output", type=Path, default=Path("results/distribution-intervention"))
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.patterns) != len(set(args.patterns)):
        parser.error("Seeds and patterns must be unique")
    for pattern in args.patterns:
        target_indices(pattern)
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("datasets", "parameters"):
        (args.output / directory).mkdir(exist_ok=True)
    started = time.perf_counter()
    windows = enumerate_windows(5)
    x_all = jnp.asarray(np.eye(SIGMA, dtype=np.float32)[windows])
    snapshots, final, data_summary = [], [], []
    functions = {"windows": windows}
    for pattern in args.patterns:
        target = target_indices(pattern)
        mask = np.arange(SIGMA)[None, :] != target[:, None]
        for condition in CONDITIONS:
            ds = build_intervention_dataset(pattern, condition)
            np.savez_compressed(args.output / "datasets" / f"{pattern}_{condition}.npz",
                                windows=ds.windows, labels=ds.labels, source=ds.source)
            data_summary.append({"pattern": pattern, "condition": condition,
                                 "positive_count": int(ds.labels.sum()),
                                 "unique_windows": len(np.unique(ds.windows, axis=0)),
                                 "hard_negative_mismatch_counts": (ds.windows[ds.source == 2] != target).sum(axis=0).tolist()})
            x_train = ds.x
            for model in ("single", "multi"):
                evaluate = jax.jit(forward if model == "single" else forward_multi)
                for seed in args.seeds:
                    run_id = f"{pattern}_{condition}_{model}_{seed}"
                    print(f"\n{run_id}", flush=True)
                    key = jax.random.PRNGKey(seed)
                    params = init_params(key, 5, SIGMA) if model == "single" else init_params_multi(key, 4, 5, SIGMA)

                    def observe(epoch, current):
                        if epoch not in (0, 1, 5, 20, 80):
                            return
                        values = np.asarray(evaluate(current, x_all)).ravel()
                        drops = single_substitution_drops(values, pattern)
                        alternatives = drops[mask].reshape(5, SIGMA - 1)
                        mean_drops = alternatives.mean(axis=1)
                        relative = (mean_drops / mean_drops.mean()).tolist() if np.all(mean_drops > 0) else None
                        record = {"pattern": pattern, "condition": condition, "model": model,
                                  "seed": seed, "epoch": epoch,
                                  **describe_function(values, pattern, windows),
                                  "substitution_drops": drops.tolist(),
                                  "mean_substitution_drops": mean_drops.tolist(),
                                  "min_substitution_drops": alternatives.min(axis=1).tolist(),
                                  "relative_mean_drops": relative}
                        snapshots.append(record)
                        functions[f"{run_id}_epoch{epoch}"] = values
                        if epoch == 80:
                            final.append({**record, "comparison_order": comparison_order_analysis(values, pattern, ds, windows)})

                    params, history = train(params, x_train, ds.labels, 80, 64,
                                            .01 if model == "single" else .005, 20., seed=0,
                                            log_every=80, loss_fn=loss_fn if model == "single" else loss_fn_multi,
                                            on_epoch=observe)
                    np.savez_compressed(args.output / "parameters" / f"{run_id}.npz",
                                        **{k: np.asarray(v) for k,v in params.items()})
                    functions[f"{run_id}_loss"] = np.array(history)
                    print(f"  FP={final[-1]['false_positives']}, FN={final[-1]['false_negatives']}, "
                          f"mean drops={np.round(final[-1]['mean_substitution_drops'],2).tolist()}", flush=True)
    root = Path(__file__).resolve().parent
    sources = ("experiment8.py", "distribution_controls.py", "exhaustive_analysis.py",
               "model.py", "train.py", "patterns.py", "uv.lock")
    result = {
        "protocol": {"patterns": args.patterns, "initialization_seeds": args.seeds,
                     "conditions": CONDITIONS, "data_seed": 0, "shuffle_seed": 0,
                     "samples": 2400, "positive_count": 120, "background_count": 1140,
                     "hard_negative_count": 1140, "hard_position_probabilities": {
                         "balanced": [.2]*5, "focus_first": [.8,.05,.05,.05,.05],
                         "focus_last": [.05,.05,.05,.05,.8]},
                     "epochs": 80, "batch_size": 64, "pos_weight": 20.,
                     "learning_rates": {"single": .01, "multi": .005}, "K_multi": 4,
                     "primary_effect": "paired change of (drop[0]-drop[4])/mean(drop) across conditions",
                     "evaluation": "all 1024 windows; search costs under exact population probabilities",
                     "scope": "isolated-window training; all conditions share class balance, background and support",
                     "search": "exact equality, stride one, no shifts; neural scores choose order only",
                     "oracle": "best of 120 fixed permutations under evaluation probabilities"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]},
        "source_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        "pattern_tables": {p: {"sp": kmp_failure(p), "Z": z_array(p)} for p in args.patterns},
        "datasets": data_summary, "checkpoints": snapshots, "final": final,
        "paired_effects": paired_effects(final), "elapsed_seconds": time.perf_counter()-started,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output / "functions.npz", **functions)
    make_figures(final, args.output)
    print(f"\nSaved {len(final)} runs, {len(snapshots)} evaluations ({result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()

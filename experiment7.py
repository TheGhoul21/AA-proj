"""Experiment 7: exhaustive function analysis and exact-search controls.

Run: .venv/bin/python experiment7.py
Artifacts: results/exhaustive/{results.json, functions.npz, checkpoints/, *.png}
The default training protocols match experiments 1 and 2; seeds vary only
initialization. All epochs in a run share the same Adam and shuffle state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import build_dataset
from exact_search import bad_character_search, kmp_counted, naive_search
from exhaustive_analysis import additive_projection, describe_function, enumerate_windows
from experiment5 import (
    build_penalty_table, build_shift_table, encode_text, generate_long_text,
    matcher_early_exit, matcher_early_exit_bm, matcher_naive,
)
from model import forward, forward_multi, init_params, init_params_multi, loss_fn, loss_fn_multi
from patterns import ALPHABET, CHAR_TO_IDX, SIGMA, kmp_failure, kmp_search, z_array
from train import train


def search_controls(pattern: str, trained_params: dict, seed: int) -> list[dict]:
    """Exact extraction has a different acceptance rule from CNN logit > 0."""
    m = len(pattern)
    target = np.array([CHAR_TO_IDX[c] for c in pattern])
    canonical = (np.arange(SIGMA)[None, :] != target[:, None]).astype(float)
    learned, recovered = build_penalty_table(np.asarray(trained_params["w"]))
    rng = random.Random(123)
    planted, _ = generate_long_text(pattern, 20000, 80, 200, seed=123)
    border = kmp_failure(pattern)[-1]
    period = m - border
    repeat = pattern[:period]
    texts = {
        "uniform": "".join(rng.choice(ALPHABET) for _ in range(20000)),
        "planted_prefix_nearmisses": planted,
        "periodic_overlap": (repeat * ((20000 + period - 1) // period))[:20000],
    }
    rows = []
    for name, text in texts.items():
        reference = kmp_search(text, pattern)
        encoded = encode_text(text)
        methods = {}
        for label, function in (
            ("naive_early_exit", naive_search), ("kmp", kmp_counted),
            ("bad_character_right_to_left", bad_character_search),
        ):
            found, ops = function(text, pattern)
            assert found == reference, (pattern, name, label)
            methods[label] = {"operations": ops, "unit": "character_comparison",
                              "agrees_with_exact": True}
        for label, pen in (("canonical", canonical), ("learned", learned)):
            shift = build_shift_table(pen, eta=0.)
            for suffix, matcher in (("full", matcher_naive), ("early_exit", matcher_early_exit)):
                found, ops = matcher(encoded, pen, tau=0.)
                methods[f"{label}_{suffix}"] = {
                    "operations": ops, "unit": "penalty_lookup",
                    "agrees_with_exact": found == reference,
                }
            found, ops = matcher_early_exit_bm(encoded, pen, shift, tau=0.)
            methods[f"{label}_left_to_right_shift"] = {
                "operations": ops, "unit": "penalty_lookup",
                "agrees_with_exact": found == reference,
            }
        rows.append({"pattern": pattern, "initialization_seed": seed,
                     "text": name, "text_seed": 123, "length": len(text),
                     "matches": len(reference), "recovered_pattern": recovered,
                     "methods": methods})
    return rows


def plot_trajectories(records: list[dict], output: Path) -> None:
    patterns = list(dict.fromkeys(r["pattern"] for r in records))
    fig, axes = plt.subplots(len(patterns), 3, figsize=(13, 4 * len(patterns)), squeeze=False)
    colors = {"single": "#2769a0", "multi": "#b44b36"}
    for row, pattern in enumerate(patterns):
        for model in colors:
            seeds = sorted({r["seed"] for r in records if r["model"] == model})
            for j, seed in enumerate(seeds):
                run = [r for r in records if (r["pattern"], r["model"], r["seed"]) == (pattern, model, seed)]
                epochs = [r["epoch"] for r in run]
                series = [
                    [r["false_positives"] + r["false_negatives"] for r in run],
                    [max(r["interaction_variance_fraction"] or 0., 1e-16) for r in run],
                    [r["separation_gap"] for r in run],
                ]
                for col, values in enumerate(series):
                    axes[row, col].plot(epochs, values, ".-", color=colors[model], alpha=.65,
                                        label=model if j == 0 else None)
        axes[row, 0].set_yscale("symlog", linthresh=1)
        axes[row, 1].set_yscale("log")
        axes[row, 2].axhline(0, color="gray", lw=.8)
        titles = ["Decision errors at logit > 0", "Nonadditive variance / total variance",
                  "Pattern score − largest nonmatch score"]
        for col, title in enumerate(titles):
            axes[row, col].set(title=f"{pattern}: {title}", xlabel="Completed epochs")
            axes[row, col].grid(alpha=.2)
        axes[row, 0].legend()
    fig.suptitle("All 1,024 windows; each line is one initialization seed", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    fig.savefig(output / "trajectories.pdf")
    plt.close(fig)


def plot_function_comparison(records: list[dict], functions: dict, output: Path) -> None:
    """Show the positive window explicitly instead of hiding it in global R²."""
    final = [r for r in records if r["model"] == "multi" and r["epoch"] == 150]
    patterns = list(dict.fromkeys(r["pattern"] for r in final))
    seeds = sorted({r["seed"] for r in final})
    fig, axes = plt.subplots(len(patterns), len(seeds), figsize=(4.2 * len(seeds), 4 * len(patterns)), squeeze=False)
    for row, pattern in enumerate(patterns):
        target = np.array([CHAR_TO_IDX[c] for c in pattern])
        positive = np.all(functions["windows"] == target, axis=1)
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            values = functions[f"{pattern}_multi_{seed}_epoch150"]
            additive = additive_projection(values, len(pattern))[2]
            ax.scatter(values[~positive], additive[~positive], s=6, alpha=.3, color="#2769a0")
            ax.scatter(values[positive], additive[positive], s=65, color="#b44b36", zorder=3,
                       label="Exact pattern")
            lo = min(values.min(), additive.min())
            hi = max(values.max(), additive.max())
            ax.plot([lo, hi], [lo, hi], color="gray", lw=.7)
            ax.axhline(0, color="gray", lw=.7, ls="--")
            ax.axvline(0, color="gray", lw=.7, ls="--")
            ax.set(title=f"{pattern}, seed {seed}", xlabel="Network logit", ylabel="Additive projection logit")
            ax.legend(fontsize=8)
    fig.suptitle("Same decision boundary (0); red point is the only positive window")
    fig.tight_layout()
    fig.savefig(output / "function_comparison.png", dpi=160)
    fig.savefig(output / "function_comparison.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 42])
    parser.add_argument("--patterns", nargs="+", default=["ABABC", "ABABA"])
    parser.add_argument("--output", type=Path, default=Path("results/exhaustive"))
    args = parser.parse_args()
    if any(len(p) != 5 or set(p) - set(ALPHABET) for p in args.patterns):
        parser.error("This controlled protocol uses five-character patterns over ABCD")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.patterns)) != len(args.patterns):
        parser.error("Seeds and patterns must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    windows = enumerate_windows(5)
    one_hot = jnp.asarray(np.eye(SIGMA, dtype=np.float32)[windows])
    records, scans, baselines = [], [], []
    functions = {"windows": windows}
    protocols = {
        "single": {"samples": 2000, "epochs": 80, "lr": .01},
        "multi": {"samples": 4000, "epochs": 150, "lr": .005},
    }
    for pattern in args.patterns:
        target = np.array([CHAR_TO_IDX[c] for c in pattern])
        # This constructed CNN has logit = 0.5 - Hamming(W, P).
        template = {"w": jnp.asarray(np.eye(SIGMA, dtype=np.float32)[target]),
                    "b": jnp.array(.5 - len(pattern))}
        template_scores = np.asarray(forward(template, one_hot)).ravel()
        baselines.append({"pattern": pattern, "kind": "constructed_exact_CNN",
                          **describe_function(template_scores, pattern, windows)})
        assert np.array_equal(template_scores > 0, np.all(windows == target, axis=1))
        functions[f"{pattern}_constructed"] = template_scores
        for model, protocol in protocols.items():
            ds = build_dataset(pattern, protocol["samples"], 50, 2, 4, seed=0)
            evaluate = jax.jit(forward if model == "single" else forward_multi)
            for seed in args.seeds:
                run_id = f"{pattern}_{model}_{seed}"
                print(f"\n{run_id}", flush=True)
                key = jax.random.PRNGKey(seed)
                params = init_params(key, 5, SIGMA) if model == "single" else init_params_multi(key, 4, 5, SIGMA)
                checkpoints = {0, 1, 5, 10, 20, 40, 80, protocol["epochs"]}

                def observe(epoch, current):
                    if epoch not in checkpoints:
                        return
                    values = np.asarray(evaluate(current, one_hot)).ravel()
                    record = {"pattern": pattern, "model": model, "seed": seed, "epoch": epoch,
                              **describe_function(values, pattern, windows)}
                    if model == "single":
                        w = np.asarray(current["w"], dtype=float)
                        pen, recovered = build_penalty_table(w)
                        reconstructed = float(current["b"]) + w.max(axis=1).sum() - pen[np.arange(5), windows].sum(axis=1)
                        mask = np.arange(SIGMA)[None, :] != target[:, None]
                        target_differences = w[np.arange(5), target, None] - w
                        record.update({"recovered_pattern": recovered,
                                       "target_margins": np.where(mask, target_differences, np.inf).min(axis=1).tolist(),
                                       "penalty_identity_max_error": float(np.max(np.abs(values - reconstructed)))})
                    records.append(record)
                    functions[f"{run_id}_epoch{epoch}"] = values
                    np.savez_compressed(args.output / "checkpoints" / f"{run_id}_epoch{epoch}.npz",
                                        **{k: np.asarray(v) for k, v in current.items()})

                params, history = train(
                    params, ds.x, ds.y, protocol["epochs"], 64, protocol["lr"],
                    20., seed=0, log_every=protocol["epochs"],
                    loss_fn=loss_fn if model == "single" else loss_fn_multi, on_epoch=observe,
                )
                functions[f"{run_id}_loss"] = np.array(history)
                if model == "single":
                    scans.extend(search_controls(pattern, params, seed))
                last = records[-1]
                print(f"  FP={last['false_positives']}, FN={last['false_negatives']}, "
                      f"interaction={last['interaction_variance_fraction']:.3g}, "
                      f"projection disagreements={last['additive_decision_disagreements']}", flush=True)

    sources = ["experiment7.py", "exhaustive_analysis.py", "exact_search.py", "train.py",
               "model.py", "data.py", "patterns.py", "experiment5.py", "uv.lock"]
    root = Path(__file__).resolve().parent
    result = {
        "protocol": {"alphabet": ALPHABET, "patterns": args.patterns, "seeds": args.seeds,
                     "training": protocols, "data_seed": 0, "shuffle_seed": 0,
                     "batch_size": 64, "pos_weight": 20., "target_length": 50,
                     "n_complete": 2, "n_partial": 4, "K_multi": 4,
                     "decision_rule": "logit > 0", "evaluation": "uniform exhaustive",
                     "projection": "descriptive orthogonal projection; no held-out fit",
                     "seed_scope": "initialization only; data and shuffle fixed",
                     "cost_scope": "online comparisons/lookups only; not hardware runtimes",
                     "extraction_rule": "argmax pattern, nonnegative penalties, tau=eta=0; different from logit > 0"},
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "devices": [str(d) for d in jax.devices()]},
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "pattern_tables": {p: {"sp": kmp_failure(p), "Z": z_array(p)} for p in args.patterns},
        "baselines": baselines, "checkpoints": records, "search_controls": scans,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "functions.npz", **functions)
    plot_trajectories(records, args.output)
    plot_function_comparison(records, functions, args.output)
    print(f"\nSaved {len(records)} checkpoints to {args.output} ({result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()

"""Re-train the three models from Experiments 1, 2, 3 and emit the figures
referenced in the paper.

Outputs everything under figures/ as PNG.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from analysis import filter_argmax_string
from data import CHAR_TO_IDX, SIGMA, build_dataset, one_hot_encode
from model import (
    forward,
    forward_multi,
    init_params,
    init_params_multi,
    loss_fn_multi,
)
from patterns import ALPHABET, kmp_failure, z_array
from train import train


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# ---------------- shared helpers ----------------

def heatmap(ax, mat, row_labels, col_labels, title, cmap="RdBu_r",
            center_zero=True, show_values=True, fmt="{:+.2f}"):
    if center_zero:
        v = float(np.abs(mat).max())
        im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=+v, aspect="auto")
    else:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    if show_values:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(mat[i, j]) < 0.6 * np.abs(mat).max()
                        else "white")
    return im


def train_single(pattern, n_epochs=80, n_train=2000, seed=42):
    train_ds = build_dataset(pattern, n_samples=n_train, target_length=50,
                             n_complete=2, n_partial=4, seed=0)
    val_ds = build_dataset(pattern, n_samples=500, target_length=50,
                           n_complete=2, n_partial=4, seed=1)
    key = jax.random.PRNGKey(seed)
    params = init_params(key, m=len(pattern), sigma=SIGMA)
    params, _ = train(params, train_ds.x, train_ds.y,
                      n_epochs=n_epochs, batch_size=64, lr=1e-2,
                      pos_weight=20.0, seed=0, log_every=n_epochs)
    return params, train_ds, val_ds


def train_multi(pattern, K=4, n_epochs=150, n_train=4000, seed=7):
    train_ds = build_dataset(pattern, n_samples=n_train, target_length=50,
                             n_complete=2, n_partial=4, seed=0)
    val_ds = build_dataset(pattern, n_samples=500, target_length=50,
                           n_complete=2, n_partial=4, seed=1)
    key = jax.random.PRNGKey(seed)
    params = init_params_multi(key, K=K, m=len(pattern), sigma=SIGMA)
    params, _ = train(params, train_ds.x, train_ds.y,
                      n_epochs=n_epochs, batch_size=64, lr=5e-3,
                      pos_weight=20.0, seed=0, log_every=n_epochs,
                      loss_fn=loss_fn_multi)
    return params, train_ds, val_ds


# ---------------- Figure 1: a sample text annotated ----------------

def fig_sample_text(pattern: str, ds, save_to: str) -> None:
    """Pick a sample, render its characters and highlight matches and partial
    prefixes."""
    # find a sample with at least one match and one partial
    sample_idx = 0
    text = ds.texts[sample_idx]
    y = ds.y[sample_idx]
    m = len(pattern)

    fig, ax = plt.subplots(figsize=(13, 1.6))
    ax.set_xlim(-0.5, len(text) - 0.5)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([])
    ax.set_xticks([])

    # mark complete-match start positions: y == 1
    match_starts = [int(i) for i in np.where(y == 1)[0]]

    # find all maximal partial-prefix matches (length >= 2, < m, not extendable)
    partials = []
    i = 0
    while i < len(text):
        # longest prefix of pattern matching at i
        j = 0
        while i + j < len(text) and j < m and text[i + j] == pattern[j]:
            j += 1
        if j >= 2 and j < m and i not in match_starts:
            partials.append((i, j))
            i += j  # skip past
        else:
            i += 1

    # background spans
    for start in match_starts:
        ax.axvspan(start - 0.5, start + m - 0.5, ymin=0.05, ymax=0.95,
                   color="#a8e6a3", alpha=0.7, zorder=0)
    for start, length in partials:
        ax.axvspan(start - 0.5, start + length - 0.5, ymin=0.1, ymax=0.55,
                   color="#ffd0a3", alpha=0.7, zorder=0)

    # characters
    for i, c in enumerate(text):
        ax.text(i, 0.5, c, ha="center", va="center", fontsize=11,
                family="monospace", zorder=2)
    # position ruler every 5
    for i in range(0, len(text), 5):
        ax.text(i, -0.3, str(i), ha="center", va="center", fontsize=7,
                color="gray")

    # legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#a8e6a3", alpha=0.7,
                 label=f"complete match of {pattern!r} (label = 1)"),
           Patch(facecolor="#ffd0a3", alpha=0.7,
                 label="partial-prefix near-miss (label = 0)")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.7),
              ncol=2, frameon=False, fontsize=9)
    ax.set_title(f"Sample text (length {len(text)}). Pattern = {pattern!r}.",
                 pad=22)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 2: filter weights vs PWM ----------------

def fig_filter_vs_pwm(pattern: str, w: np.ndarray, save_to: str) -> None:
    pwm = one_hot_encode(pattern)
    rows = [f"pos {i} ({pattern[i]})" for i in range(len(pattern))]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    heatmap(axes[0], pwm, rows, list(ALPHABET),
            f"Ground-truth PWM\n(one-hot of pattern {pattern!r})",
            center_zero=True, fmt="{:.0f}")
    heatmap(axes[1], w, rows, list(ALPHABET),
            f"Learned single-filter weights\n(after training)",
            center_zero=True)
    fig.suptitle(f"What the single-filter CNN learns on pattern {pattern!r}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 3: logit distribution ----------------

def fig_logit_distribution(pattern: str, params, val_ds, save_to: str) -> None:
    logits = np.array(forward(params, jnp.asarray(val_ds.x))).ravel()
    y = val_ds.y.ravel()
    tp_mask = y == 1
    neg_mask = y == 0

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.linspace(logits.min(), logits.max(), 80)
    ax.hist(logits[neg_mask], bins=bins, color="#888888", alpha=0.6,
            label=f"non-matches (n={neg_mask.sum()})", log=True)
    ax.hist(logits[tp_mask], bins=bins, color="#2c7a2c", alpha=0.85,
            label=f"matches (n={tp_mask.sum()})", log=True)
    ax.axvline(0.0, color="red", lw=1, ls="--", label="threshold = 0")
    ax.set_xlabel("CNN logit")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Logit distribution on the validation set, pattern {pattern!r}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 4: penalty table heatmaps ----------------

def fig_penalty_tables(structures: list[dict], save_to: str) -> None:
    fig, axes = plt.subplots(1, len(structures), figsize=(4.4 * len(structures), 3.6))
    if len(structures) == 1:
        axes = [axes]
    for ax, s in zip(axes, structures):
        p = s["pattern"]
        rows = [f"pos {i} ({p[i]})" for i in range(len(p))]
        # use a one-sided cmap because penalties are >= 0
        v = float(s["pen"].max())
        im = ax.imshow(s["pen"], cmap="Reds", vmin=0, vmax=v, aspect="auto")
        ax.set_xticks(range(SIGMA)); ax.set_xticklabels(list(ALPHABET))
        ax.set_yticks(range(len(p))); ax.set_yticklabels(rows)
        for i in range(s["pen"].shape[0]):
            for j in range(s["pen"].shape[1]):
                val = s["pen"][i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if val < 0.6 * v else "white")
        ax.set_title(f"Penalty table for {p!r}\n"
                     f"(pen[i, c] = w[i, P[i]] - w[i, c])")
    fig.suptitle("The structure: a real-valued generalised "
                 "bad-character table", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 5: identity check ----------------

def fig_decomposition_identity(structures: list[dict], val_xs, save_to: str) -> None:
    fig, axes = plt.subplots(1, len(structures), figsize=(4.4 * len(structures), 3.6))
    if len(structures) == 1:
        axes = [axes]
    for ax, s, val_x in zip(axes, structures, val_xs):
        p = s["pattern"]
        m = len(p)
        params = {"w": jnp.asarray(s["w"]), "b": jnp.array(s["b"])}
        cnn_logits = np.array(forward(params, jnp.asarray(val_x))).ravel()

        char_idx = val_x.argmax(axis=2)
        N = val_x.shape[0]
        n_pos = val_x.shape[1] - m + 1
        recon = np.empty((N, n_pos))
        for n in range(N):
            for j in range(n_pos):
                wi = char_idx[n, j:j + m]
                recon[n, j] = s["pattern_logit"] - sum(s["pen"][i, wi[i]] for i in range(m))
        recon = recon.ravel()
        err = np.abs(cnn_logits - recon).max()

        ax.scatter(recon, cnn_logits, s=4, alpha=0.4, color="#2c5fb0")
        lim_lo = min(recon.min(), cnn_logits.min()) - 1
        lim_hi = max(recon.max(), cnn_logits.max()) + 1
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8,
                label="y = x")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("decomposition: pattern_logit - sum_i pen[i, W[i]]")
        ax.set_ylabel("actual CNN logit")
        ax.set_title(f"Pattern {p!r}\n"
                     f"max |error| over {len(cnn_logits)} windows: {err:.1e}")
        ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Identity: CNN logit = additive penalty decomposition",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 6: multi-filter collapses to PSSM ----------------

def fig_multifilter_pssm(pattern: str, save_to: str) -> None:
    print(f"\n>>> Training K=4 multi-filter model on {pattern!r} ...")
    params, _train_ds, val_ds = train_multi(pattern, K=4, n_epochs=150)
    val_logits = np.array(forward_multi(params, jnp.asarray(val_ds.x)))

    char_idx = val_ds.x.argmax(axis=2)
    m = len(pattern)
    sigma = SIGMA
    N, n_pos = val_logits.shape
    A_rows = []
    b_rows = []
    for n in range(N):
        for j in range(n_pos):
            row = np.zeros(m * sigma + 1, dtype=np.float32)
            row[-1] = 1.0
            for i in range(m):
                row[i * sigma + char_idx[n, j + i]] = 1.0
            A_rows.append(row); b_rows.append(val_logits[n, j])
    A = np.stack(A_rows); bv = np.array(b_rows)
    sol, *_ = np.linalg.lstsq(A, bv, rcond=None)
    pred = A @ sol
    err = bv - pred
    r2 = 1 - err.var() / bv.var()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    ax.scatter(pred, bv, s=4, alpha=0.3, color="#2c5fb0")
    lo, hi = min(pred.min(), bv.min()), max(pred.max(), bv.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("best additive PSSM-style fit")
    ax.set_ylabel("multi-filter (K=4 + ReLU) logit")
    ax.set_title(f"K=4 multi-filter logit vs best additive fit\nR^2 = {r2:.4f}")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    ax.hist(err, bins=80, color="#888888")
    ax.set_xlabel("residual (multi-filter logit - additive fit)")
    ax.set_ylabel("count")
    ax.set_title(f"Residuals (std={err.std():.2f}, "
                 f"target std={bv.std():.2f}, max |e|={np.abs(err).max():.2f})")
    fig.suptitle(f"Despite ReLU, the K=4 model is empirically PSSM-equivalent  "
                 f"(pattern {pattern!r})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- Figure 7: margin per position vs sp ----------------

def fig_margin_vs_sp(structures: list[dict], save_to: str) -> None:
    fig, axes = plt.subplots(1, len(structures), figsize=(4.6 * len(structures), 3.6),
                             sharey=False)
    if len(structures) == 1:
        axes = [axes]
    for ax, s in zip(axes, structures):
        p = s["pattern"]
        m = len(p)
        idx = np.arange(m)
        width = 0.35
        ax.bar(idx - width / 2, s["margin"], width=width, color="#2c5fb0",
               label="margin (learned)")
        ax2 = ax.twinx()
        ax2.bar(idx + width / 2, s["sp"], width=width, color="#c33",
                alpha=0.7, label="sp[i] (KMP)")
        ax2.bar(idx + width / 2, [0] * m, width=0)  # spacer
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{i}\n({p[i]})" for i in range(m)])
        ax.set_xlabel("kernel position")
        ax.set_ylabel("learned margin")
        ax2.set_ylabel("sp[i] (KMP failure)")
        ax2.set_yticks(range(max(s["sp"]) + 1))
        # combined legend
        lines, labels = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lab2, loc="upper left", frameon=False,
                  fontsize=9)
        ax.set_title(f"Pattern {p!r}")
    fig.suptitle("Per-position margin vs KMP failure function sp[i]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- main ----------------

def main() -> None:
    print(">>> Training single-filter on 'ABABC' ...")
    p1, train1, val1 = train_single("ABABC", n_epochs=80)
    print(">>> Training single-filter on 'ABABA' ...")
    p3, train3, val3 = train_single("ABABA", n_epochs=80)

    structures = []
    for pattern, params, val in [("ABABC", p1, val1), ("ABABA", p3, val3)]:
        w = np.array(params["w"])
        b = float(params["b"])
        m = len(pattern)
        P_idx = np.array([CHAR_TO_IDX[c] for c in pattern])
        diag = np.array([w[i, P_idx[i]] for i in range(m)])
        margin = np.array([w[i, P_idx[i]] - np.delete(w[i], P_idx[i]).max()
                           for i in range(m)])
        pattern_logit = float(diag.sum() + b)
        pen = np.empty_like(w)
        for i in range(m):
            pen[i] = w[i, P_idx[i]] - w[i, :]
        structures.append({
            "pattern": pattern, "w": w, "b": b,
            "diag": diag, "margin": margin,
            "pattern_logit": pattern_logit, "pen": pen,
            "sp": kmp_failure(pattern), "Z": z_array(pattern),
            "P_recovered": filter_argmax_string(w),
        })

    print("\n>>> Generating figures ...")
    fig_sample_text("ABABC", train1, f"{FIG_DIR}/fig1_sample_text.png")
    fig_filter_vs_pwm("ABABC", structures[0]["w"], f"{FIG_DIR}/fig2_filter_vs_pwm.png")
    fig_logit_distribution("ABABC", p1, val1,
                           f"{FIG_DIR}/fig3_logit_distribution.png")
    fig_penalty_tables(structures, f"{FIG_DIR}/fig4_penalty_tables.png")
    fig_decomposition_identity(structures, [val1.x, val3.x],
                               f"{FIG_DIR}/fig5_decomposition_identity.png")
    fig_multifilter_pssm("ABABC", f"{FIG_DIR}/fig6_multifilter_pssm.png")
    fig_margin_vs_sp(structures, f"{FIG_DIR}/fig7_margin_vs_sp.png")
    print(f"\nDone. Figures saved under {FIG_DIR}/")


if __name__ == "__main__":
    main()

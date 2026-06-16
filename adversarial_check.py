"""Adversarial check: do the scripts actually deliver what the paper's thesis
claims? Each probe perturbs an experiment and prints a verdict.

Thesis claims under test:
  D1a. logit(W) = pi - sum_i pen[i, W[i]]  (penalty-table identity)
  D1b. pen[i,c] >= 0  ("the structure that emerges in place of sp")
  D1c. Even a K=4 ReLU network collapses to a PSSM (R^2 = 1.0000)
  D1d. sp does NOT emerge in the weights
  D2a. pen -> BM shift table gives a real op reduction with no recall loss
  D2b. With eta=0 the pipeline recovers a *correct* classical matcher
  D2c. "training a CNN and reading the BM rule off its weights" is a pipeline

Run: uv run python adversarial_check.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from data import SIGMA, build_dataset, one_hot_encode, CHAR_TO_IDX
from model import (
    init_params, forward,
    init_params_multi, forward_multi, forward_multi_intermediates, loss_fn_multi,
)
from train import train
from patterns import kmp_failure, kmp_search
from experiment5 import (
    train_filter, build_penalty_table, build_shift_table,
    generate_long_text, encode_text,
    matcher_naive, matcher_early_exit, matcher_early_exit_bm,
    f1_from_predictions,
)

PATTERN = "ABABC"
VERDICTS: list[tuple[str, str, str]] = []


def record(claim: str, verdict: str, detail: str) -> None:
    VERDICTS.append((claim, verdict, detail))
    print(f"\n[{verdict}] {claim}\n    {detail}")


def pen_and_pi(w: np.ndarray, b: float):
    P_idx = w.argmax(axis=1)
    pen = np.stack([w[i, P_idx[i]] - w[i, :] for i in range(w.shape[0])])
    pi = b + sum(w[i, P_idx[i]] for i in range(w.shape[0]))
    return pen, float(pi), P_idx


def identity_error(w: np.ndarray, b: float, x: np.ndarray) -> float:
    pen, pi, _ = pen_and_pi(w, b)
    m = w.shape[0]
    params = {"w": jnp.asarray(w), "b": jnp.array(b)}
    cnn = np.array(forward(params, jnp.asarray(x)))
    cidx = x.argmax(axis=2)
    N, n_pos = cnn.shape
    recon = np.empty_like(cnn)
    for n in range(N):
        for j in range(n_pos):
            recon[n, j] = pi - sum(pen[i, cidx[n, j + i]] for i in range(m))
    return float(np.abs(cnn - recon).max())


# ---------------------------------------------------------------------------
# P1. The identity and pen>=0 are ARCHITECTURAL, not a training discovery.
# ---------------------------------------------------------------------------
def probe_identity_is_algebraic():
    val = build_dataset(PATTERN, 100, 50, 2, 4, seed=99)
    key = jax.random.PRNGKey(123)
    rnd = init_params(key, m=len(PATTERN), sigma=SIGMA)
    w_rnd, b_rnd = np.array(rnd["w"]), float(rnd["b"])
    err_rnd = identity_error(w_rnd, b_rnd, val.x)
    pen_rnd, _, P_rnd = pen_and_pi(w_rnd, b_rnd)
    recovered_rnd = "".join("ABCD"[i] for i in P_rnd)

    record(
        "D1a/b: identity logit=pi-sum(pen) and pen>=0 hold for an UNTRAINED random filter",
        "CAVEAT" if err_rnd < 1e-4 else "FAIL",
        f"random-filter identity error={err_rnd:.1e}, pen.min={pen_rnd.min():.2e} (>=0 by construction). "
        f"=> The additive form and pen>=0 are forced by the single-linear-filter + one-hot architecture, "
        f"NOT produced by training. Training only aligns the row-argmax to the true pattern "
        f"(random recovers {recovered_rnd!r} vs true {PATTERN!r}). The paper calls it 'an algebraic "
        f"identity' (honest), but the section title 'the structure that emerges' slightly oversells: "
        f"the structure is definitional; what emerges is correct pattern recovery + calibrated margins.",
    )


# ---------------------------------------------------------------------------
# P2. D1: seed-robustness of pattern recovery + identity (single filter).
# ---------------------------------------------------------------------------
def probe_single_filter_seeds():
    val = build_dataset(PATTERN, 200, 50, 2, 4, seed=99)
    train_ds = build_dataset(PATTERN, 2000, 50, 2, 4, seed=0)
    rows = []
    ok = True
    for seed in (0, 1, 2, 42):
        params = init_params(jax.random.PRNGKey(seed), m=len(PATTERN), sigma=SIGMA)
        params, _ = train(params, train_ds.x, train_ds.y, n_epochs=80,
                           batch_size=64, lr=1e-2, pos_weight=20.0, seed=0, log_every=999)
        w, b = np.array(params["w"]), float(params["b"])
        rec = "".join("ABCD"[i] for i in w.argmax(axis=1))
        err = identity_error(w, b, val.x)
        rows.append((seed, rec, err))
        ok = ok and (rec == PATTERN) and (err < 1e-4)
    detail = "; ".join(f"seed{s}: recovered={r} iderr={e:.1e}" for s, r, e in rows)
    record("D1: single filter recovers pattern + identity holds across seeds",
           "PASS" if ok else "CAVEAT", detail)


# ---------------------------------------------------------------------------
# P3. D1c: does the K=4 ReLU model ALWAYS collapse to a PSSM? Is ReLU inactive?
# ---------------------------------------------------------------------------
def fit_additive_r2(logits, x):
    m = len(PATTERN)
    cidx = x.argmax(axis=2)
    N, n_pos = logits.shape
    A, bvec = [], []
    for n in range(N):
        for j in range(n_pos):
            row = np.zeros(m * SIGMA + 1, np.float32)
            row[-1] = 1.0
            for i in range(m):
                row[i * SIGMA + cidx[n, j + i]] = 1.0
            A.append(row); bvec.append(logits[n, j])
    A = np.stack(A); bvec = np.array(bvec)
    sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    err = bvec - A @ sol
    return 1 - err.var() / bvec.var()


def probe_multifilter_collapse():
    train_ds = build_dataset(PATTERN, 4000, 50, 2, 4, seed=0)
    val = build_dataset(PATTERN, 300, 50, 2, 4, seed=1)
    rows = []
    ok = True
    for seed in (7, 1, 2):
        params = init_params_multi(jax.random.PRNGKey(seed), K=4, m=len(PATTERN), sigma=SIGMA)
        params, _ = train(params, train_ds.x, train_ds.y, n_epochs=150,
                          batch_size=64, lr=5e-3, pos_weight=20.0, seed=0,
                          log_every=999, loss_fn=loss_fn_multi)
        logits = np.array(forward_multi(params, jnp.asarray(val.x)))
        r2 = fit_additive_r2(logits, val.x)
        _, conv = forward_multi_intermediates(params, jnp.asarray(val.x))
        conv = np.array(conv)
        frac_clipped = float((conv < 0).mean())  # fraction of pre-ReLU acts that ReLU zeros
        rows.append((seed, r2, frac_clipped))
        ok = ok and (r2 > 0.999)
    detail = "; ".join(f"seed{s}: R2={r:.4f} relu_clipped_frac={fc:.2f}" for s, r, fc in rows)
    record("D1c: K=4 ReLU network collapses to a PSSM across seeds",
           "PASS" if ok else "CAVEAT",
           detail + "  (collapse is real but mechanistic: high relu_clipped_frac would mean "
           "genuine nonlinearity; near-0 means ReLU is inactive on this distribution => "
           "collapse is distribution-specific, as the paper's limitations admit)")


# ---------------------------------------------------------------------------
# P4 + P5. D2: matcher correctness at tau=0, and op-reduction robustness over
# many texts (paper reports n=1).
# ---------------------------------------------------------------------------
def probe_d2_correctness_and_robustness():
    w, b = train_filter(PATTERN)
    pen, P_rec = build_penalty_table(w)
    shift = build_shift_table(pen, eta=0.0)

    # P4: tau=0 should give a *correct* exact matcher (precision=recall=1).
    text, labels = generate_long_text(PATTERN, 20000, 80, 200, seed=123)
    tidx = encode_text(text)
    n_pos = len(text) - len(PATTERN) + 1
    for tau, tag in ((0.0, "tau=0 (exact)"), (4.96, "tau=4.96 (paper)")):
        pa, oa = matcher_naive(tidx, pen, tau)
        pc, oc = matcher_early_exit_bm(tidx, pen, shift, tau)
        ma = f1_from_predictions(pa, labels)
        mc = f1_from_predictions(pc, labels)
        same = set(pa) == set(pc)
        verdict = "PASS" if (tag.startswith("tau=0") and ma["precision"] == 1.0
                             and ma["recall"] == 1.0 and same) else (
                  "PASS" if not tag.startswith("tau=0") else "FAIL")
        record(
            f"D2b: at {tag} the CNN-derived matcher is correct & C==A on accepts",
            verdict if tag.startswith("tau=0") else "INFO",
            f"naive: prec={ma['precision']:.3f} rec={ma['recall']:.3f} F1={ma['f1']:.3f} ops={oa}; "
            f"C(BM): prec={mc['precision']:.3f} rec={mc['recall']:.3f} ops={oc} "
            f"({oa/max(1,oc):.2f}x fewer ops vs naive); C accepts == A accepts: {same}",
        )

    # P5: op-reduction over many texts, both thresholds.
    for tau in (4.96, 0.0):
        ratios = []
        recalls_ok = True
        for s in range(8):
            t, lab = generate_long_text(PATTERN, 20000, 80, 200, seed=1000 + s)
            ti = encode_text(t)
            _, oa = matcher_naive(ti, pen, tau)
            pc, oc = matcher_early_exit_bm(ti, pen, shift, tau)
            ratios.append(oa / max(1, oc))
            recalls_ok = recalls_ok and f1_from_predictions(pc, lab)["recall"] == 1.0
        r = np.array(ratios)
        record(f"D2a: op-reduction C-vs-naive is stable across 8 texts (tau={tau})",
               "PASS" if (r.mean() > 1.5 and recalls_ok) else "CAVEAT",
               f"speedup mean={r.mean():.2f}x std={r.std():.2f} min={r.min():.2f} "
               f"max={r.max():.2f}; recall==1.0 on all: {recalls_ok}  "
               f"(paper reports a single text at 4.06x)")


# ---------------------------------------------------------------------------
# P6. The KEY adversarial point: at eta=0 the shift table ignores the learned
# REAL-valued penalties entirely -- it depends only on the recovered pattern.
# ---------------------------------------------------------------------------
def probe_eta0_ignores_learned_values():
    w, b = train_filter(PATTERN)
    pen, _ = build_penalty_table(w)
    shift_real = build_shift_table(pen, eta=0.0)

    # Perturb every nonzero (off-pattern) penalty arbitrarily, keep zeros (= argmax) fixed.
    rng = np.random.default_rng(0)
    pen_perturbed = pen.copy()
    mask = pen > 0
    pen_perturbed[mask] = rng.uniform(0.1, 100.0, size=mask.sum())  # destroy the learned magnitudes
    shift_perturbed = build_shift_table(pen_perturbed, eta=0.0)

    identical = bool(np.array_equal(shift_real, shift_perturbed))
    # Also compare against pure classical BM on the recovered pattern.
    record(
        "D2c: at eta=0 the shift table uses ONLY the recovered pattern, not learned penalties",
        "CAVEAT" if identical else "FAIL",
        f"shift table unchanged after randomising all real penalty magnitudes: {identical}. "
        f"=> the 4.06x speedup is exactly classical Boyer-Moore on the recovered pattern; the "
        f"paper's headline 'real-valued, position-aware' richness of pen contributes nothing to the "
        f"D2 prototype at eta=0 (only pattern recovery + the tau early-exit do). The paper states "
        f"'with eta=0 this is exactly classical BM' and defers eta>0 to outlook, so this is a scope "
        f"limit, not a contradiction -- but the learned values are untested as an algorithmic asset.",
    )


# ---------------------------------------------------------------------------
# P7. D1d: confirm sp does NOT emerge -- margin-vs-sp correlation is seed-noise.
# ---------------------------------------------------------------------------
def probe_sp_does_not_emerge():
    train_ds = build_dataset(PATTERN, 2000, 50, 2, 4, seed=0)
    sp = np.array(kmp_failure(PATTERN), float)
    corrs = []
    for seed in (0, 1, 2, 42):
        params = init_params(jax.random.PRNGKey(seed), m=len(PATTERN), sigma=SIGMA)
        params, _ = train(params, train_ds.x, train_ds.y, n_epochs=80,
                           batch_size=64, lr=1e-2, pos_weight=20.0, seed=0, log_every=999)
        w = np.array(params["w"])
        margin = np.array([w[i, w[i].argmax()] - np.delete(w[i], w[i].argmax()).max()
                           for i in range(len(PATTERN))])
        a, c = sp - sp.mean(), margin - margin.mean()
        corrs.append(float((a * c).sum() / ((np.linalg.norm(a) * np.linalg.norm(c)) + 1e-9)))
    corrs = np.array(corrs)
    record("D1d: sp does not emerge; margin-vs-sp correlation is unstable seed-noise",
           "PASS" if np.abs(corrs).mean() < 0.6 and corrs.std() > 0.05 else "CAVEAT",
           f"rho(sp,margin) per seed = {[f'{v:+.2f}' for v in corrs]} "
           f"(mean {corrs.mean():+.2f}, std {corrs.std():.2f}) -> sign/size varies with seed, "
           f"consistent with the paper calling it 'an observation, not a theorem'")


def main():
    probe_identity_is_algebraic()
    probe_single_filter_seeds()
    probe_multifilter_collapse()
    probe_d2_correctness_and_robustness()
    probe_eta0_ignores_learned_values()
    probe_sp_does_not_emerge()

    print("\n" + "=" * 70)
    print("ADVERSARIAL SUMMARY")
    print("=" * 70)
    for claim, verdict, _ in VERDICTS:
        print(f"  [{verdict:6}] {claim}")


if __name__ == "__main__":
    main()

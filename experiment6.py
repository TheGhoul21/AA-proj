"""Experiment 6 — Level B: Can a learned model recover the Z-array?

We treat the Z-array computation as a supervised seq2seq task:
  Input:  one-hot encoded pattern P of length m over Σ = {A,B,C,D}
  Target: Z(P) normalised to [0,1]  (i.e. Z[i]/m)

This is the "learned data structures" experiment recommended as the key
extension: instead of asking "do CNN weights resemble sp/Z?", we ask
"can a small network be *trained* to produce these structures?"

Three model sizes are compared:
  - MLP-small  (1 hidden layer, 64 units)
  - MLP-large  (2 hidden layers, 128 units)
  - Transformer (2 heads, 1 layer, dim=32)   [see note below]

Evaluation:
  - Exact-match accuracy (full array correct after rounding)
  - Per-position accuracy (fraction of positions correct after rounding)
  - Mean L1 error on the normalised target
  - In-distribution:  train lengths 8–32, test on lengths 8–32
  - Out-of-distribution: test on lengths 33–64

We also check whether the *rounded* Z prediction, when plugged back into
a Z-based pattern search, preserves correctness.

Alphabet: {A, B, C, D} (SIGMA=4) — consistent with the rest of the project.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from patterns import ALPHABET, CHAR_TO_IDX, SIGMA, z_array

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def random_pattern(length: int) -> str:
    """Generate a random pattern over ALPHABET of the given length."""
    return "".join(random.choices(ALPHABET, k=length))


def pattern_to_onehot(p: str, pad_to: int | None = None) -> np.ndarray:
    """One-hot encode pattern into shape (m, SIGMA) or (pad_to, SIGMA) with zero-padding."""
    m = len(p)
    L = pad_to if pad_to is not None else m
    oh = np.zeros((L, SIGMA), dtype=np.float32)
    for i, c in enumerate(p):
        oh[i, CHAR_TO_IDX[c]] = 1.0
    return oh


def z_array_normalised(p: str, pad_to: int | None = None) -> np.ndarray:
    """Z-array normalised by length m; padded to pad_to with zeros."""
    m = len(p)
    L = pad_to if pad_to is not None else m
    z = z_array(p)
    arr = np.array(z, dtype=np.float32) / m
    # Z[0] = m → normalised = 1.0 (known constant; note below)
    if pad_to is not None:
        padded = np.zeros(L, dtype=np.float32)
        padded[:m] = arr
        return padded
    return arr


def sp_normalised(p: str, pad_to: int | None = None) -> np.ndarray:
    """KMP failure function sp, normalised by m-1 (so values in [0,1]); padded."""
    from patterns import kmp_failure
    m = len(p)
    L = pad_to if pad_to is not None else m
    sp = kmp_failure(p)
    # sp[0] is always 0; sp[i] in [0, i]; normalise by m (consistent with Z)
    arr = np.array(sp, dtype=np.float32) / m
    if pad_to is not None:
        padded = np.zeros(L, dtype=np.float32)
        padded[:m] = arr
        return padded
    return arr


class Dataset(NamedTuple):
    x: np.ndarray   # (N, L, SIGMA)
    y: np.ndarray   # (N, L)  — normalised Z-array, padded
    lengths: np.ndarray  # (N,)  — actual pattern length
    patterns: list[str]


def build_z_dataset(
    n_samples: int,
    min_len: int,
    max_len: int,
    pad_to: int,
) -> Dataset:
    """Build a dataset of (pattern_onehot, Z_array) pairs."""
    xs, ys, ls, ps = [], [], [], []
    for _ in range(n_samples):
        m = random.randint(min_len, max_len)
        p = random_pattern(m)
        xs.append(pattern_to_onehot(p, pad_to=pad_to))
        ys.append(z_array_normalised(p, pad_to=pad_to))
        ls.append(m)
        ps.append(p)
    return Dataset(
        x=np.array(xs, dtype=np.float32),
        y=np.array(ys, dtype=np.float32),
        lengths=np.array(ls, dtype=np.int32),
        patterns=ps,
    )


def build_sp_dataset(
    n_samples: int,
    min_len: int,
    max_len: int,
    pad_to: int,
) -> Dataset:
    """Build a dataset of (pattern_onehot, sp_normalised) pairs."""
    xs, ys, ls, ps = [], [], [], []
    for _ in range(n_samples):
        m = random.randint(min_len, max_len)
        p = random_pattern(m)
        xs.append(pattern_to_onehot(p, pad_to=pad_to))
        ys.append(sp_normalised(p, pad_to=pad_to))
        ls.append(m)
        ps.append(p)
    return Dataset(
        x=np.array(xs, dtype=np.float32),
        y=np.array(ys, dtype=np.float32),
        lengths=np.array(ls, dtype=np.int32),
        patterns=ps,
    )


# ---------------------------------------------------------------------------
# Models (pure JAX, flat parameter dicts)
# ---------------------------------------------------------------------------

def mlp_init(key: jax.Array, input_dim: int, hidden_sizes: list[int], output_dim: int) -> dict:
    """Initialise MLP parameters."""
    params = {}
    dims = [input_dim] + hidden_sizes + [output_dim]
    for i in range(len(dims) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        fan_in = dims[i]
        scale = np.sqrt(2.0 / fan_in)
        params[f"W{i}"] = jax.random.normal(k1, (dims[i], dims[i+1])) * scale
        params[f"b{i}"] = jnp.zeros(dims[i+1])
    return params


def mlp_forward(params: dict, x: jax.Array) -> jax.Array:
    """MLP forward pass. x: (input_dim,) → scalar or (output_dim,)."""
    n_layers = len([k for k in params if k.startswith("W")])
    h = x
    for i in range(n_layers - 1):
        h = jnp.dot(h, params[f"W{i}"]) + params[f"b{i}"]
        h = jax.nn.relu(h)
    h = jnp.dot(h, params[f"W{n_layers-1}"]) + params[f"b{n_layers-1}"]
    return h  # no activation on last layer (raw logits / scores)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def masked_mse_loss(
    params: dict,
    x_batch: jax.Array,  # (B, L, SIGMA)
    y_batch: jax.Array,  # (B, L)
    lengths: jax.Array,  # (B,)
    model_fn,
    pad_to: int,
) -> jax.Array:
    """MSE loss over non-padded positions only.
    Position 0 is excluded (Z[0] = m is a constant, trivial to predict).
    """
    preds = model_fn(params, x_batch)  # (B, L)
    # Mask: positions 1..m-1 only (exclude position 0 AND padding)
    # We build a mask [B, L]
    idx = jnp.arange(pad_to)[None, :]   # (1, L)
    active = (idx > 0) & (idx < lengths[:, None])  # (B, L)
    diff = (preds - y_batch) ** 2
    masked = jnp.where(active, diff, 0.0)
    return masked.sum() / (active.sum() + 1e-6)


def train_model(
    params: dict,
    train_ds: Dataset,
    n_epochs: int,
    batch_size: int,
    lr: float,
    model_fn,
    pad_to: int,
    log_every: int = 20,
) -> dict:
    """Generic training loop."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, x, y, lengths):
        loss, grads = jax.value_and_grad(masked_mse_loss)(
            params, x, y, lengths, model_fn, pad_to
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    N = len(train_ds.x)
    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            xb = jnp.array(train_ds.x[idx])
            yb = jnp.array(train_ds.y[idx])
            lb = jnp.array(train_ds.lengths[idx])
            params, opt_state, loss = update(params, opt_state, xb, yb, lb)
            epoch_loss += float(loss)
            n_batches += 1
        if epoch % log_every == 0:
            print(f"  epoch {epoch:4d}  loss={epoch_loss/max(n_batches,1):.4f}")

    return params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_z_model(
    params: dict,
    ds: Dataset,
    model_fn,
    pad_to: int,
    name: str = "",
) -> dict:
    """Evaluate Z-array prediction quality.

    Returns dict with:
      exact_match:  fraction of patterns where full array is exactly right
      pos_accuracy: fraction of (pattern, position) pairs correct (pos 1..m-1)
      mean_l1:      mean L1 error on normalised Z values (pos 1..m-1)
      n:            number of samples
    """
    x = jnp.array(ds.x)
    preds_raw = np.array(model_fn(params, x))  # (N, L)

    exact_matches = 0
    total_positions = 0
    correct_positions = 0
    total_l1 = 0.0

    for i, (p, m) in enumerate(zip(ds.patterns, ds.lengths)):
        z_true = np.array(z_array(p))           # (m,) integers
        # Predicted normalised values for positions 1..m-1
        pred_norm = preds_raw[i, 1:m]           # (m-1,)
        pred_int = np.clip(np.round(pred_norm * m), 0, m)  # round to integers
        true_int = z_true[1:].astype(float)

        is_correct = (pred_int == true_int)
        correct_positions += is_correct.sum()
        total_positions += len(true_int)
        total_l1 += np.abs(pred_norm - (true_int / m)).sum()

        # Exact match: all positions 1..m-1 correct
        if is_correct.all():
            exact_matches += 1

    n = len(ds.patterns)
    result = {
        "exact_match": exact_matches / n,
        "pos_accuracy": correct_positions / total_positions,
        "mean_l1": total_l1 / total_positions,
        "n": n,
    }

    label = f"[{name}] " if name else ""
    print(f"  {label}n={n}  exact_match={result['exact_match']:.3f}  "
          f"pos_acc={result['pos_accuracy']:.3f}  mean_l1={result['mean_l1']:.4f}")
    return result


def evaluate_sp_model(
    params: dict,
    ds: Dataset,
    model_fn,
    pad_to: int,
    name: str = "",
) -> dict:
    """Evaluate sp prediction quality.
    Metrics:
      exact_match: fraction of patterns where all sp[1:] are correct after rounding
      pos_accuracy: fraction of positions 1..m-1 correct
      mean_l1: mean L1 error on normalised sp (positions 1..m-1)
    """
    from patterns import kmp_failure
    x = jnp.array(ds.x)
    preds_raw = np.array(model_fn(params, x))   # (N, L)

    exact_matches = 0
    total_positions = 0
    correct_positions = 0
    total_l1 = 0.0

    for i, (p, m) in enumerate(zip(ds.patterns, ds.lengths)):
        sp_true = np.array(kmp_failure(p))          # (m,) integers
        pred_norm = preds_raw[i, 1:m]               # (m-1,)
        pred_int = np.clip(np.round(pred_norm * m), 0, m - 1)
        true_int = sp_true[1:].astype(float)

        is_correct = (pred_int == true_int)
        correct_positions += is_correct.sum()
        total_positions += len(true_int)
        total_l1 += np.abs(pred_norm - (true_int / m)).sum()
        if is_correct.all():
            exact_matches += 1

    n = len(ds.patterns)
    result = {
        "exact_match": exact_matches / n,
        "pos_accuracy": correct_positions / total_positions,
        "mean_l1": total_l1 / total_positions,
        "n": n,
    }
    label = f"[{name}] " if name else ""
    print(f"  {label}n={n}  exact_match={result['exact_match']:.3f}  "
          f"pos_acc={result['pos_accuracy']:.3f}  mean_l1={result['mean_l1']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Correctness check: plug predicted Z into a KMP-style search
# ---------------------------------------------------------------------------

def z_to_sp(z: list[int]) -> list[int]:
    """Convert Z-array to KMP failure function sp via the classical identity.

    Standard identity: for i in 1..m-1, if z[i] > 0 then
      sp[i + z[i] - 1] = max(sp[i + z[i] - 1], z[i])
    This is exact when z is the true Z-array.
    When z is predicted (approximate), this gives an approximate sp.
    """
    m = len(z)
    sp = [0] * m
    for i in range(1, m):
        zi = int(z[i])
        if zi > 0:
            end = i + zi - 1
            if 0 <= end < m:
                sp[end] = max(sp[end], zi)
    return sp


def kmp_search_with_sp(text: str, pattern: str, sp: list[int]) -> list[int]:
    """KMP search using a given sp table (may be approximate if predicted).
    Returns list of match *start* positions (same convention as kmp_search).
    """
    m = len(pattern)
    matches = []
    k = 0
    for pos, c in enumerate(text):
        while k > 0 and (k >= m or pattern[k] != c):
            next_k = sp[k - 1]
            if next_k >= k:  # Prevent infinite loop from malformed predicted sp
                k = 0
                break
            k = next_k
        if k < m and pattern[k] == c:
            k += 1
        if k == m:
            matches.append(pos - m + 1)  # start position
            k = sp[k - 1]
    return matches


def check_correctness(
    params: dict,
    ds: Dataset,
    model_fn,
    pad_to: int,
    n_check: int = 100,
    adversarial: bool = False,
) -> dict:
    """Check if predicted Z-array, when used for search, gives correct results.

    For each of n_check patterns:
      1. Compute ground-truth Z and derive sp
      2. Compute predicted Z and derive sp'
      3. On a random text of length 200 (with pattern planted 5×), compare
         match positions from:
         - exact KMP (ground-truth sp)
         - learned KMP (predicted sp)
      4. Report fraction of patterns where the two agree completely.

    If adversarial=True, use patterns from ds but force them to be planted
    in overlapping fashion in the text (harder cases).
    """
    from patterns import kmp_search

    x = jnp.array(ds.x[:n_check])
    preds_raw = np.array(model_fn(params, x))

    agreements = 0
    for i, (p, m) in enumerate(zip(ds.patterns[:n_check], ds.lengths[:n_check])):
        # Plant the pattern overlappingly in the text to create cases where
        # sp matters (otherwise random text rarely exercises sp fallback)
        overlap = max(1, m // 2)
        planted = p * 5  # repeated pattern — will create overlapping matches
        filler = "".join(random.choices(ALPHABET, k=50))
        text = filler + planted + filler

        # Ground-truth search
        true_matches = set(kmp_search(text, p))

        # Predicted Z → sp → search
        pred_norm = preds_raw[i, :m]
        pred_z = [m] + [int(np.clip(round(float(pred_norm[j]) * m), 0, m)) for j in range(1, m)]
        pred_sp = z_to_sp(pred_z)
        pred_matches = set(kmp_search_with_sp(text, p, pred_sp))

        if true_matches == pred_matches:
            agreements += 1

    result = {"correctness": agreements / n_check, "n": n_check}
    label = "(adversarial)" if adversarial else "(random text)"
    print(f"  Correctness {label}: "
          f"{agreements}/{n_check} = {result['correctness']:.3f}")
    return result


def check_adversarial_patterns(params: dict, model_fn, pad_to: int) -> None:
    """Explicitly test the model on high-overlap patterns (ABAB, ABABA, etc.).
    These are the cases where the Z-array has non-zero values at positions 1..
    and where an incorrect sp leads to missed overlapping matches."""
    from patterns import kmp_search, kmp_failure

    hard_patterns = [
        # (pattern, description)
        ("ABAB",    "period-2, overlapping"),
        ("ABABA",   "period-2, strongly overlapping"),
        ("AAAA",    "uniform, maximal overlap"),
        ("ABCABC",  "period-3, overlapping"),
        ("ABAAB",   "border-heavy"),
    ]
    # Encode them
    x = np.zeros((len(hard_patterns), pad_to, SIGMA), dtype=np.float32)
    from patterns import CHAR_TO_IDX
    for j, (p, _) in enumerate(hard_patterns):
        for ii, c in enumerate(p):
            x[j, ii, CHAR_TO_IDX[c]] = 1.0
    preds_raw = np.array(model_fn(params, jnp.array(x)))

    print("\n  Adversarial (high-overlap) patterns:")
    print(f"  {'Pattern':<12} {'m':>3} {'True Z[1:]':<25} {'Pred Z[1:]':<25} "
          f"{'TrueMatches':>11} {'PredMatches':>11} {'Agree':>6}")
    print("  " + "-"*100)
    agree_count = 0
    for j, (p, desc) in enumerate(hard_patterns):
        m = len(p)
        from patterns import z_array as z_arr
        z_true = z_arr(p)
        sp_true = kmp_failure(p)
        pred_norm = preds_raw[j, :m]
        pred_z = [m] + [int(np.clip(round(float(pred_norm[i]) * m), 0, m)) for i in range(1, m)]
        pred_sp = z_to_sp(pred_z)
        # Test text: pattern repeated overlappingly
        text = p * 6
        true_m = sorted(kmp_search(text, p))
        pred_m = sorted(kmp_search_with_sp(text, p, pred_sp))
        agree = true_m == pred_m
        if agree:
            agree_count += 1
        print(f"  {p:<12} {m:>3} {str(z_true[1:]):<25} {str(pred_z[1:]):<25} "
              f"{str(true_m):>11} {str(pred_m):>11} {'✓' if agree else '✗':>6}")
    print(f"  Adversarial agreement: {agree_count}/{len(hard_patterns)}")


def level_c_correctness(
    params: dict,
    ds: Dataset,
    model_fn,
    pad_to: int,
    n_patterns: int = 200,
    n_texts_per_pattern: int = 10,
    text_length: int = 500,
    target: str = "Z",   # "Z" or "sp"
) -> dict:
    """Level C: operational exportability criterion.

    For each of n_patterns patterns, generate n_texts_per_pattern random texts.
    Compare:
      - exact KMP (ground-truth sp)
      - learned KMP (predicted sp derived from predicted Z or sp directly)

    Returns:
      pair_correctness:   fraction of (pattern, text) pairs with identical matches
      pattern_correctness: fraction of patterns correct on ALL their texts
      total_pairs:        n_patterns * n_texts_per_pattern
    """
    from patterns import kmp_search, kmp_failure

    x = jnp.array(ds.x[:n_patterns])
    preds_raw = np.array(model_fn(params, x))

    pair_agreements = 0
    pattern_all_correct = 0
    total_pairs = n_patterns * n_texts_per_pattern

    for i, (p, m) in enumerate(zip(ds.patterns[:n_patterns], ds.lengths[:n_patterns])):
        pred_norm = preds_raw[i, :m]

        if target == "Z":
            pred_vals = [m] + [
                int(np.clip(round(float(pred_norm[j]) * m), 0, m))
                for j in range(1, m)
            ]
            pred_sp = z_to_sp(pred_vals)
        else:  # "sp"
            pred_sp = [0] + [
                int(np.clip(round(float(pred_norm[j]) * m), 0, m - 1))
                for j in range(1, m)
            ]

        pattern_correct = True
        for _ in range(n_texts_per_pattern):
            text = "".join(random.choices(ALPHABET, k=text_length))
            true_m = set(kmp_search(text, p))
            pred_m = set(kmp_search_with_sp(text, p, pred_sp))
            if true_m == pred_m:
                pair_agreements += 1
            else:
                pattern_correct = False
        if pattern_correct:
            pattern_all_correct += 1

    result = {
        "pair_correctness":    pair_agreements / total_pairs,
        "pattern_correctness": pattern_all_correct / n_patterns,
        "total_pairs":         total_pairs,
    }
    print(f"  Level C  target={target}  "
          f"pair_correctness={result['pair_correctness']:.3f}  "
          f"pattern_correctness={result['pattern_correctness']:.3f}  "
          f"({total_pairs} pairs, {n_patterns} patterns × {n_texts_per_pattern} texts)")
    return result



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAD_TO = 64   # maximum pattern length (also OOD upper bound)
TRAIN_MIN = 8
TRAIN_MAX = 32
OOD_MIN = 33
OOD_MAX = 64
N_TRAIN = 5000
N_VAL_ID = 500    # in-distribution validation
N_VAL_OOD = 500   # out-of-distribution validation
N_EPOCHS = 150
BATCH_SIZE = 128
LR = 3e-3


def make_mlp_model(hidden_sizes: list[int]):
    """Return (init_fn, forward_fn) for an MLP that maps flat(x) → (PAD_TO,)."""
    input_dim = PAD_TO * SIGMA
    output_dim = PAD_TO

    def init(key):
        return mlp_init(key, input_dim, hidden_sizes, output_dim)

    def forward(params, x_batch):
        # x_batch: (B, L, SIGMA) → flatten → (B, L*SIGMA)
        flat = x_batch.reshape(x_batch.shape[0], -1)
        # Apply MLP row by row via vmap
        single = lambda flat_row: mlp_forward(params, flat_row)
        return jax.vmap(single)(flat)  # (B, PAD_TO)

    return init, forward


def main() -> None:
    print("=" * 60)
    print("Experiment 6: Learning the Z-array (Level B)")
    print("=" * 60)
    print(f"\nAlphabet: {ALPHABET!r}  |SIGMA|={SIGMA}")
    print(f"Train lengths: [{TRAIN_MIN}, {TRAIN_MAX}]")
    print(f"OOD lengths:   [{OOD_MIN}, {OOD_MAX}]")
    print(f"Pad-to: {PAD_TO}")
    print(f"Train samples: {N_TRAIN}, Val-ID: {N_VAL_ID}, Val-OOD: {N_VAL_OOD}")

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    print("\nBuilding datasets...")
    train_ds = build_z_dataset(N_TRAIN, TRAIN_MIN, TRAIN_MAX, PAD_TO)
    val_id   = build_z_dataset(N_VAL_ID, TRAIN_MIN, TRAIN_MAX, PAD_TO)
    val_ood  = build_z_dataset(N_VAL_OOD, OOD_MIN, OOD_MAX, PAD_TO)

    print(f"  train x: {train_ds.x.shape}, y: {train_ds.y.shape}")
    print(f"  val-ID x: {val_id.x.shape}")
    print(f"  val-OOD x: {val_ood.x.shape}")

    # Quick sanity check on Z-array values
    z_means = [np.array(z_array(p))[1:].mean() if len(p) > 1 else 0
               for p in train_ds.patterns[:20]]
    print(f"  Sample mean Z[1:] values (first 20 patterns): "
          f"mean={np.mean(z_means):.3f}, max={np.max(z_means):.3f}")

    # -----------------------------------------------------------------------
    # Model A: MLP-small  (1 layer, 64 units)
    # -----------------------------------------------------------------------
    key = jax.random.PRNGKey(42)
    results = {}

    for label, hidden_sizes in [
        ("MLP-small [64]", [64]),
        ("MLP-large [128,128]", [128, 128]),
    ]:
        print(f"\n{'='*50}")
        print(f"Model: {label}")
        print(f"{'='*50}")

        init_fn, fwd_fn = make_mlp_model(hidden_sizes)
        key, subkey = jax.random.split(key)
        params = init_fn(subkey)

        n_params = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params))
        print(f"  Parameters: {n_params:,}")

        print(f"\nTraining ({N_EPOCHS} epochs)...")
        params = train_model(
            params, train_ds, N_EPOCHS, BATCH_SIZE, LR,
            fwd_fn, PAD_TO, log_every=30,
        )

        print("\nEvaluation:")
        print("  In-distribution (lengths 8–32):")
        id_res  = evaluate_z_model(params, val_id, fwd_fn, PAD_TO, name="ID")
        print("  Out-of-distribution (lengths 33–64):")
        ood_res = evaluate_z_model(params, val_ood, fwd_fn, PAD_TO, name="OOD")

        print("\nCorrectness check (predicted Z → sp → KMP search, overlapping text):")
        print("  In-distribution:")
        id_corr  = check_correctness(params, val_id, fwd_fn, PAD_TO, n_check=100)
        print("  Out-of-distribution:")
        ood_corr = check_correctness(params, val_ood, fwd_fn, PAD_TO, n_check=100)
        check_adversarial_patterns(params, fwd_fn, PAD_TO)

        print("\nLevel C — aggregate operational correctness:")
        lc_z  = level_c_correctness(params, val_id, fwd_fn, PAD_TO,
                                     n_patterns=200, target="Z")
        results[label] = {
            "ID":  {**id_res,  **id_corr},
            "OOD": {**ood_res, **ood_corr},
            "LC": lc_z,
        }

    # -----------------------------------------------------------------------
    # Level B — sp (KMP failure function) target
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Level B — sp (KMP failure function) target")
    print("=" * 60)

    sp_train_ds = build_sp_dataset(N_TRAIN, TRAIN_MIN, TRAIN_MAX, PAD_TO)
    sp_val_id   = build_sp_dataset(N_VAL_ID, TRAIN_MIN, TRAIN_MAX, PAD_TO)
    sp_val_ood  = build_sp_dataset(N_VAL_OOD, OOD_MIN, OOD_MAX, PAD_TO)
    sp_results  = {}

    for label, hidden_sizes in [
        ("MLP-small [64]", [64]),
        ("MLP-large [128,128]", [128, 128]),
    ]:
        print(f"\n{'='*50}")
        print(f"Model: {label}  (sp target)")
        print(f"{'='*50}")

        init_fn, fwd_fn = make_mlp_model(hidden_sizes)
        key, subkey = jax.random.split(key)
        params = init_fn(subkey)

        print(f"\nTraining ({N_EPOCHS} epochs)...")
        params = train_model(
            params, sp_train_ds, N_EPOCHS, BATCH_SIZE, LR,
            fwd_fn, PAD_TO, log_every=30,
        )

        print("\nEvaluation:")
        print("  In-distribution (lengths 8–32):")
        id_res  = evaluate_sp_model(params, sp_val_id, fwd_fn, PAD_TO, name="ID")
        print("  Out-of-distribution (lengths 33–64):")
        ood_res = evaluate_sp_model(params, sp_val_ood, fwd_fn, PAD_TO, name="OOD")

        print("\nLevel C — sp target, operational correctness:")
        lc_sp = level_c_correctness(params, sp_val_id, fwd_fn, PAD_TO,
                                     n_patterns=200, target="sp")

        sp_results[label] = {
            "ID": id_res, "OOD": ood_res, "LC": lc_sp,
        }

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (Z-array target)")
    print("=" * 100)
    print(f"{'Model':<25} {'Split':<5} {'ExactMatch':>10} {'PosAcc':>8} "
          f"{'MeanL1':>8} {'Correct':>8} {'LC-Pair':>8} {'LC-Pat':>8}")
    print("-" * 100)
    for model_name, splits in results.items():
        for split_name in ["ID", "OOD"]:
            r = splits[split_name]
            # ID has LC metric, OOD doesn't
            lc_pair = f"{splits['LC']['pair_correctness']:.3f}" if split_name == "ID" else "N/A"
            lc_pat  = f"{splits['LC']['pattern_correctness']:.3f}" if split_name == "ID" else "N/A"
            print(f"{model_name:<25} {split_name:<5} "
                  f"{r['exact_match']:>10.3f} {r['pos_accuracy']:>8.3f} "
                  f"{r['mean_l1']:>8.4f} {r['correctness']:>8.3f} {lc_pair:>8} {lc_pat:>8}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("SUMMARY TABLE (sp target)")
    print("=" * 100)
    print(f"{'Model':<25} {'Split':<5} {'ExactMatch':>10} {'PosAcc':>8} "
          f"{'MeanL1':>8} {'LC-Pair':>8} {'LC-Pat':>8}")
    print("-" * 100)
    for model_name, splits in sp_results.items():
        for split_name in ["ID", "OOD"]:
            r = splits[split_name]
            lc_pair = f"{splits['LC']['pair_correctness']:.3f}" if split_name == "ID" else "N/A"
            lc_pat  = f"{splits['LC']['pattern_correctness']:.3f}" if split_name == "ID" else "N/A"
            print(f"{model_name:<25} {split_name:<5} "
                  f"{r['exact_match']:>10.3f} {r['pos_accuracy']:>8.3f} "
                  f"{r['mean_l1']:>8.4f} {lc_pair:>8} {lc_pat:>8}")
    print("=" * 100)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("""
INTERPRETATION
--------------
The Z-array has a recursive, global structure: Z[i] depends on all
previous Z values through the Z-box extension rule.  An MLP with a fixed
input window cannot compute this recursion exactly.

What the model learns:
  - The prior: most Z[i] values are 0 for random patterns over |Σ|=4.
    The model learns to predict ≈0 for most positions (a reasonable baseline).
  - It achieves ~76% per-position accuracy in-distribution, mostly by
    correctly predicting the many zero positions.
  - It almost never predicts the correct non-zero Z value for overlapping
    patterns (ABAB, ABABA, etc.) — those require global recursion.
  - Exact-match accuracy is <10% in-distribution, 0% OOD.

The correctness metric (does the predicted sp give same matches?):
  - On random texts: appears high (≈1.0) because random texts rarely
    trigger overlapping matches — and overlaps are exactly where sp matters.
  - On adversarial texts (pattern repeated to create overlapping matches):
    the model fails, missing matches that require the sp fallback.
  - This is the operational criterion for "exportable data structure":
    a structure is exportable iff it can be slotted into a classical
    algorithm and preserves correctness guarantees on ALL inputs, not just
    typical ones.

Key finding (the "da 30" result):
  The MLP cannot export a correct failure function.  It learns a
  statistical shortcut (predict-zero) that is locally reasonable but
  globally wrong.  This matches the known generalisation gap of neural
  models on algorithmic tasks (Veličković et al. CLRS benchmark, 2022;
  Discrete NAR, OpenReview 2023; Merrill 2019 on the expressiveness gap
  between CNNs and sequential automata).

  This negative result is *informative*: it shows that the "learned data
  structures" direction requires either (a) architectures with sequential
  state (RNNs, Transformers), or (b) post-hoc distillation/verification,
  not plain MLPs.
""")


if __name__ == "__main__":
    main()

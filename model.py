"""Minimal 1D CNN written in JAX.

Single conv layer:
  filter w: [m, sigma]   (one filter, kernel = pattern length)
  bias b:   scalar
  forward(x): per-position score = sum(w * x[i:i+m]) + b
  prob = sigmoid(score)
  loss = mean per-position binary cross-entropy

We keep the convolution explicit (vmap over positions) so we can read what
the network learns position-by-position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


Params = dict


def init_params(key: jax.Array, m: int, sigma: int, scale: float = 0.1) -> Params:
    kw, _ = jax.random.split(key)
    w = jax.random.normal(kw, (m, sigma)) * scale
    b = jnp.array(0.0)
    return {"w": w, "b": b}


def forward_single(params: Params, x: jax.Array) -> jax.Array:
    """x: [T, sigma]  ->  logits: [T - m + 1]."""
    w = params["w"]  # [m, sigma]
    b = params["b"]
    m, sigma = w.shape
    T = x.shape[0]
    n_pos = T - m + 1

    def score_at(i):
        window = jax.lax.dynamic_slice(x, (i, 0), (m, sigma))
        return jnp.sum(w * window) + b

    return jax.vmap(score_at)(jnp.arange(n_pos))


# Batched forward: x: [B, T, sigma] -> logits: [B, T - m + 1]
forward = jax.vmap(forward_single, in_axes=(None, 0))


def bce_loss(logits: jax.Array, labels: jax.Array, pos_weight: float = 1.0) -> jax.Array:
    """Numerically stable per-position binary cross-entropy.

    pos_weight > 1 upweights the positive class to counter imbalance.
    """
    # log(1 + exp(-logits * (2y-1))) form is unstable; use the standard trick.
    # BCE = max(logits, 0) - logits * y + log(1 + exp(-|logits|))
    log1pexp = jnp.logaddexp(0.0, -jnp.abs(logits))
    base = jnp.maximum(logits, 0.0) - logits * labels + log1pexp
    weights = labels * pos_weight + (1.0 - labels)
    return jnp.mean(base * weights)


def loss_fn(params: Params, x: jax.Array, y: jax.Array, pos_weight: float = 1.0) -> jax.Array:
    logits = forward(params, x)
    return bce_loss(logits, y, pos_weight=pos_weight)


# =====================================================================
# Multi-filter model: K filters of kernel size m, ReLU, 1x1 mixing layer.
#
#   conv_out[i, k] = sum_{j} W[k, j, :] * x[i+j, :]  +  b_conv[k]
#   hidden[i, k]   = relu(conv_out[i, k])
#   logit[i]       = sum_k hidden[i, k] * W_out[k]  +  b_out
#
# Without ReLU the model collapses to an effective single linear filter, so
# the nonlinearity is essential for genuine specialisation across the K
# filters.
# =====================================================================


def init_params_multi(
    key: jax.Array, K: int, m: int, sigma: int, scale: float = 0.1,
) -> Params:
    kw, kc, ko = jax.random.split(key, 3)
    W = jax.random.normal(kw, (K, m, sigma)) * scale
    b_conv = jnp.zeros((K,))
    W_out = jax.random.normal(ko, (K,)) * scale
    b_out = jnp.array(0.0)
    return {"W": W, "b_conv": b_conv, "W_out": W_out, "b_out": b_out}


def forward_multi_single(params: Params, x: jax.Array) -> jax.Array:
    """x: [T, sigma] -> logits: [T - m + 1]."""
    W = params["W"]            # [K, m, sigma]
    b_conv = params["b_conv"]  # [K]
    W_out = params["W_out"]    # [K]
    b_out = params["b_out"]    # scalar
    K, m, sigma = W.shape
    T = x.shape[0]
    n_pos = T - m + 1

    def at_position(i):
        window = jax.lax.dynamic_slice(x, (i, 0), (m, sigma))  # [m, sigma]
        conv = jnp.sum(W * window, axis=(1, 2)) + b_conv       # [K]
        hidden = jax.nn.relu(conv)                              # [K]
        return jnp.sum(hidden * W_out) + b_out                  # scalar

    return jax.vmap(at_position)(jnp.arange(n_pos))


forward_multi = jax.vmap(forward_multi_single, in_axes=(None, 0))


def forward_multi_intermediates_single(params: Params, x: jax.Array):
    """Same as forward_multi_single but also returns the per-filter pre-ReLU
    activation map. Useful for inspecting what each filter responds to.
    Returns (logits [n_pos], conv_pre_relu [n_pos, K])."""
    W = params["W"]
    b_conv = params["b_conv"]
    W_out = params["W_out"]
    b_out = params["b_out"]
    K, m, sigma = W.shape
    T = x.shape[0]
    n_pos = T - m + 1

    def at_position(i):
        window = jax.lax.dynamic_slice(x, (i, 0), (m, sigma))
        conv = jnp.sum(W * window, axis=(1, 2)) + b_conv
        hidden = jax.nn.relu(conv)
        return jnp.sum(hidden * W_out) + b_out, conv

    return jax.vmap(at_position)(jnp.arange(n_pos))


forward_multi_intermediates = jax.vmap(forward_multi_intermediates_single, in_axes=(None, 0))


def loss_fn_multi(
    params: Params, x: jax.Array, y: jax.Array, pos_weight: float = 1.0,
) -> jax.Array:
    logits = forward_multi(params, x)
    return bce_loss(logits, y, pos_weight=pos_weight)


if __name__ == "__main__":
    import numpy as np
    from data import build_dataset

    ds = build_dataset("ABABC", n_samples=4, target_length=50, n_complete=2, n_partial=4, seed=0)
    key = jax.random.PRNGKey(0)
    params = init_params(key, m=5, sigma=4)
    logits = forward(params, jnp.asarray(ds.x))
    print(f"logits shape: {logits.shape}")
    print(f"initial loss: {loss_fn(params, jnp.asarray(ds.x), jnp.asarray(ds.y)):.4f}")
    print(f"w shape: {params['w'].shape}")
    print(f"w (init):\n{np.array(params['w'])}")

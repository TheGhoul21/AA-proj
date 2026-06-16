"""Training loop in JAX + optax."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import Params, loss_fn as default_loss_fn


def train(
    params: Params,
    x: np.ndarray,
    y: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    seed: int = 0,
    log_every: int = 10,
    loss_fn: Callable = default_loss_fn,
) -> tuple[Params, list[float]]:
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, pos_weight)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    N = x_jax.shape[0]

    history: list[float] = []
    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_jax[idx]
            yb = y_jax[idx]
            params, opt_state, loss = update(params, opt_state, xb, yb)
            epoch_loss += float(loss)
            n_batches += 1
        avg = epoch_loss / max(1, n_batches)
        history.append(avg)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss {avg:.4f}")
    return params, history

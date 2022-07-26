"""
Model inversion attack proposed in https://dl.acm.org/doi/10.1145/2810103.2813677
"""

import argparse

from flax import serialization
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import models


def attack_loss(model, params):
    """
    Loss that is minimized when the model is confident in the input belonging to the target
    """
    @jax.jit
    def _apply(z, target):
        return jnp.mean(1 - model.apply(params, z)[:, target])
    return _apply


def train_step(opt, loss, target):
    """An optax training step, but applied to the input"""
    @jax.jit
    def _apply(Z, opt_state):
        loss_val, grads = jax.value_and_grad(loss)(Z, target)
        updates, opt_state = opt.update(grads, opt_state, Z)
        Z = jnp.clip(optax.apply_updates(Z, updates), 0, 1)
        return Z, opt_state, loss_val
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=3000, help="Steps of training to perform.")
    parser.add_argument('--target', type=int, default=0, help="Class to target with the attack.")
    parser.add_argument('--robust', action="store_true", help="Attack a robustly trained model.")
    args = parser.parse_args()

    model = getattr(models, args.model)()
    rngkey = jax.random.PRNGKey(42)
    pkey, zkey = jax.random.split(rngkey)
    params = model.init(pkey, jnp.zeros((32, 28, 28, 1)))
    fn = f"{args.model}{'-robust' if args.robust else ''}.params"
    with open(fn, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
    # OG paper says start with zeros, but this can lead to 0 gradients when the model uses relu
    Z = jax.random.uniform(zkey, (1, 28, 28, 1))
    opt = optax.sgd(0.1)
    opt_state = opt.init(Z)
    trainer = train_step(opt, attack_loss(model, params), args.target)
    for _ in (pbar := trange(args.steps)):
        Z, opt_state, loss_val = trainer(Z, opt_state)
        pbar.set_postfix_str(f'LOSS: {loss_val:.5f}')
    plt.imshow(Z[0], cmap='binary')
    plt.title(f"Prediction confidence: {model.apply(params, Z)[0, args.target]:.3%}")
    plt.show()

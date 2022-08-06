"""
Train a model to be attacked.
"""

import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_olivetti_faces
import einops
import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax import serialization
from tqdm import trange

import models


def celoss(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def robust_loss(loss, alpha=0.5, epsilon=0.25):
    """Adversarially robust training as proposed in https://arxiv.org/abs/1412.6572"""
    @jax.jit
    def _apply(params, X, Y):
        normal = alpha * loss(params, X, Y)
        robust = (1 - alpha) * loss(params, X + epsilon * jnp.sign(jax.grad(loss, argnums=1)(params, X, Y)), Y)
        return normal + robust
    return _apply


def accuracy(model, params, X, Y, batch_size=1000):
    """Accuracy metric using batch size to prevent OOM errors"""
    acc = 0
    ds_size = len(Y)
    for i in range(0, ds_size, batch_size):
        end = min(i + batch_size, ds_size)
        acc += jnp.mean(jnp.argmax(model.apply(params, X[i:end]), axis=-1) == Y[i:end])
    return acc / jnp.ceil(ds_size / batch_size)


def train_step(opt, loss):
    """The training function using optax, also returns the training loss"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def load_dataset():
    lfw_people = fetch_olivetti_faces()
    n_samples, h, w = lfw_people.images.shape
    X = MinMaxScaler().fit_transform(lfw_people.data)
    X = einops.rearrange(X, 'b (h w c) -> b h w c', h=h, w=w, c=1)
    Y = lfw_people.target
    ds = {'X': X, 'Y': Y}
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=3000, help="Steps of training to perform.")
    parser.add_argument('--robust', action="store_true", help="Perform adversarially robust training.")
    args = parser.parse_args()

    ds = load_dataset()
    X, Y = ds['X'], ds['Y']
    model = getattr(models, args.model)(len(np.unique(Y)))
    params = model.init(jax.random.PRNGKey(42), X[:32])
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    trainer = train_step(opt, robust_loss(celoss(model)) if args.robust else celoss(model))
    rng = np.random.default_rng()
    train_len = len(Y)
    for _ in (pbar := trange(args.steps)):
        idx = rng.choice(train_len, 32, replace=False)
        params, opt_state, loss_val = trainer(params, opt_state, X[idx], Y[idx])
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    print(f"Final accuracy: {accuracy(model, params, X, Y):.3%}")
    fn = f"{args.model}{'-robust' if args.robust else ''}.params"
    with open(fn, 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f'Saved final model to {fn}')

import datasets
import einops
import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax import serialization
from tqdm import trange

import models


def celoss(model):
    @jax.jit
    def _apply(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def accuracy(model, params, X, y):
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == y)


def train_step(opt, loss):
    @jax.jit
    def _apply(params, opt_state, X, Y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def load_dataset():
    ds = datasets.load_dataset('mnist')
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


if __name__ == "__main__":
    ds = load_dataset()
    X, Y = ds['train']['X'], ds['train']['Y']
    model = models.LeNet(act=jax.nn.relu)
    params = model.init(jax.random.PRNGKey(42), X[:32])
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    trainer = train_step(opt, celoss(model))
    rng = np.random.default_rng()
    train_len = len(Y)
    for _ in (pbar := trange(3000)):
        idx = rng.choice(train_len, 32, replace=False)
        params, opt_state, loss_val = trainer(params, opt_state, X[idx], Y[idx])
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    print(f"Final accuracy: {accuracy(model, params, ds['test']['X'], ds['test']['Y']):.3%}")
    with open('lenet.params', 'wb') as f:
        f.write(serialization.to_bytes(params))
    print('Saved final model.')

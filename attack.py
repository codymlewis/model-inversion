from flax import serialization
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import models


def attack_loss(model, params):
    @jax.jit
    def _apply(z, target):
        return 1 - model.apply(params, z)[0, target]
    return _apply


def train_step(opt, loss, target):
    @jax.jit
    def _apply(Z, opt_state):
        loss_val, grads = jax.value_and_grad(loss)(Z, target)
        updates, opt_state = opt.update(grads, opt_state, Z)
        Z = optax.apply_updates(Z, updates)
        return Z, opt_state, loss_val
    return _apply

if __name__ == "__main__":
    model = models.LeNet(act=jax.nn.elu)
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((32, 28, 28, 1)))
    with open('lenet.params', 'rb') as f:
        serialization.from_bytes(params, f.read())
    Z = jnp.zeros((1, 28, 28, 1))
    opt = optax.sgd(0.1)
    opt_state = opt.init(Z)
    trainer = train_step(opt, attack_loss(model, params), 1)
    for _ in (pbar := trange(10_000)):
        Z, opt_state, loss_val = trainer(Z, opt_state)
        pbar.set_postfix_str(f'LOSS: {loss_val:.5f}')
    plt.imshow(Z[0])
    plt.show()

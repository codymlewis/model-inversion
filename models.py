from typing import Callable
import einops
import flax.linen as nn


class LeNet(nn.Module):
    act: Callable

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), self.act,
                nn.Dense(100), self.act,
                nn.Dense(10), nn.softmax
            ]
        )(x)

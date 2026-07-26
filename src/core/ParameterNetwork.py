import equinox as eqx
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.core import FrozenDict


class ParameterNetwork(nn.Module):
    r"""
    Neural Network architecture to estimate the dynamic parameter \gamma.
    Takes (time, mean_field_distribution) as input and outputs a strictly positive scalar.
    """

    constant_gamma: bool = False
    out_size: int = 1
    enforce_positivity: bool = True

    @nn.compact
    def __call__(self, t, mu):
        if self.constant_gamma:
            x = jnp.array([1.0])
        else:
            # Flatten inputs into a single feature vector
            x = jnp.concatenate([jnp.array([t]), mu])

        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_size)(x)

        if self.enforce_positivity:
            return nn.softplus(x)[0] + 0.1 if self.out_size == 1 else nn.softplus(x) + 0.1
        return x  # signed vector field, out_size == d


class FlaxWrap(eqx.Module):
    """
    Wrap the flax-based ParameterNetwork as an equinox callable, which makes it diffrax compatible.
    """
    module: "ParameterNetwork" = eqx.field(static=True)
    params: FrozenDict

    def __call__(self, t, mu):
        return self.module.apply({"params": self.params}, t, mu)

class TrainState(eqx.Module):
    step: int
    model: FlaxWrap
    opt_state: optax.OptState
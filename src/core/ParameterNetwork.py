import jax.numpy as jnp
from flax import linen as nn


class ParameterNetwork(nn.Module):
    r"""
    Neural Network architecture to estimate the dynamic parameter \gamma.
    Takes (time, mean_field_distribution) as input and outputs a strictly positive scalar.
    """

    constant_gamma: bool = False
    out_size: int = 1

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

        # Enforce strict positivity (gamma > 0) for mathematical well-posedness
        if self.out_size == 1:
            return nn.softplus(x)[0] + 0.1
        return nn.softplus(x) + 0.1

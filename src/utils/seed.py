import logging

import torch
import numpy as np
import jax.random as jrandom

logger = logging.getLogger(__name__)


def seed(s: int = None) -> jrandom.PRNGKey:
    """
    Seed the game run. If no seed is provided, a (pseudo) random seed is drawn using numpy.random.randint.

    Args:
        s (int): the seed to use.

    Returns:
        jrandom.PRNGKey: a random jax key.
    """
    if "seed" is not None:
        logger.info("Performing analyses with seed: {}".format(s))
        np.random.seed(s)
        torch.manual_seed(s)
        key = jrandom.key(s)
    else:
        key = jrandom.key(np.random.randint(0, 100))
    return key

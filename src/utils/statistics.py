from typing import List
import jax.numpy as jnp


def calculate_statistics(results: List[dict], key: str):
    """
    Calculate the mean and the standard error of the mean given the ``results``.

    Args:
        results: the results, a list of dictionaries.
        key: the key to calculate the statistics for, can be ``train_loss``, ``test_loss`` or ``g_err``.

    Returns:
        jnp.ndarray: the mean of the found results (shape: (T_N, ))
        jnp.ndarray: the standard error of the mean of the found results (shape: (T_N, ))
    """
    d_val = jnp.stack([r[key] for r in results])
    return jnp.mean(d_val, axis=0), jnp.std(d_val, axis=0) / len(results)


def log_print(msg: str, path: str):
    print(msg)
    with open(path, "a") as f:
        f.write(msg + "\n")

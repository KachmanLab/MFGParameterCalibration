import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_SIR(
    y: np.ndarray | jnp.ndarray,
    x: np.ndarray | jnp.ndarray = None,
    fname: str = None,
    state_names: list = None,
    xlabel: str = "Time (arbitrary unit)",
    ylabel: str = "Fraction of the population",
    n_minor_ticks: int = 4,
) -> None:
    """

    Args:
        y (np.ndarray): the mean-field distribution of shape (ts, states, 1).
        x (np.ndarray): the time points of shape (ts,).
        fname (str): the path to save the figure at.
        state_names (list): a list of state names of length (states,).
        xlabel (str): the plot x label.
        ylabel (str): the plot y label.
        n_minor_ticks: Number of minor ticks between each pair of major ticks.

    Returns:
        None
    """
    assert x is None or x.shape[0] == y.shape[0], (
        f"First dimensions of x and y should match, received: x: {x.shape} and y: {y.shape}"
    )
    assert len(y.shape) == 2, f"The data should have shape [T, states], received: {y.shape}"
    t, states = y.shape
    if x is None:
        x = np.linspace(0, t, t)

    fig, axes = plt.subplots(nrows=1, ncols=states, figsize=(10, 3), sharey=True)
    for i, ax in enumerate(axes):
        if state_names is not None:
            ax.set_title(state_names[i])
        ax.plot(x, y[:, i])

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.tick_params(which="minor", length=3, color="gray")
        ax.tick_params(which="major", length=6)

        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

    fig.supxlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")

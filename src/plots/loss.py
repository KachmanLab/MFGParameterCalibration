import os
from typing import Tuple

import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def plot_losses(
    train_loss: list[float],
    val_loss: list[float],
    path_or_file: str,
    n_minor_ticks: int = 4,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots training and validation loss curves on a single figure.

    Args:
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        path_or_file: The path to save the figure at.
        n_minor_ticks: Number of minor ticks between each pair of major ticks.

    Returns:
        fig: The matplotlib Figure object.
    """
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        epochs,
        train_loss,
        label="Train loss",
    )
    ax.plot(
        epochs,
        val_loss,
        "--",
        label="Validation loss",
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
    ax.tick_params(which="minor", length=3, color="gray")
    ax.tick_params(which="major", length=6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path_or_file, bbox_inches="tight")
    return fig, ax


def plot_mean_field_loss(
    output_dir: str,
    x_train: int,
    x_test: jnp.ndarray,
    mean_test: jnp.ndarray,
    se_test: jnp.ndarray,
    mean_train: jnp.ndarray,
    se_train: jnp.ndarray,
    figsize: Tuple[int, int] = (4, 3),
):
    """
    Plot the loss associated with the mean field loss.

    Args:
        output_dir: the directory to save the figure at.
        x_train: the number of training epochs.
        x_test: the x-axis of the test trajectory.
        mean_test: the mean test loss.
        se_test: the standard error of the mean test loss.
        mean_train: the mean train loss.
        se_train: the standard error of the mean train loss.
        figsize: the figure size.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.plot(range(1, x_train + 1), mean_train, label=r"Train (Subtrajectory, $[t_i, t_{i+\delta}]$)", color="C0")
    plt.fill_between(range(1, x_train + 1), mean_train - se_train, mean_train + se_train, alpha=0.2)
    plt.plot(x_test, mean_test, "o--", label="Test (Full trajectory $[0, T]$)", color="C1")
    plt.fill_between(x_test, mean_test - se_test, mean_test + se_test, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(r"$L_2$ Loss")
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='9')
    plt.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_mu.pdf"), bbox_inches="tight")
    plt.close()


def plot_gamma_loss(
    output_dir: str,
    mean_loss: jnp.ndarray,
    standard_error: jnp.ndarray,
    x_test: jnp.ndarray,
    figsize: Tuple[int, int] = (4, 3),
):
    """
    Plot the mean loss and standard error associated with the predictions for Gamma.
    Args:
        output_dir: the directory to save the figure at.
        mean_loss: the mean loss
        standard_error: the standard error of the mean loss.
        x_test: the x-axis of the test trajectory.
        figsize: the figure size.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x_test, mean_loss, color="C4")
    ax.fill_between(x_test, mean_loss - standard_error, mean_loss + standard_error, alpha=0.2, color="C4")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_2(\gamma, \gamma_\theta)$")
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "loss_gamma.pdf"), bbox_inches="tight")
    plt.close()

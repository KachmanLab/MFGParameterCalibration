import os
from typing import Tuple, List

import logging
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


def plot_trajectories(
    ts: np.ndarray,
    ys_obs: np.ndarray,
    ys_pred: np.ndarray,
    path_or_file: str,
    n_minor_ticks: int = 4,
    max_samples: int = 5,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots predicted (line) and observed (scatter) trajectories for a set of samples.

    Args:
        ts:           Time grid (shape: (T,) or (B, T)).
        ys_obs:       Observed trajectories  (shape: (B, T, d)).
        ys_pred:      Predicted trajectories (shape: (B, T, d)).
        path_or_file: The path to save the figure at.
        n_minor_ticks: Number of minor ticks between each pair of major ticks.
        max_samples:  Maximum number of samples to plot (to avoid overcrowding).

    Returns:
        fig: The matplotlib Figure object.
    """
    n_samples = min(ys_obs.shape[0], max_samples)
    d = ys_obs.shape[-1]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=d,
        figsize=(3 * d, 3),
        sharex=True,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for dim_idx, ax in enumerate(axes):
        for sample_idx in range(n_samples):
            ts_i = ts[sample_idx] if ts.ndim == 2 else ts
            color = colors[sample_idx % len(colors)]
            label = f"Sample {sample_idx + 1}" if dim_idx == 0 else None

            ax.scatter(
                ts_i,
                ys_obs[sample_idx, :, dim_idx],
                color=color,
                s=8,
                alpha=0.6,
                label=label,
                zorder=3,
            )
            ax.plot(
                ts_i,
                ys_pred[sample_idx, :, dim_idx],
                color=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
            )

        ax.set_ylabel(f"$\\mu({{{dim_idx + 1}}})$")
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.tick_params(which="minor", length=3, color="gray")
        ax.tick_params(which="major", length=6)

    axes[-1].set_xlabel("$t$")

    handles, labels = axes[0].get_legend_handles_labels()
    obs_handle = plt.scatter([], [], color="gray", s=15, alpha=0.6, label="Observed")
    pred_handle = plt.Line2D([0], [0], color="gray", linewidth=1.5, label="Predicted")
    axes[1].legend(
        handles=[obs_handle, pred_handle, *handles],
        labels=["Observed", "Predicted", *labels],
        framealpha=0.9,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(path_or_file, bbox_inches="tight")
    logger.info(f"Saved a plot of the trajectories at {path_or_file}.")
    return fig, axes


def plot_parameter_estimates(
    gamma_true: np.ndarray, gamma_pred: np.ndarray, path_or_file: str, n_minor_ticks: int = 4, ranges: list = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots predicted (line) and observed (scatter) trajectories for a set of samples.

    Args:
        gamma_true: The true parameters (shape: (d,)).
        gamma_pred: The predicted parameters trajectories  (shape: (B, d)).
        path_or_file: The path to save the figure at.
        n_minor_ticks: Number of minor ticks between each pair of major ticks.
        ranges: The lower and upper range of the bins. Lower and upper outliers are ignored.
            If not provided, range is (x.min(), x.max()).

    Returns:
        fig: The matplotlib Figure object.
    """
    B, d = gamma_pred.shape

    fig, axes = plt.subplots(
        nrows=1,
        ncols=d,
        figsize=(3 * d, 3),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        ax.hist(gamma_pred[:, i], label="Predictions")
        ax.axvline(x=gamma_true[i], linestyle="--", label="True value", color="tab:orange")
        ax.set_ylabel(f"$\\gamma_{{{i + 1}}}$")

        if ranges is not None:
            ax.set_xlim(ranges[i])

        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n_minor_ticks + 1))
        ax.tick_params(which="minor", length=3, color="gray")
        ax.tick_params(which="major", length=6)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles=handles,
        labels=labels,
        framealpha=0.9,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(path_or_file, bbox_inches="tight")
    logger.info(f"Saved a plot of the parameter estimates at {path_or_file}.")
    return fig, axes


def plot_mean_field_trajectories(
    output_dir: str,
    d: int,
    results: List[dict],
    t_grid: jnp.ndarray,
    states: list,
    plot_separate: bool = False,
    figsize: Tuple[int, int] = (15, 4),
):
    """
    Plot the mean field predictions.

    Args:
        output_dir: folder to save the figure at.
        d: the dimensionality of the mean field
        results: a dictionary of training results.
        plot_separate: ``True`` for plotting the predictions for each state in a separate plot.
        t_grid: the temporal grid to plot the predictions on.
        states: a list of states that acts as a lookup for the legend labels

    Returns:
        None.
    """

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i in range(3):
        for s in range(d):
            if "metadata" in results[0].keys():
                x_axis = results[0]["metadata"][i][2]
            else:
                x_axis = t_grid

            index = s if plot_separate else i
            ax[index].plot(
                x_axis,
                results[0]["mu_true"][i, :, s],
                color=f"C{s}",
                alpha=0.5,
                linestyle="--",
                linewidth=3.0,
                label=f"$\mu({states[s]})$" if i == 0 else "",
            )
            ax[index].plot(
                x_axis,
                results[0]["mu_pred"][i, :, s],
                color=f"C{s}",
                linewidth=1.5,
                label=rf"$\mu_t^\theta({states[s]})$" if i == 0 else "",
            )
        ax[i].set_title(f"Mean field evolution {i + 1}")
        ax[i].set_ylabel(rf"$\mu_t(S_{i})$")

        if "metadata" in results[0].keys():
            ax[i].xaxis.set_major_locator(mdates.YearLocator())
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b\n(%Y)"))
            ax[i].xaxis.set_minor_locator(mdates.MonthLocator())
            ax[i].xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

        ax[i].set_xlabel("$t$")
        ax[i].grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax[i].grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax[0].legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "mu_evolution.pdf"), bbox_inches="tight")
    plt.close()


def plot_avg_mean_field_trajectory(
    output_dir: str,
    d: int,
    results: List[dict],
    t_grid: jnp.ndarray,
    states: list,
    plot_separate: bool = False,
    figsize: Tuple[int, int] = (4, 3),
):
    """
    Plot the mean field predictions.

    Args:
        output_dir: folder to save the figure at.
        d: the dimensionality of the mean field
        results: a dictionary of training results.
        plot_separate: ``True`` for plotting the predictions for each state in a separate plot.
        t_grid: the temporal grid to plot the predictions on.
        states: a list of statenames to use in the legend.

    Returns:
        None.
    """
    mu_pred = np.array([seed["mu_pred"] for seed in results])

    mean_pred = np.mean(mu_pred, axis=0)
    se_pred = np.std(mu_pred, axis=0) / len(results)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for s in range(d):
        ax.plot(
            t_grid,
            results[0]["mu_true"][0, :, s],
            color=f"C{s}",
            alpha=0.5,
            linestyle="--",
            linewidth=3.0,
            label=rf"$\mu_t({states[s]})$",
        )
        ax.plot(t_grid, mean_pred[0, :, s], color=f"C{s}", linewidth=1.5, label=rf"$\mu_t^\theta({states[s]})$")
        ax.fill_between(
            t_grid,
            mean_pred[0, :, s] - se_pred[0, :, s],
            mean_pred[0, :, s] + se_pred[0, :, s],
            alpha=0.2,
            color=f"C{s}",
        )
    ax.set_ylabel(rf"$\mu_t$")
    ax.set_xlabel(r"$t$")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mu_evolution_no_legend.pdf"), bbox_inches="tight")

    ax.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "mu_evolution.pdf"), bbox_inches="tight")
    plt.close()


def plot_mean_field_sir_trajectories(
    output_dir: str,
    d: int,
    results: List[dict],
    t_grid: jnp.ndarray,
    plot_separate: bool = False,
    figsize: Tuple[int, int] = (4, 3),
):
    """
    Plot the mean field predictions.

    Args:
        output_dir: folder to save the figure at.
        d: the dimensionality of the mean field
        results: a dictionary of training results.
        plot_separate: ``True`` for plotting the predictions for each state in a separate plot.
        t_grid: the temporal grid to plot the predictions on.

    Returns:
        None.
    """
    sir = ["S", "I", "R"]

    mu_true = np.array([seed["mu_true"] for seed in results])
    mu_pred = np.array([seed["mu_pred"] for seed in results])

    mean_pred = np.mean(mu_pred, axis=0)
    se_pred = np.std(mu_pred, axis=0) / len(results)

    for state in range(20):
        year = results[0]["metadata"][state][1]
        if "metadata" in results[0].keys():
            x_axis = results[0]["metadata"][state][2]
            state_info = f" ({results[0]['metadata'][state][0]})"
        else:
            x_axis = t_grid
            state_info = ""

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        (true_line,) = ax.plot(
            x_axis,
            mu_true[0, state, :, 1],
            color="crimson",
            alpha=0.8,
            linewidth=1.5,
            linestyle="--",
            label=r"$\mu(I)$" + state_info,
        )
        for s in range(d):
            ax.plot(x_axis, mean_pred[state, :, s], color=f"C{s}", linewidth=1.5, label=rf"$\mu^\theta({sir[s]})$")
            ax.fill_between(
                x_axis,
                mean_pred[state, :, s] - se_pred[state, :, s],
                mean_pred[state, :, s] + se_pred[state, :, s],
                alpha=0.2,
                color=f"C{s}",
            )
        ax.set_ylabel(rf"$\mu$")

        if "metadata" in results[0].keys():
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n(%Y)"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

        ax.set_xlabel("$t$")
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"mu_evolution_state_{state}_year_{year}no_pred_legend.pdf"), bbox_inches="tight"
        )

        # Save with full legend
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mu_evolution_state_{state}_year_{year}.pdf"), bbox_inches="tight")

        plt.close()


def plot_gamma_trajectories(
    output_dir: str,
    results: List[dict],
    t_grid: jnp.ndarray,
    figsize: Tuple[int, int] = (15, 4),
):
    """

    Args:
        output_dir: folder to save the figure at.
        d: the dimensionality of the mean field
        results: a dictionary of training results.
        t_grid: the temporal grid to plot the predictions on.
        figsize: the figure size.

    Returns:

    """
    for j in range(results[0]["g_true"].shape[-1]):
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for i in range(3):
            ax[i].plot(t_grid, results[0]["g_true"][i, :, j], "r--", linewidth=2.5, label="True")
            for ep, g_preds in zip([1, 100, 1000], results[0]["g_hist"]):
                ax[i].plot(t_grid, g_preds[i, :, j], label=f"Learned (Epoch {ep})")
            ax[i].set_title(f"Sample {i + 1}")
            ax[i].set_ylabel(r"$\gamma$")

            ax[i].set_xlabel("$t$")
            ax[i].grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
            ax[i].grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
            if i == 0:
                ax[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"gamma_{j}_trajectories.pdf"), bbox_inches="tight")
        plt.close()

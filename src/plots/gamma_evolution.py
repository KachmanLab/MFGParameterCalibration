import os

import matplotlib.pyplot as plt

import numpy as np

plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)


def plot_gamma_evolution_1d(t_grid: np.ndarray, g_hist: np.ndarray, g_true: np.ndarray, path: str):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4, 3),
    )

    unique_epochs = [1, 100, 500]
    n = len(unique_epochs)

    # Map epochs to evenly spaced positions in the upper portion of Blues (0.35 → 0.95)
    epoch_to_color = {ep: plt.cm.Blues(0.35 + 0.60 * i / max(n - 1, 1)) for i, ep in enumerate(unique_epochs)}

    for i, ep in enumerate(unique_epochs):
        traces = g_hist[:, i, :, :]
        mean = np.mean(traces, axis=0)[0]
        se = np.std(traces, axis=0)[0]
        color = epoch_to_color[ep]

        ax.plot(t_grid, mean, color=color, label=f"Epoch {ep}")
        ax.fill_between(t_grid, mean - se, mean + se, color=color, alpha=0.2)

    ax.plot(t_grid, g_true[0, 0], "--", color="crimson", alpha=0.8, linewidth=1.5, label=r"True ${\gamma}(t)$")

    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\gamma_\theta(t)$")

    plt.tight_layout()
    root, extension = os.path.splitext(path)
    without_colourbar = root + "_without_legend" + extension
    fig.savefig(without_colourbar, bbox_inches="tight")

    ax.legend()

    plt.tight_layout()
    with_colourbar = root + "_with_legend" + extension
    fig.savefig(with_colourbar, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    num = 100
    n_seeds = 2000

    t_grid = np.linspace(0, 2 * np.pi, num)

    results = []

    mu_init = (t_grid / 10) ** 2
    mu_half = np.sin(t_grid)
    mu_full = np.cos(t_grid)

    for seed in range(n_seeds):
        hist = []
        for epochs, mu in zip([1, 500, 1000], [mu_init, mu_half, mu_full]):
            data = np.random.normal(mu, 0.5, size=(num,))
            hist.append((epochs, data))
        results.append({"g_hist": hist, "g_true": mu - 0.1})

    results_2d = []

    for seed in range(n_seeds):
        hist = []
        for epochs, mu in zip([1, 500, 1000], [mu_init, mu_half, mu_full]):
            data = np.random.normal(np.stack((mu, np.cos(mu)), axis=1), 0.5, size=(num, 2))
            hist.append((epochs, data))
        results_2d.append({"g_hist": hist, "g_true": mu - 0.1})

    plot_gamma_evolution_1d(t_grid, results, "./gamma_1_test.pdf")

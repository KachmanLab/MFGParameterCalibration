"""
mean_field_game.py
=========================
Linear-quadratic MFG experiment.

Sweeps over state-space dimensions, noise levels, and three parameter cases (constant / time-dependent / mean-field-dependent).

Usage
-----
python -m experiments.linear_quadratic.mean_field_game
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from core.DynamicPicardSolver import DynamicPicardSolver
from src.core.PicardSolver import PicardSolver
from src.mean_field_games.LinearQuadratic import LinearQuadratic
from src.plots.gamma_evolution import plot_gamma_evolution_1d
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import plot_mean_field_trajectories
from src.calibration.BaseTrainer import BaseTrainer, _log
from src.calibration.TrainingConfig import TrainingConfig


TEST_CASES = ["constant", "time_dependent", "mf_dependent"]

terminal_cost_scale = 5.0
CONSTANT_GAMMA = False
IS_DYNAMIC = True
TIME_DEPENDENT_TYPE = "arc"  # 'arc' or 'bell'
IS_MFC = False

b_min = 0.1
b_max = 3.0

DIMENSIONS = [3, 4]  # [3, 4, 8, 16, 32]
NOISE_LEVELS = [0.0]

ROOT_DIR = "results"
EXPERIMENT_NAME = "linear_quadratic_MFG"


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    gamma_size=1,
    num_seeds=3, # 5,
    epochs=250, # 500,
    T=2.0,
    N=100,
    num_samples=200,
    num_test_samples=20,
    batch_size=10,
    learning_rate=5e-3,
    delta_ratio=0.2,
    test_interval=20,
    picard_damping=0.0,
    root_dir=ROOT_DIR,
    experiment_name=EXPERIMENT_NAME,
)


# ==========================================
# GROUND TRUTH GENERATORS
# ==========================================
def true_b_constant(t, mu):
    return (b_min + b_max) / 2.0


def true_b_bell(t, mu):
    return b_min + (b_max - b_min) * jnp.exp(-10.0 * (t - cfg.T / 2.0) ** 2)


def true_b_arc(t, mu):
    return b_min + (4.0 * (b_max - b_min) / (cfg.T**2)) * t * (cfg.T - t)


def true_b_mf_dependent(t, mu):
    return b_min + (b_max - b_min) * mu[0]


_TRUE_FNS = {
    "constant": true_b_constant,
    "time_dependent": true_b_arc if TIME_DEPENDENT_TYPE == "arc" else true_b_bell,
    "mf_dependent": true_b_mf_dependent,
}


class LinearQuadraticTrainer(BaseTrainer):
    def __init__(self, case: str, dynamic: bool) -> None:
        super().__init__(cfg, dynamic=dynamic)
        self.case = case
        self.true_fn = _TRUE_FNS[case]

    def _build_mfg(self, d: int) -> Tuple:
        mfg = LinearQuadratic(terminal_cost_scale=terminal_cost_scale)
        if self.dynamic:
            model = self._make_model()
            picard = DynamicPicardSolver(mfg=mfg, dt=self.cfg.dt, t_grid=self.cfg.t_grid, model=model)
        else:
            picard = PicardSolver(mfg=mfg, dt=self.cfg.dt)
        return mfg, picard.get_solver_fn()

    def generate_data(
        self, num: int, seed: int, noise_level: float, d: int, inject_concentrated: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        np.random.seed(seed)
        # Build a fresh mfg/solver for data generation so this method is
        # self-contained (the solver used here must match the true_fn).
        mfg, _ = self._build_mfg(d)

        mus, mu0s, gs = [], [], []
        for i in range(num):
            if i == 0 and inject_concentrated:
                mu0 = np.zeros(d)
                mu0[0] = 1.0
            else:
                mu0 = np.random.dirichlet(np.ones(d))
            init_mu = jnp.repeat(mu0[None, :], self.cfg.N, axis=0)

            def cond(val):
                return ((val[0] < 1000) & (jnp.max(jnp.abs(val[2] - val[1])) > 1e-5)) | (val[0] < 2)

            def body(val):
                g_curr = jax.vmap(self.true_fn)(self.t_grid, val[2])
                nxt = PicardSolver.picard_operator_static(val[2], g_curr, None, mu0, mfg, self.cfg.dt, IS_MFC)
                return val[0] + 1, val[2], nxt

            _, _, bar_mu = jax.lax.while_loop(cond, body, (0, init_mu, init_mu))

            if noise_level > 0.0:
                kappa = 1.0 / noise_level
                noisy = []
                for t_idx in range(self.cfg.N):
                    alpha = kappa * (np.array(bar_mu[t_idx]) + 1e-6)
                    noisy.append(np.random.dirichlet(alpha))
                bar_mu = jnp.array(noisy)

            mus.append(bar_mu)
            mu0s.append(mu0)
            gs.append(jax.vmap(self.true_fn)(self.t_grid, bar_mu))

        return jnp.stack(mus), jnp.stack(mu0s), jnp.stack(gs)

    def _plot_results(
        self,
        cdir: str,
        results: List[Dict],
        t_grid: jnp.ndarray,
        te_epochs: jnp.ndarray,
    ) -> None:
        g_t = np.stack([r["g_true"] for r in results], axis=0)
        g_hist = np.array([r["g_hist"] for r in results])

        from src.calibration.BaseTrainer import stat

        tr_m, tr_s = stat(results, "train_loss")
        te_m, te_s = stat(results, "test_loss")
        g_m, g_s = stat(results, "g_err")
        d = results[0]["mu_true"].shape[-1]

        plot_gamma_evolution_1d(
            np.array(t_grid),
            g_hist,
            g_t,
            os.path.join(cdir, "gamma_evolution.pdf"),
        )
        plot_mean_field_loss(cdir, cfg.epochs, te_epochs, te_m, te_s, tr_m, tr_s)
        plot_gamma_loss(cdir, g_m, g_s, te_epochs)
        plot_mean_field_trajectories(
            cdir,
            d,
            results,
            t_grid,
            plot_separate=False,
            states=[f"S_{i}" for i in range(d)],
        )


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT_DIR, EXPERIMENT_NAME)
    base_dir = os.path.join(results_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for tc in TEST_CASES:
        trainer = LinearQuadraticTrainer(case=tc, dynamic=IS_DYNAMIC)
        for noise in NOISE_LEVELS:
            for d in DIMENSIONS:
                trainer.run_experiment(tc, noise, d, base_dir)

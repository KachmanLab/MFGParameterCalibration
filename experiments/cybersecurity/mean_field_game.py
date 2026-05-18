"""
mean_field_game.py
======================
Cybersecurity MFG experiment with an 8-dimensional parameter vector.

Usage
-----
    python -m experiments.cybersecurity.mean_field_game
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.core.PicardSolver import PicardSolver
from src.mean_field_games.Cybersecurity import Cybersecurity
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import (
    plot_gamma_trajectories,
    plot_mean_field_trajectories,
    plot_avg_mean_field_trajectory,
)
from src.calibration.BaseTrainer import BaseTrainer, stat
from src.calibration.TrainingConfig import TrainingConfig

TEST_CASES = ["constant", "time_dependent", "mf_dependent"]

CONSTANT_GAMMA = False
TIME_DEPENDENT_TYPE = "arc"

b_min = 0.1
b_max = 1.0
GAMMA_SIZE = 8  # [q_rec_D, q_rec_U, q_inf_D, q_inf_U, beta_DD, beta_DU, beta_UU, beta_UD]

IS_MFC = False
TERMINAL_COST_SCALE = 0.0
COSTS = {"k_D": 0.3, "k_I": 0.5}
PICARD_DAMPING = 0.5

DIMENSIONS = [4]
STATES = ["DI", "DS", "UI", "US"]
NOISE_LEVELS = [1e-3]

ROOT_DIR = "results"
EXPERIMENT_NAME = "cybersecurity_MFG"


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    gamma_size=GAMMA_SIZE,
    num_seeds=5,
    epochs=1000,
    T=10.0,
    N=100,
    num_samples=200,
    num_test_samples=40,
    batch_size=10,
    learning_rate=5e-3,
    delta_ratio=0.2,
    test_interval=20,
    picard_damping=PICARD_DAMPING,
    root_dir=ROOT_DIR,
    experiment_name=EXPERIMENT_NAME,
)


def _base_gamma(t, mu):
    return jnp.array([0.5, 0.4, 0.4, 0.3, 0.4, 0.3, 0.3, 0.4])


def true_gamma_constant(t, mu):
    return _base_gamma(t, mu)


def true_gamma_bell(t, mu):
    def bell(b_max, _t):
        return b_max * jnp.exp(-10.0 * (_t - cfg.T / 2.0) ** 2)

    return jax.vmap(lambda b: bell(b, t))(_base_gamma(t, mu))


def true_gamma_arc(t, mu):
    def arc(b_max, _t):
        return (4.0 * b_max / cfg.T**2) * _t * (cfg.T - _t)

    return jax.vmap(lambda b: arc(b, t))(_base_gamma(t, mu))


def true_gamma_mf_dependent(t, mu):
    def mf(b_max):
        return b_max * mu[0]

    return jax.vmap(mf)(_base_gamma(t, mu))


_TRUE_FNS = {
    "constant": true_gamma_constant,
    "time_dependent": true_gamma_arc if TIME_DEPENDENT_TYPE == "arc" else true_gamma_bell,
    "mf_dependent": true_gamma_mf_dependent,
}


class CybersecurityTrainer(BaseTrainer):
    def __init__(self, case: str) -> None:
        super().__init__(cfg)
        self.case = case
        self.true_fn = _TRUE_FNS[case]

    def _build_mfg(self, d: int) -> Tuple:
        mfg = Cybersecurity(costs=COSTS)
        picard = PicardSolver(mfg=mfg, dt=cfg.dt, is_mfc=IS_MFC, damping=PICARD_DAMPING)
        return mfg, picard.get_solver_fn()

    def generate_data(
        self, num: int, seed: int, noise_level: float, d: int, inject_concentrated: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        np.random.seed(seed)
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
                dg = jax.vmap(jax.grad(self.true_fn, argnums=1))(self.t_grid, val[2]) if IS_MFC else None
                nxt = PicardSolver.picard_operator_static(val[2], g_curr, dg, mu0, mfg, self.cfg.dt, IS_MFC)
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

    def _get_checkpoint_path(self, folder: str, seed: int) -> str:
        return os.path.join(os.getcwd(), folder, f"best_checkpoint_seed_{seed}/")

    def _plot_results(
        self,
        cdir: str,
        results: List[Dict],
        t_grid: jnp.ndarray,
        te_epochs: jnp.ndarray,
    ) -> None:
        tr_m, tr_s = stat(results, "train_loss")
        te_m, te_s = stat(results, "test_loss")
        g_m, g_s = stat(results, "g_err")
        d = results[0]["mu_true"].shape[-1]

        plot_mean_field_loss(cdir, cfg.epochs, te_epochs, te_m, te_s, tr_m, tr_s)
        plot_gamma_loss(cdir, g_m, g_s, te_epochs)
        plot_gamma_trajectories(cdir, results, t_grid)
        plot_mean_field_trajectories(
            cdir,
            d,
            results,
            t_grid,
            plot_separate=False,
            states=STATES,
        )
        plot_avg_mean_field_trajectory(
            cdir,
            d,
            results,
            t_grid,
            plot_separate=False,
            states=STATES,
        )


if __name__ == "__main__":
    if IS_MFC:
        raise NotImplementedError("MFC requires Jacobian calculation for >1 parameters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT_DIR, EXPERIMENT_NAME)
    base_dir = os.path.join(results_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for tc in TEST_CASES:
        trainer = CybersecurityTrainer(case=tc)
        for noise in NOISE_LEVELS:
            trainer.run_experiment(tc, noise, DIMENSIONS[0], base_dir)

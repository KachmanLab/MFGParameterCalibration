"""
PDE.py
================
SIR Agent-Based Model experiment.  Generates synthetic trajectories from a system of partial differential equations
and trains the MFG parameter network against them.

Usage
-----
    python -m experiments.SIR.PDE
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.core.PicardSolver import PicardSolver
from src.mean_field_games.SIR import SIR
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import plot_gamma_trajectories, plot_mean_field_trajectories
from src.calibration.BaseTrainer import BaseTrainer, stat
from src.calibration.TrainingConfig import TrainingConfig


CONSTANT_GAMMA = False
TIME_DEPENDENT_TYPE = "arc"
TEST_CASES = ["constant", "time_dependent", "mf_dependent"]

GAMMA_SIZE = 2
IS_MFC = False
TERMINAL_COST_SCALE = 0.0
beta_min = 0.2
beta_max = 0.6

gamma_min = 0.05
gamma_max = 0.2

c_alpha = 1  # Running costs for reducing the contact factor
c_infected = 0.1  # Running costs for being infected
PICARD_DAMPING = 0.0  # Damping factor for Picard iteration (0.0 = no damping, >0.0 = damped)

T = 40.0
N = 100

DIMENSIONS = [3]
NOISE_LEVELS = [0.0]
NUM_AGENTS = [150, 1500, 3000]


ROOT_DIR = "results"
EXPERIMENT_NAME = "SIR_PDE"


def true_gamma_constant(t, mu):
    """
    Create constant parameters

    Args:
        t: the time point.
        mu: the mean field at t.

    Returns:
        jnp.ndarray: the transition rates
    """
    return jnp.array([(beta_max + beta_min) / 2, (gamma_max + gamma_min) / 2])


def true_gamma_bell(t, mu):
    """
    Create a smooth time-dependent bell

    Args:
        t: the time point.
        mu: the mean field at t.

    Returns:
        jnp.ndarray: the transition rates
    """
    beta = beta_min + (beta_max - beta_min) * jnp.exp(-10.0 * (t - T / 2.0) ** 2)
    gamma = gamma_min + (gamma_max - gamma_min) * jnp.exp(-10.0 * (t - T / 2.0) ** 2)
    return jnp.ndarray([beta, gamma])


def true_gamma_arc(t: float, mu: jnp.ndarray) -> jnp.ndarray:
    """
    Create a smooth time-dependent arc

    Args:
        t: the time point.
        mu: the mean field at t.

    Returns:
        jnp.ndarray: the transition rates
    """
    beta = beta_min + (4.0 * (beta_max - beta_min) / T**2) * t * (T - t)
    gamma = gamma_min + (4.0 * (gamma_max - gamma_min) / T**2) * t * (T - t)
    return jnp.array([beta, gamma])


def true_gamma_mf_dependent(
    t: float,
    mu: jnp.ndarray,
) -> jnp.ndarray:
    """
    Create mean-field dependent parameters

    Args:
        t: the time point.
        mu: the mean field at t.

    Returns:
        jnp.ndarray: the transition rates
    """
    return jnp.array([gamma_min + (gamma_max - gamma_min) * mu[0], beta_min + (beta_max - beta_min) * mu[1]])


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    gamma_size=GAMMA_SIZE,
    num_seeds=5,
    epochs=1000,
    T=T,
    N=N,
    num_samples=500,
    num_test_samples=50,
    batch_size=10,
    learning_rate=1e-3,
    delta_ratio=0.2,
    test_interval=20,
    picard_damping=PICARD_DAMPING,
    root_dir=ROOT_DIR,
    experiment_name=EXPERIMENT_NAME,
)

_TRUE_FNS = {
    "constant": true_gamma_constant,
    "time_dependent": true_gamma_arc if TIME_DEPENDENT_TYPE == "arc" else true_gamma_bell,
    "mf_dependent": true_gamma_mf_dependent,
}


class PDETrainer(BaseTrainer):
    def __init__(self, case: str) -> None:
        super().__init__(cfg)
        self.case = case
        self.true_fn = _TRUE_FNS[case]

    def _build_mfg(self, d: int) -> Tuple:
        mfg = SIR(c_alpha, c_infected)
        picard = PicardSolver(mfg=mfg, dt=self.cfg.dt, is_mfc=self.cfg.is_mfc, damping=self.cfg.picard_damping)
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

    @staticmethod
    def _plot_results(
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
            states=["S", "I", "R"],
        )


if __name__ == "__main__":
    if IS_MFC:
        raise NotImplementedError("MFC requires Jacobian calculation for >1 parameters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT_DIR, EXPERIMENT_NAME)
    base_dir = os.path.join(results_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for tc in TEST_CASES:
        trainer = PDETrainer(case=tc)
        for noise in NOISE_LEVELS:
            trainer.run_experiment(tc, noise, DIMENSIONS[0], base_dir)

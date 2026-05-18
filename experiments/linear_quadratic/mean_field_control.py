"""
mean_field_control.py
=========================
Linear-quadratic MFC experiment.

Sweeps over state-space dimensions, noise levels, and three parameter cases (constant / time-dependent / mean-field-dependent).
Demonstrates the versatility of the implicit differentiation method for the Picard solver.

Usage
-----
python -m experiments.linear_quadratic.mean_field_control
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.core.PicardSolver import PicardSolver
from src.mean_field_games.LinearQuadratic import LinearQuadratic
from src.plots.gamma_evolution import plot_gamma_evolution_1d
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import plot_mean_field_trajectories
from src.calibration.BaseTrainer import BaseTrainer
from src.calibration.TrainingConfig import TrainingConfig


TEST_CASES = ["constant"]

CONSTANT_GAMMA = False
TIME_DEPENDENT_TYPE = "arc"  # 'arc' or 'bell'


IS_MFC = True  # Toggle between Mean Field Game (False) and Mean Field Control (True)
TERMINAL_COST_SCALE = 0.0  # Scale of the state-dependent terminal cost
CONGESTION_WEIGHT = 2  # Weight c of the congestion penalty c * mu(x)^p
CONGESTION_POWER = 2  # Power p for the congestion term mu(x)^p
CONTROL_WEIGHT = 1.0  # Multiplier for the quadratic control penalty
TERMINAL_CONGESTION_WEIGHT = 0.0  # Weight for congestion penalty specifically at terminal time
DESTINATION_CONGESTION_WEIGHT = 0.0  # Multiplier \kappa for crowding resistance at destination
BASE_TRANSITION_RATE = 0.3  # Natural "free" transition rate when no effort is exerted
PICARD_DAMPING = 0.5  # Damping for the Picard solver

b_min = 0.1
b_max = 3.0

DIMENSIONS = [3]
NOISE_LEVELS = [0.0]

ROOT_DIR = "results"
EXPERIMENT_NAME = "linear_quadratic_MFC"


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    gamma_size=1,
    num_seeds=5,
    epochs=1000,
    T=1.0,
    N=100,
    num_samples=50,
    num_test_samples=10,
    batch_size=10,
    learning_rate=1e-3,
    delta_ratio=0.2,
    test_interval=20,
    picard_damping=PICARD_DAMPING,
    root_dir=ROOT_DIR,
    experiment_name=EXPERIMENT_NAME,
)


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


class LinearQuadraticMFC(BaseTrainer):
    def __init__(self, case: str) -> None:
        super().__init__(cfg)
        self.case = case
        self.true_fn = _TRUE_FNS[case]

    def _build_mfg(self, d: int) -> Tuple:
        mfg = LinearQuadratic(
            terminal_cost_scale=TERMINAL_COST_SCALE,
            congestion_weight=CONGESTION_WEIGHT,
            congestion_power=CONGESTION_POWER,
            control_weight=CONTROL_WEIGHT,
            terminal_congestion_weight=TERMINAL_CONGESTION_WEIGHT,
            destination_congestion_weight=DESTINATION_CONGESTION_WEIGHT,
            base_transition_rate=BASE_TRANSITION_RATE,
        )
        picard = PicardSolver(mfg=mfg, dt=self.cfg.dt, is_mfc=self.cfg.is_mfc, damping=self.cfg.picard_damping)
        return mfg, picard.get_solver_fn()

    def generate_data(
        self,
        num: int,
        seed: int,
        noise_level: float,
        d: int,
        inject_concentrated: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        np.random.seed(seed)

        # Build a fresh mfg/solver for data generation so this method is
        # self-contained (the solver used here must match the true_fn).
        mfg, _ = self._build_mfg(d)
        mus, mu0s, gs = [], [], []
        for i in range(num):
            if inject_concentrated and i == 0:
                mu0 = np.zeros(d)
                mu0[0] = 1.0
            else:
                mu0 = np.random.dirichlet(np.ones(d))
            init_mu = jnp.repeat(mu0[None, :], self.cfg.N, axis=0)

            def cond(val):
                return (val[0] < 1000) & (jnp.max(jnp.abs(val[2] - val[1])) > 1e-5) | (val[0] < 2)

            def body(val):
                g_curr = jax.vmap(self.true_fn)(self.t_grid, val[2])
                dg_dmu_curr = jax.vmap(jax.grad(self.true_fn, argnums=1))(self.t_grid, val[2])
                nxt = PicardSolver.picard_operator_static(val[2], g_curr, dg_dmu_curr, mu0, mfg, self.cfg.dt, IS_MFC)
                return val[0] + 1, val[2], nxt

            _, _, bar_mu = jax.lax.while_loop(cond, body, (0, init_mu, init_mu))

            if noise_level > 0.0:
                kappa = 1.0 / noise_level
                noisy_mu = []
                for t_idx in range(self.cfg.N):
                    # Add small epsilon to prevent alpha from being exactly zero
                    alpha = kappa * (np.array(bar_mu[t_idx]) + 1e-6)
                    noisy_mu.append(np.random.dirichlet(alpha))
                bar_mu = jnp.array(noisy_mu)

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
        trainer = LinearQuadraticMFC(case=tc)
        for noise in NOISE_LEVELS:
            for d in DIMENSIONS:
                trainer.run_experiment(tc, noise, d, base_dir)

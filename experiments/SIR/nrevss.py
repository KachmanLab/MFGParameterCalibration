"""
nrevss.py
=======================
SIR experiment fitted to the ICL NREVSS real-world flu dataset. It is recommended to learn the cost parameters
alongside the epidemiological parameters.

Please specify the data path ``DATA_PATH`` below.

Usage
-----
    python -m experiments.SIR.nrevss.py
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.core.PicardSolver import PicardSolver
from src.data.ICL_NREVSS import get_ICL_NREVSS_data_from_path
from src.mean_field_games.SIR import SIR
from src.core.ParameterNetwork import ParameterNetwork
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import plot_gamma_trajectories, plot_mean_field_sir_trajectories
from src.calibration.BaseTrainer import BaseTrainer, _log, stat
from src.calibration.TrainingConfig import TrainingConfig


CONSTANT_GAMMA = False

DATA_PATH = r"data/ICL_NREVSS_Clinical_Labs.csv"
GAMMA_SIZE = 2
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0

IS_MFC = False
TERMINAL_COST_SCALE = 0.0
c_alpha = 0.1  # Default running cost for avoiding behaviour
c_infected = 1  # Default running cost for being infected
LEARN_COSTS = True  # When True the network outputs GAMMA_SIZE + 2 values (recommended)

PICARD_DAMPING = 0.0
OBSERVED_INDEX = [1]  # Only the infected compartment is observed

DIMENSIONS = 3

ROOT_DIR = "results"
EXPERIMENT_NAME = "SIR_ICL_NREVSS"


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    # Network outputs SIR params + optional cost params
    gamma_size=GAMMA_SIZE + (2 if LEARN_COSTS else 0),
    gamma_loss_idx=GAMMA_SIZE,
    num_seeds=5,
    epochs=1000,
    # T, N, and num_samples are determined by the dataset, overridden in generate_data
    T=1,
    N=1,
    num_samples=1,
    num_test_samples=1,
    batch_size=2,
    learning_rate=1e-3,
    delta_ratio=0.2,
    test_interval=20,
    picard_damping=PICARD_DAMPING,
    root_dir=ROOT_DIR,
    experiment_name=EXPERIMENT_NAME,
)


class NrevssTrainer(BaseTrainer):
    def __init__(self) -> None:
        self._mu_train, self._mu0_train, self._g_train, self._meta_train = get_ICL_NREVSS_data_from_path(
            DATA_PATH, train=True
        )
        self._mu_test, self._mu0_test, self._g_test, self._meta_test = get_ICL_NREVSS_data_from_path(
            DATA_PATH, train=False
        )

        # Patch config with real dataset sizes
        cfg.T = float(self._mu_train[0, -1, 0])
        cfg.N = self._mu_train.shape[1]
        cfg.num_samples = self._mu_train.shape[0]
        cfg.num_test_samples = self._mu_test.shape[0]

        super().__init__(cfg)

    def _build_mfg(self, d: int) -> Tuple:
        mfg = SIR(c_alpha, c_infected, learn_cost=LEARN_COSTS)
        picard = PicardSolver(mfg=mfg, dt=self.cfg.dt, is_mfc=self.cfg.is_mfc, damping=self.cfg.picard_damping)
        return mfg, picard.get_solver_fn()

    def generate_data(
        self, num: int, seed: int, noise_level: float, d: int, inject_concentrated: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the cached real-world data (train or test based on *num*)."""
        if num == cfg.num_samples:
            return self._mu_train, self._mu0_train, self._g_train
        return self._mu_test, self._mu0_test, self._g_test

    def _build_loss_fn(self, model: ParameterNetwork, solver_fn: Any, d: int):
        def loss_fn(params, mu_o, mu0_o, s_idx):
            g_p = self._pred_g_traj(model, params, self.t_grid, mu_o)
            mu_p = solver_fn(g_p, None, jnp.repeat(mu0_o[None, :], self.cfg.N, axis=0), mu0_o)
            sl = lambda x: jax.lax.dynamic_slice(x, (s_idx, 0), (self.cfg.delta_steps, d))
            obs = lambda x: x[:, OBSERVED_INDEX]
            residual = jnp.mean(jnp.sum((obs(sl(mu_p)) - obs(sl(mu_o))) ** 2))
            return residual, (mu_p, g_p)

        return loss_fn

    def _build_full_eval(self, model: ParameterNetwork, solver_fn: Any):
        def full_eval(params, mo, m0):
            gp = self._pred_g_traj(model, params, self.t_grid, mo)
            mp = solver_fn(gp, None, jnp.repeat(m0[None, :], self.cfg.N, axis=0), m0)
            obs = lambda x: x[:, OBSERVED_INDEX]
            return jnp.mean(jnp.sum((obs(mp) - obs(mo)) ** 2)), mp, gp

        return full_eval

    def _get_checkpoint_path(self, folder: str, seed: int) -> str:
        return os.path.join(os.getcwd(), folder, f"best_checkpoint_seed_{seed}/")

    def _log_cost_params(self, results: List[Dict], log_path: str) -> None:
        g_p = np.stack([r["g_pred"] for r in results], axis=0)
        c_a = g_p[..., -2]
        c_I = g_p[..., -1]
        _log(
            f"mean c_alpha: {np.mean(c_a):.4f}  mean c_I: {np.mean(c_I):.4f}",
            log_path,
        )

    def run_experiment(
        self,
        case: str,
        noise_level: float,
        d: int,
        base_dir: str,
        log_gamma: bool=False,
    ) -> None:
        super().run_experiment(case, noise_level, d, base_dir, log_gamma=log_gamma)

        noise_str = f"{noise_level:.1e}" if noise_level > 0 else "0.0"
        cdir = os.path.join(base_dir, f"dim-{d}", f"noise-{noise_str}", f"mfc-{cfg.is_mfc}", case)
        log_path = os.path.join(cdir, "training.log")

        # Print statistics beyond the standard base class logging
        data = np.load(os.path.join(cdir, "data.npz"))
        g_p = data["g_p"]
        c_a = g_p[..., -2]
        c_I = g_p[..., -1]
        _log(
            f"mean c_alpha: {np.mean(c_a):.4f}  mean c_I: {np.mean(c_I):.4f}",
            log_path,
        )

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
        results[0]['metadata'] = self._meta_test
        plot_mean_field_sir_trajectories(cdir, d, results, t_grid, plot_separate=False)


if __name__ == "__main__":
    if IS_MFC:
        raise NotImplementedError("MFC requires Jacobian calculation for >1 parameters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT_DIR, EXPERIMENT_NAME)
    os.makedirs(results_dir, exist_ok=True)
    base_dir = os.path.join(results_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    trainer = NrevssTrainer()
    trainer.run_experiment(EXPERIMENT_NAME, 0.0, DIMENSIONS, base_dir)

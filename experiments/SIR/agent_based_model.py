"""
agent_based_model.py
================
SIR Agent-Based Model experiment.  Generates synthetic trajectories from an agent based model and trains the MFG
parameter network against them.

Usage
-----
    python -m experiments.SIR.agent_based_model
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.core.PicardSolver import PicardSolver
from src.data.agent_based_model import SIR_ABM_JAX, sample_ABM_init
from src.mean_field_games.SIR import SIR
from src.plots.loss import plot_mean_field_loss, plot_gamma_loss
from src.plots.predictions import plot_gamma_trajectories, plot_mean_field_trajectories
from src.calibration.BaseTrainer import BaseTrainer, stat
from src.calibration.TrainingConfig import TrainingConfig


CONSTANT_GAMMA = False
GAMMA_SIZE = 2
IS_MFC = False
TERMINAL_COST_SCALE = 0.0
c_alpha = 1
c_infected = 0.1
PICARD_DAMPING = 0.8
N = 100

DIMENSIONS = [3]
NOISE_LEVELS = [0.0]
NUM_AGENTS = [150, 1500, 3000]

LOAD_PRE_GENERATED = False
FLEE = False  # Set to true to make susceptible agents avoid infected agents.
suffix = "_avoid" if FLEE else ""

DATA_PATH = rf"ABM_DATA/SIR_data{suffix}.h5"

ROOT_DIR = "results"
EXPERIMENT_NAME = "SIR_ABM"

ABM_CONFIG = {
    "N": 3000,
    "space": [10, 10],
    "is_periodic": True,
    "r_infectious": 0.2,
    "p_infect": 0.3,
    "t_infectious": 14,
    "sigma_s": 0.02,
    "sigma_i": 0.01,
    "sigma_r": 0.02,
    "sigma": 0.1,
    "num_steps": N,
    "flee": FLEE,
}


cfg = TrainingConfig(
    constant_gamma=CONSTANT_GAMMA,
    is_mfc=IS_MFC,
    gamma_size=GAMMA_SIZE,
    num_seeds=5,
    epochs=1000,
    T=20.0,
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


class ABMTrainer(BaseTrainer):
    def __init__(self, cdir: str) -> None:
        super().__init__(cfg)
        self._cdir = cdir

    def _build_mfg(self, d: int) -> Tuple:
        mfg = SIR(c_alpha, c_infected)
        picard = PicardSolver(mfg=mfg, dt=cfg.dt, is_mfc=IS_MFC, damping=PICARD_DAMPING)
        return mfg, picard.get_solver_fn()

    def generate_data(
        self,
        num: int,
        seed: int,
        noise_level: float,
        d: int,
        inject_concentrated: bool = False,  # TODO
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if LOAD_PRE_GENERATED:
            return self._load_pregenerated(num)

        label = "train" if num == cfg.num_samples else "test"
        filename = os.path.join(self._cdir, f"{label}_data")
        return self._generate_abm_data(num, filename)

    @staticmethod
    def _load_pregenerated(num: int) -> Tuple[np.array, np.array, np.array]:
        import h5py as h5

        with h5.File(DATA_PATH, "r") as f:
            data = np.array(f["SIR"]["true_counts"]).squeeze()
        mu = jnp.array(data[:num])
        mu0 = mu[:, 0, :]
        g = jnp.zeros((num, cfg.N, GAMMA_SIZE))
        return mu, mu0, g

    @staticmethod
    def _generate_abm_data(n_samples: int, filename: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        d = DIMENSIONS[0]
        data = jnp.empty((n_samples, cfg.N, d))
        for m in tqdm(range(n_samples), desc="Generating SIR ABM data"):
            ABM = SIR_ABM_JAX(**ABM_CONFIG, **sample_ABM_init(ABM_CONFIG["N"]))
            params = jnp.array([ABM.params["p_infect"], ABM.params["t_infectious"]])
            for i in range(cfg.N):
                ABM.run_single(params)
                counts = ABM.counts()
                data = data.at[m, i, :].set(counts / ABM.N)

        np.savez_compressed(file=filename, data=data)
        mu0 = data[:, 0, :]
        g = jnp.zeros((n_samples, cfg.N, GAMMA_SIZE))
        return data, mu0, g

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
    os.makedirs(results_dir, exist_ok=True)
    base_dir = os.path.join(results_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for agent_count in NUM_AGENTS:
        ABM_CONFIG["N"] = agent_count
        experiment_dir = os.path.join(base_dir, f"agents_{agent_count}")
        os.makedirs(experiment_dir, exist_ok=True)
        for d in DIMENSIONS:
            trainer = ABMTrainer(cdir=experiment_dir)
            trainer.run_experiment(EXPERIMENT_NAME, NOISE_LEVELS[0], d, experiment_dir)

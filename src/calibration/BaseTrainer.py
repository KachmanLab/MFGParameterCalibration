"""
src/calibration/BaseTrainer.py
============================
Abstract base class that encapsulates the training loop shared across all MFG
parameter-estimation experiments.

Subclasses must implement:
  - generate_data(num, seed, noise_level, d)  →  (mu, mu0, g)
  - _plot_results(cdir, results, t_grid, te_epochs)  →  None

Subclasses may override:
  - _build_mfg(d)          →  (mfg_model, solver_fn)
  - _get_checkpoint_path(folder, seed)  →  str | None   (None = no checkpointing)
  - _restore_best(state, checkpoint_path)  →  state
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from src.core.ParameterNetwork import ParameterNetwork
from src.calibration.TrainingConfig import TrainingConfig


def _log(msg: str, log_path: str) -> None:
    """Print ``msg`` to stdout and append it to ``log_path``."""
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def stat(results: List[Dict], key: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return (mean, standard-error) across seeds for a scalar-array result key.
    """
    stacked = jnp.stack([r[key] for r in results])
    n = stacked.shape[0]
    return jnp.mean(stacked, axis=0), jnp.std(stacked, axis=0) / jnp.sqrt(n)


class BaseTrainer:
    """
    Shared training class for all MFG inverse-problem experiments.

    Parameters
    ----------
    cfg:
        A :class:`TrainingConfig` instance carrying all hyperparameters.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg

        # Derived time grid (reused for all experiments)
        self.t_grid: jnp.ndarray = jnp.linspace(0, cfg.T, cfg.N)

    def generate_data(
        self,
        num: int,
        seed: int,
        noise_level: float,
        d: int,
        inject_concentrated: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate or load ``num`` trajectories. For the real-world datasets, ``num`` is dependent on the data at hand,
        and the parameter here won't be considered.

        Returns
        -------
        mu   : shape (num, N, d)  – mean-field trajectories
        mu0  : shape (num, d)     – initial conditions
        g    : shape (num, N, gamma_size) – ground-truth parameter trajectories
                                            (zeros when unknown, e.g. real data)
        """
        pass

    @staticmethod
    def _plot_results(
        cdir: str,
        results: List[Dict],
        t_grid: jnp.ndarray,
        te_epochs: jnp.ndarray,
    ) -> None:
        """Produce all experiment-specific plots and save them to ``cdir``."""
        pass

    def _build_mfg(self, d: int) -> Tuple[Any, Any]:
        """
        Instantiate the MFG model and its Picard solver function.

        Returns
        -------
        mfg_model : the MFG model object
        solver_fn : the callable returned by ``PicardSolver.get_solver_fn()``
        """
        pass

    def _get_checkpoint_path(self, folder: str, seed: int) -> Optional[str]:
        """
        Return an *absolute* path to store the best checkpoint for this seed,
        or ``None`` to disable checkpointing entirely.

        The default implementation returns ``None`` (no checkpointing).
        Override in subclasses that need early-stopping via best-val tracking.
        """
        pass

    @staticmethod
    def _restore_best(state: Any, checkpoint_path: Optional[str]) -> Any:
        """
        Restore the best checkpoint saved during training.

        The default falls back to the final ``state`` when no path is given or
        when restoration fails.
        """
        if checkpoint_path is None:
            return state
        abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
        try:
            with ocp.CheckpointManager(checkpoint_path) as mngr:
                return mngr.restore(
                    mngr.latest_step(),
                    args=ocp.args.StandardRestore(abstract_state),
                )
        except Exception:
            return state

    def _maybe_save_checkpoint(
        self,
        state: Any,
        current_val: float,
        best_val: float,
        epoch: int,
        checkpoint_path: Optional[str],
    ) -> float:
        """
        Save a checkpoint when *current_val* beats *best_val* (after the first
        third of training).  Returns the updated ``best_val``.
        """
        if checkpoint_path is None:
            return best_val
        if current_val < best_val and epoch > self.cfg.epochs // 3:
            with ocp.CheckpointManager(
                ocp.test_utils.erase_and_create_empty(checkpoint_path),
                options=ocp.CheckpointManagerOptions(max_to_keep=1),
            ) as mngr:
                mngr.save(state.step, args=ocp.args.StandardSave(state))
            return current_val
        return best_val

    def _make_model(self) -> ParameterNetwork:
        return ParameterNetwork(
            constant_gamma=self.cfg.constant_gamma,
            out_size=self.cfg.gamma_size,
        )

    def _pred_g_traj(
        self,
        model: ParameterNetwork,
        params: Any,
        t_t: jnp.ndarray,
        mu_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict a full parameter trajectory of shape (N, gamma_size).

        When ``constant_gamma`` is True the network is called once and the
        result broadcast over time.
        """
        if self.cfg.constant_gamma:
            v = model.apply({"params": params}, 0.0, jnp.zeros(mu_t.shape[-1]))
            return jnp.broadcast_to(v, (t_t.shape[0], *v.shape))
        return jax.vmap(lambda t, m: model.apply({"params": params}, t, m))(t_t, mu_t)

    def _single_pred_g(
        self,
        model: ParameterNetwork,
        params: Any,
        t: float,
        mu: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single-time-step prediction used for MFC Jacobian computation."""
        if self.cfg.constant_gamma:
            return model.apply({"params": params}, 0.0, jnp.zeros(mu.shape[-1]))
        return model.apply({"params": params}, t, mu)

    def _build_loss_fn(
        self,
        model: ParameterNetwork,
        solver_fn: Any,
        d: int,
    ):
        """
        Return a JAX-traceable ``loss_fn(params, mu_o, mu0_o, s_idx)``
        closure over the current experiment's model and solver.

        Subclasses that need a different loss (e.g. partial observation as in
        the ICL-NREVSS experiment) should override this method.
        """
        cfg = self.cfg
        t_grid = self.t_grid

        def loss_fn(params, mu_o, mu0_o, s_idx):
            g_p = self._pred_g_traj(model, params, t_grid, mu_o)

            dg_dmu_p = None
            if cfg.is_mfc:
                dg_dmu_p = jax.vmap(
                    jax.grad(
                        lambda t, m: self._single_pred_g(model, params, t, m),
                        argnums=1,
                    )
                )(t_grid, mu_o)

            mu_p = solver_fn(g_p, dg_dmu_p, jnp.repeat(mu0_o[None, :], cfg.N, axis=0), mu0_o)

            sl = lambda x: jax.lax.dynamic_slice(x, (s_idx, 0), (cfg.delta_steps, d))
            residual = jnp.mean(jnp.sum((sl(mu_p) - sl(mu_o)) ** 2, axis=1))
            return residual, (mu_p, g_p)

        return loss_fn

    def _build_full_eval(
        self,
        model: ParameterNetwork,
        solver_fn: Any,
    ):
        """
        Return a ``full_eval(params, mu_o, mu0)`` closure for test-time evaluation (no sub-trajectory slicing).
        """
        cfg = self.cfg
        t_grid = self.t_grid

        def full_eval(params, mo, m0):
            gp = self._pred_g_traj(model, params, t_grid, mo)

            dgp = None
            if cfg.is_mfc:
                dgp = jax.vmap(
                    jax.grad(
                        lambda t, m: self._single_pred_g(model, params, t, m),
                        argnums=1,
                    )
                )(t_grid, mo)

            mp = solver_fn(gp, dgp, jnp.repeat(m0[None, :], cfg.N, axis=0), m0)
            return jnp.mean(jnp.sum((mp - mo) ** 2, axis=1)), mp, gp

        return full_eval

    def train_single_seed(
        self,
        seed: int,
        noise_level: float,
        d: int,
        mfg_model: Any,
        solver_fn: Any,
        log_path: str,
        inject_concentrated: bool = False,
        log_gamma: bool = True,
    ) -> Dict:
        """
        Full data-generation + training pipeline for one random seed.

        Returns a dict with keys:
          train_loss, test_loss, g_err,
          mu_true, mu_pred, g_true, g_pred, g_hist
        """

        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        _log(
            f"  [Seed {seed}] Generating data ({self.cfg.num_samples} train, {self.cfg.num_test_samples} test)...",
            log_path,
        )
        mu_train, mu0_train, g_train = self.generate_data(self.cfg.num_samples, seed, noise_level, d)
        mu_test, mu0_test, g_test = self.generate_data(
            self.cfg.num_test_samples, seed + 10_000, noise_level, d, inject_concentrated
        )

        model = self._make_model()
        key, init_key = jax.random.split(key)
        params = model.init(init_key, 0.0, jnp.zeros(d))["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(self.cfg.learning_rate),
        )

        folder = os.path.dirname(log_path)
        ckpt_path = self._get_checkpoint_path(folder, seed)

        loss_fn = self._build_loss_fn(model, solver_fn, d)
        full_eval = self._build_full_eval(model, solver_fn)

        @jax.jit
        def train_step(st, b_mu, b_mu0, b_s):
            def batch_loss(p):
                losses, _ = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(p, b_mu, b_mu0, b_s)
                return jnp.mean(losses)

            loss, grads = jax.value_and_grad(batch_loss)(st.params)
            return st.apply_gradients(grads=grads), loss

        train_hist: List[float] = []
        test_hist: List[float] = []
        g_err_hist: List[float] = []
        gamma_preds_hist: List = []

        best_val = jnp.inf
        rng = jax.random.PRNGKey(seed)
        start_time = time.time()

        _log(f"  [Seed {seed}] Starting training for {self.cfg.epochs} epochs...", log_path)

        snapshot_epochs = {1, 100, self.cfg.epochs}

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            for i in range(self.cfg.num_samples // self.cfg.batch_size):
                rng, sk = jax.random.split(rng)
                b_s = jax.random.randint(sk, (self.cfg.batch_size,), 0, self.cfg.N - self.cfg.delta_steps)
                state, l = train_step(
                    state,
                    mu_train[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size],
                    mu0_train[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size],
                    b_s,
                )
                epoch_loss += float(l)
            train_hist.append(epoch_loss / (self.cfg.num_samples // self.cfg.batch_size))

            if epoch % self.cfg.test_interval == 0 or epoch == 1:
                metrics = jax.vmap(full_eval, in_axes=(None, 0, 0))(state.params, mu_test, mu0_test)
                test_hist.append(float(jnp.mean(metrics[0])))

                if self.cfg.gamma_loss_idx is not None:
                    g_err_hist.append(
                        float(jnp.mean(jnp.sum((metrics[2][:, :, : self.cfg.gamma_loss_idx] - g_test) ** 2, axis=-1)))
                    )
                else:
                    g_err_hist.append(float(jnp.mean(jnp.sum((metrics[2] - g_test) ** 2, axis=-1))))
                elapsed = time.time() - start_time
                if log_gamma:
                    _log(
                        f"  [Seed {seed}] Epoch {epoch}/{self.cfg.epochs}"
                        f" | Mu L2 (Train): {train_hist[-1]:.3e}"
                        f" | Mu L2 (Test): {test_hist[-1]:.3e}"
                        f" | Gamma L2: {g_err_hist[-1]:.6f}"
                        f" | Time: {elapsed:.1f}s",
                        log_path,
                    )
                else:
                    _log(
                        f"  [Seed {seed}] Epoch {epoch}/{self.cfg.epochs}"
                        f" | Mu L2 (Train): {train_hist[-1]:.3e}"
                        f" | Mu L2 (Test): {test_hist[-1]:.3e}"
                        f" | Time: {elapsed:.1f}s",
                        log_path,
                    )

                best_val = self._maybe_save_checkpoint(state, test_hist[-1], best_val, epoch, ckpt_path)

            if epoch in snapshot_epochs:
                gp = jax.vmap(lambda mo: self._pred_g_traj(model, state.params, self.t_grid, mo))(mu_test)
                gamma_preds_hist.append(np.array(gp))

        best_state = self._restore_best(state, ckpt_path)
        _, final_mu, final_g = jax.vmap(full_eval, in_axes=(None, 0, 0))(best_state.params, mu_test, mu0_test)

        return {
            "train_loss": jnp.array(train_hist),
            "test_loss": jnp.array(test_hist),
            "g_err": jnp.array(g_err_hist),
            "mu_true": mu_test,
            "mu_pred": final_mu,
            "g_true": g_test,
            "g_pred": final_g,
            "g_hist": gamma_preds_hist,
        }

    def run_experiment(
        self,
        case: str,
        noise_level: float,
        d: int,
        base_dir: str,
        log_gamma: bool = True,
    ) -> None:
        """
        Run *num_seeds* independent seeds, aggregate statistics, save data
        and call ``_plot_results``.
        """
        cfg = self.cfg
        noise_str = f"{noise_level:.1e}" if noise_level > 0 else "0.0"
        cdir = os.path.join(base_dir, f"dim-{d}", f"noise-{noise_str}", f"mfc-{cfg.is_mfc}", case)
        os.makedirs(cdir, exist_ok=True)

        log_path = os.path.join(cdir, "training.log")
        with open(log_path, "w") as f:
            f.write(f"--- Experiment {case.upper()} Starting (Noise: {noise_level}, d: {d}, MFC: {cfg.is_mfc}) ---\n")

        print(f"\n--- {case.upper()} | Noise: {noise_str} | Dim: {d} | MFC: {cfg.is_mfc} ---")

        mfg_model, solver_fn = self._build_mfg(d)

        results = [
            self.train_single_seed(42 + i, noise_level, d, mfg_model, solver_fn, log_path, log_gamma=log_gamma) for i in range(cfg.num_seeds)
        ]

        tr_m, tr_s = stat(results, "train_loss")
        te_m, te_s = stat(results, "test_loss")
        g_m, g_s = stat(results, "g_err")

        mu_t = np.stack([r["mu_true"] for r in results], axis=0)
        mu_p = np.stack([r["mu_pred"] for r in results], axis=0)
        g_t = np.stack([r["g_true"] for r in results], axis=0)
        g_p = np.stack([r["g_pred"] for r in results], axis=0)
        g_hist = np.array([r["g_hist"] for r in results])

        mu_l2 = np.mean((mu_t - mu_p) ** 2, axis=0)
        if self.cfg.gamma_loss_idx is not None:
            g_l2 = np.mean((g_t - g_p[:, :, :, :self.cfg.gamma_loss_idx]) ** 2, axis=0)
        else:
            g_l2 = np.mean((g_t - g_p) ** 2, axis=0)

        if log_gamma:
            _log(
                f"Noise level: {noise_level}\n"
                f"\tmu:    {np.mean(mu_l2):.3e} ± {np.std(mu_l2):.3e}\n"
                f"\tgamma: {np.mean(g_l2):.3e} ± {np.std(g_l2):.3e}",
                log_path,
            )
        else:
            _log(
                f"Noise level: {noise_level}\n"
                f"\tmu:    {np.mean(mu_l2):.3e} ± {np.std(mu_l2):.3e}\n",
                log_path,
            )

        np.savez_compressed(
            os.path.join(cdir, "data.npz"),
            t=np.array(self.t_grid),
            tr_m=tr_m,
            tr_s=tr_s,
            te_m=te_m,
            te_s=te_s,
            g_m=g_m,
            g_s=g_s,
            mu_t=mu_t,
            mu_p=mu_p,
            g_t=g_t,
            g_p=g_p,
            g_hist=g_hist,
            noise_level=noise_level,
            d=d,
            epochs=cfg.epochs,
            num_seeds=cfg.num_seeds,
            is_mfc=cfg.is_mfc,
        )

        te_epochs = jnp.concatenate(
            [
                jnp.array([1]),
                jnp.arange(cfg.test_interval, cfg.epochs + 1, cfg.test_interval),
            ]
        )
        self._plot_results(cdir, results, self.t_grid, te_epochs)

        print(f"\n--- EXPERIMENT {case.upper()} COMPLETE ---")
        print(f"Results saved in: {cdir}\n")

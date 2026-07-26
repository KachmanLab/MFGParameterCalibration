"""
src/calibration/DiffEqTrainer.py
============================
Abstract base class that encapsulates the training loop for differential equation-based parameter
calibration experiments.

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
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import orbax.checkpoint as ocp

from core.ParameterNetwork import ParameterNetwork, FlaxWrap, TrainState
from src.calibration.BaseTrainer import _log, BaseTrainer, stat
from src.calibration.TrainingConfig import DiffEqConfig


class DiffEqTrainer(BaseTrainer):
    """
    Shared training class for all neural ODE-based MFG experiments.

    Parameters
    ----------
    cfg:
        A :class:`DiffEqConfig` instance carrying all hyperparameters.
    """

    def __init__(
        self,
        cfg: DiffEqConfig,
        mfg_model: Any,
        solver_fn: Any,
        model_out_size: int = 1,
        enforce_positivity: bool = True,
        predict_gamma: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.mfg_model = mfg_model
        self.solver_fn = solver_fn
        self.model_out_size = model_out_size
        self.enforce_positivity = enforce_positivity
        self.predict_gamma = predict_gamma

    @staticmethod
    def _restore_best(state: TrainState, checkpoint_path: Optional[str]) -> TrainState:
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

    def loss_fn(self, model: FlaxWrap, solver_fn: Any, d: int, mu_o, mu0_o, s_idx):
        cfg = self.cfg
        t_grid = self.t_grid

        mu_p = solver_fn(model, None, jnp.repeat(mu0_o[None, :], cfg.N, axis=0), mu0_o)

        sl = lambda x: jax.lax.dynamic_slice(x, (s_idx, 0), (cfg.delta_steps, d))
        residual = jnp.mean(jnp.sum((sl(mu_p) - sl(mu_o)) ** 2, axis=1))

        if self.predict_gamma:
            g_p = jax.vmap(model)(t_grid, mu_p)
        else:
            g_p = jnp.zeros((cfg.N, 0))  # placeholder, consistent shape across seeds/batches

        return residual, (mu_p, g_p)

    def full_eval(self, model: FlaxWrap, solver_fn: Any, mo, m0):
        """
        Return a closure for test-time evaluation (no sub-trajectory slicing).
        """
        cfg = self.cfg
        t_grid = self.t_grid

        mu_p = solver_fn(model, None, jnp.repeat(m0[None, :], cfg.N, axis=0), m0)
        g_p = jax.vmap(model)(t_grid, mu_p) if self.predict_gamma else jnp.zeros((cfg.N, 0))
        return jnp.mean(jnp.sum((mu_p - mo) ** 2, axis=1)), mu_p, g_p

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

        key, init_key = jax.random.split(key)
        model = ParameterNetwork(constant_gamma=False, out_size=self.model_out_size, enforce_positivity=self.enforce_positivity,)
        params = model.init(init_key, 0.0, jnp.zeros(d))["params"]
        model = FlaxWrap(module=model, params=params)

        opt = optax.adam(self.cfg.learning_rate)
        state = opt.init(model.params)
        train_state = TrainState(step=0, model=model, opt_state=state)

        folder = os.path.dirname(log_path)
        ckpt_path = self._get_checkpoint_path(folder, seed)

        @jax.jit
        def train_step(m: FlaxWrap, opt_state, b_mu, b_mu0, b_s):
            def batch_loss(m):
                losses, _ = jax.vmap(
                    lambda mu, mu0, _b_s: self.loss_fn(m, solver_fn, d, mu, mu0, _b_s), in_axes=(0, 0, 0)
                )(b_mu, b_mu0, b_s)
                return jnp.mean(losses)

            loss, grads = jax.value_and_grad(batch_loss)(m)
            updates, opt_state = opt.update(grads.params, opt_state, m.params)
            new_params = eqx.apply_updates(m.params, updates)
            m = eqx.tree_at(lambda g: g.params, m, new_params)
            return m, opt_state, loss

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
                model, state, l = train_step(
                    m=model,
                    opt_state=state,
                    b_mu=mu_train[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size],
                    b_mu0=mu0_train[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size],
                    b_s=b_s,
                )
                epoch_loss += float(l)
            train_hist.append(epoch_loss / (self.cfg.num_samples // self.cfg.batch_size))

            if epoch % self.cfg.test_interval == 0 or epoch == 1:
                metrics = jax.vmap(lambda m, s, mo, m0: self.full_eval(m, s, mo, m0), in_axes=(None, None, 0, 0))(
                    model, solver_fn, mu_test, mu0_test
                )
                test_hist.append(float(jnp.mean(metrics[0])))

                if self.predict_gamma:
                    if self.cfg.gamma_loss_idx is not None:
                        g_err_hist.append(
                            float(
                                jnp.mean(jnp.sum((metrics[2][:, :, :self.cfg.gamma_loss_idx] - g_test) ** 2, axis=-1)))
                        )
                    else:
                        g_err_hist.append(float(jnp.mean(jnp.sum((metrics[2] - g_test) ** 2, axis=-1))))
                else:
                    g_err_hist.append(0.0)

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

                train_state = TrainState(step=epoch, model=model, opt_state=state)
                best_val = self._maybe_save_checkpoint(train_state, test_hist[-1], best_val, epoch, ckpt_path)

            if epoch in snapshot_epochs:
                _, _, g_p = jax.vmap(lambda m, s, mo, m0: self.full_eval(m, s, mo, m0), in_axes=(None, None, 0, 0))(
                    model, solver_fn, mu_test, mu0_test
                )
                gamma_preds_hist.append(np.array(g_p))

        best_state = self._restore_best(train_state, ckpt_path)
        model = best_state.model
        _, final_mu, final_g = jax.vmap(lambda m, s, mo, m0: self.full_eval(m, s, mo, m0), in_axes=(None, None, 0, 0))(
            model, solver_fn, mu_test, mu0_test
        )

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
        mfg_model = self.mfg_model
        solver_fn = self.solver_fn

        noise_str = f"{noise_level:.1e}" if noise_level > 0 else "0.0"
        cdir = os.path.join(base_dir, f"dim-{d}", f"noise-{noise_str}", f"mfc-{cfg.is_mfc}", case)
        os.makedirs(cdir, exist_ok=True)

        log_path = os.path.join(cdir, "training.log")
        with open(log_path, "w") as f:
            f.write(f"--- Experiment {case.upper()} Starting (Noise: {noise_level}, d: {d}) ---\n")

        print(f"\n--- {case.upper()} | Noise: {noise_str} | Dim: {d} | Predict gamma: {self.predict_gamma} ---")

        results = [
            self.train_single_seed(42 + i, noise_level, d, mfg_model, solver_fn, log_path, log_gamma=log_gamma)
            for i in range(cfg.num_seeds)
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
        if self.predict_gamma:
            if self.cfg.gamma_loss_idx is not None:
                g_l2 = np.mean((g_t - g_p[:, :, :, : self.cfg.gamma_loss_idx]) ** 2, axis=0)
            else:
                g_l2 = np.mean((g_t - g_p) ** 2, axis=0)
        else:
            g_l2 = 0

        if log_gamma:
            _log(
                f"Noise level: {noise_level}\n"
                f"\tmu:    {np.mean(mu_l2):.3e} ± {np.std(mu_l2):.3e}\n"
                f"\tgamma: {np.mean(g_l2):.3e} ± {np.std(g_l2):.3e}",
                log_path,
            )
        else:
            _log(
                f"Noise level: {noise_level}\n\tmu:    {np.mean(mu_l2):.3e} ± {np.std(mu_l2):.3e}\n",
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

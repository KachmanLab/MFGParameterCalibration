from typing import Any

import jax
import jax.numpy as jnp

from core.ParameterNetwork import ParameterNetwork
from src.core.PicardSolver import PicardSolver


class DynamicPicardSolver(PicardSolver):
    """
    Fixed-point solver for parametric mean field games.
    Encapsulates the iteration logic and the implicit differentiation wrapper.
    """

    def __init__(self, mfg, dt: float, t_grid: jnp.ndarray, model: ParameterNetwork, max_iter=1000, tol=1e-5, damping=0.0, constant_gamma: bool = False):
        super().__init__(mfg, dt, max_iter, tol, False, damping,)
        self.constant_gamma = constant_gamma
        self.t_grid = t_grid
        self.model = model

    def _pred_g_traj(
        self,
        params: Any,
        t_t: jnp.ndarray,
        mu_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict a full parameter trajectory of shape (N, gamma_size).

        When ``constant_gamma`` is True the network is called once and the
        result broadcast over time.
        """
        if self.constant_gamma:
            v = self.model.apply({"params": params}, 0.0, jnp.zeros(mu_t.shape[-1]))
            return jnp.broadcast_to(v, (t_t.shape[0], *v.shape))
        return jax.vmap(lambda t, m: self.model.apply({"params": params}, t, m))(t_t, mu_t)


    def dynamic_solve_hjb(self, mu_traj, gamma_traj):
        """
        Backward Euler integration of HJB.
        """
        u_terminal = self.mfg.g(gamma_traj[-1], mu_traj[-1])

        def scan_fn(u_next, state_curr):
            mu_curr, gamma_curr = state_curr

            Q = self.mfg.Q(gamma_curr, u_next, mu_curr)
            alpha = self.mfg.optimal_control(mu_curr, u_next, gamma_curr)
            f = self.mfg.f(gamma_curr, u_next, mu_curr, alpha)

            # sum_y a_{xy} (u_y - u_x)
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)

            u_prev = u_next + self.dt * (f + Hamiltonian)

            return u_prev, u_prev

        # Iterate backwards in time: compute u_{N-2} down to u_0
        # We need mu and gamma from indices N-2 down to 0.
        mu_prev_steps = mu_traj[:-1][::-1]
        gamma_prev_steps = gamma_traj[:-1][::-1]

        _, u_rev = jax.lax.scan(scan_fn, u_terminal, (mu_prev_steps, gamma_prev_steps))

        # u_rev contains [u_{N-2}, u_{N-3}, ..., u_0]
        # Reverse to get [u_0, ..., u_{N-2}] and append u_terminal
        u_traj = jnp.concatenate([u_rev[::-1], u_terminal[None, :]], axis=0)
        return u_traj

    def solve_fpk(self, u_traj, gamma_traj, mu_0):
        """
        Forward Euler integration of FPK.
        """

        def scan_fn(mu_prev, state_curr):
            u_curr, gamma_curr = state_curr
            Q = self.mfg.Q(gamma_curr, u_curr, mu_prev)
            # dot(mu)_x = sum_{y!=x} mu_y a_{yx} - mu_x sum_{y!=x} a_{xy}
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            return mu_next, mu_next

        # Iterate forwards in time
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, (u_traj[:-1], gamma_traj[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)

    def dynamic_picard_operator(self, mu_traj, gamma_traj, mu_0):
        """
        The operator G: mu -> mu_next
        """
        u_traj = self.dynamic_solve_hjb(mu_traj, gamma_traj)
        return self.solve_fpk(u_traj, gamma_traj, mu_0)

    def dynamic_fixed_point(self, init_mu, params: Any, mu_0):
        """
        Finds the fixed point of G iteratively.
        """
        def cond_fun(val):
            i, prev, curr = val
            diff = jnp.max(jnp.abs(curr - prev))
            return (i < self.max_iter) & (diff > self.tol) | (i < 2)

        def body_fun(val):
            i, prev, curr = val
            # Predict the gamma trajectory based on the updated mu_traj (curr)
            next_gamma = self._pred_g_traj(params, self.t_grid, curr)
            next_mu = self.dynamic_picard_operator(curr, next_gamma, mu_0)
            next_mu_damped = (1.0 - self.damping) * next_mu + self.damping * curr
            return i + 1, curr, next_mu_damped

        gamma_traj = self._pred_g_traj(params, self.t_grid, init_mu)
        initial_next = self.dynamic_picard_operator(init_mu, gamma_traj, mu_0)
        initial_next_damped = (1.0 - self.damping) * initial_next + self.damping * init_mu

        _, _, final_mu = jax.lax.while_loop(cond_fun, body_fun, (0, init_mu, initial_next_damped))
        return final_mu


    def get_solver_fn(self):
        """
        Returns a pure JAX function that solves the fixed point and has a custom_vjp
        for implicit differentiation through the fixed point. This version does not require
        Gamma to be pre-evaluated. Instead, it features a fully coupled version that evaluates
        the \gamma_\theta on the solver-generated trajectory.
        """

        @jax.custom_vjp
        def solve_mfg(params, init_mu, mu_0):  # init_mu -> initial guess for mu (traj)
            return self.dynamic_fixed_point(init_mu, params, mu_0)

        def fwd(params, init_mu, mu_0):
            mu_star = solve_mfg(params, init_mu, mu_0)
            return mu_star, (mu_star, params, mu_0)

        def bwd(res, g):
            mu_star, params, mu_0 = res

            def coupled_operator(m, p):
                gm = self._pred_g_traj(p, self.t_grid, m)
                return self.dynamic_picard_operator(m, gm, mu_0)

            _, vjp_func = jax.vjp(coupled_operator, mu_star, params)

            def adjoint_cond(val):
                i, prev_w, w = val
                diff = jnp.max(jnp.abs(w - prev_w))
                return (i < 1000) & (diff > 1e-5) | (i < 2)

            def adjoint_body(val):
                i, prev_w, w = val
                w_mu_vjp, _ = vjp_func(w)  # dF/dmu^T applied to w
                w_next = w_mu_vjp + g
                w_next_damped = (1.0 - self.damping) * w_next + self.damping * w
                return i + 1, w, w_next_damped

            init_w = jnp.zeros_like(g)
            _, _, w_star = jax.lax.while_loop(adjoint_cond, adjoint_body, (0, init_w, g))

            _, params_vjp = vjp_func(w_star)  # dF/dparams^T applied to w*
            return params_vjp, None, None

        solve_mfg.defvjp(fwd, bwd)
        return solve_mfg
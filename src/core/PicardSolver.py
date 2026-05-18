import jax
import jax.numpy as jnp


class PicardSolver:
    """
    Fixed-point solver for parametric mean field games.
    Encapsulates the iteration logic and the implicit differentiation wrapper.
    """

    def __init__(self, mfg, dt, max_iter=1000, tol=1e-5, is_mfc=False, damping=0.0):
        self.mfg = mfg
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.is_mfc = is_mfc
        self.damping = damping

    def solve_hjb(self, mu_traj, gamma_traj, dg_dmu_traj):
        """
        Backward Euler integration of HJB.
        """
        N, d = mu_traj.shape
        u_terminal = self.mfg.g(gamma_traj[-1], mu_traj[-1])
        if self.is_mfc:
            u_terminal += self.mfg.mfc_extra_term_terminal(mu_traj[-1])

        def scan_fn(u_next, state_curr):
            mu_curr, gamma_curr, dg_dmu_curr = state_curr
            Q = self.mfg.Q(gamma_curr, u_next, mu_curr)
            alpha = self.mfg.optimal_control(mu_curr, u_next, gamma_curr)
            f = self.mfg.f(gamma_curr, u_next, mu_curr, alpha)

            # sum_y a_{xy} (u_y - u_x)
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)

            u_prev = u_next + self.dt * (f + Hamiltonian)
            if self.is_mfc:
                u_prev += self.dt * self.mfg.mfc_extra_term(gamma_curr, dg_dmu_curr, u_next, mu_curr)

            return u_prev, u_prev

        # Iterate backwards in time: compute u_{N-2} down to u_0
        # We need mu and gamma from indices N-2 down to 0.
        mu_prev_steps = mu_traj[:-1][::-1]
        gamma_prev_steps = gamma_traj[:-1][::-1]
        if dg_dmu_traj is not None:
            dg_dmu_prev_steps = dg_dmu_traj[:-1][::-1]
        else:
            dg_dmu_prev_steps = None

        _, u_rev = jax.lax.scan(scan_fn, u_terminal, (mu_prev_steps, gamma_prev_steps, dg_dmu_prev_steps))

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

    def picard_operator(self, mu_traj, gamma_traj, dg_dmu_traj, mu_0):
        """
        The operator G: mu -> mu_next
        """
        u_traj = self.solve_hjb(mu_traj, gamma_traj, dg_dmu_traj)
        return self.solve_fpk(u_traj, gamma_traj, mu_0)

    def custom_fixed_point(self, init_mu, gamma_traj, dg_dmu_traj, mu_0):
        """
        Finds the fixed point of G iteratively.
        """

        def cond_fun(val):
            i, prev, curr = val
            diff = jnp.max(jnp.abs(curr - prev))
            return (i < self.max_iter) & (diff > self.tol) | (i < 2)

        def body_fun(val):
            i, prev, curr = val
            next_mu = self.picard_operator(curr, gamma_traj, dg_dmu_traj, mu_0)
            next_mu_damped = (1.0 - self.damping) * next_mu + self.damping * curr
            return i + 1, curr, next_mu_damped

        initial_next = self.picard_operator(init_mu, gamma_traj, dg_dmu_traj, mu_0)
        initial_next_damped = (1.0 - self.damping) * initial_next + self.damping * init_mu

        _, _, final_mu = jax.lax.while_loop(cond_fun, body_fun, (0, init_mu, initial_next_damped))
        return final_mu

    @staticmethod
    def picard_operator_static(mu_prev, gamma, dg_dmu, mu_0, mfg_model, dt, is_mfc):
        """
        Self-contained static version of the Picard operator for use inside while_loop.
        """
        # 1. HJB Backward
        u_terminal = mfg_model.g(gamma[-1], mu_prev[-1])
        if is_mfc:
            u_terminal += mfg_model.mfc_extra_term_terminal(mu_prev[-1])

        def scan_hjb(u_next, state_curr):
            m_c, g_c, dg_c = state_curr
            Q = mfg_model.Q(g_c, u_next, m_c)
            alpha = mfg_model.optimal_control(m_c, u_next, g_c)
            f = mfg_model.f(g_c, u_next, m_c, alpha)
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)
            u_prev_step = u_next + dt * (f + Hamiltonian)

            if is_mfc:
                u_prev_step += dt * mfg_model.mfc_extra_term(g_c, dg_c, u_next, m_c)

            return u_prev_step, u_prev_step

        if is_mfc:
            _, u_rev = jax.lax.scan(scan_hjb, u_terminal, (mu_prev[:-1][::-1], gamma[:-1][::-1], dg_dmu[:-1][::-1]))
        else:
            _, u_rev = jax.lax.scan(scan_hjb, u_terminal, (mu_prev[:-1][::-1], gamma[:-1][::-1], None))

        u_traj = jnp.concatenate([u_rev[::-1], u_terminal[None, :]], axis=0)

        # 2. FPK Forward
        def scan_fpk(m_p, state_curr):
            u_c, g_c = state_curr
            Q = mfg_model.Q(g_c, u_c, m_p)
            influx = jnp.sum(m_p[:, None] * Q, axis=0)
            outflux = m_p * jnp.sum(Q, axis=1)
            mu_next = m_p + dt * (influx - outflux)
            return mu_next, mu_next

        _, mu_fwd = jax.lax.scan(scan_fpk, mu_0, (u_traj[:-1], gamma[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)

    def get_solver_fn(self):
        """
        Returns a pure JAX function that solves the fixed point and has a custom_vjp
        for implicit differentiation through the fixed point.
        """

        @jax.custom_vjp
        def solve_mfg(gamma, dg_dmu, init_mu, mu_0):
            return self.custom_fixed_point(init_mu, gamma, dg_dmu, mu_0)

        def fwd(gamma, dg_dmu, init_mu, mu_0):
            mu_star = solve_mfg(gamma, dg_dmu, init_mu, mu_0)
            return mu_star, (mu_star, gamma, dg_dmu, mu_0)

        def bwd(res, g):
            mu_star, gamma, dg_dmu, mu_0 = res
            # Implicit differentiation: (I - dG/dmu)^-1 * dG/dgamma
            _, vjp_func = jax.vjp(lambda m, gm, dgm: self.picard_operator(m, gm, dgm, mu_0), mu_star, gamma, dg_dmu)

            def adjoint_cond(val):
                i, prev_w, w = val
                diff = jnp.max(jnp.abs(w - prev_w))
                return (i < 1000) & (diff > 1e-5) | (i < 2)

            def adjoint_body(val):
                i, prev_w, w = val
                # w_mu_vjp is dG/dmu^T * w
                w_mu_vjp, _, _ = vjp_func(w)

                # Apply the exact same damping to the adjoint loop to ensure convergence!
                # We want to solve w = J^T w + g.
                # Damped update: w_next = (1 - alpha) * (J^T w + g) + alpha * w
                w_next = w_mu_vjp + g
                w_next_damped = (1.0 - self.damping) * w_next + self.damping * w

                return i + 1, w, w_next_damped

            init_w = jnp.zeros_like(g)
            _, _, w_star = jax.lax.while_loop(adjoint_cond, adjoint_body, (0, init_w, g))

            # Now compute gamma_vjp and dg_dmu_vjp
            _, gamma_vjp, dg_dmu_vjp = vjp_func(w_star)
            return (gamma_vjp, dg_dmu_vjp, None, None)

        solve_mfg.defvjp(fwd, bwd)
        return solve_mfg


class BasicPicardSolver:
    """
    Fixed-point solver for parametric mean field games.
    Encapsulates the iteration logic and the implicit differentiation wrapper.
    """

    def __init__(self, mfg, dt, max_iter=1000, tol=1e-5):
        self.mfg = mfg
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def solve_hjb(self, mu_traj, gamma_traj):
        """
        Backward Euler integration of HJB.
        """
        N, d = mu_traj.shape
        u_terminal = self.mfg.g(gamma_traj[-1], mu_traj[-1])

        def scan_fn(u_next, state_curr):
            mu_curr, gamma_curr = state_curr
            Q = self.mfg.Q(gamma_curr, u_next)
            f = self.mfg.f(gamma_curr, u_next, mu_curr)

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
            Q = self.mfg.Q(gamma_curr, u_curr)
            # dot(mu)_x = sum_{y!=x} mu_y a_{yx} - mu_x sum_{y!=x} a_{xy}
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            return mu_next, mu_next

        # Iterate forwards in time
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, (u_traj[:-1], gamma_traj[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)

    def picard_operator(self, mu_traj, gamma_traj, mu_0):
        """
        The operator G: mu -> mu_next
        """
        u_traj = self.solve_hjb(mu_traj, gamma_traj)
        return self.solve_fpk(u_traj, gamma_traj, mu_0)

    def custom_fixed_point(self, init_mu, gamma_traj, mu_0):
        """
        Finds the fixed point of G iteratively.
        """

        def cond_fun(val):
            i, prev, curr = val
            diff = jnp.max(jnp.abs(curr - prev))
            return (i < self.max_iter) & (diff > self.tol) | (i < 2)

        def body_fun(val):
            i, prev, curr = val
            return i + 1, curr, self.picard_operator(curr, gamma_traj, mu_0)

        _, _, final_mu = jax.lax.while_loop(
            cond_fun, body_fun, (0, init_mu, self.picard_operator(init_mu, gamma_traj, mu_0))
        )
        return final_mu

    @staticmethod
    def picard_operator_static(mu_prev, gamma, mu_0, mfg_model, dt):
        """
        Self-contained static version of the Picard operator for use inside while_loop.
        """
        # 1. HJB Backward
        u_terminal = mfg_model.g(gamma[-1], mu_prev[-1])

        def scan_hjb(u_next, state_curr):
            m_c, g_c = state_curr
            Q = mfg_model.Q(g_c, u_next)
            f = mfg_model.f(g_c, u_next, m_c)
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)
            u_prev_step = u_next + dt * (f + Hamiltonian)
            return u_prev_step, u_prev_step

        _, u_rev = jax.lax.scan(scan_hjb, u_terminal, (mu_prev[:-1][::-1], gamma[:-1][::-1]))
        u_traj = jnp.concatenate([u_rev[::-1], u_terminal[None, :]], axis=0)

        # 2. FPK Forward
        def scan_fpk(m_p, state_curr):
            u_c, g_c = state_curr
            Q = mfg_model.Q(g_c, u_c)
            influx = jnp.sum(m_p[:, None] * Q, axis=0)
            outflux = m_p * jnp.sum(Q, axis=1)
            mu_next = m_p + dt * (influx - outflux)
            return mu_next, mu_next

        _, mu_fwd = jax.lax.scan(scan_fpk, mu_0, (u_traj[:-1], gamma[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)

    def get_solver_fn(self):
        """
        Returns a pure JAX function that solves the fixed point and has a custom_vjp
        for implicit differentiation through the fixed point.
        """

        @jax.custom_vjp
        def solve_mfg(gamma, init_mu, mu_0):
            return self.custom_fixed_point(init_mu, gamma, mu_0)

        def fwd(gamma, init_mu, mu_0):
            mu_star = solve_mfg(gamma, init_mu, mu_0)
            return mu_star, (mu_star, gamma, mu_0)

        def bwd(res, g):
            mu_star, gamma, mu_0 = res
            # Implicit differentiation: (I - dG/dmu)^-1 * dG/dgamma
            # We solve the adjoint equation (I - dG/dmu)^T * w = g
            # Then gamma_vjp = w^T * dG/dgamma
            _, vjp_func = jax.vjp(lambda m, gm: self.picard_operator(m, gm, mu_0), mu_star, gamma)

            def adjoint_cond(val):
                i, prev_w, w = val
                diff = jnp.max(jnp.abs(w - prev_w))
                return (i < 1000) & (diff > 1e-5) | (i < 2)

            def adjoint_body(val):
                i, prev_w, w = val
                # We need dG/dmu^T * w, which is the first output of the VJP of G(mu, gamma)
                w_mu_vjp, _ = vjp_func(w)
                return i + 1, w, w_mu_vjp + g

            init_w = jnp.zeros_like(g)
            _, _, w_star = jax.lax.while_loop(adjoint_cond, adjoint_body, (0, init_w, g))

            # Now compute gamma_vjp = w_star^T * dG/dgamma
            _, gamma_vjp = vjp_func(w_star)
            return (gamma_vjp, None, None)

        solve_mfg.defvjp(fwd, bwd)
        return solve_mfg

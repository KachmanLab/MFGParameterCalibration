"""
MFG model for bike-sharing with destination congestion (Variant B).
Also includes the forward-only solver for the baseline model.
"""
import jax
import jax.numpy as jnp


class BikeShareMFG:
    """
    Mean-field game model for bike-sharing with destination congestion.
    
    Running cost: f(x, a, mu) = b * sum_{y!=x} (a_xy - q_base_xy)^2 + c * sum_{y!=x} mu(y) * a_xy
    Terminal cost: g = 0
    Optimal control: a*_xy = max(0, q_base_xy - (c*mu(y) + u(y) - u(x)) / (2b))
    
    Nesting: when c=0 and g=0, u=0 and a* = q_base, recovering the forward ODE.
    """
    
    def __init__(self, d):
        self.d = d
        # Precompute the mask for off-diagonal entries
        # This maps a flat vector of d*(d-1) values to a d x d matrix
        mask = jnp.ones((d, d)) - jnp.eye(d)
        self.off_diag_mask = mask
    
    def _flat_to_matrix(self, flat_rates):
        """Convert flat vector of d*(d-1) positive rates to a d x d matrix with zeros on diagonal."""
        d = self.d
        # Fill off-diagonal entries row by row
        # flat_rates[0..d-2] -> row 0 (columns 1..d-1)
        # flat_rates[d-1..2d-3] -> row 1 (columns 0, 2..d-1)
        # etc.
        mat = jnp.zeros((d, d))
        idx = 0
        # Use a static unrolled approach that JAX can trace
        for i in range(d):
            for j in range(d):
                if i != j:
                    mat = mat.at[i, j].set(flat_rates[idx])
                    idx += 1
        return mat
    
    def Q(self, gamma_base, u, mu_curr, b, c):
        """
        Compute the optimal transition rate matrix.
        
        Args:
            gamma_base: (d*(d-1),) raw base rate parameters -> softplus -> q_base
            u: (d,) value function
            mu_curr: (d,) current mean-field distribution
            b: scalar, control effort weight (> 0)
            c: scalar, destination congestion weight (>= 0)
        
        Returns:
            Q: (d, d) transition rate matrix (off-diagonal >= 0, rows sum to 0)
        """
        d = self.d
        q_base = jax.nn.softplus(gamma_base)
        Q_base = self._flat_to_matrix(q_base)
        
        # Optimal control: a*_xy = max(0, q_base_xy - (c*mu(y) + u(y) - u(x)) / (2*b))
        Delta_u = u[None, :] - u[:, None]  # Delta_u[x, y] = u(y) - u(x)
        correction = (c * mu_curr[None, :] + Delta_u) / (2.0 * b)
        A = Q_base - correction
        
        # Clip at 0 and zero out diagonal
        A = jnp.clip(A, 0.0, 100.0)
        A = A * self.off_diag_mask
        
        return A
    
    def Q_base_only(self, gamma_base):
        """Compute base rate matrix (no control). Used by the forward-only baseline."""
        q_base = jax.nn.softplus(gamma_base)
        return self._flat_to_matrix(q_base)
    
    def f(self, gamma_base, u, mu_curr, b, c):
        """Running cost vector f(x) for all states x."""
        d = self.d
        Q_opt = self.Q(gamma_base, u, mu_curr, b, c)
        Q_base = self.Q_base_only(gamma_base)
        
        # control effort: b * sum_{y!=x} (a_xy - q_base_xy)^2
        effort = b * jnp.sum((Q_opt - Q_base) ** 2 * (1.0 - jnp.eye(d)), axis=1)
        
        # destination congestion: c * sum_{y!=x} mu(y) * a_xy
        dest_cong = c * jnp.sum(mu_curr[None, :] * Q_opt * (1.0 - jnp.eye(d)), axis=1)
        
        return effort + dest_cong
    
    def g(self, gamma_base, mu_T):
        """Terminal cost (= 0)."""
        return jnp.zeros(self.d)


class ForwardSolver:
    """
    Forward-only ODE solver for the mean-field dynamics baseline.
    Integrates dot(mu) = mu * Q(gamma) forward in time using Euler.
    No HJB, no Picard iteration.
    """
    
    def __init__(self, mfg_model, dt):
        self.mfg = mfg_model
        self.dt = dt
    
    def solve_forward(self, gamma_traj, mu_0):
        """
        Solve the forward Kolmogorov equation.
        
        Args:
            gamma_traj: (N, d*(d-1)) base rate parameters at each time step
            mu_0: (d,) initial distribution
        
        Returns:
            mu_traj: (N, d) mean-field trajectory
        """
        def scan_fn(mu_prev, gamma_curr):
            Q = self.mfg.Q_base_only(gamma_curr)
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            # Project back to simplex (ensure non-negative and sum to 1)
            mu_next = jnp.clip(mu_next, 0.0, None)
            mu_next = mu_next / jnp.sum(mu_next)
            return mu_next, mu_next
        
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, gamma_traj[:-1])
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)
    
    def get_solver_fn(self):
        """Returns a pure JAX function with custom_vjp for implicit differentiation."""
        @jax.custom_vjp
        def solve(gamma_traj, mu_0):
            return self.solve_forward(gamma_traj, mu_0)
        
        def fwd(gamma_traj, mu_0):
            mu_star = solve(gamma_traj, mu_0)
            return mu_star, (mu_star, gamma_traj, mu_0)
        
        def bwd(res, g):
            mu_star, gamma_traj, mu_0 = res
            # Direct differentiation through the forward solver
            # (no implicit differentiation needed — just let JAX autodiff through scan)
            _, vjp_fn = jax.vjp(lambda gt, m0: self.solve_forward(gt, m0), 
                                gamma_traj, mu_0)
            gamma_grad, mu0_grad = vjp_fn(g)
            return (gamma_grad, mu0_grad)
        
        solve.defvjp(fwd, bwd)
        return solve


class BikeSharePicardSolver:
    """
    Picard iteration solver for the bike-sharing MFG.
    Solves the coupled HJB-FPK system with destination congestion.
    """
    
    def __init__(self, mfg_model, dt, max_iter=200, tol=1e-5, damping=0.5):
        self.mfg = mfg_model
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.d = mfg_model.d
    
    def solve_hjb(self, mu_traj, gamma_traj, b, c, w_g):
        """Backward integration of HJB."""
        N, d = mu_traj.shape
        u_terminal = w_g  # Terminal cost
        
        def scan_fn(u_next, state_curr):
            mu_curr, gamma_curr = state_curr
            Q = self.mfg.Q(gamma_curr, u_next, mu_curr, b, c)
            f_val = self.mfg.f(gamma_curr, u_next, mu_curr, b, c)
            
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)
            
            u_prev = u_next + self.dt * (f_val + Hamiltonian)
            return u_prev, u_prev
        
        mu_prev_steps = mu_traj[:-1][::-1]
        gamma_prev_steps = gamma_traj[:-1][::-1]
        
        _, u_rev = jax.lax.scan(scan_fn, u_terminal, (mu_prev_steps, gamma_prev_steps))
        u_traj = jnp.concatenate([u_rev[::-1], u_terminal[None, :]], axis=0)
        return u_traj
    
    def solve_fpk(self, u_traj, gamma_traj, mu_0, b, c):
        """Forward integration of FPK with optimal control."""
        def scan_fn(mu_prev, state_curr):
            u_curr, gamma_curr = state_curr
            Q = self.mfg.Q(gamma_curr, u_curr, mu_prev, b, c)
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            mu_next = jnp.clip(mu_next, 0.0, None)
            mu_next = mu_next / jnp.sum(mu_next)
            return mu_next, mu_next
        
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, (u_traj[:-1], gamma_traj[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)
    
    def picard_operator(self, mu_traj, gamma_traj, mu_0, b, c, w_g):
        """One step of Picard iteration: mu -> mu_next."""
        u_traj = self.solve_hjb(mu_traj, gamma_traj, b, c, w_g)
        return self.solve_fpk(u_traj, gamma_traj, mu_0, b, c)
    
    def custom_fixed_point(self, init_mu, gamma_traj, mu_0, b, c, w_g):
        """Find fixed point of the Picard operator with damping."""
        def cond_fun(val):
            i, prev, curr = val
            diff = jnp.max(jnp.abs(curr - prev))
            return (i < self.max_iter) & (diff > self.tol) | (i < 2)
        
        def body_fun(val):
            i, prev, curr = val
            next_mu = self.picard_operator(curr, gamma_traj, mu_0, b, c, w_g)
            next_mu_damped = (1.0 - self.damping) * next_mu + self.damping * curr
            return i + 1, curr, next_mu_damped
        
        initial_next = self.picard_operator(init_mu, gamma_traj, mu_0, b, c, w_g)
        initial_next_damped = (1.0 - self.damping) * initial_next + self.damping * init_mu
        
        _, _, final_mu = jax.lax.while_loop(
            cond_fun, body_fun,
            (0, init_mu, initial_next_damped)
        )
        return final_mu
    
    def get_solver_fn(self):
        """Returns a JAX function with custom_vjp for implicit differentiation."""
        @jax.custom_vjp
        def solve_mfg(gamma_traj, mu_0, b, c, w_g):
            N = gamma_traj.shape[0]
            d = self.d
            init_mu = jnp.repeat(mu_0[None, :], N, axis=0)
            return self.custom_fixed_point(init_mu, gamma_traj, mu_0, b, c, w_g)
        
        def fwd(gamma_traj, mu_0, b, c, w_g):
            mu_star = solve_mfg(gamma_traj, mu_0, b, c, w_g)
            return mu_star, (mu_star, gamma_traj, mu_0, b, c, w_g)
        
        def bwd(res, g):
            mu_star, gamma_traj, mu_0, b, c, w_g = res
            
            _, vjp_func = jax.vjp(
                lambda m, gt, bb, cc, wg: self.picard_operator(m, gt, mu_0, bb, cc, wg),
                mu_star, gamma_traj, b, c, w_g
            )
            
            def adjoint_cond(val):
                i, prev_w, w = val
                diff = jnp.max(jnp.abs(w - prev_w))
                return (i < 500) & (diff > 1e-5) | (i < 2)
            
            def adjoint_body(val):
                i, prev_w, w = val
                w_mu_vjp, _, _, _, _ = vjp_func(w)
                w_next = w_mu_vjp + g
                w_next_damped = (1.0 - self.damping) * w_next + self.damping * w
                return i + 1, w, w_next_damped
            
            init_w = jnp.zeros_like(g)
            _, _, w_star = jax.lax.while_loop(adjoint_cond, adjoint_body, (0, init_w, g))
            
            _, gamma_vjp, b_vjp, c_vjp, wg_vjp = vjp_func(w_star)
            return (gamma_vjp, None, b_vjp, c_vjp, wg_vjp)
        
        solve_mfg.defvjp(fwd, bwd)
        return solve_mfg

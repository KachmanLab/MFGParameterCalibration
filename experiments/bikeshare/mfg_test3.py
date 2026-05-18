import jax
import jax.numpy as jnp
import numpy as np

class BikeShareMFG:
    def __init__(self, d=6):
        self.d = d
        
    def _flat_to_matrix(self, flat_rates):
        d = self.d
        mat = jnp.zeros((d, d))
        mask = ~np.eye(d, dtype=bool)
        mat = mat.at[mask].set(flat_rates)
        return mat

    def Q_base_only(self, gamma_base):
        q_base = jax.nn.softplus(gamma_base)
        return self._flat_to_matrix(q_base)
        
    def Q(self, gamma_base, gamma_cost, u, b):
        q_base = jax.nn.softplus(gamma_base)
        q_base_mat = self._flat_to_matrix(q_base)
        gamma_cost_mat = self._flat_to_matrix(gamma_cost)
        
        Delta_u = u[None, :] - u[:, None]
        a_star = q_base_mat + (gamma_cost_mat - Delta_u) / (2.0 * b)
        a_star = jax.nn.relu(a_star)
        return a_star
    
    def f(self, gamma_base, gamma_cost, u, b):
        Q_opt = self.Q(gamma_base, gamma_cost, u, b)
        Q_base = self.Q_base_only(gamma_base)
        gamma_cost_mat = self._flat_to_matrix(gamma_cost)
        
        effort = b * jnp.sum((Q_opt - Q_base) ** 2 * (1.0 - jnp.eye(self.d)), axis=1)
        linear_cost = - jnp.sum(gamma_cost_mat * Q_opt * (1.0 - jnp.eye(self.d)), axis=1)
        
        return effort + linear_cost
    
    def g(self, gamma_base, mu_T):
        return jnp.zeros(self.d)


class ForwardSolver:
    def __init__(self, mfg_model, dt):
        self.mfg = mfg_model
        self.dt = dt
    
    def solve_forward(self, gamma_traj, mu_0):
        def scan_fn(mu_prev, gamma_curr):
            Q = self.mfg.Q_base_only(gamma_curr)
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            mu_next = jnp.clip(mu_next, 0.0, None)
            mu_next = mu_next / jnp.sum(mu_next)
            return mu_next, mu_next
        
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, gamma_traj[:-1])
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)
    
    def get_solver_fn(self):
        @jax.custom_vjp
        def solve(gamma_traj, mu_0):
            return self.solve_forward(gamma_traj, mu_0)
        
        def fwd(gamma_traj, mu_0):
            mu_star = solve(gamma_traj, mu_0)
            return mu_star, (mu_star, gamma_traj, mu_0)
        
        def bwd(res, g):
            mu_star, gamma_traj, mu_0 = res
            _, vjp_fn = jax.vjp(lambda gt, m0: self.solve_forward(gt, m0), 
                                gamma_traj, mu_0)
            gamma_grad, mu0_grad = vjp_fn(g)
            return (gamma_grad, mu0_grad)
        
        solve.defvjp(fwd, bwd)
        return solve


class BikeSharePicardSolver:
    def __init__(self, mfg_model, dt, cost_apply_fn, t_grid, max_iter=200, tol=1e-5, damping=0.5):
        self.mfg = mfg_model
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.d = mfg_model.d
        self.cost_apply_fn = cost_apply_fn
        self.t_grid = t_grid
    
    def solve_hjb(self, mu_traj, gamma_base_traj, gamma_cost_traj, b, w_g):
        N, d = mu_traj.shape
        u_terminal = w_g
        
        def scan_fn(u_next, state_curr):
            mu_curr, gamma_base_curr, gamma_cost_curr = state_curr
            Q = self.mfg.Q(gamma_base_curr, gamma_cost_curr, u_next, b)
            f_val = self.mfg.f(gamma_base_curr, gamma_cost_curr, u_next, b)
            
            Delta_u = u_next[None, :] - u_next[:, None]
            Hamiltonian = jnp.sum(Q * Delta_u, axis=1)
            
            u_prev = u_next + self.dt * (f_val + Hamiltonian)
            return u_prev, u_prev
        
        mu_prev_steps = mu_traj[:-1][::-1]
        gamma_base_prev_steps = gamma_base_traj[:-1][::-1]
        gamma_cost_prev_steps = gamma_cost_traj[:-1][::-1]
        
        _, u_rev = jax.lax.scan(scan_fn, u_terminal, (mu_prev_steps, gamma_base_prev_steps, gamma_cost_prev_steps))
        u_traj = jnp.concatenate([u_rev[::-1], u_terminal[None, :]], axis=0)
        return u_traj
    
    def solve_fpk(self, u_traj, gamma_base_traj, gamma_cost_traj, mu_0, b):
        def scan_fn(mu_prev, state_curr):
            u_curr, gamma_base_curr, gamma_cost_curr = state_curr
            Q = self.mfg.Q(gamma_base_curr, gamma_cost_curr, u_curr, b)
            influx = jnp.sum(mu_prev[:, None] * Q, axis=0)
            outflux = mu_prev * jnp.sum(Q, axis=1)
            mu_next = mu_prev + self.dt * (influx - outflux)
            mu_next = jnp.clip(mu_next, 0.0, None)
            mu_next = mu_next / jnp.sum(mu_next)
            return mu_next, mu_next
        
        _, mu_fwd = jax.lax.scan(scan_fn, mu_0, (u_traj[:-1], gamma_base_traj[:-1], gamma_cost_traj[:-1]))
        return jnp.concatenate([mu_0[None, :], mu_fwd], axis=0)
    
    def picard_operator(self, mu_traj, gamma_base_traj, cost_nn_params, mu_0, b, w_g):
        gamma_cost_traj = jax.vmap(self.cost_apply_fn, in_axes=(None, 0, 0))(cost_nn_params, self.t_grid, mu_traj)
        u_traj = self.solve_hjb(mu_traj, gamma_base_traj, gamma_cost_traj, b, w_g)
        return self.solve_fpk(u_traj, gamma_base_traj, gamma_cost_traj, mu_0, b)
    
    def custom_fixed_point(self, init_mu, gamma_base_traj, cost_nn_params, mu_0, b, w_g):
        def cond_fun(val):
            i, prev, curr = val
            diff = jnp.max(jnp.abs(curr - prev))
            return (i < self.max_iter) & (diff > self.tol) | (i < 2)
        
        def body_fun(val):
            i, prev, curr = val
            next_mu = self.picard_operator(curr, gamma_base_traj, cost_nn_params, mu_0, b, w_g)
            next_mu_damped = (1.0 - self.damping) * next_mu + self.damping * curr
            return i + 1, curr, next_mu_damped
        
        initial_next = self.picard_operator(init_mu, gamma_base_traj, cost_nn_params, mu_0, b, w_g)
        initial_next_damped = (1.0 - self.damping) * initial_next + self.damping * init_mu
        
        _, _, final_mu = jax.lax.while_loop(
            cond_fun, body_fun,
            (0, init_mu, initial_next_damped)
        )
        return final_mu
    
    def get_solver_fn(self):
        @jax.custom_vjp
        def solve_mfg(gamma_base_traj, cost_nn_params, mu_0, b, w_g):
            N = gamma_base_traj.shape[0]
            d = self.d
            init_mu = jnp.repeat(mu_0[None, :], N, axis=0)
            return self.custom_fixed_point(init_mu, gamma_base_traj, cost_nn_params, mu_0, b, w_g)
        
        def fwd(gamma_base_traj, cost_nn_params, mu_0, b, w_g):
            mu_star = solve_mfg(gamma_base_traj, cost_nn_params, mu_0, b, w_g)
            return mu_star, (mu_star, gamma_base_traj, cost_nn_params, mu_0, b, w_g)
        
        def bwd(res, g):
            mu_star, gamma_base_traj, cost_nn_params, mu_0, b, w_g = res
            
            _, vjp_func = jax.vjp(
                lambda m, gt, cnp, bb, wg: self.picard_operator(m, gt, cnp, mu_0, bb, wg),
                mu_star, gamma_base_traj, cost_nn_params, b, w_g
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
            
            _, gamma_base_vjp, cost_nn_vjp, b_vjp, wg_vjp = vjp_func(w_star)
            return (gamma_base_vjp, cost_nn_vjp, None, b_vjp, wg_vjp)
        
        solve_mfg.defvjp(fwd, bwd)
        return solve_mfg

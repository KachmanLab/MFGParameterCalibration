import os
import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from train_test3 import GammaNetwork, CostNetwork, compute_gamma_trajectory, load_data
from mfg_test3 import BikeShareMFG, ForwardSolver, BikeSharePicardSolver

def train_quick(D, T, N, dt, train_data, t_grid):
    print("Training models quickly for Zero-Shot evaluation...")
    mfg_model = BikeShareMFG(d=D)
    gamma_network = GammaNetwork(d=D, complexity="linear")
    cost_network = CostNetwork(d=D)
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    gamma_params = gamma_network.init(k1, jnp.array([0.0]))
    cost_params = cost_network.init(k2, jnp.array([0.0]), jnp.zeros(D))
    cost_b_params = {'log_b': jnp.log(jnp.exp(1.0) - 1.0), 'w_g': jnp.zeros(D)}
    
    # Train Baseline
    opt_base = optax.adam(5e-3)
    opt_state_base = opt_base.init(gamma_params)
    forward_solver = ForwardSolver(mfg_model, dt)
    
    @jax.jit
    def train_base(p, opt_s, mu_obs):
        def loss(p_):
            traj = compute_gamma_trajectory(p_, gamma_network, t_grid)
            return jnp.mean((forward_solver.solve_forward(traj, mu_obs[0]) - mu_obs)**2)
        l, g = jax.value_and_grad(loss)(p)
        u, opt_s = opt_base.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s
        
    for ep in range(100):
        for i in range(train_data.shape[0]):
            gamma_params, opt_state_base = train_base(gamma_params, opt_state_base, train_data[i])
            
    # Train MFG
    all_mfg_params = {'gamma_nn': gamma_params, 'cost_nn': cost_params, 'cost': cost_b_params}
    opt_mfg = optax.adam(5e-3)
    opt_state_mfg = opt_mfg.init(all_mfg_params)
    picard = BikeSharePicardSolver(mfg_model, dt, cost_network.apply, t_grid, 100, 1e-4, 0.5)
    solver_fn = picard.get_solver_fn()
    
    @jax.jit
    def train_mfg(p, opt_s, mu_obs):
        def loss(p_):
            b = jax.nn.softplus(p_['cost']['log_b'])
            traj = compute_gamma_trajectory(p_['gamma_nn'], gamma_network, t_grid)
            pred = solver_fn(traj, p_['cost_nn'], mu_obs[0], b, p_['cost']['w_g'])
            return jnp.mean((pred - mu_obs)**2)
        l, g = jax.value_and_grad(loss)(p)
        u, opt_s = opt_mfg.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s
        
    for ep in range(100):
        for i in range(train_data.shape[0]):
            all_mfg_params, opt_state_mfg = train_mfg(all_mfg_params, opt_state_mfg, train_data[i])
            
    return gamma_params, all_mfg_params, gamma_network, cost_network
    mfg_model = BikeShareMFG(d=D)
def main():
    train_data, test_data, _, metadata = load_data()
    D = metadata['d']
    N = metadata['N']
    T = 1.0
    dt = T / N
    t_grid = jnp.linspace(0, T, N)
    
    # Quick train
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_test4_intervention_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    base_params, mfg_params, base_net, cost_net = train_quick(D, T, N, dt, train_data, t_grid)
    mfg_model = BikeShareMFG(d=D)
    
    mu0 = jnp.array(test_data[0, 0, :])
    
    shock_start, shock_end = 35, 45
    shock_station = 1
    
    print(f"Simulating Intervention at Station {shock_station} from step {shock_start} to {shock_end}")

    # 1. Baseline
    def forward_step_local(mu_prev, q_base_mat, dt):
        influx = jnp.sum(mu_prev[:, None] * q_base_mat, axis=0)
        outflux = mu_prev * jnp.sum(q_base_mat, axis=1)
        mu_next = mu_prev + dt * (influx - outflux)
        mu_next = jnp.clip(mu_next, 0.0, None)
        return mu_next / jnp.sum(mu_next)

    mu_baseline = [mu0]
    mu_curr = mu0
    for i in range(N):
        t = jnp.array([i * dt])
        q_base_flat = base_net.apply(base_params, t)
        q_base_mat = jax.nn.softplus(mfg_model._flat_to_matrix(q_base_flat))
        
        if shock_start <= i <= shock_end:
            q_base_mat = q_base_mat.at[:, shock_station].set(0.0)
            
        mu_next = forward_step_local(mu_curr, q_base_mat, dt)
        mu_curr = mu_next
        mu_baseline.append(mu_curr)
    mu_baseline = jnp.stack(mu_baseline)
    
    # 2. MFG
    b_val = jax.nn.softplus(mfg_params['cost']['log_b'])
    
    def custom_hjb_step(u_next, mu_curr, t, params, b_val, dt):
        q_base_flat = base_net.apply(params['gamma_nn'], t)
        c_flat = cost_net.apply(params['cost_nn'], t, mu_curr)
        
        step_idx = jnp.floor(t[0] / dt).astype(jnp.int32)
        in_shock = jnp.logical_and(step_idx >= shock_start, step_idx <= shock_end)
        
        # Add shock penalty to the cost flat vector? 
        # Actually it's easier to use the mfg_model functions, which expect flat inputs.
        # But we want to modify the cost *matrix*. Let's modify c_flat before passing it.
        # We know c_matrix = _flat_to_matrix(c_flat). We can add a penalty to the flat vector directly.
        # But where are the indices for station 1?
        # The mask in _flat_to_matrix is `~np.eye(d, dtype=bool)`.
        # It flattens the non-diagonal elements row by row.
        # Let's just modify the matrix inside the mfg_model logic manually:
        
        q_base_mat = mfg_model._flat_to_matrix(jax.nn.softplus(q_base_flat))
        c_matrix = mfg_model._flat_to_matrix(c_flat)
        
        Delta_u = u_next[None, :] - u_next[:, None]
        a_star = q_base_mat + (c_matrix - Delta_u) / (2.0 * b_val)
        a_star = jax.nn.relu(a_star)
        
        a_star = jax.lax.cond(in_shock, lambda a: a.at[:, shock_station].set(0.0), lambda a: a, a_star)
        
        effort = b_val * jnp.sum((a_star - q_base_mat) ** 2 * (1.0 - jnp.eye(D)), axis=1)
        linear_cost = - jnp.sum(c_matrix * a_star * (1.0 - jnp.eye(D)), axis=1)
        f_val = effort + linear_cost
        
        # State penalty: massive POSITIVE running cost for BEING at the station during the shock.
        # In the HJB: u(t) = u(t+dt) + dt*(f + H). Positive f => large u => high cost-to-go.
        # This propagates backward, making Delta_u[x, shock] large, so a_star[x, shock] -> 0 via ReLU.
        state_penalty = jnp.zeros(D)
        state_penalty = jax.lax.cond(in_shock, lambda p: p.at[shock_station].set(1000.0), lambda p: p, state_penalty)
        f_val = f_val + state_penalty
        
        Hamiltonian = jnp.sum(a_star * Delta_u, axis=1)
        return u_next + dt * (f_val + Hamiltonian), a_star
        
    def custom_picard(mu0, params, b_val, max_iter=200):
        mu_traj = jnp.broadcast_to(mu0, (N + 1, D))
        for _ in range(max_iter):
            u_traj = [jnp.zeros(D)]
            a_traj = []
            for i in reversed(range(N)):
                t = jnp.array([i * dt])
                u_curr, a_star = custom_hjb_step(u_traj[-1], mu_traj[i], t, params, b_val, dt)
                u_traj.append(u_curr)
                a_traj.append(a_star)
            u_traj = jnp.stack(u_traj[::-1])
            a_traj = jnp.stack(a_traj[::-1])
            
            new_mu_traj = [mu0]
            for i in range(N):
                new_mu_traj.append(forward_step_local(new_mu_traj[-1], a_traj[i], dt))
            new_mu_traj = jnp.stack(new_mu_traj)
            
            diff = jnp.max(jnp.abs(new_mu_traj - mu_traj))
            mu_traj = 0.5 * new_mu_traj + 0.5 * mu_traj
            if diff < 1e-4: break
        return mu_traj
        
    print("Running Picard Solver for Scenario Planning...")
    mu_mfg = custom_picard(mu0, mfg_params, b_val)
    
    os.makedirs("../../PAPER/figures/bikeshare_merged", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.linspace(6, 22, N + 1)
    ax.plot(hours, mu_baseline[:, shock_station], '-', color='#115E59', linewidth=3, label='MF Dynamics (Reactive)')
    ax.plot(hours, mu_mfg[:, shock_station], '-', color='#7209B7', linewidth=3, label='MFG (Anticipatory)')
    
    s_start, s_end = 6 + (shock_start / N) * 16, 6 + (shock_end / N) * 16
    ax.axvspan(s_start, s_end, color='red', alpha=0.2, label='Announced Station Closure')
    
    ax.set_xlabel("Hour of Day", fontsize=14, fontweight='bold')
    ax.set_ylabel(f"Mass at Station {shock_station} (μ)", fontsize=14, fontweight='bold')
    ax.set_title("Zero-Shot Causal Intervention: Announced Station Closure", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "intervention_station1.pdf"))
    fig.savefig("../../PAPER/figures/bikeshare_merged/intervention_station1.pdf")
    print(f"Saved plot to {results_dir} and PAPER/figures/bikeshare_merged/")

if __name__ == "__main__":
    main()

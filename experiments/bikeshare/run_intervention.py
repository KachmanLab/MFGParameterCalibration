"""
Intervention simulation for bike-sharing MFG.

Runs a counterfactual scenario (station closure) using both:
  - MF Dynamics (baseline): purely reactive, forward-only propagation
  - MFG (ours): anticipatory, equilibrium-based propagation

Produces one plot per station showing the mass trajectory under intervention,
plus one combined panel figure.

Usage:
  python run_intervention.py --scenario 1   # Station 1 closure, mid-day
  python run_intervention.py --scenario 2   # Station 2 closure, morning rush
  python run_intervention.py --test3-dir results/run_test3_XXXX  # load params from test3

If --test3-dir is not given, the script trains from scratch (slower).
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from datetime import datetime

from train_test3 import GammaNetwork, CostNetwork, compute_gamma_trajectory, load_data
from mfg_test3 import BikeShareMFG, ForwardSolver, BikeSharePicardSolver

# ==============================================================================
# Scenario definitions
# ==============================================================================
SCENARIOS = {
    1: {
        'name': 'Station 1 closure (mid-day)',
        'shock_station': 1,
        'shock_start': 35,
        'shock_end': 45,
        'description': 'Broadway & E 14 St closed from ~14:45 to ~17:30',
    },
    2: {
        'name': 'Station 2 closure (morning rush)',
        'shock_station': 2,
        'shock_start': 5,
        'shock_end': 15,
        'description': '8 Ave & W 31 St closed from ~7:15 to ~9:45',
    },
}

# Station metadata (indices -> names)
STATION_NAMES = {
    0: "W 21 St & 6 Ave",
    1: "Broadway & E 14 St",
    2: "8 Ave & W 31 St",
    3: "West St & Chambers St",
    4: "Lafayette St & E 8 St",
    5: "External",
}

# ==============================================================================
# Configuration
# ==============================================================================
D = 6
T = 1.0
N = 64
dt = T / (N - 1)

# ==============================================================================
# Forward step helper
# ==============================================================================
def forward_step(mu_prev, q_mat, dt_val):
    """One step of the forward Kolmogorov equation."""
    influx = jnp.sum(mu_prev[:, None] * q_mat, axis=0)
    outflux = mu_prev * jnp.sum(q_mat, axis=1)
    mu_next = mu_prev + dt_val * (influx - outflux)
    mu_next = jnp.clip(mu_next, 0.0, None)
    return mu_next / jnp.sum(mu_next)


# ==============================================================================
# Training (fallback if no test3 checkpoint)
# ==============================================================================
def train_quick(train_data, t_grid):
    """Quick training for both baseline and MFG models."""
    print("Training models from scratch for intervention evaluation...")
    mfg_model = BikeShareMFG(d=D)
    gamma_network = GammaNetwork(d=D, complexity="linear")
    cost_network = CostNetwork(d=D)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    gamma_params = gamma_network.init(k1, jnp.array([0.0]))
    cost_params = cost_network.init(k2, jnp.array([0.0]), jnp.zeros(D))
    cost_b_params = {'log_b': jnp.log(jnp.exp(1.0) - 1.0), 'w_g': jnp.zeros(D)}

    # --- Train Baseline ---
    opt_base = optax.adam(5e-3)
    opt_state_base = opt_base.init(gamma_params)
    forward_solver = ForwardSolver(mfg_model, dt)

    @jax.jit
    def train_base(p, opt_s, mu_obs):
        def loss(p_):
            traj = compute_gamma_trajectory(p_, gamma_network, t_grid)
            return jnp.mean((forward_solver.solve_forward(traj, mu_obs[0]) - mu_obs)**2)
        l, g = jax.value_and_grad(loss)(p)
        u, opt_s_new = opt_base.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s_new

    print("  Training baseline (100 epochs)...")
    for ep in range(100):
        for i in range(train_data.shape[0]):
            gamma_params, opt_state_base = train_base(gamma_params, opt_state_base, train_data[i])

    # --- Train MFG ---
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
        u, opt_s_new = opt_mfg.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s_new

    print("  Training MFG (100 epochs)...")
    for ep in range(100):
        for i in range(train_data.shape[0]):
            all_mfg_params, opt_state_mfg = train_mfg(all_mfg_params, opt_state_mfg, train_data[i])

    return gamma_params, all_mfg_params


# ==============================================================================
# Load parameters from a test3 run
# ==============================================================================
def load_params_from_test3(test3_dir, seed=0):
    """Load trained baseline and MFG parameters from a test3 results directory."""
    bl_path = os.path.join(test3_dir, "baseline", f"params_seed{seed}.pkl")
    mfg_path = os.path.join(test3_dir, "mfg", f"params_seed{seed}.pkl")

    if not os.path.exists(bl_path) or not os.path.exists(mfg_path):
        raise FileNotFoundError(
            f"Could not find params_seed{seed}.pkl in {test3_dir}/baseline/ and {test3_dir}/mfg/. "
            f"Re-run test3 first (which now saves parameters)."
        )

    with open(bl_path, "rb") as f:
        base_params = jax.tree.map(jnp.array, pickle.load(f))
    with open(mfg_path, "rb") as f:
        mfg_params = jax.tree.map(jnp.array, pickle.load(f))

    print(f"Loaded trained parameters from {test3_dir} (seed {seed})")
    return base_params, mfg_params


# ==============================================================================
# Simulate Baseline (MF Dynamics) under intervention
# ==============================================================================
def simulate_baseline(mu0, base_params, gamma_network, mfg_model, scenario):
    """Forward-only simulation with the shock applied reactively."""
    shock_station = scenario['shock_station']
    shock_start = scenario['shock_start']
    shock_end = scenario['shock_end']

    mu_traj = [mu0]
    mu_curr = mu0
    for i in range(N):
        t = jnp.array([i * dt])
        q_base_flat = gamma_network.apply(base_params, t)
        q_base_mat = jax.nn.softplus(mfg_model._flat_to_matrix(q_base_flat))

        if shock_start <= i <= shock_end:
            q_base_mat = q_base_mat.at[:, shock_station].set(0.0)

        mu_curr = forward_step(mu_curr, q_base_mat, dt)
        mu_traj.append(mu_curr)

    return jnp.stack(mu_traj)


# ==============================================================================
# Simulate MFG (anticipatory) under intervention
# ==============================================================================
def simulate_mfg(mu0, mfg_params, gamma_network, cost_network, mfg_model, scenario,
                 max_iter=200, damping=0.5, tol=1e-4):
    """
    MFG equilibrium with the shock modeled as:
      1. Physical constraint: transition rates TO the closed station are zeroed.
      2. State penalty: a large running cost for being at the closed station during the shock.
    This ensures backward-looking HJB propagation causes agents to anticipate the closure.
    """
    shock_station = scenario['shock_station']
    shock_start = scenario['shock_start']
    shock_end = scenario['shock_end']

    b_val = jax.nn.softplus(mfg_params['cost']['log_b'])

    def custom_hjb_step(u_next, mu_curr, t_val, params, b, dt_val):
        q_base_flat = gamma_network.apply(params['gamma_nn'], t_val)
        c_flat = cost_network.apply(params['cost_nn'], t_val, mu_curr)

        step_idx = jnp.floor(t_val[0] / dt_val).astype(jnp.int32)
        in_shock = jnp.logical_and(step_idx >= shock_start, step_idx <= shock_end)

        q_base_mat = mfg_model._flat_to_matrix(jax.nn.softplus(q_base_flat))
        c_matrix = mfg_model._flat_to_matrix(c_flat)

        Delta_u = u_next[None, :] - u_next[:, None]
        a_star = q_base_mat + (c_matrix - Delta_u) / (2.0 * b)
        a_star = jax.nn.relu(a_star)

        # Physical constraint: zero out rates TO the closed station
        a_star = jax.lax.cond(in_shock, lambda a: a.at[:, shock_station].set(0.0), lambda a: a, a_star)

        effort = b * jnp.sum((a_star - q_base_mat) ** 2 * (1.0 - jnp.eye(D)), axis=1)
        linear_cost = -jnp.sum(c_matrix * a_star * (1.0 - jnp.eye(D)), axis=1)
        f_val = effort + linear_cost

        # State penalty: large positive running cost for BEING at the station during the shock.
        # Propagates backward via HJB, making Delta_u[x, shock] large => a_star[x, shock] -> 0.
        state_penalty = jnp.zeros(D)
        state_penalty = jax.lax.cond(
            in_shock, lambda p: p.at[shock_station].set(1000.0), lambda p: p, state_penalty
        )
        f_val = f_val + state_penalty

        Hamiltonian = jnp.sum(a_star * Delta_u, axis=1)
        return u_next + dt_val * (f_val + Hamiltonian), a_star

    def custom_picard(mu0_local, params, b, max_it, damp, tol_val):
        mu_traj = jnp.broadcast_to(mu0_local, (N + 1, D))
        for it in range(max_it):
            # Backward pass (HJB)
            u_traj = [jnp.zeros(D)]
            a_traj = []
            for i in reversed(range(N)):
                t_val = jnp.array([i * dt])
                u_curr, a_star = custom_hjb_step(u_traj[-1], mu_traj[i], t_val, params, b, dt)
                u_traj.append(u_curr)
                a_traj.append(a_star)
            u_traj = jnp.stack(u_traj[::-1])
            a_traj = jnp.stack(a_traj[::-1])

            # Forward pass (FPK)
            new_mu_traj = [mu0_local]
            for i in range(N):
                new_mu_traj.append(forward_step(new_mu_traj[-1], a_traj[i], dt))
            new_mu_traj = jnp.stack(new_mu_traj)

            diff = jnp.max(jnp.abs(new_mu_traj - mu_traj))
            mu_traj = damp * mu_traj + (1.0 - damp) * new_mu_traj
            if diff < tol_val:
                print(f"    Picard converged at iteration {it+1} (diff={float(diff):.2e})")
                break

        return mu_traj

    print("  Running MFG Picard solver for intervention scenario...")
    return custom_picard(mu0, mfg_params, b_val, max_iter, damping, tol)


# ==============================================================================
# Plotting
# ==============================================================================
def plot_intervention(mu_baseline, mu_mfg, scenario, results_dir, hours):
    """
    Produce:
      1. One plot per station
      2. One combined panel figure with all stations
    """
    shock_station = scenario['shock_station']
    shock_start = scenario['shock_start']
    shock_end = scenario['shock_end']

    s_start = 6 + (shock_start / N) * 16
    s_end = 6 + (shock_end / N) * 16

    # --- Individual station plots ---
    for station_idx in range(D):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(hours, mu_baseline[:, station_idx], '-', color='#115E59', linewidth=2.5,
                label='MF Dynamics (Reactive)')
        ax.plot(hours, mu_mfg[:, station_idx], '-', color='#7209B7', linewidth=2.5,
                label='MFG (Anticipatory)')

        ax.axvspan(s_start, s_end, color='red', alpha=0.15, label='Station Closure Window')

        station_label = STATION_NAMES.get(station_idx, f"Station {station_idx}")
        ax.set_xlabel("Hour of Day", fontsize=13)
        ax.set_ylabel(f"Mass at {station_label} (μ)", fontsize=13)
        ax.set_title(f"Intervention: {station_label}", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')

        fig.tight_layout()
        fname = f"intervention_station{station_idx}.pdf"
        fig.savefig(os.path.join(results_dir, fname))
        plt.close(fig)

    # --- Combined panel figure ---
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    closed_name = STATION_NAMES.get(shock_station, f"Station {shock_station}")
    fig.suptitle(f"Counterfactual Intervention: Closure of {closed_name}", fontsize=15, fontweight='bold')

    for station_idx in range(D):
        ax = axes.flat[station_idx]
        ax.plot(hours, mu_baseline[:, station_idx], '-', color='#115E59', linewidth=2,
                label='MF Dynamics (Reactive)')
        ax.plot(hours, mu_mfg[:, station_idx], '-', color='#7209B7', linewidth=2,
                label='MFG (Anticipatory)')
        ax.axvspan(s_start, s_end, color='red', alpha=0.15)

        station_label = STATION_NAMES.get(station_idx, f"Station {station_idx}")
        ax.set_title(station_label, fontsize=11)
        ax.set_xlabel("Hour", fontsize=10)
        ax.set_ylabel("μ", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        if station_idx == 0:
            ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "intervention_all_stations.pdf"))
    plt.close(fig)

    print(f"  Saved {D} individual plots + combined panel to {results_dir}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run intervention (counterfactual) scenarios.")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2],
                        help="Scenario ID: 1 = Station 1 mid-day closure, 2 = Station 2 morning rush")
    parser.add_argument("--test3-dir", type=str, default=None,
                        help="Path to test3 results directory to load pre-trained params. "
                             "If not provided, trains from scratch.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for parameter loading")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    print("=" * 60)
    print(f"Intervention Scenario {args.scenario}: {scenario['name']}")
    print(f"  {scenario['description']}")
    print("=" * 60)

    train_data, test_data, _, metadata = load_data()
    t_grid = jnp.linspace(0, T, N)

    # Timestamp and results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.scenario == 1:
        test_label = "test4_intervention1"
    else:
        test_label = "test5_intervention2"
    results_dir = os.path.join("results", f"run_{test_label}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    config = {
        'scenario': args.scenario,
        'scenario_name': scenario['name'],
        'shock_station': scenario['shock_station'],
        'shock_start': scenario['shock_start'],
        'shock_end': scenario['shock_end'],
        'test3_dir': args.test3_dir,
        'seed': args.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Model setup
    mfg_model = BikeShareMFG(d=D)
    gamma_network = GammaNetwork(d=D, complexity="linear")
    cost_network = CostNetwork(d=D)

    # Load or train parameters
    if args.test3_dir is not None:
        base_params, mfg_params = load_params_from_test3(args.test3_dir, seed=args.seed)
    else:
        base_params, mfg_params = train_quick(train_data, t_grid)

    # Initial condition from first test day
    mu0 = jnp.array(test_data[0, 0, :])

    # --- Run simulations ---
    print("\n--- Baseline (MF Dynamics) ---")
    mu_baseline = simulate_baseline(mu0, base_params, gamma_network, mfg_model, scenario)

    print("\n--- MFG (Anticipatory) ---")
    mu_mfg = simulate_mfg(mu0, mfg_params, gamma_network, cost_network, mfg_model, scenario)

    # --- Save trajectories ---
    np.save(os.path.join(results_dir, "mu_baseline.npy"), np.array(mu_baseline))
    np.save(os.path.join(results_dir, "mu_mfg.npy"), np.array(mu_mfg))

    # --- Plot ---
    hours = np.linspace(metadata['start_hour'], metadata['end_hour'], N + 1)
    plot_intervention(mu_baseline, mu_mfg, scenario, results_dir, hours)

    # Also save to paper figures
    paper_fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "PAPER", "figures", "bikeshare_merged")
    os.makedirs(paper_fig_dir, exist_ok=True)
    for fname in os.listdir(results_dir):
        if fname.endswith(".pdf"):
            import shutil
            src = os.path.join(results_dir, fname)
            dst = os.path.join(paper_fig_dir, f"scenario{args.scenario}_{fname}")
            shutil.copy2(src, dst)
    print(f"  Copied plots to {paper_fig_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

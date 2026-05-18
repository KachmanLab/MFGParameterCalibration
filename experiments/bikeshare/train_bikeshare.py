"""
Training script for Citi Bike experiment (v4).
Key changes from v3:
  1. Added relative L2 error tracking.
  2. Convergence tracking for train AND test every interval.
  3. Plotted convergence of both L2 and relative L2 with train/test separation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
import os
import time
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from mfg_bikeshare import BikeShareMFG, ForwardSolver, BikeSharePicardSolver

# ==========================================
# HYPERPARAMETERS
# ==========================================

RUN_BASELINE = True
RUN_MFG = True

MODEL_COMPLEXITY = "linear"  # Will be overridden by args

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

D = 6
T = 1.0
N = 64
dt = T / (N - 1)

INIT_B = 1.0
INIT_C = 0.5

PICARD_MAX_ITER = 200
PICARD_TOL = 1e-5
PICARD_DAMPING = 0.5

EPOCHS = 400
LEARNING_RATE = 5e-3
BATCH_SIZE = 5
TEST_INTERVAL = 20
NUM_SEEDS = 5

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ==========================================
# NEURAL NETWORK
# ==========================================
class GammaNetwork(nn.Module):
    d: int
    complexity: str = "linear"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, t):
        n_rates = self.d * (self.d - 1)
        if self.complexity == "constant":
            gamma = self.param('gamma_const', nn.initializers.zeros, (n_rates,))
            return gamma
        elif self.complexity == "linear":
            x = t.reshape(-1)
            x = nn.Dense(n_rates)(x)
            return x
        else:
            x = t.reshape(-1)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
            x = nn.Dense(n_rates)(x)
            return x

def load_data():
    train_data = np.load(os.path.join(DATA_DIR, "train_data.npy"))
    test_data = np.load(os.path.join(DATA_DIR, "test_data.npy"))
    ood_path = os.path.join(DATA_DIR, "test_ood_data.npy")
    test_ood_data = np.load(ood_path) if os.path.exists(ood_path) else None
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return jnp.array(train_data), jnp.array(test_data), (jnp.array(test_ood_data) if test_ood_data is not None else None), metadata

def compute_gamma_trajectory(params, network, t_grid):
    def scan_fn(_, t):
        gamma = network.apply(params, t)
        return None, gamma
    _, gamma_traj = jax.lax.scan(scan_fn, None, t_grid)
    return gamma_traj

def loss_baseline(params, network, forward_solver, t_grid, mu_observed):
    mu_0 = mu_observed[0]
    gamma_traj = compute_gamma_trajectory(params, network, t_grid)
    mu_pred = forward_solver.solve_forward(gamma_traj, mu_0)
    return jnp.mean((mu_pred - mu_observed) ** 2)

def loss_mfg(params, cost_params, network, mfg_solver_fn, t_grid, mu_observed):
    mu_0 = mu_observed[0]
    b = jax.nn.softplus(cost_params['log_b'])
    c = jax.nn.softplus(cost_params['log_c'])
    w_g = cost_params['w_g']
    gamma_traj = compute_gamma_trajectory(params, network, t_grid)
    mu_pred = mfg_solver_fn(gamma_traj, mu_0, b, c, w_g)
    return jnp.mean((mu_pred - mu_observed) ** 2)

def train_model(model_name, train_data, test_data, test_ood_data, t_grid, results_dir, seed=0):
    print(f"\n{'='*60}")
    print(f"Training: {model_name} (Seed {seed})")
    print(f"{'='*60}")
    
    is_mfg = (model_name == "MFG")
    mfg_model = BikeShareMFG(d=D)
    network = GammaNetwork(d=D, complexity=MODEL_COMPLEXITY)
    
    key = jax.random.PRNGKey(seed)
    dummy_t = jnp.array([0.0])
    nn_params = network.init(key, dummy_t)
    
    if is_mfg:
        picard_solver = BikeSharePicardSolver(mfg_model, dt, PICARD_MAX_ITER, PICARD_TOL, PICARD_DAMPING)
        solver_fn = picard_solver.get_solver_fn()
        cost_params = {
            'log_b': jnp.log(jnp.exp(jnp.array(INIT_B)) - 1.0),
            'log_c': jnp.log(jnp.exp(jnp.array(INIT_C)) - 1.0),
            'w_g': jnp.zeros(D),
        }
        all_params = {'nn': nn_params, 'cost': cost_params}
        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(all_params)
        
        @jax.jit
        def train_step(all_params, opt_state, mu_obs):
            def loss_fn(p):
                return loss_mfg(p['nn'], p['cost'], network, solver_fn, t_grid, mu_obs)
            loss_val, grads = jax.value_and_grad(loss_fn)(all_params)
            updates, new_opt_state = optimizer.update(grads, opt_state, all_params)
            new_params = optax.apply_updates(all_params, updates)
            return new_params, new_opt_state, loss_val
        
        @jax.jit
        def eval_single(all_params, mu_obs):
            mu_0 = mu_obs[0]
            b = jax.nn.softplus(all_params['cost']['log_b'])
            c = jax.nn.softplus(all_params['cost']['log_c'])
            w_g = all_params['cost']['w_g']
            gamma_traj = compute_gamma_trajectory(all_params['nn'], network, t_grid)
            mu_pred = solver_fn(gamma_traj, mu_0, b, c, w_g)
            l2 = jnp.mean((mu_pred - mu_obs) ** 2)
            rel_l2 = jnp.mean(((mu_pred - mu_obs)**2) / (mu_obs**2 + 1e-4))
            return l2, rel_l2, mu_pred
    else:
        forward_solver = ForwardSolver(mfg_model, dt)
        all_params = nn_params
        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(all_params)
        
        @jax.jit
        def train_step(all_params, opt_state, mu_obs):
            def loss_fn(p):
                return loss_baseline(p, network, forward_solver, t_grid, mu_obs)
            loss_val, grads = jax.value_and_grad(loss_fn)(all_params)
            updates, new_opt_state = optimizer.update(grads, opt_state, all_params)
            new_params = optax.apply_updates(all_params, updates)
            return new_params, new_opt_state, loss_val
        
        @jax.jit
        def eval_single(all_params, mu_obs):
            mu_0 = mu_obs[0]
            gamma_traj = compute_gamma_trajectory(all_params, network, t_grid)
            mu_pred = forward_solver.solve_forward(gamma_traj, mu_0)
            l2 = jnp.mean((mu_pred - mu_obs) ** 2)
            rel_l2 = jnp.mean(((mu_pred - mu_obs)**2) / (mu_obs**2 + 1e-4))
            return l2, rel_l2, mu_pred
    
    n_train = train_data.shape[0]
    
    epoch_history = []
    train_history_l2 = []
    train_history_rel = []
    test_history_l2 = []
    test_history_rel = []
    
    key = jax.random.PRNGKey(seed + 100)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)
        
        for i in range(n_train):
            idx = perm[i]
            all_params, opt_state, loss_val = train_step(all_params, opt_state, train_data[idx])
        
        if (epoch + 1) % TEST_INTERVAL == 0 or epoch == 0:
            tr_eval = [eval_single(all_params, train_data[i]) for i in range(n_train)]
            te_eval = [eval_single(all_params, test_data[i]) for i in range(test_data.shape[0])]
            
            t_l2 = np.mean([float(x[0]) for x in tr_eval])
            t_rel = np.mean([float(x[1]) for x in tr_eval])
            te_l2 = np.mean([float(x[0]) for x in te_eval])
            te_rel = np.mean([float(x[1]) for x in te_eval])
            
            epoch_history.append(epoch + 1)
            train_history_l2.append(t_l2)
            train_history_rel.append(t_rel)
            test_history_l2.append(te_l2)
            test_history_rel.append(te_rel)
            
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | Train L2: {t_l2:.6f} | Test L2: {te_l2:.6f} | {elapsed:.1f}s")
            
    model_dir = os.path.join(results_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # Save trained parameters for later reuse
    import pickle
    with open(os.path.join(model_dir, f"params_seed{seed}.pkl"), "wb") as f:
        pickle.dump(jax.tree.map(np.array, all_params), f)
    
    np.save(os.path.join(model_dir, f"history_epochs_seed{seed}.npy"), np.array(epoch_history))
    np.save(os.path.join(model_dir, f"history_train_l2_seed{seed}.npy"), np.array(train_history_l2))
    np.save(os.path.join(model_dir, f"history_train_rel_seed{seed}.npy"), np.array(train_history_rel))
    np.save(os.path.join(model_dir, f"history_test_l2_seed{seed}.npy"), np.array(test_history_l2))
    np.save(os.path.join(model_dir, f"history_test_rel_seed{seed}.npy"), np.array(test_history_rel))
    
    for ds_name, ds_data in [("train", train_data), ("test_id", test_data), ("test_ood", test_ood_data)]:
        if ds_data is None: continue
        predictions, l2s, rel_l2s = [], [], []
        for i in range(ds_data.shape[0]):
            l2, rel_l2, pred = eval_single(all_params, ds_data[i])
            predictions.append(np.array(pred))
            l2s.append(float(l2))
            rel_l2s.append(float(rel_l2))
        np.save(os.path.join(model_dir, f"{ds_name}_predictions_seed{seed}.npy"), np.array(predictions))
        np.save(os.path.join(model_dir, f"{ds_name}_l2s_seed{seed}.npy"), np.array(l2s))
        np.save(os.path.join(model_dir, f"{ds_name}_rel_l2s_seed{seed}.npy"), np.array(rel_l2s))
    
    if is_mfg:
        b_final = float(jax.nn.softplus(all_params['cost']['log_b']))
        c_final = float(jax.nn.softplus(all_params['cost']['log_c']))
        w_g_final = [float(x) for x in all_params['cost']['w_g']]
        with open(os.path.join(model_dir, f"cost_params_seed{seed}.json"), "w") as f:
            json.dump({'b': b_final, 'c': c_final, 'w_g': w_g_final}, f, indent=2)

def plot_results(results_dir, metadata):
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    models = {}
    for name in ["baseline", "mfg"]:
        model_dir = os.path.join(results_dir, name)
        if not os.path.exists(model_dir): continue
        
        m = {'hist_epochs': [], 'hist_tr_l2': [], 'hist_tr_rel': [], 'hist_te_l2': [], 'hist_te_rel': [],
             'train_l2s': [], 'test_id_l2s': [], 'test_ood_l2s': [],
             'train_rel_l2s': [], 'test_id_rel_l2s': [], 'test_ood_rel_l2s': []}
        
        pred_seed = 0
        for ds in ["train", "test_id", "test_ood"]:
            pred_path = os.path.join(model_dir, f"{ds}_predictions_seed{pred_seed}.npy")
            if os.path.exists(pred_path):
                m[f'{ds}_predictions'] = np.load(pred_path)
        
        for seed in range(NUM_SEEDS):
            for k, file_suffix in [('hist_epochs', 'epochs'), ('hist_tr_l2', 'train_l2'), 
                                   ('hist_tr_rel', 'train_rel'), ('hist_te_l2', 'test_l2'), 
                                   ('hist_te_rel', 'test_rel')]:
                path = os.path.join(model_dir, f"history_{file_suffix}_seed{seed}.npy")
                if os.path.exists(path):
                    m[k].append(np.load(path))
            
            for ds in ["train", "test_id", "test_ood"]:
                l2_path = os.path.join(model_dir, f"{ds}_l2s_seed{seed}.npy")
                if os.path.exists(l2_path):
                    m[f'{ds}_l2s'].append(np.mean(np.load(l2_path)))
                rel_path = os.path.join(model_dir, f"{ds}_rel_l2s_seed{seed}.npy")
                if os.path.exists(rel_path):
                    m[f'{ds}_rel_l2s'].append(np.mean(np.load(rel_path)))
                    
        for k in m.keys():
            if isinstance(m[k], list) and len(m[k]) > 0:
                m[k] = np.array(m[k])
        models[name] = m
    
    # 1. Convergence Plots
    for metric_key, ylabel, title_suffix in [('l2', 'L2 Error', 'Absolute L2'), ('rel', 'Relative L2 Error', 'Relative L2')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, data in models.items():
            if len(data[f'hist_tr_{metric_key}']) == 0: continue
            label_base = "MF Dynamics" if name == "baseline" else "MFG"
            
            # Darker colors
            color = '#115E59' if name == "baseline" else '#991B1B'
            
            epochs = data['hist_epochs'][0]
            
            # Train curve (solid)
            tr_mean = np.mean(data[f'hist_tr_{metric_key}'], axis=0)
            tr_sem = np.std(data[f'hist_tr_{metric_key}'], axis=0) / np.sqrt(NUM_SEEDS)
            ax.semilogy(epochs, tr_mean, '-', color=color, label=f"{label_base} (Train)", linewidth=2)
            ax.fill_between(epochs, tr_mean - tr_sem, tr_mean + tr_sem, color=color, alpha=0.4)
            
            # Test curve (dashed)
            te_mean = np.mean(data[f'hist_te_{metric_key}'], axis=0)
            te_sem = np.std(data[f'hist_te_{metric_key}'], axis=0) / np.sqrt(NUM_SEEDS)
            ax.semilogy(epochs, te_mean, '--', color=color, label=f"{label_base} (Test)", linewidth=2)
            ax.fill_between(epochs, te_mean - te_sem, te_mean + te_sem, color=color, alpha=0.25)
            
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Convergence of {title_suffix} (Mean ± SEM, N={NUM_SEEDS})", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"convergence_{metric_key}.pdf"))
        plt.close(fig)
        
    # 2. Test comparison bar charts (L2 and Rel)
    for metric_key, ylabel in [('l2s', 'L2 Error'), ('rel_l2s', 'Relative L2 Error')]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax_idx, ds in enumerate(["test_id", "test_ood"]):
            ax = axes[ax_idx]
            names, means, sems = [], [], []
            for name, data in models.items():
                key = f'{ds}_{metric_key}'
                if key not in data or len(data[key]) == 0: continue
                names.append("MF Dynamics" if name == "baseline" else "MFG")
                means.append(np.mean(data[key]))
                sems.append(np.std(data[key]) / np.sqrt(NUM_SEEDS))
            if names:
                colors = ['#115E59', '#991B1B'][:len(names)]
                ax.bar(names, means, yerr=sems, capsize=5, color=colors, alpha=0.85)
                ax.set_ylabel(ylabel)
                title = "In-Distribution Test" if ds == "test_id" else "Out-of-Distribution Test"
                ax.set_title(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"test_comparison_{metric_key}.pdf"))
        plt.close(fig)
    
    # 3. Trajectories
    d = metadata['d']
    hours = np.linspace(metadata['start_hour'], metadata['end_hour'], metadata['N'])
    for ds_name, ds_label in [("test_id", "Test ID"), ("test_ood", "Test OOD")]:
        ds_file = "test_data.npy" if ds_name == "test_id" else "test_ood_data.npy"
        ds_path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(ds_path): continue
        ds_data = np.load(ds_path)
        
        for day_idx in range(min(2, ds_data.shape[0])):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f"{ds_label} Day {day_idx + 1} (Seed 0): Predicted vs Observed μ(t)", fontsize=14)
            for state_idx in range(d):
                ax = axes.flat[state_idx]
                ax.plot(hours, ds_data[day_idx, :, state_idx], 'k--', linewidth=2, label='Observed', alpha=0.8)
                for name, data in models.items():
                    key = f'{ds_name}_predictions'
                    if key not in data: continue
                    label = "MF Dynamics" if name == "baseline" else "MFG"
                    color = '#4ECDC4' if name == "baseline" else '#FF6B6B'
                    ax.plot(hours, data[key][day_idx, :, state_idx], color=color, linewidth=1.5, label=label, alpha=0.8)
                state_label = f"Station {state_idx}" if state_idx < d - 1 else "External"
                ax.set_title(state_label, fontsize=10)
                ax.set_xlabel("Hour")
                ax.set_ylabel("μ")
                if state_idx == 0: ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"trajectories_{ds_name}_day{day_idx+1}.pdf"))
            plt.close(fig)
            
    # Print summary
    print("\n" + "="*90)
    print(f"{'':20s} {'MF Dynamics':>25s} {'MFG':>25s} {'Improvement':>12s}")
    print("="*90)
    for ds in ["train", "test_id", "test_ood"]:
        for m_key, m_label in [('l2s', 'L2 Error'), ('rel_l2s', 'Rel. L2')]:
            key = f'{ds}_{m_key}'
            if key in models['baseline'] and key in models['mfg']:
                bl_mean = np.mean(models['baseline'][key])
                bl_sem = np.std(models['baseline'][key]) / np.sqrt(NUM_SEEDS)
                mg_mean = np.mean(models['mfg'][key])
                mg_sem = np.std(models['mfg'][key]) / np.sqrt(NUM_SEEDS)
                imp = (bl_mean - mg_mean) / bl_mean * 100
                label = f"{ds} ({m_label})"
                print(f"{label:20s} {bl_mean:10.5f} ± {bl_sem:8.5f} {mg_mean:10.5f} ± {mg_sem:8.5f} {imp:11.2f}%")
    print("="*90)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=str, default="linear")
    args = parser.parse_args()
    
    global MODEL_COMPLEXITY
    MODEL_COMPLEXITY = args.complexity

    print("=" * 60)
    print(f"Citi Bike: complexity={MODEL_COMPLEXITY}")
    print(f"Seeds: {NUM_SEEDS}, Epochs: {EPOCHS}")
    print("=" * 60)
    
    train_data, test_data, test_ood_data, metadata = load_data()
    t_grid = jnp.linspace(0, T, N)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_label = "test1_constant" if MODEL_COMPLEXITY == "constant" else "test2_linear"
    results_dir = os.path.join(RESULTS_DIR, f"run_{test_label}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    config = {
        'D': D, 'T': T, 'N': N, 'dt': float(dt),
        'MODEL_COMPLEXITY': MODEL_COMPLEXITY,
        'EPOCHS': EPOCHS, 'LEARNING_RATE': LEARNING_RATE,
        'NUM_SEEDS': NUM_SEEDS,
        'INIT_B': INIT_B, 'INIT_C': INIT_C,
        'PICARD_DAMPING': PICARD_DAMPING,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if RUN_BASELINE:
        for seed in range(NUM_SEEDS):
            train_model("baseline", train_data, test_data, test_ood_data, t_grid, results_dir, seed=seed)
    
    if RUN_MFG:
        for seed in range(NUM_SEEDS):
            train_model("MFG", train_data, test_data, test_ood_data, t_grid, results_dir, seed=seed)
    
    plot_results(results_dir, metadata)

if __name__ == "__main__":
    main()

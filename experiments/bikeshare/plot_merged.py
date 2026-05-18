import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

RESULTS_DIR = "results"
NUM_SEEDS = 5
DATA_DIR = "data/processed"

def get_latest_run(prefix):
    dirs = glob.glob(os.path.join(RESULTS_DIR, f"run_{prefix}_*"))
    if not dirs: return None
    return sorted(dirs)[-1]  # Get the most recent one

def load_run_data(run_dir, model_name):
    model_dir = os.path.join(run_dir, model_name)
    if not os.path.exists(model_dir): return None
    
    m = {'hist_epochs': [], 'hist_tr_l2': [], 'hist_tr_rel': [], 'hist_te_l2': [], 'hist_te_rel': [],
         'train_l2s': [], 'test_id_l2s': [], 'test_ood_l2s': [],
         'train_rel_l2s': [], 'test_id_rel_l2s': [], 'test_ood_rel_l2s': []}
    
    for ds in ["train", "test_id", "test_ood"]:
        pred_path = os.path.join(model_dir, f"{ds}_predictions_seed0.npy")
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
            if os.path.exists(l2_path): m[f'{ds}_l2s'].append(np.mean(np.load(l2_path)))
            rel_path = os.path.join(model_dir, f"{ds}_rel_l2s_seed{seed}.npy")
            if os.path.exists(rel_path): m[f'{ds}_rel_l2s'].append(np.mean(np.load(rel_path)))
                
    for k in m.keys():
        if isinstance(m[k], list) and len(m[k]) > 0:
            m[k] = np.array(m[k])
    return m

def plot_group(fig_dir, group_name, models_dict, metadata, title_prefix=""):
    os.makedirs(fig_dir, exist_ok=True)
    d = metadata['d']
    hours = np.linspace(metadata['start_hour'], metadata['end_hour'], metadata['N'])
    
    colors = ['#115E59', '#E76F51', '#7209B7', '#43AA8B'] # Baseline, MFG1, MFG2
    
    # 1. Convergence
    for metric_key, ylabel, title_suffix in [('l2', 'L2 Error', 'Absolute L2'), ('rel', 'Relative L2 Error', 'Relative L2')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        color_idx = 0
        for label_name, data in models_dict.items():
            if len(data[f'hist_tr_{metric_key}']) == 0: continue
            color = colors[color_idx]
            epochs = data['hist_epochs'][0]
            
            tr_mean = np.mean(data[f'hist_tr_{metric_key}'], axis=0)
            tr_sd = np.std(data[f'hist_tr_{metric_key}'], axis=0)
            ax.semilogy(epochs, tr_mean, '-', color=color, label=f"{label_name} (Train)", linewidth=3)
            ax.fill_between(epochs, tr_mean - tr_sd, tr_mean + tr_sd, color=color, alpha=0.3)
            
            te_mean = np.mean(data[f'hist_te_{metric_key}'], axis=0)
            te_sd = np.std(data[f'hist_te_{metric_key}'], axis=0)
            ax.semilogy(epochs, te_mean, '--', color=color, label=f"{label_name} (Test)", linewidth=3)
            ax.fill_between(epochs, te_mean - te_sd, te_mean + te_sd, color=color, alpha=0.15)
            color_idx += 1
            
        ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f"{title_prefix}Convergence of {title_suffix} (Mean ± SD, N={NUM_SEEDS})", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"convergence_{metric_key}_{group_name}.pdf"))
        plt.close(fig)

    # 2. Trajectories
    for ds_name, ds_label in [("test_id", "Test ID"), ("test_ood", "Test OOD")]:
        ds_file = "test_data.npy" if ds_name == "test_id" else "test_ood_data.npy"
        ds_path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(ds_path): continue
        ds_data = np.load(ds_path)
        
        for day_idx in range(min(2, ds_data.shape[0])):
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f"{title_prefix}{ds_label} Day {day_idx + 1} (Seed 0): Predicted vs Observed μ(t)", fontsize=18, fontweight='bold')
            for state_idx in range(d):
                ax = axes.flat[state_idx]
                ax.plot(hours, ds_data[day_idx, :, state_idx], 'k--', linewidth=3, label='Observed', alpha=0.7)
                color_idx = 0
                for label_name, data in models_dict.items():
                    key = f'{ds_name}_predictions'
                    if key not in data: continue
                    color = colors[color_idx]
                    ax.plot(hours, data[key][day_idx, :, state_idx], color=color, linewidth=3, label=label_name, alpha=0.85)
                    color_idx += 1
                state_label = f"Station {state_idx}" if state_idx < d - 1 else "External"
                ax.set_title(state_label, fontsize=14, fontweight='bold')
                ax.set_xlabel("Hour of Day", fontsize=12)
                ax.set_ylabel("Mass (μ)", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                if state_idx == 0: ax.legend(fontsize=12)
                ax.grid(True, alpha=0.4, linestyle='--')
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            fig.savefig(os.path.join(fig_dir, f"trajectories_{ds_name}_day{day_idx+1}_{group_name}.pdf"))
            plt.close(fig)

    # 3. Error over time
    for ds_name, ds_label in [("train", "Train"), ("test_id", "Test ID"), ("test_ood", "Test OOD")]:
        ds_file = "train_data.npy" if ds_name == "train" else ("test_data.npy" if ds_name == "test_id" else "test_ood_data.npy")
        ds_path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(ds_path): continue
        ds_data = np.load(ds_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        color_idx = 0
        for label_name, data in models_dict.items():
            key = f'{ds_name}_predictions'
            if key not in data: continue
            preds = data[key]
            mse_over_time = np.mean((preds - ds_data)**2, axis=(0, 2))
            color = colors[color_idx]
            ax.plot(hours, mse_over_time, '-', color=color, linewidth=3, label=label_name)
            color_idx += 1
            
        ax.set_xlabel("Hour of Day", fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
        ax.set_title(f"{title_prefix}{ds_label}: Prediction Error Over Time", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"error_over_time_{ds_name}_{group_name}.pdf"))
        plt.close(fig)

def plot_bar_charts(fig_dir, models_all):
    # models_all is a dict of group_name -> models_dict
    os.makedirs(fig_dir, exist_ok=True)
    
    # Separate ID and OOD into different plots
    for ds_name, ds_label in [("test_id", "In-Distribution (Weekdays)"), ("test_ood", "Out-of-Distribution (Weekends)")]:
        for metric_key, ylabel in [('l2s', 'L2 Error'), ('rel_l2s', 'Relative L2 Error')]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{ds_label} - {ylabel}", fontsize=16, fontweight='bold')
            
            # Left Plot: Test 1 (Constant)
            ax = axes[0]
            names, means, sds = [], [], []
            if 'test1' in models_all:
                for name, data in models_all['test1'].items():
                    key = f'{ds_name}_{metric_key}'
                    if key in data and len(data[key]) > 0:
                        names.append(name)
                        means.append(np.mean(data[key]))
                        sds.append(np.std(data[key]))
            if names:
                colors = ['#115E59', '#E76F51'][:len(names)]
                ax.bar(names, means, yerr=sds, capsize=6, color=colors, alpha=0.9, width=0.6, error_kw={'linewidth': 2})
                ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
                ax.set_title("Test 1: Constant Base Rates", fontsize=14, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, axis='y', alpha=0.4, linestyle='--')
                
            # Right Plot: Test 2/3 (Linear)
            ax = axes[1]
            names, means, sds = [], [], []
            if 'test23' in models_all:
                for name, data in models_all['test23'].items():
                    key = f'{ds_name}_{metric_key}'
                    if key in data and len(data[key]) > 0:
                        names.append(name)
                        means.append(np.mean(data[key]))
                        sds.append(np.std(data[key]))
            if names:
                colors = ['#115E59', '#E76F51', '#7209B7'][:len(names)]
                ax.bar(names, means, yerr=sds, capsize=6, color=colors, alpha=0.9, width=0.6, error_kw={'linewidth': 2})
                ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
                ax.set_title("Test 2 & 3: Linear Base Rates", fontsize=14, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.grid(True, axis='y', alpha=0.4, linestyle='--')
                
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            fig.savefig(os.path.join(fig_dir, f"test_comparison_{metric_key}_{ds_name}.pdf"))
            plt.close(fig)

from datetime import datetime

def main():
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = os.path.join(RESULTS_DIR, f"merged_plots_{timestamp}")
    os.makedirs(fig_dir, exist_ok=True)
    
    # We look for the prefixes defined by the new run_all.py test structure
    run_test1 = get_latest_run("test1_constant")
    if not run_test1: run_test1 = get_latest_run("constant") # Fallback to old name
    
    run_test2 = get_latest_run("test2_linear")
    if not run_test2: run_test2 = get_latest_run("linear")
    
    run_test3 = get_latest_run("test3_nncost")
    if not run_test3: run_test3 = get_latest_run("test3")
    
    models_all = {}
    
    # Group 1: Test 1
    if run_test1:
        print(f"Loading Test 1 from {run_test1}...")
        m1 = load_run_data(run_test1, "baseline")
        m2 = load_run_data(run_test1, "mfg")
        models_test1 = {"MF Dynamics": m1, "MFG": m2}
        plot_group(fig_dir, "test1", models_test1, metadata, title_prefix="Test 1 (Constant): ")
        models_all['test1'] = models_test1
        
    # Group 2: Test 2 & 3 merged
    if run_test2 and run_test3:
        print(f"Loading Test 2 from {run_test2} and Test 3 from {run_test3}...")
        m_base = load_run_data(run_test2, "baseline") # They share the same baseline
        m_mfg2 = load_run_data(run_test2, "mfg")
        m_mfg3 = load_run_data(run_test3, "mfg")
        
        models_test23 = {
            "MF Dynamics": m_base,
            "MFG (Scalar Cost)": m_mfg2,
            "MFG (General Cost)": m_mfg3
        }
        plot_group(fig_dir, "test23", models_test23, metadata, title_prefix="Test 2 & 3 (Linear): ")
        models_all['test23'] = models_test23
        
    print("Generating separated test comparison bar charts...")
    plot_bar_charts(fig_dir, models_all)
    print("Done! Plots saved to", fig_dir)

if __name__ == "__main__":
    main()

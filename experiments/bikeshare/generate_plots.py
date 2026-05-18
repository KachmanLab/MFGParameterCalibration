#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

NUM_SEEDS = 5

def find_data_dir():
    # Try a few common relative paths assuming script is in a results folder
    candidates = [
        "../../data/processed",
        "../data/processed",
        "../../../data/processed",
        "data/processed"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "metadata.json")):
            return c
    return None

def main():
    results_dir = "."
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    data_dir = find_data_dir()
    if data_dir is None:
        print("Warning: Could not find data/processed directory. Trajectory and Error-over-time plots will be skipped.")
    else:
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
    models = {}
    for name in ["baseline", "mfg"]:
        model_dir = os.path.join(results_dir, name)
        if not os.path.exists(model_dir): 
            continue
        
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
        
    if not models:
        print("No model data found. Make sure you run this script inside a results directory containing 'baseline' and/or 'mfg' folders.")
        return

    print("Generating convergence plots...")
    for metric_key, ylabel, title_suffix in [('l2', 'L2 Error', 'Absolute L2'), ('rel', 'Relative L2 Error', 'Relative L2')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, data in models.items():
            if len(data[f'hist_tr_{metric_key}']) == 0: continue
            label_base = "MF Dynamics" if name == "baseline" else "MFG"
            color = '#2A9D8F' if name == "baseline" else '#E76F51'
            epochs = data['hist_epochs'][0]
            
            tr_mean = np.mean(data[f'hist_tr_{metric_key}'], axis=0)
            tr_sd = np.std(data[f'hist_tr_{metric_key}'], axis=0)
            ax.semilogy(epochs, tr_mean, '-', color=color, label=f"{label_base} (Train)", linewidth=3)
            ax.fill_between(epochs, tr_mean - tr_sd, tr_mean + tr_sd, color=color, alpha=0.3)
            
            te_mean = np.mean(data[f'hist_te_{metric_key}'], axis=0)
            te_sd = np.std(data[f'hist_te_{metric_key}'], axis=0)
            ax.semilogy(epochs, te_mean, '--', color=color, label=f"{label_base} (Test)", linewidth=3)
            ax.fill_between(epochs, te_mean - te_sd, te_mean + te_sd, color=color, alpha=0.2)
            
        ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f"Convergence of {title_suffix} (Mean ± SD, N={NUM_SEEDS})", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"convergence_{metric_key}.pdf"))
        plt.close(fig)
        
    print("Generating bar charts...")
    for metric_key, ylabel in [('l2s', 'L2 Error'), ('rel_l2s', 'Relative L2 Error')]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax_idx, ds in enumerate(["test_id", "test_ood"]):
            ax = axes[ax_idx]
            names, means, sds = [], [], []
            for name, data in models.items():
                key = f'{ds}_{metric_key}'
                if key not in data or len(data[key]) == 0: continue
                names.append("MF Dynamics" if name == "baseline" else "MFG")
                means.append(np.mean(data[key]))
                sds.append(np.std(data[key]))
            if names:
                colors = ['#2A9D8F', '#E76F51'][:len(names)]
                bars = ax.bar(names, means, yerr=sds, capsize=6, color=colors, alpha=0.9, width=0.6, error_kw={'linewidth': 2})
                ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
                title = "In-Distribution Test" if ds == "test_id" else "Out-of-Distribution Test"
                ax.set_title(title, fontsize=16, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, axis='y', alpha=0.4, linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"test_comparison_{metric_key}.pdf"))
        plt.close(fig)

    if data_dir is not None:
        print("Generating trajectory and error-over-time plots...")
        d = metadata['d']
        hours = np.linspace(metadata['start_hour'], metadata['end_hour'], metadata['N'])
        
        # Trajectories
        for ds_name, ds_label in [("test_id", "Test ID"), ("test_ood", "Test OOD")]:
            ds_file = "test_data.npy" if ds_name == "test_id" else "test_ood_data.npy"
            ds_path = os.path.join(data_dir, ds_file)
            if not os.path.exists(ds_path): continue
            ds_data = np.load(ds_path)
            
            for day_idx in range(min(2, ds_data.shape[0])):
                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                fig.suptitle(f"{ds_label} Day {day_idx + 1} (Seed 0): Predicted vs Observed μ(t)", fontsize=18, fontweight='bold')
                for state_idx in range(d):
                    ax = axes.flat[state_idx]
                    ax.plot(hours, ds_data[day_idx, :, state_idx], 'k--', linewidth=3, label='Observed', alpha=0.7)
                    for name, data in models.items():
                        key = f'{ds_name}_predictions'
                        if key not in data: continue
                        label = "MF Dynamics" if name == "baseline" else "MFG"
                        color = '#2A9D8F' if name == "baseline" else '#E76F51'
                        ax.plot(hours, data[key][day_idx, :, state_idx], color=color, linewidth=3, label=label, alpha=0.85)
                    state_label = f"Station {state_idx}" if state_idx < d - 1 else "External"
                    ax.set_title(state_label, fontsize=14, fontweight='bold')
                    ax.set_xlabel("Hour of Day", fontsize=12)
                    ax.set_ylabel("Mass (μ)", fontsize=12)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    if state_idx == 0: ax.legend(fontsize=12)
                    ax.grid(True, alpha=0.4, linestyle='--')
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                fig.savefig(os.path.join(fig_dir, f"trajectories_{ds_name}_day{day_idx+1}.pdf"))
                plt.close(fig)

        # Error over time
        for ds_name, ds_label in [("train", "Train"), ("test_id", "Test ID"), ("test_ood", "Test OOD")]:
            ds_file = "train_data.npy" if ds_name == "train" else ("test_data.npy" if ds_name == "test_id" else "test_ood_data.npy")
            ds_path = os.path.join(data_dir, ds_file)
            if not os.path.exists(ds_path): continue
            ds_data = np.load(ds_path)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, data in models.items():
                key = f'{ds_name}_predictions'
                if key not in data: continue
                preds = data[key]
                mse_over_time = np.mean((preds - ds_data)**2, axis=(0, 2))
                
                label = "MF Dynamics" if name == "baseline" else "MFG"
                color = '#2A9D8F' if name == "baseline" else '#E76F51'
                ax.plot(hours, mse_over_time, '-', color=color, linewidth=3, label=label)
                
            ax.set_xlabel("Hour of Day", fontsize=14, fontweight='bold')
            ax.set_ylabel("Mean Squared Error", fontsize=14, fontweight='bold')
            ax.set_title(f"{ds_label}: Prediction Error Over Time", fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.4, linestyle='--')
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"error_over_time_{ds_name}.pdf"))
            plt.close(fig)

    print(f"All plots successfully generated in ./{fig_dir}/")

if __name__ == "__main__":
    main()

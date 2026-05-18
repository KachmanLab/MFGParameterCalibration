import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from collections import defaultdict
import re


def find_latest_results_dir():
    dirs = [d for d in glob.glob("results-*") if os.path.isdir(d) and d != "results-noise-sweep"]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def extract_dim_level(folder_name):
    match = re.search(r"dim-(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return None


def main():
    target_noise = 0.0
    base_dir = None

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        target_noise = float(sys.argv[2])

    if not base_dir:
        base_dir = find_latest_results_dir()

    if not base_dir or not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return

    print(f"Analyzing dimensional sweep in directory: {base_dir} for noise level: {target_noise}")

    # Data structure: test_case -> lists of metrics
    metrics_by_case = defaultdict(lambda: {"d": [], "mu_m": [], "mu_s": [], "g_m": [], "g_s": []})

    # Expected noise folder name
    target_noise_str = f"noise-{target_noise:.1e}" if target_noise > 0 else "noise-0.0"

    for dim_folder in os.listdir(base_dir):
        if not dim_folder.startswith("dim-"):
            continue
        dim_path = os.path.join(base_dir, dim_folder)

        noise_path = os.path.join(dim_path, target_noise_str)
        if not os.path.isdir(noise_path):
            print(f"Warning: Noise folder {target_noise_str} not found in {dim_folder}")
            continue

        for testcase in os.listdir(noise_path):
            case_path = os.path.join(noise_path, testcase)
            if not os.path.isdir(case_path):
                continue

            data_path = os.path.join(case_path, "data.npz")
            if not os.path.exists(data_path):
                continue

            data = np.load(data_path)
            d_val = int(data["d"]) if "d" in data else extract_dim_level(dim_folder)

            if d_val is None:
                continue

            mc = metrics_by_case[testcase]
            mc["d"].append(d_val)
            mc["mu_m"].append(data["te_m"][-1])
            mc["mu_s"].append(data["te_s"][-1])
            mc["g_m"].append(data["g_m"][-1])
            mc["g_s"].append(data["g_s"][-1])

    if not metrics_by_case:
        print("No valid data found to plot.")
        return

    # Generate one plot per test case
    for testcase, mc in metrics_by_case.items():
        sorted_indices = np.argsort(mc["d"])
        dimensions = np.array(mc["d"])[sorted_indices]
        mu_means = np.array(mc["mu_m"])[sorted_indices]
        mu_sems = np.array(mc["mu_s"])[sorted_indices]
        gamma_means = np.array(mc["g_m"])[sorted_indices]
        gamma_sems = np.array(mc["g_s"])[sorted_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Mu Error
        ax1.errorbar(dimensions, mu_means, yerr=mu_sems, fmt="o-", capsize=5, color="C0", label="Mu Test Error")
        ax1.set_yscale("log")
        ax1.set_xticks(dimensions)
        ax1.set_xlabel("State Space Dimension ($d$)")
        ax1.set_ylabel(r"Final $L_2$ Error ($\mu$)")
        ax1.set_title(f"Scaling with Dimension: Mean Field ({testcase})")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.legend()

        # Subplot 2: Gamma Error
        ax2.errorbar(dimensions, gamma_means, yerr=gamma_sems, fmt="s-", capsize=5, color="C1", label="Gamma Error")
        ax2.set_yscale("log")
        ax2.set_xticks(dimensions)
        ax2.set_xlabel("State Space Dimension ($d$)")
        ax2.set_ylabel(r"Final $L_2$ Error ($\gamma$)")
        ax2.set_title(f"Scaling with Dimension: Parameter ({testcase})")
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(base_dir, f"robustness_dimension_{testcase}_noise{target_noise}.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"Dimension robustness plot for '{testcase}' saved to {plot_path}")


if __name__ == "__main__":
    main()

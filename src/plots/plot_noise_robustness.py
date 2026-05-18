import os
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import glob
from collections import defaultdict

plt.rc("axes", labelsize=14)
plt.rc("legend", fontsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)


def extract_noise_level(folder_name):
    """Extracts the numerical noise level from folder names like 'results-noise1e-3'."""
    match = re.search(r"noise([\d\.e-]+)", folder_name)
    if match:
        return float(match.group(1))
    return None


def find_latest_results_dir():
    dirs = [d for d in glob.glob("results-*") if os.path.isdir(d) and d != "results-noise-sweep"]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = find_latest_results_dir()

    if not base_dir or not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return

    print(f"Analyzing sweep in directory: {base_dir}")

    for dim_folder in os.listdir(base_dir):
        if not dim_folder.startswith("dim-"):
            continue
        dim_path = os.path.join(base_dir, dim_folder)

        # Data structure: (testcase, subcase) -> lists of metrics
        metrics_by_case = defaultdict(lambda: {"noise": [], "mu_m": [], "mu_s": [], "g_m": [], "g_s": []})

        for noise_folder in os.listdir(dim_path):
            if not noise_folder.startswith("noise-"):
                continue

            noise_path = os.path.join(dim_path, noise_folder)
            if not os.path.isdir(noise_path):
                continue

            for testcase in os.listdir(noise_path):
                case_path = os.path.join(noise_path, testcase)
                if not os.path.isdir(case_path):
                    continue

                # for subcase in os.listdir(case_path):
                #     subcase_path = os.path.join(case_path, subcase)
                #     if not os.path.isdir(subcase_path):
                #         continue
                subcase = testcase
                subcase_path = case_path
                data_path = os.path.join(subcase_path, "data.npz")
                if not os.path.exists(data_path):
                    print(f"Warning: No data.npz found in {subcase_path}")
                    continue

                data = np.load(data_path)
                noise = float(data["noise_level"]) if "noise_level" in data else extract_noise_level(noise_folder)

                if noise is None:
                    print(f"Skipping {subcase_path}: Could not determine noise level.")
                    continue

                key = (testcase, subcase)
                mc = metrics_by_case[key]
                mc["noise"].append(np.abs(noise))

                print(data["g_hist"].shape)

                mu_pred = data["mu_p"]
                mu_true = data["mu_t"]
                mu_l2 = np.mean((mu_pred - mu_true) ** 2, axis=0)

                gamma_true = data["g_t"]
                gamma_pred = data["g_p"]
                gamma_l2 = np.mean((gamma_true - gamma_pred) ** 2, axis=0)

                # print(noise, np.mean(mu_l2), np.std(mu_l2), np.mean(gamma_l2), np.std(gamma_l2))

                mc["mu_m"].append(np.mean(mu_l2))
                mc["mu_s"].append(np.std(mu_l2))
                mc["g_m"].append(np.mean(gamma_l2))
                mc["g_s"].append(np.std(gamma_l2))

                # if subcase == "mf_dependent":
                # print(subcase, noise, np.std(gamma_l2))

        if not metrics_by_case:
            print(f"No valid data found in {dim_folder} to plot.")
            continue

        for (testcase, subcase), mc in metrics_by_case.items():
            sorted_indices = np.argsort(mc["noise"])
            noise_levels = np.array(mc["noise"])[sorted_indices]
            mu_means = np.array(mc["mu_m"])[sorted_indices]
            mu_sems = np.array(mc["mu_s"])[sorted_indices]
            gamma_means = np.array(mc["g_m"])[sorted_indices]
            gamma_sems = np.array(mc["g_s"])[sorted_indices]

            if subcase == "mf_dependent":
                print(gamma_sems)

            fig, ax1 = plt.subplots(1, 1, figsize=(4, 3))

            mu_yerr_lower = np.minimum(mu_sems, mu_means * 0.9999)
            ax1.errorbar(
                noise_levels,
                mu_means,
                markersize=3,
                yerr=[mu_yerr_lower, mu_sems],
                fmt="o-",
                capsize=5,
                color="C0",
                label=r"$L_2(\mu_t, \mu_t^\theta)$",
            )

            gamma_yerr_lower = np.minimum(gamma_sems, gamma_means * 0.9999)
            ax1.errorbar(
                noise_levels,
                gamma_means,
                markersize=3,
                yerr=[gamma_yerr_lower, gamma_sems],
                fmt="s-",
                capsize=5,
                color="C1",
                label=r"$L_2(\gamma_t, \gamma_t^\theta)$",
            )

            if any(n > 0 for n in noise_levels):
                ax1.set_xscale("symlog", linthresh=1e-5)
            ax1.set_yscale("log")
            ax1.set_xlabel("Noise Level (Dirichlet)")
            ax1.set_ylabel(r"$L_2$")
            # ax1.set_title(f"Robustness of Mean Field Recovery\n({testcase}/{subcase}, {dim_folder})")
            ax1.grid(True, which="both", ls="-", alpha=0.2)
            ax1.margins(y=0.15)
            ax1.legend()
            plt.tight_layout()
            plot_path = os.path.join(base_dir, f"robustness_noise_mu_{testcase}_{subcase}_{dim_folder}.pdf")
            plt.savefig(plot_path)
            plt.close()

            fig, ax2 = plt.subplots(1, 1, figsize=(4, 3))
            ax2.errorbar(
                noise_levels, gamma_means, yerr=gamma_sems, fmt="s-", capsize=5, color="C1", label="Gamma Error"
            )
            if any(n > 0 for n in noise_levels):
                ax2.set_xscale("symlog", linthresh=1e-5)
            ax2.set_yscale("log")
            ax2.set_xlabel("Noise Level (Dirichlet)")
            ax2.set_ylabel(r"$L_2(\gamma_t, \gamma_t^\theta)$")
            # ax2.set_title(f"Robustness of Parameter Identification\n({testcase}/{subcase}, {dim_folder})")
            ax2.grid(True, which="both", ls="-", alpha=0.2)
            ax2.margins(y=0.15)
            # ax2.legend()

            plt.tight_layout()
            plot_path = os.path.join(base_dir, f"robustness_noise_gamma_{testcase}_{subcase}_{dim_folder}.pdf")
            plt.savefig(plot_path)
            plt.close()
            print(f"Robustness plots for '{testcase}/{subcase}' ({dim_folder}) saved to {plot_path}")


if __name__ == "__main__":
    main()

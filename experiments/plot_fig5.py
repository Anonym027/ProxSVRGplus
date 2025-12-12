# experiments/plot_fig5.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path to allow importing from plot_fig4
sys.path.append(str(Path(__file__).resolve().parent))
from plot_fig4 import load_hist, get_sfo_over_n, collect_curves

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def find_best_b(dataset: str,
                algo_suffix: str,
                n_samples: int,
                b_list):
    """
    Leverages the logic from plot_fig4.collect_curves to find the best 'b'.
    
    Returns the batch size 'b' that leads to the fastest convergence for a
    given dataset and algorithm combination.
    """
    _, _, best_b = collect_curves(
        dataset=dataset,
        algo_suffix=algo_suffix,
        n_samples=n_samples,
        b_list=b_list,
    )
    return best_b


def main():
    # Fallback values for n if not found in the history dictionary.
    N_A9A = 32561
    N_MNIST = 60000

    # Use the same grid of batch sizes 'b' as in fig4.
    b_list = [1, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384]

    fig, axes = plt.subplots(2, 1, figsize=(5, 7), sharex=True)

    # =========================================================
    # 1) a9a
    # =========================================================
    dataset = "a9a"
    n_default = N_A9A

    # Find the best 'b' for ProxSVRG and ProxSVRG+.
    best_b_svrg_a9a = find_best_b(dataset, "proxsvrg", n_default, b_list)
    best_b_plus_a9a = find_best_b(dataset, "proxsvrgplus", n_default, b_list)

    # ProxGD/ProxSGD are independent of 'b', so any result file can be used as a reference.
    b_ref = b_list[3]

    hist_gd = load_hist(dataset, b_ref, "proxgd")
    hist_sgd = load_hist(dataset, b_ref, "proxsgd")
    hist_svrg = load_hist(dataset, best_b_svrg_a9a, "proxsvrg")
    hist_plus = load_hist(dataset, best_b_plus_a9a, "proxsvrgplus")

    ax = axes[0]

    x_gd = get_sfo_over_n(hist_gd, n_default)
    y_gd = np.array(hist_gd["objective"], dtype=float)
    ax.plot(x_gd, y_gd, "ko-", label="ProxGD")  # Black, solid line with circle markers

    x_sgd = get_sfo_over_n(hist_sgd, n_default)
    y_sgd = np.array(hist_sgd["objective"], dtype=float)
    ax.plot(x_sgd, y_sgd, "go--", label="ProxSGD")  # Green, dashed line with circle markers

    x_svrg = get_sfo_over_n(hist_svrg, n_default)
    y_svrg = np.array(hist_svrg["objective"], dtype=float)
    ax.plot(x_svrg, y_svrg, "b-", label=f"ProxSVRG (b={best_b_svrg_a9a})")

    x_plus = get_sfo_over_n(hist_plus, n_default)
    y_plus = np.array(hist_plus["objective"], dtype=float)
    ax.plot(x_plus, y_plus, "r--", label=f"ProxSVRG+ (b={best_b_plus_a9a})")

    ax.set_title("a9a")
    ax.set_ylabel("Function value")
    ax.legend(loc="upper right")
    ax.grid(True)

    # =========================================================
    # 2) MNIST
    # =========================================================
    dataset = "mnist"
    n_default = N_MNIST

    best_b_svrg_mnist = find_best_b(dataset, "proxsvrg", n_default, b_list)
    best_b_plus_mnist = find_best_b(dataset, "proxsvrgplus", n_default, b_list)

    hist_gd = load_hist(dataset, b_ref, "proxgd")
    hist_sgd = load_hist(dataset, b_ref, "proxsgd")
    hist_svrg = load_hist(dataset, best_b_svrg_mnist, "proxsvrg")
    hist_plus = load_hist(dataset, best_b_plus_mnist, "proxsvrgplus")

    ax = axes[1]

    x_gd = get_sfo_over_n(hist_gd, n_default)
    y_gd = np.array(hist_gd["objective"], dtype=float)
    ax.plot(x_gd, y_gd, "ko-", label="ProxGD")

    x_sgd = get_sfo_over_n(hist_sgd, n_default)
    y_sgd = np.array(hist_sgd["objective"], dtype=float)
    ax.plot(x_sgd, y_sgd, "go--", label="ProxSGD")

    x_svrg = get_sfo_over_n(hist_svrg, n_default)
    y_svrg = np.array(hist_svrg["objective"], dtype=float)
    ax.plot(x_svrg, y_svrg, "b-", label=f"ProxSVRG (b={best_b_svrg_mnist})")

    x_plus = get_sfo_over_n(hist_plus, n_default)
    y_plus = np.array(hist_plus["objective"], dtype=float)
    ax.plot(x_plus, y_plus, "r--", label=f"ProxSVRG+ (b={best_b_plus_mnist})")

    ax.set_title("MNIST")
    ax.set_ylabel("Function value")
    ax.set_xlabel("#SFO / n")
    ax.legend(loc="upper right")
    ax.grid(True)

    plt.tight_layout()

    out_path = RESULTS_DIR / "fig5.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Figure saved to:", out_path)

    plt.show()


if __name__ == "__main__":
    main()
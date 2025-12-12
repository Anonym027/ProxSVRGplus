# experiments/plot_fig3.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 0) Path and Style Setup =========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_hist(dataset: str, b: int, algo_suffix: str):
    """
    dataset: "a9a" or "mnist"
    b: minibatch size
    algo_suffix: "proxgd", "proxsgd", "proxsvrg", "proxsvrgplus"
    """
    fname = f"results_{dataset}_b{b}_{algo_suffix}.npy"
    path = RESULTS_DIR / fname
    hist = np.load(path, allow_pickle=True).item()
    return hist


def get_sfo_over_n(hist, n_default=1.0, shift_to_zero=True):
    """
    Compute the x-axis values, #SFO / n, and optionally shift the first point to 0.

    This helps align the plot with the 0-6 range typically seen in papers like
    the ProxSVRG+ paper (Figure 3).

    If the history dictionary does not contain 'n', it falls back to n_default.
    """
    sfo = np.array(hist["sfo"], dtype=float)

    # Read 'n' from history; if not present, use the provided default value.
    n = None
    for key in ["n", "num_samples", "N"]:
        if key in hist:
            n = float(hist[key])
            break
    if n is None:
        n = float(n_default)

    x = sfo / n  # #SFO / n

    if shift_to_zero and x.size > 0:
        # Shift the first point to 0 (e.g., from 1.0 to 0.0) for alignment.
        x = x - x[0]
        # Protect against minor numerical errors (e.g., -1e-16).
        x = np.maximum(x, 0.0)

    return x


def main():
    # Fallback values for 'n' if they are not found in the history dictionary.
    N_A9A = 32561   # The value of 'n' printed when running the experiment.
    N_MNIST = 60000  # This can be modified if make_mnist_nnpca_problem crops the dataset.

    b_list = [4, 64, 256]
    algo_suffixes = ["proxgd", "proxsgd", "proxsvrg", "proxsvrgplus"]
    algo_labels = {
        "proxgd": "ProxGD",
        "proxsgd": "ProxSGD",
        "proxsvrg": "ProxSVRG",
        "proxsvrgplus": "ProxSVRG+",
    }
    algo_styles = {
        "proxgd": {"marker": "o", "linestyle": "-",  "linewidth": 1.5},
        "proxsgd": {"marker": "o", "linestyle": "--", "linewidth": 1.5},
        "proxsvrg": {"marker": None, "linestyle": "-",  "linewidth": 1.5},
        "proxsvrgplus": {"marker": None, "linestyle": "--", "linewidth": 1.5},
    }

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), sharex=True, sharey=True
    )

    # ========= 1) a9a: First Row of Plots =========
    dataset = "a9a"
    for j, b in enumerate(b_list):
        ax = axes[0, j]
        for algo in algo_suffixes:
            hist = load_hist(dataset, b, algo)
            x = get_sfo_over_n(hist, n_default=N_A9A, shift_to_zero=True)
            # To align with the paper's scale, you can multiply the objective by n.
            # y = np.array(hist["objective"], dtype=float) * N_A9A
            y = np.array(hist["objective"], dtype=float)
            style = algo_styles[algo]
            ax.plot(
                x,
                y,
                label=algo_labels[algo],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )

        ax.set_title(f"a9a (b={b})")
        if j == 0:
            ax.set_ylabel("Function value")

        # Each subplot has its own legend for clarity.
        ax.legend(loc="upper right", fontsize=8)

    # ========= 2) MNIST: Second Row of Plots =========
    dataset = "mnist"
    for j, b in enumerate(b_list):
        ax = axes[1, j]
        for algo in algo_suffixes:
            hist = load_hist(dataset, b, algo)
            x = get_sfo_over_n(hist, n_default=N_MNIST, shift_to_zero=True)
            # y = np.array(hist["objective"], dtype=float) * N_MNIST
            y = np.array(hist["objective"], dtype=float)
            style = algo_styles[algo]
            ax.plot(
                x,
                y,
                label=algo_labels[algo],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )

        ax.set_title(f"MNIST (b={b})")
        if j == 0:
            ax.set_ylabel("Function value")
        ax.set_xlabel("#SFO / n")

        # Each subplot has its own legend for clarity.
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

     # === Save figure to the results directory ===
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = RESULTS_DIR / "fig3.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Figure saved to:", fig_path)

    plt.show()


if __name__ == "__main__":
    main()
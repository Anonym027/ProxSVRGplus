# experiments/plot_fig4.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_hist(dataset: str, b: int, algo_suffix: str):
    """
    dataset: "a9a" or "mnist"
    algo_suffix: "proxsvrg" or "proxsvrgplus"
    """
    fname = f"results_{dataset}_b{b}_{algo_suffix}.npy"
    path = RESULTS_DIR / fname
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find results file: {path}")
    hist = np.load(path, allow_pickle=True).item()
    return hist


def get_sfo_over_n(hist, n_default: float) -> np.ndarray:
    """Compute #SFO / n for x-axis."""
    sfo = np.array(hist["sfo"], dtype=float)

    # If 'n' is stored in history, use it; otherwise, use the provided n_default.
    n = None
    for key in ("n", "num_samples", "N"):
        if key in hist:
            n = float(hist[key])
            break
    if n is None:
        n = float(n_default)

    return sfo / n


def collect_curves(
    dataset: str,
    algo_suffix: str,
    n_samples: int,
    b_list,
):
    """
    Read all curves for a given dataset and algorithm across all batch sizes 'b'.
    
    This function selects the best 'b' based on the criterion of fastest 
    convergence to a near-optimal function value.

    Returns
    -------
    xs_dict : dict
        Dictionary of x-axis sequences (#SFO/n), keyed by 'b'.
    ys_dict : dict
        Dictionary of y-axis sequences (function value), keyed by 'b'.
    best_b : int
        The batch size that resulted in the fastest convergence.
    """
    xs_dict = {}
    ys_dict = {}

    # 1) First, load all curves from the result files.
    for b in b_list:
        try:
            hist = load_hist(dataset, b, algo_suffix)
        except FileNotFoundError:
            # If a result file for a specific 'b' is missing, skip it.
            continue

        x = get_sfo_over_n(hist, n_default=n_samples)
        y = np.array(hist["objective"], dtype=float)

        xs_dict[b] = x
        ys_dict[b] = y

    if not ys_dict:
        raise RuntimeError(f"No curves found for {dataset}, {algo_suffix}")

    # 2) Find the best overall final function value across all curves.
    final_vals = [ys[-1] for ys in ys_dict.values()]
    v_star = min(final_vals)

    # Use a relative tolerance of 0.01% to define "near-optimal".
    tol = 0.0001
    v_target = v_star + tol * abs(v_star)

    # 3) For each curve, find the #SFO/n required to first reach the target value.
    hit_sfo = {}
    for b, ys in ys_dict.items():
        xs = xs_dict[b]
        idx = np.where(ys <= v_target)[0]
        if len(idx) > 0:
            hit_sfo[b] = xs[idx[0]]
        else:
            # This curve did not converge to the target level.
            hit_sfo[b] = np.inf

    # 4) Select the 'b' that converged fastest. If none converged to the target,
    # fall back to the one with the best final objective value.
    if all(np.isinf(v) for v in hit_sfo.values()):
        bs = list(ys_dict.keys())
        finals = [ys_dict[b][-1] for b in bs]
        best_b = bs[int(np.argmin(finals))]
    else:
        best_b = min(hit_sfo, key=hit_sfo.get)

    return xs_dict, ys_dict, best_b


def plot_panel(ax, dataset: str, algo_suffix: str, n_samples: int, b_list, title: str):
    """
    Draws a single panel on a given `ax` object.

    - Automatically selects the best 'b' by calling collect_curves.
    - Style for the best 'b' curve: solid line with markers.
    - Style for curves with b < best_b: dashed lines.
    - Style for curves with b > best_b: solid lines.
    """
    xs_dict, ys_dict, best_b = collect_curves(
        dataset=dataset,
        algo_suffix=algo_suffix,
        n_samples=n_samples,
        b_list=b_list,
    )

    # Plot curves in ascending order of 'b'.
    for b in sorted(xs_dict.keys()):
        x = xs_dict[b]
        y = ys_dict[b]

        if b == best_b:
            ls = "-"         # Solid line
            marker = "o"     # With markers
            lw = 1.8
        elif b < best_b:
            ls = "--"        # b < best_b: dashed line
            marker = None
            lw = 1.2
        else:
            ls = "-"         # b > best_b: solid line
            marker = None
            lw = 1.2

        ax.plot(
            x,
            y,
            linestyle=ls,
            marker=marker,
            linewidth=lw,
            label=f"b={b}",
        )

    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)


def main():
    # Fallback values for n (if not found in history).
    N_A9A = 32561
    N_MNIST = 60000

    # List of batch sizes 'b' as in Figure 4 of the paper.
    b_list = [1, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

    # ===== a9a: ProxSVRG+ (top-left) =====
    plot_panel(
        ax=axes[0, 0],
        dataset="a9a",
        algo_suffix="proxsvrgplus",
        n_samples=N_A9A,
        b_list=b_list,
        title="a9a (ProxSVRG+)",
    )
    axes[0, 0].set_ylabel("Function value")

    # ===== a9a: ProxSVRG (top-right) =====
    plot_panel(
        ax=axes[0, 1],
        dataset="a9a",
        algo_suffix="proxsvrg",
        n_samples=N_A9A,
        b_list=b_list,
        title="a9a (ProxSVRG)",
    )

    # ===== MNIST: ProxSVRG+ (bottom-left) =====
    plot_panel(
        ax=axes[1, 0],
        dataset="mnist",
        algo_suffix="proxsvrgplus",
        n_samples=N_MNIST,
        b_list=b_list,
        title="MNIST (ProxSVRG+)",
    )
    axes[1, 0].set_xlabel("#SFO / n")
    axes[1, 0].set_ylabel("Function value")

    # ===== MNIST: ProxSVRG (bottom-right) =====
    plot_panel(
        ax=axes[1, 1],
        dataset="mnist",
        algo_suffix="proxsvrg",
        n_samples=N_MNIST,
        b_list=b_list,
        title="MNIST (ProxSVRG)",
    )
    axes[1, 1].set_xlabel("#SFO / n")

    plt.tight_layout()

    # Save the figure to the results directory.
    out_path = RESULTS_DIR / "fig4.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Figure saved to:", out_path)

    plt.show()


if __name__ == "__main__":
    main()
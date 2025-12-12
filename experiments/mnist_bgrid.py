# experiments/mnist_bgrid.py

import os
import numpy as np
from pathlib import Path

from proxsvrgplus.problems.datasets import make_mnist_nnpca_problem
from proxsvrgplus.optim.ProxGD import prox_gd
from proxsvrgplus.optim.ProxSGD import prox_sgd
from proxsvrgplus.optim.ProxSVRG import prox_svrg
from proxsvrgplus.optim.ProxSVRGplus import prox_svrg_plus


def main():
    # ===== 0) Setup: Paths and Data =====
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    os.makedirs(results_dir, exist_ok=True)

    mnist_path = data_dir / "mnist"
    problem = make_mnist_nnpca_problem(mnist_path)

    d = problem.d
    n = problem.n
    L = problem.L

    print(f"[MNIST NN-PCA] n = {n}, d = {d}, L = {L:.6f}")

    x0 = np.ones(d) / np.sqrt(d)

    B = n // 5
    print(f"Use outer batch size B = n/5 = {B}")

    b_list = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384]
    max_epochs = 6 

    for b in b_list:
        print("\n" + "=" * 70)
        print(f"[MNIST] Running all algorithms with minibatch b = {b}")
        print("=" * 70)

        eta_gd = 1.0 / L
        eta_sgd = 1.0 / (2.0 * L)
        eta_svrg = (b ** 1.5) / (3.0 * L * n)
        eta_plus = 1.0 / (6.0 * L)

        print(
            f"Step sizes: "
            f"eta_gd={eta_gd:.4e}, eta_sgd={eta_sgd:.4e}, "
            f"eta_svrg={eta_svrg:.4e}, eta_plus={eta_plus:.4e}"
        )

        # ===== 1) Run ProxGD =====
        print("\n[ProxGD]")
        x_gd, hist_gd = prox_gd(
            problem=problem,
            x0=x0,
            stepsize=eta_gd,
            max_epochs=max_epochs,
            log_every=1.0,
        )
        print("  final objective:", hist_gd.objective[-1])
        out_gd = results_dir / f"results_mnist_b{b}_proxgd.npy"
        np.save(out_gd, hist_gd.as_dict())
        print("  saved to:", out_gd)

        # ===== 2) Run ProxSGD =====
        print("\n[ProxSGD]")
        x_sgd, hist_sgd = prox_sgd(
            problem=problem,
            x0=x0,
            stepsize=eta_sgd,
            batch_size=b,
            max_epochs=max_epochs,
            log_every=1.0,
            eta_prime=None,
            seed=615,
        )
        print("  final objective:", hist_sgd.objective[-1])
        out_sgd = results_dir / f"results_mnist_b{b}_proxsgd.npy"
        np.save(out_sgd, hist_sgd.as_dict())
        print("  saved to:", out_sgd)

        # ===== 3) Run ProxSVRG =====
        print("\n[ProxSVRG]")
        inner_iters = n//b
        x_svrg, hist_svrg = prox_svrg(
            problem=problem,
            x0=x0,
            stepsize=eta_svrg,
            max_epochs=max_epochs,
            inner_iters=inner_iters,
            batch_size=b,
            log_every=1.0,
            seed=615,
        )
        print("  final objective:", hist_svrg.objective[-1])
        out_svrg = results_dir / f"results_mnist_b{b}_proxsvrg.npy"
        np.save(out_svrg, hist_svrg.as_dict())
        print("  saved to:", out_svrg)

        # ===== 4) Run ProxSVRG+ =====
        print("\n[ProxSVRG+]")
        outer_batch_size = B
        inner_batch_size = b
        epoch_length = inner_iters

        x_plus, hist_plus = prox_svrg_plus(
            problem=problem,
            x0=x0,
            stepsize=eta_plus,
            max_epochs=max_epochs,
            outer_batch_size=outer_batch_size,
            inner_batch_size=inner_batch_size,
            epoch_length=epoch_length,
            log_every=1.0,
            seed=615,
        )
        print("  final objective:", hist_plus.objective[-1])
        out_plus = results_dir / f"results_mnist_b{b}_proxsvrgplus.npy"
        np.save(out_plus, hist_plus.as_dict())
        print("  saved to:", out_plus)


if __name__ == "__main__":
    main()
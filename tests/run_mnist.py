# tests/run_mnist.py

import os
import numpy as np
from pathlib import Path

from proxsvrgplus.problems.datasets import make_mnist_nnpca_problem
from proxsvrgplus.optim.ProxGD import prox_gd
from proxsvrgplus.optim.ProxSGD import prox_sgd
from proxsvrgplus.optim.ProxSVRG import prox_svrg
from proxsvrgplus.optim.ProxSVRGplus import prox_svrg_plus

# ===== 0) Path and Data Setup =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


mnist_path = DATA_DIR / "mnist"

problem = make_mnist_nnpca_problem(mnist_path)

d = problem.d
# A feasible initial point, same as used for a9a.
x0 = np.ones(d) / np.sqrt(d)


# ===== 1) Run ProxGD =====
print("=== ProxGD on MNIST (NN-PCA) ===")
x_gd, hist_gd = prox_gd(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10,
    log_every=1.0,
)

print("ProxGD final objective:", hist_gd.objective[-1])
print("ProxGD final grad_map_norm_sq:", hist_gd.grad_map_norm_sq[-1])

out_gd = RESULTS_DIR / "results_mnist_proxgd.npy"
np.save(out_gd, hist_gd.as_dict())
print("ProxGD history saved to:", out_gd)


# ===== 2) Run ProxSGD =====
print("\n=== ProxSGD on MNIST (NN-PCA) ===")
x_sgd, hist_sgd = prox_sgd(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    batch_size=128,      # MNIST has more samples, so a larger batch size can be used.
    max_epochs=10,
    log_every=1.0,
    eta_prime=None,
    seed=615,
)

print("ProxSGD final objective:", hist_sgd.objective[-1])
print("ProxSGD final grad_map_norm_sq:", hist_sgd.grad_map_norm_sq[-1])

out_sgd = RESULTS_DIR / "results_mnist_proxsgd.npy"
np.save(out_sgd, hist_sgd.as_dict())
print("ProxSGD history saved to:", out_sgd)


# ===== 3) Run ProxSVRG =====
print("\n=== ProxSVRG on MNIST (NN-PCA) ===")
x_svrg, hist_svrg = prox_svrg(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10,
    inner_iters=problem.n // 10,   # Use n/10, same as in the a9a test.
    batch_size=128,
    log_every=1.0,
    seed=615,
)

print("ProxSVRG final objective:", hist_svrg.objective[-1])
print("ProxSVRG final grad_map_norm_sq:", hist_svrg.grad_map_norm_sq[-1])

out_svrg = RESULTS_DIR / "results_mnist_proxsvrg.npy"
np.save(out_svrg, hist_svrg.as_dict())
print("ProxSVRG history saved to:", out_svrg)


# ===== 4) Run ProxSVRG+ =====
print("\n=== ProxSVRG+ on MNIST (NN-PCA) ===")
outer_batch_size = min(problem.n, 2048)  # B
inner_batch_size = 128                   # b
epoch_length = problem.n // 10           # m

x_plus, hist_plus = prox_svrg_plus(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10,
    outer_batch_size=outer_batch_size,
    inner_batch_size=inner_batch_size,
    epoch_length=epoch_length,
    log_every=1.0,
    seed=615,
)

print("ProxSVRG+ final objective:", hist_plus.objective[-1])
print("ProxSVRG+ final grad_map_norm_sq:", hist_plus.grad_map_norm_sq[-1])

out_plus = RESULTS_DIR / "results_mnist_proxsvrgplus.npy"
np.save(out_plus, hist_plus.as_dict())
print("ProxSVRG+ history saved to:", out_plus)
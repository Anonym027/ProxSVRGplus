# tests/run_a9a.py

import os
import numpy as np
from pathlib import Path

from proxsvrgplus.problems.datasets import make_a9a_nnpca_problem
from proxsvrgplus.optim.ProxGD import prox_gd
from proxsvrgplus.optim.ProxSGD import prox_sgd
from proxsvrgplus.optim.ProxSVRG import prox_svrg
from proxsvrgplus.optim.ProxSVRGplus import prox_svrg_plus

# ===== 0) Path and Data Setup =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

a9a_path = DATA_DIR / "a9a.txt"
problem = make_a9a_nnpca_problem(a9a_path)

d = problem.d
n = problem.n
L = problem.L    # Pre-calculated Lipschitz constant from the NNPCAProblem.

print(f"a9a NN-PCA: n = {n}, d = {d}, L = {L:.6f}")

# A simple feasible initial point (non-negative with an L2 norm of 1).
x0 = np.ones(d) / np.sqrt(d)

# ===== 0.5) Batch Size Parameters: B and b =====
# As per the paper, B = n/5. For b, you can run experiments separately
# for values like 4, 64, and 256 to reproduce Figure 3.
B = n // 5
b = 4   # Set to 4 initially. Change to 64 or 256 and re-run to reproduce the columns of Figure 3.

# ===== 0.6) Step Sizes based on the paper's formulas =====
eta_gd   = 1.0 / L
eta_sgd  = 1.0 / (2.0 * L)
eta_svrg = (b ** 1.5) / (3.0 * L * n)
eta_plus = 1.0 / (6.0 * L)

print(f"Using batch sizes: B = {B}, b = {b}")
print(f"Step sizes: eta_gd={eta_gd:.4e}, eta_sgd={eta_sgd:.4e}, "
      f"eta_svrg={eta_svrg:.4e}, eta_plus={eta_plus:.4e}")

# ===== 1) Run ProxGD =====
print("\n=== ProxGD on a9a (NN-PCA) ===")
x_gd, hist_gd = prox_gd(
    problem=problem,
    x0=x0,
    stepsize=eta_gd,
    max_epochs=10,
    log_every=1,
)

print("ProxGD final objective:", hist_gd.objective[-1])
print("ProxGD final grad_map_norm_sq:", hist_gd.grad_map_norm_sq[-1])

out_gd = RESULTS_DIR / "results_a9a_proxgd.npy"
np.save(out_gd, hist_gd.as_dict())
print("ProxGD history saved to:", out_gd)

# ===== 2) Run ProxSGD =====
print("\n=== ProxSGD on a9a (NN-PCA) ===")
x_sgd, hist_sgd = prox_sgd(
    problem=problem,
    x0=x0,
    stepsize=eta_sgd,
    batch_size=b,
    max_epochs=10.0,
    log_every=1.0,
    eta_prime=None,   # Use a constant step size.
    seed=615,
)

print("ProxSGD final objective:", hist_sgd.objective[-1])
print("ProxSGD final grad_map_norm_sq:", hist_sgd.grad_map_norm_sq[-1])

out_sgd = RESULTS_DIR / "results_a9a_proxsgd.npy"
np.save(out_sgd, hist_sgd.as_dict())
print("ProxSGD history saved to:", out_sgd)

# ===== 3) Run ProxSVRG =====
print("\n=== ProxSVRG on a9a (NN-PCA) ===")
# inner_iters corresponds to the epoch length m. Theoretically, m is often
# chosen around sqrt(b), but this is not strictly fixed in the paper's
# experiments. We use a conservative choice here that can be tuned later.
inner_iters = max(1, n // 10)

x_svrg, hist_svrg = prox_svrg(
    problem=problem,
    x0=x0,
    stepsize=eta_svrg,
    max_epochs=10.0,
    inner_iters=inner_iters,
    batch_size=b,
    log_every=1.0,
    seed=615,
)

print("ProxSVRG final objective:", hist_svrg.objective[-1])
print("ProxSVRG final grad_map_norm_sq:", hist_svrg.grad_map_norm_sq[-1])

out_svrg = RESULTS_DIR / "results_a9a_proxsvrg.npy"
np.save(out_svrg, hist_svrg.as_dict())
print("ProxSVRG history saved to:", out_svrg)

# ===== 4) Run ProxSVRG+ =====
print("\n=== ProxSVRG+ on a9a (NN-PCA) ===")

outer_batch_size = B      # Outer batch size B
inner_batch_size = b      # Inner mini-batch size b
epoch_length = inner_iters  # Same as ProxSVRG for now; other values for m can be tested later.

x_plus, hist_plus = prox_svrg_plus(
    problem=problem,
    x0=x0,
    stepsize=eta_plus,
    max_epochs=10.0,
    outer_batch_size=outer_batch_size,
    inner_batch_size=inner_batch_size,
    epoch_length=epoch_length,
    log_every=1.0,
    seed=615,
)

print("ProxSVRG+ final objective:", hist_plus.objective[-1])
print("ProxSVRG+ final grad_map_norm_sq:", hist_plus.grad_map_norm_sq[-1])

out_plus = RESULTS_DIR / "results_a9a_proxsvrgplus.npy"
np.save(out_plus, hist_plus.as_dict())
print("ProxSVRG+ history saved to:", out_plus)
# tests/run_vmf.py

import os
import numpy as np
from pathlib import Path

from proxsvrgplus.problems.vmf_sim import make_vmf_nnpca_problem
from proxsvrgplus.optim.ProxGD import prox_gd
from proxsvrgplus.optim.ProxSGD import prox_sgd
from proxsvrgplus.optim.ProxSVRG import prox_svrg
from proxsvrgplus.optim.ProxSVRGplus import prox_svrg_plus


def cosine_alignment(x: np.ndarray, x_star: np.ndarray) -> float:
    """
    Calculates the cosine similarity between x and x_star.

    This is used to measure the quality of the direction recovery for the
    synthetically generated vMF problem.
    """
    x = np.asarray(x, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    if np.linalg.norm(x) == 0 or np.linalg.norm(x_star) == 0:
        return 0.0
    return float(
        np.dot(x, x_star)
        / (np.linalg.norm(x) * np.linalg.norm(x_star))
    )


# ===== 0) vMF Simulation Parameters =====
# Using parameters from the project proposal, e.g., n=5000, d=200, kappa=10.
n = 5000
d = 200
kappa = 10.0
seed = 615

print(f"Simulating vMF-like data: n={n}, d={d}, kappa={kappa}, seed={seed}")

problem, x_star = make_vmf_nnpca_problem(
    n=n,
    d=d,
    kappa=kappa,
    seed=seed,
)

d = problem.d
# A simple feasible initial point (non-negative with an L2 norm of 1).
x0 = np.ones(d) / np.sqrt(d)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 1) Run ProxGD =====
print("=== ProxGD on vMF (NN-PCA) ===")
x_gd, hist_gd = prox_gd(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10,   # Note: max_epochs should be an integer for prox_gd.
    log_every=1,
)

align_gd = cosine_alignment(x_gd, x_star)
print("ProxGD final objective:", hist_gd.objective[-1])
print("ProxGD final grad_map_norm_sq:", hist_gd.grad_map_norm_sq[-1])
print("ProxGD alignment with x_star:", align_gd)

out_gd = RESULTS_DIR / "results_vmf_proxgd.npy"
np.save(out_gd, hist_gd.as_dict())
print("ProxGD history saved to:", out_gd)


# ===== 2) Run ProxSGD =====
print("\n=== ProxSGD on vMF (NN-PCA) ===")
x_sgd, hist_sgd = prox_sgd(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    batch_size=64,
    max_epochs=10.0,
    log_every=1.0,
    eta_prime=None,
    seed=seed,
)

align_sgd = cosine_alignment(x_sgd, x_star)
print("ProxSGD final objective:", hist_sgd.objective[-1])
print("ProxSGD final grad_map_norm_sq:", hist_sgd.grad_map_norm_sq[-1])
print("ProxSGD alignment with x_star:", align_sgd)

out_sgd = RESULTS_DIR / "results_vmf_proxsgd.npy"
np.save(out_sgd, hist_sgd.as_dict())
print("ProxSGD history saved to:", out_sgd)


# ===== 3) Run ProxSVRG =====
print("\n=== ProxSVRG on vMF (NN-PCA) ===")
x_svrg, hist_svrg = prox_svrg(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10.0,
    inner_iters=problem.n // 10,
    batch_size=64,
    log_every=1.0,
    seed=seed,
)

align_svrg = cosine_alignment(x_svrg, x_star)
print("ProxSVRG final objective:", hist_svrg.objective[-1])
print("ProxSVRG final grad_map_norm_sq:", hist_svrg.grad_map_norm_sq[-1])
print("ProxSVRG alignment with x_star:", align_svrg)

out_svrg = RESULTS_DIR / "results_vmf_proxsvrg.npy"
np.save(out_svrg, hist_svrg.as_dict())
print("ProxSVRG history saved to:", out_svrg)


# ===== 4) Run ProxSVRG+ =====
print("\n=== ProxSVRG+ on vMF (NN-PCA) ===")
outer_batch_size = min(problem.n, 1024)  # Outer batch size B
inner_batch_size = 64                    # Inner mini-batch size b
epoch_length = problem.n // 10           # Number of inner iterations m

x_plus, hist_plus = prox_svrg_plus(
    problem=problem,
    x0=x0,
    stepsize=1.0,
    max_epochs=10.0,
    outer_batch_size=outer_batch_size,
    inner_batch_size=inner_batch_size,
    epoch_length=epoch_length,
    log_every=1.0,
    seed=seed,
)

align_plus = cosine_alignment(x_plus, x_star)
print("ProxSVRG+ final objective:", hist_plus.objective[-1])
print("ProxSVRG+ final grad_map_norm_sq:", hist_plus.grad_map_norm_sq[-1])
print("ProxSVRG+ alignment with x_star:", align_plus)

out_plus = RESULTS_DIR / "results_vmf_proxsvrgplus.npy"
np.save(out_plus, hist_plus.as_dict())
print("ProxSVRG+ history saved to:", out_plus)
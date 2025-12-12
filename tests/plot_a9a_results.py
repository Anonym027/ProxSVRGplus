# tests/plot_a9a_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 0) 路径设置 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

file_gd   = RESULTS_DIR / "results_a9a_proxgd.npy"
file_sgd  = RESULTS_DIR / "results_a9a_proxsgd.npy"
file_svrg = RESULTS_DIR / "results_a9a_proxsvrg.npy"
file_plus = RESULTS_DIR / "results_a9a_proxsvrgplus.npy"

hist_gd   = np.load(file_gd,   allow_pickle=True).item()
hist_sgd  = np.load(file_sgd,  allow_pickle=True).item()
hist_svrg = np.load(file_svrg, allow_pickle=True).item()
hist_plus = np.load(file_plus, allow_pickle=True).item()

# 为了写起来方便，放进一个 dict
methods = {
    "ProxGD":    hist_gd,
    "ProxSGD":   hist_sgd,
    "ProxSVRG":  hist_svrg,
    "ProxSVRG+": hist_plus,
}

# ===== 1) objective vs SFO =====
plt.figure()
for name, h in methods.items():
    sfo = np.array(h["sfo"], dtype=float)
    obj = np.array(h["objective"], dtype=float)
    plt.plot(sfo / sfo[-1], obj, label=name)

plt.xlabel("SFO / final SFO")  # 或者改成 "SFO"
plt.ylabel("Objective value")
plt.title("a9a + NN-PCA: Objective vs SFO")
plt.legend()
plt.grid(True)

# ===== 2) grad_map_norm_sq vs SFO (log scale) =====
plt.figure()
for name, h in methods.items():
    sfo = np.array(h["sfo"], dtype=float)
    gmap = np.array(h["grad_map_norm_sq"], dtype=float)
    plt.plot(sfo / sfo[-1], gmap, label=name)

plt.xlabel("SFO / final SFO")
plt.ylabel("||G_eta(x)||^2")
plt.title("a9a + NN-PCA: Grad-map norm^2 vs SFO")
plt.yscale("log")  # 用对数尺度更容易看下降趋势
plt.legend()
plt.grid(True)

plt.show()
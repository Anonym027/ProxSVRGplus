# Project: ProxSVRG+ for a Non-convex Stochastic Optimization Course

## 1. Project Background
This is a project for a master's level course on non-convex stochastic optimization.
The goal is to implement and compare ProxGD, ProxSGD, ProxSVRG, and ProxSVRG+ on several datasets (vMF synthetic data, a9a, MNIST).

The approximate code structure is as follows:

- `src/optim/`
  - `prox_gd.py`
  - `prox_sgd.py`
  - `prox_svrg.py`
  - `prox_svrg_plus.py`
- `src/problems/`
  - `nn_pca.py`      # f, h, grad_i, prox_h for NN-PCA
  - `vmf_sim.py`     # vMF data generation
  - `datasets.py`    # a9a / MNIST preprocessing
- `experiments/`
  - `run_vmf.py`
  - `run_a9a.py`
  - `run_mnist.py`
- `tests/`
  - `test_nn_pca_basic.py`
  - `test_optim_convergence.py`
- `report/`
  - `report.tex` or `.ipynb`

## 2. How I Expect Gemini to Assist Me

1.  Treat this as a Python-based research project for a graduate-level optimization course.
2.  Prioritize providing **clear, well-commented Python code**.
3.  Do not arbitrarily modify filenames or directory structures unless specifically requested.
4.  When suggesting code changes, show only **relevant functions or snippets**, not entire files.
5.  Mathematical derivations can be explained briefly; the focus should be on implementation details (loops, vectorization, logging, testing, etc.).

## 3. Code Style Preferences

- Explanations should be professional and clear. **All code comments must be in English.**
- Use type hints reasonably.
- Follow general PEP8 guidelines, but do not be overly pedantic about formatting.
- The use of libraries like `numpy`, `torch`, and `matplotlib` is expected when needed.

## 4. Experiments and Evaluation

- Please help design and implement comparative experiments, including:
  - Objective value vs. SFO calls.
  - Gradient mapping norm vs. SFO calls.
  - Runtime vs. accuracy.
- Datasets:
  - vMF synthetic data with varying `n` and `d`.
  - a9a (binary classification).
  - MNIST (classification or NN-PCA).

When I refer to "running the vMF experiment" or similar phrases, assume I mean using scripts like `experiments/run_vmf.py` and related tools.

## 5. Things to Avoid

- Do not invent new algorithms beyond the scope of ProxGD/ProxSGD/ProxSVRG/ProxSVRG+ unless explicitly asked.
- Do not delete existing files.
- Do not assume a GPU is available unless I specify that it can be used.
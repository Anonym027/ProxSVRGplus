# ProxSVRG+ for Non-convex Optimization

This project is a Python implementation of several proximal stochastic gradient methods for non-convex finite-sum optimization problems. It was developed as part of a graduate course on optimization, primarily based on the paper 'A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization' by Li and Li (2018).

The main problem of interest is Non-negative Principal Component Analysis (NN-PCA). The algorithms are tested on synthetic von Mises-Fisher (vMF) data, as well as the `a9a` and `MNIST` datasets.

## GenAI Tutorial

📘 See [GenAI_Tutorial.md](./GenAI_Tutorial.md) for a detailed, step-by-step tutorial documenting how generative AI tools (Gemini CLI, ChatGPT) were used throughout the development, debugging, packaging, and validation of this project.

---

## How to Use This Project

There are two primary ways to use this project, depending on your goal.

### Option 1: As an Installable Python Library

This is the recommended approach if you want to use the implemented optimization algorithms in your own projects.

1.  **Install the package directly from GitHub:**
    ```bash
    pip install git+https://github.com/Anonym027/ProxSVRGplus.git
    ```

2.  **Use the algorithms in your code:**
    ```python
    import numpy as np
    from proxsvrgplus.optim import prox_sgd
    from proxsvrgplus.problems.datasets import make_a9a_nnpca_problem
    
    # Your code to create a problem instance...
    # problem = ... 
    # x0 = ...
    
    # Use the algorithm
    # x_final, history = prox_sgd(problem, x0, ...)
    ```

3.  **Verify Installation with the CLI Demo:**
    After installation, you can run a self-contained demo. This command uses a bundled version of the `a9a.txt` dataset and proves that the package is installed and working correctly.
    ```bash
    proxsvrgplus-demo run-demo
    ```

### Option 2: To Reproduce the Research Experiments

This is the recommended approach for developers, reviewers, or anyone who wants to run the exact experiments described in this project and generate the plots.

1.  **Clone the full repository:**
    ```bash
    git clone https://github.com/Anonym027/ProxSVRGplus.git
    cd ProxSVRGplus
    ```

2.  **Set up a virtual environment and install in editable mode:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the package with development dependencies
    pip install -e .[dev]
    ```

3.  **Run the experiment scripts:**
    You can now run the scripts located in the `experiments/` directory.
    ```bash
    # Run the a9a experiment grid
    python experiments/a9a_bgrid.py

    # Run the MNIST experiment grid
    python experiments/mnist_bgrid.py
    ```
    *Note: For experiment reproducibility, please ensure `a9a.txt` and `mnist` are present in the top-level `data/` directory.*

4.  **Generate plots:**
    After the experiments are finished, you can generate the summary figures.
    ```bash
    python experiments/plot_fig3.py
    ```

---

## Project Structure

- **`src/proxsvrgplus/`**: The core, installable Python package.
- **`data/`**: Directory for storing experiment data. The large `mnist` dataset will be downloaded here.
- **`experiments/`**: Scripts to run experiments and generate plots.
- **`results/`**: Default output directory for `.npy` data files and figures.
- **`tests/`**: Legacy runnable scripts used during development.

---

## Datasets

The `a9a` and `MNIST` datasets used in this project can be downloaded from:
[https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

Specifically, `a9a.txt` is found under the "binary" section, and `mnist.bz2` (which needs to be decompressed) is found under the "multiclass" section.

---

## Citation

If you find this work useful, please consider citing the original paper:

```bibtex
@article{li2018simple,
  title={A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization},
  author={Li, Zhize and Li, Jian},
  journal={arXiv preprint arXiv:1802.04477},
  year={2018}
}
```
[View on arXiv](https://arxiv.org/abs/1802.04477)

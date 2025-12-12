# ProxSVRG+ for Non-convex Optimization

This project is a Python implementation of several proximal stochastic gradient methods for non-convex finite-sum optimization problems. It was developed as part of a graduate course on optimization, primarily based on the paper 'A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization' by Li and Li (2018). The primary algorithms implemented are ProxGD, ProxSGD, ProxSVRG, and ProxSVRG+.

The main problem of interest is Non-negative Principal Component Analysis (NN-PCA). The algorithms are tested on synthetic von Mises-Fisher (vMF) data, as well as the `a9a` and `MNIST` datasets.

## Quickstart

After installing the package, you can run a quick demonstration on the `a9a` dataset directly from your terminal.

```bash
# First, ensure you have installed the package in editable mode.
# Then, from the project's root directory:
proxsvrgplus-demo run-demo
```

This command runs a lightweight CLI demo that executes ProxSGD for a few epochs
on the a9a dataset and prints logs to the console. It is intended as a quick
sanity check that the package and CLI are installed correctly.


## Installation

To set up the project, first clone the repository. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/Anonym027/ProxSVRGplus.git
cd ProxSVRGplus

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the package and its dependencies in editable mode
pip install -e .[dev]
```

This command installs the package in "editable" mode (`-e`) and includes the development dependencies (`[dev]`), such as `pytest`. This is required to run the scripts and the CLI demo.

Alternatively, the package can be installed directly from GitHub without
editable mode using:

```bash
pip install git+https://github.com/Anonym027/ProxSVRGplus.git
```

## Project Structure

- **`src/proxsvrgplus/`**: Contains the core, installable Python package.
  - **`optim/`**: Implementations of the optimization algorithms (ProxGD, ProxSGD, ProxSVRG, ProxSVRG+).
  - **`problems/`**: Code for defining the NN-PCA problem, loading datasets, and simulating data.
  - **`cli.py`**: Defines the command-line interface for the `proxsvrgplus-demo` command.
- **`data/`**: Directory for storing datasets like `a9a.txt`.
- **`experiments/`**: Scripts to run the main experiments and generate plots.
- **`results/`**: Default output directory for `.npy` data files and figures.
- **`tests/`**: Legacy runnable scripts used during development for executing experiments and plotting results (not unit tests).

## How to Run Experiments

The main experiments are grids of runs over different mini-batch sizes (`b`). You can run them using the scripts in the `experiments/` directory.

For example, to run the `a9a` experiment across all configured batch sizes:

```bash
python experiments/a9a_bgrid.py
```

Similarly, for the `MNIST` dataset:

```bash
python experiments/mnist_bgrid.py
```

The numerical results (`.npy` files) will be saved in the `results/` directory.

## How to Generate Plots

After running the experiments, you can generate the summary figures (similar to those in the original paper) using the `plot_*.py` scripts.

To generate Figure 3, which compares all algorithms on both datasets for specific batch sizes:

```bash
python experiments/plot_fig3.py
```

## Implemented Algorithms

- **Proximal Gradient Descent (ProxGD)**
- **Proximal Stochastic Gradient Descent (ProxSGD)**
- **Proximal SVRG (ProxSVRG)**
- **ProxSVRG+**

These are implemented in the `src/proxsvrgplus/optim/` directory.

## Citation

If you find this work useful or use the provided algorithms, please consider citing the original paper:

```bibtex
@article{li2018simple,
  title={A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization},
  author={Li, Zhize and Li, Jian},
  journal={arXiv preprint arXiv:1802.04477},
  year={2018}
}
```

Or in a more readable format:

Zhize Li, Jian Li. "A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization." *arXiv preprint arXiv:1802.04477* (2018).

[View on arXiv](https://arxiv.org/abs/1802.04477)

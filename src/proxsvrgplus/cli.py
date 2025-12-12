# src/proxsvrgplus/cli.py

import argparse
from pathlib import Path
import numpy as np

# To locate the data file, we need the project root.
# This assumes the CLI is run after installation, so we can't
# rely on __file__ pointing to the src directory in the same way.
# A robust way is to package the data or have the user provide a path.
# For this demo, we'll assume the data exists in a 'data' folder
# relative to where the command is run.

from .problems.datasets import make_a9a_nnpca_problem
from .optim.ProxSGD import prox_sgd

def run_a9a_demo():
    """
    Runs a small demonstration of ProxSGD on the a9a dataset.
    """
    print("=======================================")
    print("=      ProxSVRG+ Demo: a9a + ProxSGD    =")
    print("=======================================\n")

    # --- 1. Find and load data ---
    # For the demo, we assume the 'data/a9a.txt' file is available
    # in the current working directory or a subdirectory.
    a9a_path = Path("./data/a9a.txt")
    if not a9a_path.exists():
        print(f"Error: Data file not found at '{a9a_path.resolve()}'")
        print("Please run this command from the project root directory where the 'data' folder exists.")
        return

    print(f"Loading a9a dataset from: {a9a_path}")
    problem = make_a9a_nnpca_problem(a9a_path)
    d = problem.d
    n = problem.n
    L = problem.L
    print(f"Problem loaded: n={n}, d={d}, L={L:.4f}\n")

    # --- 2. Set up optimization ---
    x0 = np.ones(d) / np.sqrt(d)
    b = 64
    eta_sgd = 1.0 / (2.0 * L)
    max_epochs_demo = 2.0

    print("Running ProxSGD with the following parameters:")
    print(f"  - stepsize (eta): {eta_sgd:.4e}")
    print(f"  - batch_size (b): {b}")
    print(f"  - max_epochs: {max_epochs_demo}\n")

    # --- 3. Run optimization ---
    x_sgd, hist_sgd = prox_sgd(
        problem=problem,
        x0=x0,
        stepsize=eta_sgd,
        batch_size=b,
        max_epochs=max_epochs_demo,
        log_every=0.5, # Log more frequently for the demo
        seed=615,
    )

    # --- 4. Print results ---
    print("\n--- Demo Finished ---")
    print(f"Final objective value: {hist_sgd.objective[-1]:.6f}")
    print(f"Final gradient mapping norm^2: {hist_sgd.grad_map_norm_sq[-1]:.6f}")

    print("\nObjective history (per 0.5 epoch):")
    for i, obj in enumerate(hist_sgd.objective):
        epoch = hist_sgd.epoch[i]
        print(f"  Epoch {epoch}: {obj:.6f}")
    print("\nQuickstart demo complete!")


def main():
    """
    Main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="ProxSVRG+ Project: Run demos and experiments."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run-demo
    demo_parser = subparsers.add_parser(
        "run-demo", help="Run a quick demonstration."
    )
    demo_parser.set_defaults(func=run_a9a_demo)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

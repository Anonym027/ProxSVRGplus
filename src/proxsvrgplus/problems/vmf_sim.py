# src/problems/vmf_sim.py

import numpy as np
from .nn_pca import NNPCAProblem


def sample_vmf_like(
    n: int,
    d: int,
    kappa: float,
    rng: np.random.Generator,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Generate n samples on the unit sphere in R^d that are
    concentrated around direction mu.

    This is an approximate vMF-like generator:
        z_i = normalize( kappa * mu + eps_i ),
    where eps_i ~ N(0, I_d).
    For larger kappa, samples are more concentrated around mu.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Dimension.
    kappa : float
        Concentration parameter (kappa >= 0).
    rng : np.random.Generator
        Random number generator.
    mu : np.ndarray, shape (d,)
        Unit vector representing the mean direction.

    Returns
    -------
    Z : np.ndarray, shape (n, d)
        Each row is a unit-norm vector on the sphere.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (d,):
        raise ValueError("mu must have shape (d,).")

    if abs(np.linalg.norm(mu) - 1.0) > 1e-6:
        raise ValueError("mu must be a unit vector.")

    eps = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    Y = kappa * mu + eps                # shape (n, d)
    # normalize rows to unit norm
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Z = Y / norms
    return Z


def make_vmf_nnpca_problem(
    n: int,
    d: int,
    kappa: float,
    seed: int = 0,
):
    """
    Construct an NNPCAProblem with vMF-like simulated data.

    Steps:
    1) Sample a ground truth direction x_star in C = {x >= 0, ||x|| <= 1}.
    2) Generate Z ~ vMF-like(mu=x_star, kappa).
    3) Return NNPCAProblem(Z) and x_star for alignment evaluation.

    Returns
    -------
    problem : NNPCAProblem
        NN-PCA problem instance built from Z.
    x_star : np.ndarray, shape (d,)
        Ground truth nonnegative unit vector used as the mean direction.
    """
    rng = np.random.default_rng(seed)

    # sample x_star: positive vector on sphere
    x_raw = rng.random(size=d)          # all positive
    x_star = x_raw / np.linalg.norm(x_raw)

    Z = sample_vmf_like(n=n, d=d, kappa=kappa, rng=rng, mu=x_star)
    problem = NNPCAProblem(Z)
    return problem, x_star

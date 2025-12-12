# src/optim/ProxGD.py

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ProxGDHistory:
    """
    Container for logging Proximal Gradient Descent (ProxGD) progress.

    Each field is a list of the same length. You can later convert this to
    numpy arrays or a pandas DataFrame for plotting.

    We also store the dataset size n so that one can directly plot SFO / n
    without hard-coding n in the plotting scripts.
    """

    def __init__(self, n: int) -> None:
        # problem size (number of samples)
        self.n: int = int(n)

        # per-iteration logs
        self.epoch: List[int] = []
        self.sfo: List[int] = []  # cumulative number of component gradient evaluations
        self.objective: List[float] = []
        self.grad_map_norm_sq: List[float] = []

    def append(self, epoch: int, sfo: int,
               objective: float, grad_map_norm_sq: float) -> None:
        self.epoch.append(int(epoch))
        self.sfo.append(int(sfo))
        self.objective.append(float(objective))
        self.grad_map_norm_sq.append(float(grad_map_norm_sq))

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the history as a plain dictionary.
        """
        return {
            "n": self.n,
            "epoch": self.epoch,
            "sfo": self.sfo,
            "objective": self.objective,
            "grad_map_norm_sq": self.grad_map_norm_sq,
        }


def prox_gd(
    problem: Any,
    x0: np.ndarray,
    stepsize: float,
    max_epochs: int,
    log_every: int = 1,
    stepsize_for_gmap: Optional[float] = None,
) -> Tuple[np.ndarray, ProxGDHistory]:
    """
    Proximal Gradient Descent (ProxGD) for problems of the form

        F(x) = (1/n) * sum_{i=1}^n f_i(x) + h(x),

    where f is smooth (with Lipschitz gradient) and h is convex,
    possibly non-smooth, but with an easy proximal operator.

    Parameters
    ----------
    problem :
        An object that represents the optimization problem and provides:
            - problem.n : int
            - problem.full_grad(x) -> np.ndarray
            - problem.prox_h(y, step) -> np.ndarray
            - problem.objective(x) -> float
    x0 : np.ndarray
        Initial point (must be a 1D array of length d).
    stepsize : float
        Step size for the ProxGD update (eta).
    max_epochs : int
        Number of epochs (full gradient evaluations).
    log_every : int, optional
        Log statistics every `log_every` epochs (including epoch 0).
    stepsize_for_gmap : float, optional
        Step size used in the gradient mapping definition G_eta(x).
        If None, we use `stepsize`.

    Returns
    -------
    x : np.ndarray
        Final iterate after `max_epochs` epochs.
    history : ProxGDHistory
        Logged optimization statistics.
    """
    if stepsize_for_gmap is None:
        stepsize_for_gmap = stepsize

    x = np.asarray(x0, dtype=float).copy()
    n = int(problem.n)

    # history now knows n, so plotting can use hist["sfo"] / hist["n"]
    history = ProxGDHistory(n=n)

    # cumulative number of component-gradient evaluations (SFO calls)
    # For ProxGD, each full_grad(x) uses n component gradients.
    sfo = 0

    # compute initial full gradient at x0
    g = problem.full_grad(x)
    sfo += n  # one full gradient = n SFO calls

    def log_state(epoch: int, x_vec: np.ndarray,
                  grad_vec: np.ndarray, sfo_count: int) -> None:
        """
        Compute objective and gradient mapping norm, and append to history.
        """
        obj = problem.objective(x_vec)

        # gradient mapping: G_eta(x) = (x - prox_h(x - eta * grad_f(x))) / eta
        y = x_vec - stepsize_for_gmap * grad_vec
        x_prox = problem.prox_h(y, stepsize_for_gmap)
        grad_map = (x_vec - x_prox) / stepsize_for_gmap
        grad_map_norm_sq = float(np.dot(grad_map, grad_map))

        history.append(epoch=epoch,
                       sfo=sfo_count,
                       objective=obj,
                       grad_map_norm_sq=grad_map_norm_sq)

    # log at "epoch 0" (after computing the first full gradient at x0)
    if log_every > 0 and (0 % log_every == 0):
        log_state(epoch=0, x_vec=x, grad_vec=g, sfo_count=sfo)

    # main loop: each epoch performs one ProxGD update using a full gradient
    for epoch in range(1, max_epochs + 1):
        # proximal gradient update:
        #   x_{k+1} = prox_h(x_k - eta * grad_f(x_k))
        x = problem.prox_h(x - stepsize * g, stepsize)

        # recompute full gradient at new point
        g = problem.full_grad(x)
        sfo += n  # another full gradient evaluation

        # logging
        if log_every > 0 and (epoch % log_every == 0):
            log_state(epoch=epoch, x_vec=x, grad_vec=g, sfo_count=sfo)

    return x, history

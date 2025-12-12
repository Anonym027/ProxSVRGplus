# src/optim/ProxSGD.py

from typing import Any, Optional, Tuple
import numpy as np

from .ProxGD import ProxGDHistory


def prox_sgd(
    problem: Any,
    x0: np.ndarray,
    stepsize: float,
    batch_size: int,
    max_epochs: float,
    log_every: float = 1.0,
    stepsize_for_gmap: Optional[float] = None,
    eta_prime: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ProxGDHistory]:
    """
    Proximal Stochastic Gradient Descent (ProxSGD) for problems of the form

        F(x) = (1/n) * sum_{i=1}^n f_i(x) + h(x),

    where f is smooth and h has an easy proximal operator.

    This implementation follows the spirit of:
      - S. Ghadimi, G. Lan, and H. Zhang (2016),
        "Mini-batch stochastic approximation methods for nonconvex
         stochastic composite optimization", Math. Program.

    and uses the same Problem interface as ProxGD:

        - problem.n : int
        - problem.full_grad(x)       -> np.ndarray
        - problem.grad_batch(x, idx) -> np.ndarray
        - problem.prox_h(y, step)    -> np.ndarray
        - problem.objective(x)       -> float

    Parameters
    ----------
    problem : object
        Problem instance as described above.
    x0 : np.ndarray
        Initial point.
    stepsize : float
        Base step size (eta).
    batch_size : int
        Mini-batch size.
    max_epochs : float
        Maximum number of "epochs" (in units of n component-gradient
        evaluations). That is, we stop when SFO / n >= max_epochs.
    log_every : float, optional
        Log statistics every `log_every` epochs (in terms of SFO / n),
        approximately.
    stepsize_for_gmap : float, optional
        Step size used for the gradient mapping G_eta(x). If None,
        we use `stepsize`.
    eta_prime : float, optional
        If not None and > 0, use a diminishing learning rate

            eta_k = stepsize / (1 + eta_prime * floor(total_iter / n)),

        where total_iter is the number of mini-batch updates.
        If None or 0, use a constant stepsize.
    seed : int, optional
        Random seed for mini-batch sampling. If None, use NumPy default.

    Returns
    -------
    x : np.ndarray
        Final iterate.
    history : ProxGDHistory
        Logged statistics: n, epoch (floor(SFO/n)), sfo, objective,
        grad_map_norm_sq.
    """
    if stepsize_for_gmap is None:
        stepsize_for_gmap = stepsize

    rng = np.random.default_rng(seed)

    x = np.asarray(x0, dtype=float).copy()
    n = int(problem.n)

    # History now carries n so plotting can use hist["sfo"] / hist["n"]
    history = ProxGDHistory(n=n)

    # sfo counts *component* gradients used:
    #   - mini-batch gradients (batch_size each)
    #   - plus any full gradients used for logging
    sfo = 0
    total_iter = 0  # number of mini-batch updates

    # ----- initial full gradient for logging -----
    g_full = problem.full_grad(x)
    sfo += n  # one full gradient = n SFO calls

    def log_state(epoch_value: float, x_vec: np.ndarray,
                  grad_vec: np.ndarray, sfo_count: int) -> None:
        """
        Compute objective and gradient mapping norm, then append to history.

        Here epoch_value is measured in "SFO / n" (i.e., approx data passes).
        We store floor(epoch_value) as an integer epoch counter, but the
        plotting should mainly rely on `sfo` (or sfo / n) instead of epoch.
        """
        obj = problem.objective(x_vec)

        # gradient mapping: G_eta(x) = (x - prox_h(x - eta * grad_f(x))) / eta
        y = x_vec - stepsize_for_gmap * grad_vec
        x_prox = problem.prox_h(y, stepsize_for_gmap)
        grad_map = (x_vec - x_prox) / stepsize_for_gmap
        grad_map_norm_sq = float(np.dot(grad_map, grad_map))

        # epoch stored as an integer "number of full passes" (floor)
        epoch_int = int(np.floor(epoch_value))

        history.append(
            epoch=epoch_int,
            sfo=sfo_count,
            objective=obj,
            grad_map_norm_sq=grad_map_norm_sq,
        )

    # log at "epoch 0" (after initial full gradient at x0)
    if log_every > 0:
        log_state(epoch_value=0.0, x_vec=x, grad_vec=g_full, sfo_count=sfo)

    # we will log again at approx log_every, 2*log_every, ...
    next_log_epoch = log_every

    # ----- main stochastic loop -----
    while True:
        # sample mini-batch indices with replacement
        indices = rng.integers(low=0, high=n, size=batch_size)
        g_batch = problem.grad_batch(x, indices)

        total_iter += 1

        # diminishing or constant step size
        if eta_prime is not None and eta_prime > 0.0:
            epoch_counter = total_iter // n
            eta_k = stepsize / (1.0 + eta_prime * float(epoch_counter))
        else:
            eta_k = stepsize

        # proximal stochastic gradient update
        x = problem.prox_h(x - eta_k * g_batch, eta_k)

        # mini-batch of size b uses b SFO calls
        sfo += batch_size
        epoch_now = sfo / float(n)  # SFO / n, consistent with the paper

        # decide whether to log / stop
        need_log = (log_every > 0 and epoch_now >= next_log_epoch) or (epoch_now >= max_epochs)

        if need_log:
            # compute full gradient for logging / gradient mapping
            g_full = problem.full_grad(x)
            sfo += n  # count this full gradient in SFO

            log_state(epoch_value=epoch_now, x_vec=x, grad_vec=g_full, sfo_count=sfo)

            next_log_epoch += log_every

        if epoch_now >= max_epochs:
            break

    return x, history

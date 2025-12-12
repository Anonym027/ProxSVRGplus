# src/optim/ProxSVRG.py

from typing import Any, Optional, Tuple
import numpy as np

from .ProxGD import ProxGDHistory


def prox_svrg(
    problem: Any,
    x0: np.ndarray,
    stepsize: float,
    max_epochs: float,
    inner_iters: int,
    batch_size: int,
    log_every: float = 1.0,
    stepsize_for_gmap: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ProxGDHistory]:
    """
    Proximal SVRG (ProxSVRG) for finite-sum composite problems

        F(x) = (1/n) * sum_{i=1}^n f_i(x) + h(x),

    with smooth (possibly nonconvex) component functions f_i and
    a proximable regularizer h.

    This is a mini-batch variant in the spirit of:
      - Reddi et al. (NIPS 2016), "Proximal stochastic methods for
        nonsmooth nonconvex finite-sum optimization".

    Problem interface (same as ProxGD / ProxSGD)
    --------------------------------------------
        problem.n : int
        problem.full_grad(x)       -> np.ndarray
        problem.grad_batch(x, idx) -> np.ndarray
        problem.prox_h(y, step)    -> np.ndarray
        problem.objective(x)       -> float

    Parameters
    ----------
    problem : object
        Problem instance as described above.
    x0 : np.ndarray
        Initial point.
    stepsize : float
        Step size for the inner-loop updates.
    max_epochs : float
        Maximum allowed number of epochs, measured in SFO units:
        epoch = (number of component gradients used) / n.
    inner_iters : int
        Number of inner-loop iterations per outer epoch (m).
    batch_size : int
        Mini-batch size for gradient-difference estimation (b).
    log_every : float, optional
        Log statistics every `log_every` epochs (approximate).
    stepsize_for_gmap : float, optional
        Step size used in the gradient mapping G_eta(x). If None,
        we use `stepsize`.
    seed : int, optional
        Random seed for mini-batch sampling.

    Returns
    -------
    x : np.ndarray
        Final iterate.
    history : ProxGDHistory
        Recorded statistics: epoch, sfo, objective, grad_map_norm_sq.
    """
    if stepsize_for_gmap is None:
        stepsize_for_gmap = stepsize

    rng = np.random.default_rng(seed)

    x_ref = np.asarray(x0, dtype=float).copy()  # \tilde{x}
    n = int(problem.n)

    history = ProxGDHistory(n=int(problem.n))

    # SFO counts component gradients:
    #   - each full_grad costs n,
    #   - each inner update costs 2 * batch_size (two batches).
    sfo = 0

    # ----- initial full gradient and logging -----
    full_grad_ref = problem.full_grad(x_ref)
    sfo += n

    def log_state(epoch_value: float, x_vec: np.ndarray,
                  grad_vec: np.ndarray, sfo_count: int) -> None:
        """Compute objective and gradient mapping norm, then append to history."""
        obj = problem.objective(x_vec)

        # Gradient mapping G_eta(x) = (x - prox_h(x - eta * grad_f(x))) / eta
        y = x_vec - stepsize_for_gmap * grad_vec
        x_prox = problem.prox_h(y, stepsize_for_gmap)
        grad_map = (x_vec - x_prox) / stepsize_for_gmap
        grad_map_norm_sq = float(np.dot(grad_map, grad_map))

        history.append(
            epoch=int(epoch_value),
            sfo=sfo_count,
            objective=obj,
            grad_map_norm_sq=grad_map_norm_sq,
        )

    # epoch measured in SFO / n
    epoch_now = sfo / float(n)
    if log_every > 0:
        log_state(epoch_value=epoch_now, x_vec=x_ref,
                  grad_vec=full_grad_ref, sfo_count=sfo)

    next_log_epoch = epoch_now + log_every

    # ----- outer loop -----
    while epoch_now < max_epochs:
        # start inner loop from reference point
        x = x_ref.copy()

        for _ in range(inner_iters):
            # sample mini-batch indices
            indices = rng.integers(low=0, high=n, size=batch_size)

            # compute gradient difference estimator:
            # grad_diff = grad_batch(x, I) - grad_batch(x_ref, I)
            grad_curr = problem.grad_batch(x, indices)
            grad_ref_batch = problem.grad_batch(x_ref, indices)
            grad_diff = grad_curr - grad_ref_batch

            # SVRG search direction
            v = full_grad_ref + grad_diff

            # proximal update
            x = problem.prox_h(x - stepsize * v, stepsize)

            # update SFO and epoch counter
            sfo += 2 * batch_size
            epoch_now = sfo / float(n)

            # logging if needed
            need_log = (log_every > 0 and epoch_now >= next_log_epoch) or (
                epoch_now >= max_epochs
            )
            if need_log:
                full_grad = problem.full_grad(x)
                sfo += n
                epoch_now = sfo / float(n)

                log_state(
                    epoch_value=epoch_now,
                    x_vec=x,
                    grad_vec=full_grad,
                    sfo_count=sfo,
                )

                next_log_epoch += log_every

            if epoch_now >= max_epochs:
                break

        # prepare next outer epoch: new reference point and its full gradient
        x_ref = x.copy()
        if epoch_now >= max_epochs:
            break

        full_grad_ref = problem.full_grad(x_ref)
        sfo += n
        epoch_now = sfo / float(n)

        if log_every > 0 and epoch_now >= next_log_epoch:
            log_state(
                epoch_value=epoch_now,
                x_vec=x_ref,
                grad_vec=full_grad_ref,
                sfo_count=sfo,
            )
            next_log_epoch += log_every

    return x_ref, history

# src/optim/ProxSVRGplus.py

from typing import Any, Optional, Tuple
import numpy as np

from .ProxGD import ProxGDHistory


def prox_svrg_plus(
    problem: Any,
    x0: np.ndarray,
    stepsize: float,
    max_epochs: float,
    outer_batch_size: int,    # B in the paper
    inner_batch_size: int,    # b in the paper
    epoch_length: int,        # m in the paper
    log_every: float = 1.0,
    stepsize_for_gmap: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ProxGDHistory]:
    """
    ProxSVRG+ (Algorithm 1 in the paper) for nonsmooth nonconvex finite-sum problems

        F(x) = (1/n) * sum_{i=1}^n f_i(x) + h(x).

    Compared with ProxSVRG, ProxSVRG+ replaces the full gradient at the beginning
    of each epoch by a mini-batch gradient with batch size B (denoted as g_s in
    the paper), and uses mini-batch differences of size b inside the epoch.

    Problem interface (same as ProxGD / ProxSGD / ProxSVRG)
    -------------------------------------------------------
        problem.n : int
        problem.full_grad(x)       -> np.ndarray
        problem.grad_batch(x, idx) -> np.ndarray
        problem.prox_h(y, step)    -> np.ndarray
        problem.objective(x)       -> float

    Parameters
    ----------
    problem : object
        Problem instance.
    x0 : np.ndarray
        Initial point.
    stepsize : float
        Step size η.
    max_epochs : float
        Maximum SFO budget measured in epochs: epoch = SFO / n.
    outer_batch_size : int
        B in Algorithm 1: batch size for the outer gradient g_s.
        When B = n, this reduces (up to parameters) to ProxSVRG.
    inner_batch_size : int
        b in Algorithm 1: mini-batch size for gradient differences.
    epoch_length : int
        m in Algorithm 1: number of inner-loop iterations per epoch.
    log_every : float, optional
        Log statistics every `log_every` epochs (in SFO / n units).
    stepsize_for_gmap : float, optional
        Step size used in the gradient mapping G_eta(x). If None, use stepsize.
    seed : int, optional
        Random seed for mini-batch sampling.

    Returns
    -------
    x : np.ndarray
        Final iterate (the last reference point x_ref).
    history : ProxGDHistory
        Recorded statistics: n, epoch, sfo, objective, grad_map_norm_sq.
    """
    if stepsize_for_gmap is None:
        stepsize_for_gmap = stepsize

    rng = np.random.default_rng(seed)

    x_ref = np.asarray(x0, dtype=float).copy()  # \tilde{x}_0
    n = int(problem.n)

    outer_batch_size = min(outer_batch_size, n)
    inner_batch_size = max(1, inner_batch_size)

    # The history object now includes n, which simplifies plotting SFO/n later.
    history = ProxGDHistory(n=int(problem.n))

    # SFO: number of component gradients used so far.
    sfo = 0

    def log_state(epoch_value: float, x_vec: np.ndarray,
                  grad_vec: np.ndarray, sfo_count: int) -> None:
        """Compute objective and gradient mapping norm, then append to history."""
        obj = problem.objective(x_vec)

        # Gradient mapping G_eta(x) = (x - prox_h(x - eta * grad_f(x))) / eta
        y = x_vec - stepsize_for_gmap * grad_vec
        x_prox = problem.prox_h(y, stepsize_for_gmap)
        grad_map = (x_vec - x_prox) / stepsize_for_gmap
        grad_map_norm_sq = float(np.dot(grad_map, grad_map))

        epoch_int = int(np.floor(epoch_value))

        history.append(
            epoch=epoch_int,
            sfo=sfo_count,
            objective=obj,
            grad_map_norm_sq=grad_map_norm_sq,
        )

    # --- initial logging: full gradient at x0 ---
    full_grad0 = problem.full_grad(x_ref)
    sfo += n                      # full_grad -> n SFO calls
    epoch_now = sfo / float(n)

    if log_every > 0:
        log_state(epoch_value=epoch_now, x_vec=x_ref,
                  grad_vec=full_grad0, sfo_count=sfo)

    next_log_epoch = epoch_now + log_every

    # ============= Outer loop over epochs s = 1,2,... ============
    while epoch_now < max_epochs:
        # Algorithm 1, Line 3: x_s^0 = \tilde{x}_{s-1}
        x = x_ref.copy()

        # Line 4: g_s = (1/B) sum_{j in I_B} grad f_j(\tilde{x}_{s-1})
        idx_B = rng.integers(low=0, high=n, size=outer_batch_size)
        g_s = problem.grad_batch(x_ref, idx_B)
        sfo += outer_batch_size         # B component gradients
        epoch_now = sfo / float(n)

        # -------------- Inner loop: t = 1,...,m --------------
        for _ in range(epoch_length):
            # sample mini-batch I_b
            idx_b = rng.integers(low=0, high=n, size=inner_batch_size)

            # grad_batch(x, I_b) - grad_batch(x_ref, I_b)
            grad_curr = problem.grad_batch(x, idx_b)
            grad_ref_batch = problem.grad_batch(x_ref, idx_b)
            grad_diff = grad_curr - grad_ref_batch

            # v_{t-1}^s = 1/b sum (grad_curr - grad_ref_batch) + g_s
            v = grad_diff + g_s

            # proximal update: x_t^s = prox_{η h}(x_{t-1}^s - η v_{t-1}^s)
            x = problem.prox_h(x - stepsize * v, stepsize)

            # inner update uses two mini-batch gradients -> 2 * inner_batch_size SFO
            sfo += 2 * inner_batch_size
            epoch_now = sfo / float(n)

            # Logging: use the full gradient to calculate the gradient mapping.
            need_log = (log_every > 0 and epoch_now >= next_log_epoch) or (
                epoch_now >= max_epochs
            )
            if need_log:
                full_grad = problem.full_grad(x)
                sfo += n                 # full_grad -> n SFO calls
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

        # Line 9: \tilde{x}_s = x_m^s
        x_ref = x.copy()

        if epoch_now >= max_epochs:
            break

        # After an epoch, another full gradient can be computed for the next log.
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


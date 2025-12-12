# src/problems/nn_pca.py

import numpy as np


class NNPCAProblem:
    """
    Nonnegative Principal Component Analysis (NN-PCA) problem:

        minimize   F(x) = f(x) + h(x)
        where
            f(x) = (1/n) * sum_{i=1}^n f_i(x),
            f_i(x) = -0.5 * (x^T z_i)^2,
            h(x) = I_{C}(x),

        C = { x in R^d : x >= 0, ||x||_2 <= 1 }.

    Z is an (n, d) data matrix.
    We (re)normalize each row to have L2-norm 1, as in the paper.
    """

    def __init__(self, Z: np.ndarray, normalize_rows: bool = True):
        """
        Parameters
        ----------
        Z : np.ndarray, shape (n, d)
            Each row is a data point z_i.
        normalize_rows : bool, default True
            If True, enforce ||z_i||_2 = 1 for all i.
        """
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array of shape (n, d).")

        self.Z = Z
        self.n, self.d = Z.shape

        # Normalize rows so that ||z_i||_2 = 1 (as done in the paper)
        if normalize_rows:
            self._normalize_rows()

        # Pre-compute Lipschitz constant L for grad f:
        # grad f(x) = -(1/n) * (Z^T Z) x,
        # so L = lambda_max( (1/n) * Z^T Z ).
        self.L = self._compute_L()

    # ===== helpers for normalization and Lipschitz constant =====

    def _normalize_rows(self):
        """Normalize each row z_i so that ||z_i||_2 = 1."""
        norms = np.linalg.norm(self.Z, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.Z = self.Z / norms

    def _compute_L(self) -> float:
        """
        Compute Lipschitz constant L of grad f:

            grad f(x) = -(1/n) * (Z^T Z) x
            => L = lambda_max( (1/n) * Z^T Z ).
        """
        S_bar = (self.Z.T @ self.Z) / self.n   # d x d
        # S_bar is symmetric; use eigvalsh for numerical stability
        eigvals = np.linalg.eigvalsh(S_bar)
        return float(eigvals[-1])

    # ===== basic components used by ProxGD / ProxSVRG / ProxSVRG+ =====

    def grad_i(self, x: np.ndarray, i: int) -> np.ndarray:
        """
        Component gradient grad f_i(x) for NN-PCA:

            f_i(x) = -0.5 * (x^T z_i)^2
            grad f_i(x) = - (x^T z_i) * z_i
        """
        z_i = self.Z[i]
        inner = float(np.dot(x, z_i))      # x^T z_i
        return -inner * z_i                # shape (d,)

    def grad_batch(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Average gradient over a minibatch:

            (1 / |I|) sum_{i in I} grad f_i(x).

        This is convenient for ProxSGD / ProxSVRG / ProxSVRG+.
        """
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("indices must be a 1D array of ints.")
        Z_batch = self.Z[indices]         # shape (b, d)
        Zx = Z_batch @ x                  # shape (b,)
        # grad_i(x) = - (x^T z_i) z_i
        # average over batch: (-1 / b) * Z_batch^T (Zx)
        b = len(indices)
        coeff = -Zx / b
        return Z_batch.T @ coeff          # shape (d,)

    def full_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Full gradient grad f(x):

            grad f(x) = (1/n) * sum_{i=1}^n grad f_i(x).
        """
        Zx = self.Z @ x                   # shape (n,)
        coeff = -Zx / self.n             # shape (n,)
        return self.Z.T @ coeff          # shape (d,)

    def prox_h(self, y: np.ndarray, step: float) -> np.ndarray:
        """
        Proximal operator of h(x) = I_C(x), i.e. projection onto:

            C = { x : x >= 0, ||x||_2 <= 1 }.

        Step size 'step' is not used, but kept for API consistency.
        """
        # enforce nonnegativity
        x = np.maximum(y, 0.0)
        # project onto L2 ball of radius 1
        norm = float(np.linalg.norm(x))
        if norm == 0.0 or norm <= 1.0:
            return x
        return x / norm

    def objective(self, x: np.ndarray) -> float:
        """
        Full objective F(x) = f(x) + h(x).

        For h(x) = I_C(x), h(x) = 0 if x in C, +infinity otherwise.
        """
        Zx = self.Z @ x
        f_val = -0.5 * float(np.mean(Zx ** 2))

        # check feasibility for C
        if np.any(x < -1e-12):
            return float("inf")
        x_pos = np.maximum(x, 0.0)
        if np.linalg.norm(x_pos) > 1.0 + 1e-8:
            return float("inf")

        return f_val

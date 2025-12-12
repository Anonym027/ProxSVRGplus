# src/problems/datasets.py

import os
import numpy as np

from .nn_pca import NNPCAProblem

try:
    from sklearn.datasets import load_svmlight_file
except ImportError as exc:
    load_svmlight_file = None


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row of matrix to have L2 norm 1.
    Rows with zero norm are left unchanged.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _load_libsvm_as_dense(path: str, dtype=np.float32) -> np.ndarray:
    """
    Load a LIBSVM-format file and return a dense feature matrix.

    Parameters
    ----------
    path : str
        Path to the LIBSVM file, for example 'data/a9a' or 'data/mnist'.
    dtype : data type
        Data type of the returned dense matrix.

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_features)
    """
    if load_svmlight_file is None:
        raise ImportError(
            "scikit-learn is required to load LIBSVM files. "
            "Install it with 'pip install scikit-learn'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"LIBSVM file not found: {path}")

    sparse_features, labels = load_svmlight_file(path)
    dense_features = sparse_features.astype(dtype).toarray()
    return dense_features


def make_a9a_nnpca_problem(path: str) -> NNPCAProblem:
    """
    Build an NNPCAProblem from the a9a dataset.

    Parameters
    ----------
    path : str
        Path to the a9a file, for example '.../ProxSVRGplus/data/a9a'.

    Returns
    -------
    problem : NNPCAProblem
    """
    features = _load_libsvm_as_dense(path)
    z_matrix = _row_normalize(features)
    return NNPCAProblem(z_matrix)


def make_mnist_nnpca_problem(path: str) -> NNPCAProblem:
    """
    Build an NNPCAProblem from the MNIST dataset in LIBSVM format.

    Parameters
    ----------
    path : str
        Path to the MNIST file, for example '.../ProxSVRGplus/data/mnist'.

    Returns
    -------
    problem : NNPCAProblem
    """
    features = _load_libsvm_as_dense(path)
    z_matrix = _row_normalize(features)
    return NNPCAProblem(z_matrix)

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance


def _normalize(matrix_1: npt.NDArray[np.float_],
               matrix_2: npt.NDArray[np.float_],
               max_value: float) -> None:
    """
    Inplace column-based normalization of the matrices so that :
        - columns with all values smaller than <max_value> in both matrices are left untouched
        - other columns are normalized by a constant value so that values fall within the range [0, <max_value>].

    Args:
        matrix_1: first matrix 2D, either numpy
        matrix_2: second matrix 2D, either numpy
        max_value: maximum allowed value in each column
    """
    # scaling factors
    scaling_factors = np.maximum(matrix_1.max(axis=0), matrix_2.max(axis=0))

    # identify columns to normalize
    columns_need_scaling = scaling_factors > max_value

    # rescale columns
    matrix_1[:, columns_need_scaling] *= max_value / scaling_factors[columns_need_scaling]
    matrix_2[:, columns_need_scaling] *= max_value / scaling_factors[columns_need_scaling]


def kanto_1d(matrix_1: npt.NDArray[np.float_], 
                matrix_2: npt.NDArray[np.float_]) -> float:
    """
    Computes the kantorovich distance between two 1D distributions.
    Wrapper over the scipy wasserstein_distance.

    Args:
        sample1: first 1d distribution
        sample2: second 1d distribution, to be compared to sample1

    Returns:
        The kantorovich distance between sample1 and sample2.
    """
    assert matrix_1.ndim == 1 and matrix_2.ndim == 1
    return wasserstein_distance(matrix_1, matrix_2)


def multigene_kanto1d(matrix_1: npt.NDArray[np.float_],
                      matrix_2: npt.NDArray[np.float_],
                      reduce: Callable[[npt.NDArray[np.float_]], float] | None = np.sum,
                      normalize: bool = False,
                      max_value: float = 10.) -> float | npt.NDArray[np.float_]:
    """
    Computes the sum of a series of kantorovich distances on cpu.

    Args:
        sample1: 2D distribution of cells x genes
        sample2: 2D distribution of cells x genes
        reduce (optional): function to use to reduce the per-gene distances. Defaults to sum.

    Returns:
        the sum of gene-wise kantorovich distances
    """
    assert matrix_1.ndim == 2 and matrix_2.ndim == 2, "expects 2D arrays"
    assert matrix_1.shape[1] == matrix_2.shape[1], "nb of distributions should be the same"

    if normalize:
        _normalize(matrix_1, matrix_2, max_value)

    nb_genes = matrix_1.shape[1]
    result = np.array([kanto_1d(matrix_1[:, i], matrix_2[:, i]) for i in range(nb_genes)])
    
    if reduce is not None:
        # sum on gpu and return
        return float(reduce(result))

    else:
        return result

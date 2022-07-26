from typing import Tuple

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike


def add_risk_free_rate_to_params(
    mu: ArrayLike, sigma: ArrayLike, C: ArrayLike, starting_value, risk_free_rate: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given geometric Brownian motion parameters, add a risk-free rate.
    """
    new_mu = np.concatenate([mu, [risk_free_rate]])
    new_sigma = np.concatenate([sigma, [0]])
    new_C = scipy.linalg.block_diag(C, 1.0)
    new_start = np.concatenate([starting_value, [1.0]])
    return new_mu, new_sigma, new_C, new_start

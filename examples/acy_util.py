from typing import Tuple

import numpy as np
import numpy.testing
import xarray as xr
from numpy.typing import ArrayLike

from qml.tools.monte_carlo.brownian_motion import (
    ASSET_DIMENSION,
    C_MAT,
    MU,
    SIGMA,
    START_VALUE,
    TIME_STEP_DIMENSION,
    TIME_STEPS,
)
from qml.tools.monte_carlo.brownian_motion_util import add_risk_free_rate_to_params

"""
Utility functions for the experiments in the paper

Alexander, Siddharth, Thomas F. Coleman, and Yuying Li.
"Minimizing CVaR and VaR for a portfolio of derivatives." Journal of Banking & Finance 30.2 (2006): 583-605.
"""

ACY_mu = np.array([0.1091, 0.0619, 0.0279, 0.0649])
ACY_initial_price = np.array([100.0, 50, 30, 100])

ACY_cov = np.array(
    [
        [0.2890, 0.0690, 0.0080, 0.0690],
        [0.0690, 0.1160, 0.0200, 0.0610],
        [0.0080, 0.0200, 0.0220, 0.0130],
        [0.0690, 0.0610, 0.0130, 0.0790],
    ]
)
np.testing.assert_allclose(ACY_cov, ACY_cov.transpose())


def compute_std_correlation_C_from_cov(cov: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = np.asarray(cov)
    num_assets = len(cov)
    std = np.sqrt(np.diag(cov))
    correlation = cov / std.reshape(1, num_assets) / std.reshape(num_assets, 1)
    C = np.linalg.cholesky(correlation)
    corr_diag = np.diag(correlation)
    assert np.allclose(corr_diag, np.ones_like(corr_diag))
    return std, correlation, C


def acy_params_with_risk_free_rate(risk_free_rate: float = 0.01, terminal_time: float = 62.5 / 250) -> xr.Dataset:
    std, correlation, C = compute_std_correlation_C_from_cov(ACY_cov)
    mu, std, C, start = add_risk_free_rate_to_params(ACY_mu, std, C, ACY_initial_price, risk_free_rate=risk_free_rate)
    return xr.Dataset(
        {
            MU: xr.DataArray(mu, dims=(ASSET_DIMENSION,)),
            SIGMA: xr.DataArray(std, dims=(ASSET_DIMENSION,)),
            C_MAT: xr.DataArray(C, dims=(ASSET_DIMENSION, ASSET_DIMENSION)),
            START_VALUE: xr.DataArray(start, dims=(ASSET_DIMENSION,)),
            TIME_STEPS: xr.DataArray([terminal_time], dims=(TIME_STEP_DIMENSION,)),
        }
    )

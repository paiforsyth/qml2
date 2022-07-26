import numpy as np
import numpy.testing
import xarray as xr
from numpy.typing import ArrayLike

from qml.tools.jax_util.types import Array
from qml.tools.monte_carlo.brownian_motion import TIME_STEP_DIMENSION, xr_vector_geometric_brownian_motion

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
ACY_std = np.sqrt(np.diag(ACY_cov))
ACY_correlation = ACY_cov / ACY_std.reshape(1, 4) / ACY_std.reshape(4, 1)
ACY_C = np.linalg.cholesky(ACY_correlation)


def generate_asset_paths(n_paths: int, time_steps: ArrayLike, key: Array) -> xr.DataArray:
    return xr_vector_geometric_brownian_motion(
        num_paths=n_paths,
        start_value=ACY_initial_price,
        mu=ACY_mu,
        sigma=ACY_std,
        C=ACY_C,
        time_steps=time_steps,
        key=key,
    )


def generate_terminal_values(n_paths: int, terminal_time: float, key: Array) -> xr.DataArray:
    result = generate_asset_paths(n_paths=n_paths, time_steps=np.array([terminal_time]), key=key)
    return result[{TIME_STEP_DIMENSION: 0}]

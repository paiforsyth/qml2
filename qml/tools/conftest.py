import numpy as np
import pytest
import scipy
import scipy.stats
import xarray as xr

from qml.tools.portfolio_optimize.cvx_optimize import efficient_frontier_non_smooth
from qml.tools.portfolo_plot.cvar_min_plot import add_risk_return_percentage_to_ds


@pytest.fixture(scope="package")
def example_required_return() -> np.ndarray:
    return np.array([1.25, 1.5, 1.75])


@pytest.fixture(scope="package")
def example_efficient_frontier(example_required_return: np.ndarray) -> xr.Dataset:
    samples = 100
    normal = scipy.stats.multivariate_normal(
        cov=np.array([[1.0, 0.0], [0.0, 10.0]]), mean=np.array([1.0, 2.0]), seed=42
    )
    frontier = efficient_frontier_non_smooth(
        confidence_level=np.array(0.95),
        instrument_price=np.array([1.0, 1.0]),
        instrument_payoff=normal.rvs(size=samples),
        required_returns=example_required_return,
    )
    frontier = add_risk_return_percentage_to_ds(frontier)
    return frontier

import jax.numpy as jnp
import numpy as np
import numpy.testing
import scipy
import scipy.stats
from cvxpy import OPTIMAL

from qml.tools.portfolio_optimize.cvx_optimize import (
    TRAINING_RETURN,
    cvar_minimize_non_smooth_problem,
    efficient_frontier_non_smooth,
)


def test_superior_asset():
    """
    CVAR minimization over a universe of a risk-free instrument and a risky instrument
    with a worse return should result in full investment in the risk-free instrument
    """
    cprob, _ = cvar_minimize_non_smooth_problem(
        confidence_level=jnp.array(0.95),
        instrument_price=jnp.array([1.0, 1.0]),
        instrument_payoff=jnp.array([[2.0, 2.0, 2.0], [0, -1.0, -3.0]]),
        required_return=jnp.array([1.0]),
        minimum_holding=np.array([0.0, 0.0]),
    )
    prob = cprob.prob
    prob.solve(verbose=True, solver="CVXOPT")
    assert prob.status == OPTIMAL
    np.testing.assert_allclose(cprob.x.value, [1.0, 0.0], rtol=1e-6, atol=1e-6)


def test_frontier():
    """
    When creating an efficient frontier, every point on the frontier should meet its required return
    """
    samples = 100
    eps = 1e-6
    normal = scipy.stats.multivariate_normal(
        cov=np.array([[1.0, 0.0], [0.0, 10.0]]), mean=np.array([1.0, 2.0]), seed=42
    )
    required_return = np.array([1.25, 1.5, 1.75])
    frontier = efficient_frontier_non_smooth(
        confidence_level=np.array(0.95),
        instrument_price=np.array([1.0, 1.0]),
        instrument_payoff=normal.rvs(size=samples).transpose(),
        required_returns=required_return,
    )
    assert np.all(frontier[TRAINING_RETURN].values >= (required_return - eps))

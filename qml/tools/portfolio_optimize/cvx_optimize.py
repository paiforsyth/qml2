import multiprocessing
from typing import Optional, Tuple

import attr
import cvxpy as cp
import numpy as np
import xarray as xr
from cvxpy import OPTIMAL
from numpy.typing import ArrayLike

from qml.tools.monte_carlo.brownian_motion import START_VALUE


@attr.s(frozen=True, auto_attribs=True)
class CVaRProblem:
    prob: cp.Problem
    x: cp.Variable
    alpha: cp.Variable


@attr.s(frozen=True, auto_attribs=True)
class CVaRAuxData:
    V_delta_bar: np.ndarray
    price: np.ndarray


def cvar_minimize_non_smooth_problem(
    confidence_level: ArrayLike,
    instrument_price: ArrayLike,
    instrument_payoff: ArrayLike,
    required_return: ArrayLike | cp.Parameter,
    minimum_holding: Optional[ArrayLike] = None,
    maximum_holding: Optional[ArrayLike] = None,
    budget: ArrayLike = np.array(1),
) -> Tuple[CVaRProblem, CVaRAuxData]:
    """

    Args:
        confidence_level: Scalar. Confidence level for CVAR minimization.  Common choices are 0.95 and 0.99
        instrument_price: (num_instruments,) vector giving purchase price of every instrument
        instrument_payoff: (num_simulations, num_instruments) array giving payoff of each instrument in each simulation.
        minimum_holding:  (num_instruments,) array giving minimum holding of each instrument
        maximum_holding:  (num_instruments,) array giving maximum holding of each instrument
        required_return: Scalar.  Required return for the portfolio
        budget:  Scalar, available budget for portfolio
    Returns:
        structure containing a cvxpy problem representing the non-smooth cvar minimization
    Gotchas:
        - matrices containing infinite values can cause CVXPY's underlying solvers to fail with cryptic errors
    References:
        Alexander, Siddharth, Thomas F. Coleman, and Yuying Li.
        "Minimizing CVaR and VaR for a portfolio of derivatives." Journal of Banking & Finance 30.2 (2006): 583-605.
    """
    instrument_payoff = np.asarray(instrument_payoff)
    instrument_payoff = instrument_payoff.transpose()
    num_instruments, num_simulations = instrument_payoff.shape

    # Define constants. use names from paper
    beta = np.asarray(confidence_level)
    m = np.asarray(num_simulations)
    V0 = np.asarray(instrument_price)
    V_delta = np.asarray(instrument_payoff - V0.reshape(num_instruments, 1))
    V_delta_bar = np.mean(V_delta, axis=-1)
    b = np.asarray(budget)
    r = required_return if isinstance(required_return, cp.Parameter) else np.asarray(required_return)
    l = np.asarray(minimum_holding) if minimum_holding is not None else None
    u = np.asarray(maximum_holding) if maximum_holding is not None else None

    # define variables
    x = cp.Variable(shape=num_instruments)
    alpha = cp.Variable()
    obj = cp.Minimize(alpha + 1 / (m * (1 - beta)) * cp.sum(cp.pos(-V_delta.transpose() @ x - alpha)))
    budget_constraint = [V0 @ x <= b]
    return_constraint = [V_delta_bar @ x >= r]
    min_constraint = [l <= x] if l is not None else []
    max_constraint = [x <= u] if u is not None else []
    constraints = budget_constraint + return_constraint + min_constraint + max_constraint
    prob = cp.Problem(obj, constraints)
    return CVaRProblem(prob, x=x, alpha=alpha), CVaRAuxData(V_delta_bar=V_delta_bar, price=V0)


TRAINING_RETURN = "training_mean_return"
TRAINING_CVAR = "training_cvar"
HOLDINGS = "holdings"
COST = "cost"

INSTRUMENT_DIMENSION = "instrument"
FRONTIER_DIMENSION = "frontier_dimension"


def cvar_minimize_non_smooth_solve(cprob: CVaRProblem, cdata: CVaRAuxData) -> xr.Dataset:
    problem = cprob.prob
    problem.solve()
    assert problem.status == OPTIMAL
    holding = cprob.x.value
    ret = cdata.V_delta_bar @ holding
    cvar = problem.value
    return xr.Dataset(
        {
            TRAINING_RETURN: np.array(ret),
            TRAINING_CVAR: np.array(cvar),
            HOLDINGS: xr.DataArray(holding, dims=INSTRUMENT_DIMENSION),
            COST: xr.DataArray(sum(holding * cdata.price)),
        }
    )


@attr.s(auto_attribs=True)
class _SolveHelper:
    problem: CVaRProblem
    problem_data: CVaRAuxData
    return_parameter: cp.Parameter

    def __call__(self, return_value: float) -> xr.Dataset:
        self.return_parameter.value = return_value
        return cvar_minimize_non_smooth_solve(cprob=self.problem, cdata=self.problem_data)


def efficient_frontier_non_smooth(
    confidence_level: ArrayLike,
    instrument_price: ArrayLike,
    instrument_payoff: ArrayLike,
    required_returns: ArrayLike,
    processess: Optional[int] = None,
    minimum_holding: Optional[ArrayLike] = None,
    maximum_holding: Optional[ArrayLike] = None,
    budget: ArrayLike = np.array(1),
) -> xr.Dataset:

    """
    Args:
        confidence_level: Scalar. Confidence level for CVAR minimization.  Common choices are 0.95 and 0.99
        instrument_price: (num_instruments,) vector giving purchase price of every instrument
        instrument_payoff: (num_instruments, num_simulations) array giving payoff of each instrument in each simulation.
        minimum_holding:  (num_instruments,) array giving minimum holding of each instrument
        maximum_holding:  (num_instruments,) array giving maximum holding of each instrument
        required_returns: (num_points):  Required return for each point on thje
        budget:  Scalar, available budget for portfolio
    Returns:
        xarray dataset describing efficient frontier
    Gotchas:
        - matrices containing infinite values can cause CVXPY's underlying solvers to fail with cryptic errors
    References:
        Alexander, Siddharth, Thomas F. Coleman, and Yuying Li.
        "Minimizing CVaR and VaR for a portfolio of derivatives." Journal of Banking & Finance 30.2 (2006): 583-605.
    """
    return_parameter = cp.Parameter()
    cprob, data = cvar_minimize_non_smooth_problem(
        confidence_level=confidence_level,
        instrument_price=instrument_price,
        instrument_payoff=instrument_payoff,
        required_return=return_parameter,
        minimum_holding=minimum_holding,
        maximum_holding=maximum_holding,
        budget=budget,
    )
    required_returns = np.asarray(required_returns)
    pool = multiprocessing.Pool(processes=processess)
    solutions = pool.map(
        _SolveHelper(problem=cprob, problem_data=data, return_parameter=return_parameter), required_returns
    )
    ds = xr.concat(solutions, dim=FRONTIER_DIMENSION)
    ds[START_VALUE] = xr.DataArray(instrument_price, dims=INSTRUMENT_DIMENSION)
    return ds

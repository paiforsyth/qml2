from typing import Optional

import attr
import cvxpy as cp
import numpy as np


@attr.s(frozen=True, auto_attribs=True)
class CVaRProblem:
    prob: cp.Problem
    x: cp.Variable
    alpha: cp.Variable


def cvar_minimize_non_smooth(
    confidence_level: np.ndarray,
    instrument_price: np.ndarray,
    instrument_payoff: np.ndarray,
    required_return: np.ndarray,
    minimum_holding: Optional[np.ndarray] = None,
    maximum_holding: Optional[np.ndarray] = None,
    budget: np.ndarray = np.array(1),
) -> CVaRProblem:
    """

    Args:
        confidence_level: Scalar. Confidence level for CVAR minimization.  Common choices are 0.95 and 0.99
        instrument_price: (num_instruments,) vector giving purchase price of every instrument
        instrument_payoff: (num_instruments, num_simulations) array giving payoff of each instrument in each simulation.
        minimum_holding:  (num_instruments,) array giving minimum holding of each instrument
        maximum_holding:  (num_instruments,) array giving maximum holding of each instrument
        required_return: Scalar.  Required return for the portfolio
        budget:  Scalar, available budget for portfolio
    Returns:
        a cvxpy problem representing the non-smooth cvar minimization
    Gotchas:
        - matrices containing infinite values can cause CVXPY's underlying solvers to fail with cryptic errors
    References:
        Alexander, Siddharth, Thomas F. Coleman, and Yuying Li.
        "Minimizing CVaR and VaR for a portfolio of derivatives." Journal of Banking & Finance 30.2 (2006): 583-605.
    """
    num_instruments = instrument_price.shape[0]
    num_simulations = instrument_payoff.shape[1]

    # Define constants. use names from paper
    beta = np.asarray(confidence_level)
    m = np.asarray(num_simulations)
    V0 = np.asarray(instrument_price)
    V_delta = np.asarray(instrument_payoff - instrument_price.reshape(num_instruments, 1))
    V_delta_bar = np.mean(V_delta, axis=-1)
    b = np.asarray(budget)
    r = np.asarray(required_return)
    l = np.asarray(minimum_holding) if minimum_holding is not None else None
    u = np.asarray(maximum_holding) if maximum_holding is not None else None

    # define variables
    x = cp.Variable(shape=num_instruments)
    alpha = cp.Variable()
    obj = cp.Minimize(alpha + 1 / (m * (1 - beta)) * cp.sum(cp.pos(-V_delta.transpose() @ x - alpha)))
    budget_constraint = [V0 @ x <= b]
    return_constraint = [V_delta_bar @ x >= r]
    min_constraint = [l <= x] if l is not None else []
    max_constriant = [x <= u] if u is not None else []
    constraints = budget_constraint + return_constraint + min_constraint + max_constriant
    prob = cp.Problem(obj, constraints)
    return CVaRProblem(prob, x=x, alpha=alpha)

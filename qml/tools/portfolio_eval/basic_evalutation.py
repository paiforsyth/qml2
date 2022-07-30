import functools
from typing import Sequence

import numpy as np
import scipy.stats
import xarray as xr
from numpy.typing import ArrayLike

STD = "std"
CVaR = "CVaR"
VaR = "VaR"
LOWER_CI_SUFFIX = "_ci_lower"
UPPER_CI_SUFFIX = "_ci_upper"


def value_at_risk(a: ArrayLike, axis: int, level: float, keepdims: bool = False) -> np.ndarray:
    """
    Args:
        a: array of PnL values
        axis: axis along which to compute VaR
        level: confidence leve of the VaR to compute.  Common choices are 0.05 and 0.1
        keepdims: whether to retain the reduced dimension with size 1 in the output array
    Returns:
        array of VaR values

    Note on conventions:  There seems to be some ambiguity in the definitions used in the literature.  What
    some people call VaR at level p, others call VaR at level (1-p).  Here we adopt the definitions of
    Cont, Rama, Romain Deguest, and Giacomo Scandolo.
    "Robustness and sensitivity analysis of risk measurement procedures." Quantitative finance 10.6 (2010): 593-606.
    In which alpha var is defined as the negative of the alpha quantile of the distribution
    """
    a2 = np.asarray(a)
    result = -np.quantile(a2, axis=axis, q=level, keepdims=keepdims)
    return np.array(result)


def conditional_value_at_risk(a: ArrayLike, axis: int, level: float) -> np.ndarray:
    """

    Args:
        a: array of PnL values
        axis: axis along which to compute CVaR
        level: confidence leve of the CVaR to compute.  Common choices are 0.05 and 0.1
        keepdims: whether to retain the reduced dimension with size 1 in the output array
    Returns:
        array of CVaR values
    """
    var = value_at_risk(a, axis=axis, level=level, keepdims=True)
    a = np.asarray(a)
    result: np.ndarray = var.squeeze(axis) + 1 / level * np.mean(np.maximum(-a - var, 0.0), axis=axis)
    return result


def standard_eval(
    sample_values: xr.DataArray,
    dim: str,
    ci_confidence_level: float = 0.05,
    var_levels: float | Sequence[float] = 0.05,
    include_ci: bool = True,
) -> xr.Dataset:
    result = xr.Dataset()
    dim_axis = sample_values.dims.index(dim)
    reduced_dims = list(sample_values.dims)
    reduced_dims.remove(dim)
    raw = sample_values.values
    statistic_dictionary = {STD: np.std}
    if isinstance(var_levels, float):
        var_levels = [var_levels]
    for var_level in var_levels:
        statistic_dictionary[CVaR + "_" + str(var_level)] = functools.partial(
            conditional_value_at_risk, level=var_level
        )
        statistic_dictionary[VaR + "_" + str(var_level)] = functools.partial(value_at_risk, level=var_level)
    for name, func in statistic_dictionary.items():
        result[name] = xr.DataArray(func(raw, axis=dim_axis), dims=reduced_dims)
        if include_ci:
            ci = scipy.stats.bootstrap(
                (raw,), statistic=func, axis=dim_axis, confidence_level=ci_confidence_level, method="basic"
            )
            result[name + LOWER_CI_SUFFIX] = xr.DataArray(ci.confidence_interval.low, dims=reduced_dims)
            result[name + UPPER_CI_SUFFIX] = xr.DataArray(ci.confidence_interval.high, dims=reduced_dims)
    return result

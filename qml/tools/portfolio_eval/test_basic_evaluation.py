import scipy.stats
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import xarray as xr

from qml.tools.portfolio_eval.basic_evalutation import LOWER_CI_SUFFIX, STD, UPPER_CI_SUFFIX, standard_eval, \
    value_at_risk, conditional_value_at_risk


def test_value_at_risk():
    a = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    assert value_at_risk(a, level=0.3, axis=0) == pytest.approx(0.3)


@given(level=st.floats(min_value=0.4, max_value=0.6), std=st.floats(min_value=1.0, max_value=3.0))
@settings(max_examples=5)
def test_var_cvar_normal(level:float, std: float):
    """
    There is an analytic formula for var and cvar for a truncated normal distribution.  We should respect this formula
    """
    n_samples = 10000
    generator = np.random.default_rng(seed=42)
    data= std*generator.standard_normal(size=n_samples)
    var = value_at_risk(data, axis=0, level=level)
    cvar = conditional_value_at_risk(data, axis=0, level=level)
    norm = scipy.stats.norm()
    expected_var = scipy.stats.norm(scale=std).ppf(level)
    beta = expected_var/std
    expected_cvar = -std*norm.pdf(beta)/(norm.cdf(beta))
    assert var == pytest.approx(expected_var, rel=0.1, abs=0.1)
    assert cvar == pytest.approx(expected_cvar,rel=0.1, abs=0.1)


def test_standard_eval():
    n_samples = 1000
    dim_name = "dim"
    generator = np.random.default_rng(seed=42)
    data = xr.DataArray(generator.standard_normal(size=n_samples), dims=(dim_name,))
    result = standard_eval(data, dim=dim_name)
    std = result[STD]
    assert 0.96 <= std
    assert std <= 1.04
    assert result[STD + LOWER_CI_SUFFIX] <= std
    assert std <= result[STD + UPPER_CI_SUFFIX]

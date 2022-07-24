import jax
import jax.numpy as jnp
import numpy.testing
from hypothesis import given
from hypothesis import strategies as st

from qml.tools.portfolio_data.black_scholes_call_portfolio import (
    black_scholes_option_portfolio,
)


@given(
    risk_free_rate=st.floats(min_value=0, max_value=1.0),
    underlying_price=st.floats(min_value=0.5, max_value=10.0),
    time=st.floats(min_value=0.0, max_value=100),
)
def test_risk_free(
    risk_free_rate: float,
    underlying_price: float,
    time: float,
):
    """
    We should be able to simulate a portfolio of the risk free asset
    """
    seed = 4
    key = jax.random.PRNGKey(seed)
    portfolio_value = black_scholes_option_portfolio(
        num_paths=1,
        mu=jnp.array([risk_free_rate]),
        sigma=jnp.array([0.0]),
        C=jnp.array([1.0]),
        time_steps=jnp.array([time]),
        key=key,
        strike=jnp.array([0.0]),
        maturities=2
        * jnp.array([time]),  # just need expiry to be beyond final timestep
        underlying=jnp.array([0]),
        risk_free_rate=jnp.array([risk_free_rate]),
        initial_price=jnp.array([underlying_price]),
    )
    portfolio_value = float(portfolio_value)
    numpy.testing.assert_allclose(
        portfolio_value,
        underlying_price * jnp.exp(risk_free_rate * time),
        atol=1e-5,
        rtol=1e-5,
    )

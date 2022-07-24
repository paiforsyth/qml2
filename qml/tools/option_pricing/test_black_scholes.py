from hypothesis import given, strategies as st
import jax.numpy as jnp
from qml.tools.option_pricing.black_scholes import price_call
import numpy as np
import numpy.testing

@given(
    underlying_price = st.floats(min_value=0, max_value=10, width=32),
    strike= st.floats(min_value=0, max_value=10, width=32),
    risk_free_rate = st.floats(min_value=0.0, max_value=0.5,width=32 ),
    vol = st.floats(min_value=0.0, max_value=0.5, width=32)
)
def test_price_at_maturity(
underlying_price: float,
strike:float,
risk_free_rate: float,
vol:float
):
   price =  price_call(
       underlying_price=jnp.array([underlying_price]),
       strike=jnp.array([strike]), time_to_maturity=jnp.array([0]),risk_free_rate=risk_free_rate,vol=vol
   )
   np.testing.assert_allclose(price, jnp.maximum(underlying_price-strike,0.0), atol=1e-6, rtol=1e-6)


@given(
    time_to_maturity = st.floats(min_value=0.0, max_value=3.0, width=32),
    strike= st.floats(min_value=0, max_value=10, width=32),
    risk_free_rate = st.floats(min_value=0.0, max_value=0.5,width=32 ),
    vol = st.floats(min_value=0.0, max_value=0.5, width=32)
)
def test_price_large_price(
        time_to_maturity: float,
        strike: float,
        risk_free_rate: float,
        vol: float
):
    underlying_price = 10.0**20
    price = price_call(
        underlying_price=jnp.array([underlying_price]),
        strike=jnp.array([strike]), time_to_maturity=jnp.array([time_to_maturity]), risk_free_rate=risk_free_rate, vol=vol
    )
    np.testing.assert_allclose(price, underlying_price - strike, atol=1e-6, rtol=1e-6)

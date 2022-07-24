import jax.lax

from qml.tools.jax_util.types import Array
import jax.numpy as jnp
import jax.scipy as jscipy

@jax.jit
def price_call(underlying_price: Array,
               strike: Array,
               time_to_maturity: Array,
               risk_free_rate: Array,
               vol: Array) -> Array:
    """"""
    return _price_call_core(
        underlying_price=underlying_price,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        vol=vol,
    )



def _price_call_core(underlying_price: Array,
               strike: Array,
               time_to_maturity: Array,
               risk_free_rate: Array,
               vol: Array) -> Array:
    eps = jnp.finfo(underlying_price.dtype).eps # use machine epsilon to avoid 0/0
    r= risk_free_rate
    S = underlying_price
    K= strike
    d1 = 1/(vol*jnp.sqrt(time_to_maturity))*(jnp.log((S+eps)/K) + (r+vol**2/2)*time_to_maturity+eps )
    d2 = d1 - vol * jnp.sqrt(time_to_maturity)
    price = jscipy.stats.norm.cdf(d1)*S - jscipy.stats.norm.cdf(d2)*K*jnp.exp(-r*time_to_maturity)
    price = jnp.maximum(price, 0.0 ) # correct numerical issues causing negative prices
    return price


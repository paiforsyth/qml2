import jax.numpy as jnp

from qml.tools.jax_util.types import Array
from qml.tools.monte_carlo.brownian_motion import vector_geometric_brownian_motion
from qml.tools.option_pricing.black_scholes import price_call


def black_scholes_option_portfolio(
    num_paths: int,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    initial_price: Array,
    C: jnp.ndarray,
    time_steps: Array,
    key: Array,
    strike: Array,
    maturities: Array,
    underlying: Array,
    risk_free_rate: Array,
) -> Array:
    """
    Args:
        num_paths: Number of paths to simulate
        mu: array of shape (num_assets,) giving (P-measure) geometric drift for each asset
        sigma: array of dimension (num_assets,) indicating volatility
        initial_price: array of dimension (num_assets,) indicating initial price for each asset
        C: array of dimension (num_assets, num_assets) such that CC^T is the correlation between assets
        time_steps: shape (num_steps,) indicating intermediate times to simulate
        key: random number generation key
        strike: array of shape (num_options,) indicating the strike for each option
        maturities:  array of shape (num_options,) indicating the maturity of each option
        underlying:  integer array of shape (num_options, ) indicating underlying asset for each option
        risk_free_rate: scalar indicating the risk-free rate
    Returns:
        array of dimension (num_paths,num_steps, num_options) indicating the price of each option at each timestep
        on each path
    """
    num_time_steps = time_steps.shape[0]
    num_options = maturities.shape[0]
    asset_prices = vector_geometric_brownian_motion(
        num_paths=num_paths, start_value=initial_price, mu=mu, C=C, time_steps=time_steps, key=key, sigma=sigma
    )  # Shape (num_paths, num_time_steps, num_assets)
    option_underlying_price = asset_prices[..., underlying]  # shape num_paths, num_time_steps, num_options
    option_vol = sigma[underlying]
    time_to_maturity = maturities - time_steps.reshape(num_time_steps, num_options)  # shape num_time steps, num_options
    return price_call(
        underlying_price=option_underlying_price,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        vol=option_vol,
    )

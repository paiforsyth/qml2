import jax
import jax.numpy as jnp
import xarray as xr
from numpy.typing import ArrayLike

from qml.tools.jax_util.types import Array


def vector_arithmetic_brownian_motion(
    num_paths: int, mu: jnp.ndarray, sigma: jnp.ndarray, C: jnp.ndarray, time_steps: Array, key: Array
) -> Array:
    """
    Args:
        Simulate a given number of paths off vector arithmetic brownian motion
        num_paths: number of paths to generate
        mu:  array of dimension (n,) indicating geometric drif
        sigma: array of dimension (n,) indicating volatility
        C: array of dimension (n,n) such that CC^T is the correlation between assets
        time_steps: shape (num_steps,) indicating intermediate times to simulate
        key: Jax Pseudo-random key
    Returns:
        array of dimension (num_paths, num_steps,n) giving simulated values of Brownian Motion
    """
    n = mu.size
    C = C.reshape(n, n)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    C = jnp.asarray(C)
    time_steps = jnp.asarray(time_steps)
    num_steps = time_steps.shape[-1]
    time_steps = time_steps.reshape(1, num_steps, 1)
    drift_array = mu.reshape(1, 1, n) * (time_steps)  # should be 1 by steps by assets
    correlated_normals = (C * (jax.random.normal(key, (num_paths, num_steps, 1, n)))).sum(
        axis=-1
    )  # dimension paths by steps by n
    time_deltas = jnp.diff(time_steps, axis=1, prepend=0.0)
    shocks = sigma.reshape(1, 1, n) * jnp.sqrt(time_deltas) * correlated_normals
    cum_shocks = jnp.cumsum(shocks, axis=1)
    return drift_array + cum_shocks


vector_arithmetic_brownian_motion = jax.jit(fun=vector_arithmetic_brownian_motion, static_argnums=(0,))


def vector_geometric_brownian_motion(
    num_paths: int, start_value: Array, mu: Array, sigma: Array, C: Array, time_steps: Array, key: Array
) -> Array:
    """
    Args:
        num_paths: number of paths to generate
        start_value : shape (,n)  indicating starting valueA for all assets
        mu:  array of dimension (n,) indicating geometric drift
        sigma: array of dimension (n,) indicating volatility
        C: array of dimension (n,n) such that CC^T is the correlation between assets
        time_steps: shape (num_steps,) indicating intermediate times to simulate
        key: Jax Pseudo-random key
    Returns:
        array of dimension (num_paths, num_steps,n) giving simulated values of Brownian Motion
    """
    arithmetic_drift = mu - (sigma**2) / 2
    n = mu.size
    abm = vector_arithmetic_brownian_motion(
        num_paths=num_paths, mu=arithmetic_drift, sigma=sigma, time_steps=time_steps, key=key, C=C
    )
    return start_value.reshape((1, 1, n)) * jnp.exp(abm)


TIME_STEP_DIMENSION = "time_step"
PATH_DIMENSION = "path"
ASSET_DIMENSION = "asset"

MU = "mu"
SIGMA = "sigma"
C_MAT = "C_matrix"
START_VALUE = "start_value"
TIME_STEPS = "time_steps"


def xr_input_vector_geometric_brownian_motion(
    num_paths: int,
    start_value: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    C: ArrayLike,
    time_steps: ArrayLike,
    key: Array,
) -> xr.DataArray:
    result = vector_geometric_brownian_motion(
        num_paths=num_paths,
        start_value=jnp.asarray(start_value),
        mu=jnp.asarray(mu),
        sigma=jnp.asarray(sigma),
        C=jnp.asarray(C),
        time_steps=jnp.asarray(time_steps),
        key=key,
    )
    return xr.DataArray(result, dims=(PATH_DIMENSION, TIME_STEP_DIMENSION, ASSET_DIMENSION))


def xr_vector_geometric_brownian_motion(num_paths: int, params: xr.Dataset, key: Array) -> xr.DataArray:
    return xr_input_vector_geometric_brownian_motion(
        num_paths=num_paths,
        start_value=params[START_VALUE],
        mu=params[MU],
        sigma=params[SIGMA],
        C=params[C_MAT],
        time_steps=params[TIME_STEPS],
        key=key,
    )

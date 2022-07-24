import jax
import jax.numpy as jnp

from qml.tools.jax_util.types import Array


def vector_arithmetic_brownian_motion(
    num_paths: int,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    C: jnp.ndarray,
    time_steps: Array,
    key: Array,
) -> Array:
    """
    Simulate a given number of paths off vector arithmetic brownian motion
    num_paths: number of paths to generate
    mu:  array of dimension (n,) indicating geometric drif
    sigma: array of dimension (n,) indicating volatility
    C: array of dimension (n,n) such that CC^T is the correlation between assets
    time_steps: shape (num_steps,) indicating intermediate times to simulate
    returns:
    array of dimension (num_paths, num_steps,n) giving simulated values of Brownian Motion
    """
    n = mu.size
    C = C.reshape(n, n)
    num_steps = time_steps.shape[-1]
    time_steps = time_steps.reshape(1, num_steps, 1)
    drift_array = mu.reshape(1, 1, n) * (time_steps)  # should be 1 by steps by assets
    # step_size = (final_time / (steps)).reshape(num_paths, 1, 1)
    correlated_normals = (
        C * (jax.random.normal(key, (num_paths, num_steps, 1, n)))
    ).sum(
        axis=-1
    )  # dimension paths by steps by n
    time_deltas = jnp.diff(time_steps, axis=1, prepend=0.0)
    shocks = sigma.reshape(1, 1, n) * jnp.sqrt(time_deltas) * correlated_normals
    cum_shocks = jnp.cumsum(shocks, axis=1)
    return drift_array + cum_shocks


vector_arithmetic_brownian_motion = jax.jit(
    fun=vector_arithmetic_brownian_motion, static_argnums=(0,)
)


def vector_geometric_brownian_motion(
    num_paths: int,
    start_value: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    C: jnp.ndarray,
    time_steps: Array,
    key: Array,
) -> Array:
    """
    num_paths: number of paths to generate
    start_value : shape (num_paths,n)  indicating starting valueA for all assets
    mu:  array of dimension (n,) indicating geometric drift
    sigma: array of dimension (n,) indicating volatility
    C: array of dimension (n,n) such that CC^T is the correlation between assets
    time_steps: shape (num_steps,) indicating intermediate times to simulate
    """
    arithmetic_drift = mu - (sigma**2) / 2
    n = mu.size
    abm = vector_arithmetic_brownian_motion(
        num_paths=num_paths,
        mu=arithmetic_drift,
        sigma=sigma,
        time_steps=time_steps,
        key=key,
        C=C,
    )
    return start_value.reshape((num_paths, 1, n)) * jnp.exp(abm)

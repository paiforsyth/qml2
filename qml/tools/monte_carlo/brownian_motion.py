import jax
import jax.numpy as np

from qml.tools.jax_util.types import Array


def vector_arithmetic_brownian_motion(
    num_paths: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    C: np.ndarray,
    steps: int,
    final_time: np.ndarray,
    key: Array,
) -> Array:
    """
    Simulate a given number of paths off vector arithmetic brownian motion
    num_paths: number of paths to generate
    mu:  array of dimension (n,) indicating geometric drif
    sigma: array of dimension (n,) indicating volatility
    C: array of dimension (n,n) such that CC^T is the correlation between assets
    final_time: shape (num_paths,) indicating time to simulate
    returns:
    array of dimension (num_paths, num_steps,n) giving simulated values of Brownian Motion
    """
    assert steps >= 1, "must use at least 1 step"
    n = mu.size
    C = C.reshape(n, n)
    time_steps = (
        np.linspace(0, final_time, steps + 1)
        .transpose()[:, 1:]
        .reshape(num_paths, steps, 1)
    )
    drift_array = mu.reshape(1, 1, n) * (
        time_steps
    )  # should be paths by states by assets
    step_size = (final_time / (steps)).reshape(num_paths, 1, 1)
    correlated_normals = (C * (jax.random.normal(key, (num_paths, steps, 1, n)))).sum(
        axis=-1
    )  # dimension paths by steps by n
    shocks = sigma.reshape(1, 1, n) * np.sqrt(step_size) * correlated_normals
    cum_shocks = np.cumsum(shocks, axis=1)
    return drift_array + cum_shocks


def vector_geometric_brownian_motion(
    num_paths: int,
    start_value: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    C: np.ndarray,
    final_time: Array,
    steps: int,
    key: Array,
) -> Array:
    """
    num_paths: number of paths to generate
    start_value : shape (num_paths,n)  indicating starting valueA for all assets
    mu:  array of dimension (n,) indicating geometric drift
    sigma: array of dimension (n,) indicating volatility
    C: array of dimension (n,n) such that CC^T is the correlation between assets
    final_time: shape (num_path,) indicating amount of time to simulate
    """
    arithmetic_drift = mu - (sigma ** 2) / 2
    n = mu.size
    abm = vector_arithmetic_brownian_motion(
        num_paths=num_paths,
        mu=arithmetic_drift,
        sigma=sigma,
        final_time=final_time,
        steps=steps,
        key=key,
        C=C,
    )
    return start_value.reshape((num_paths, 1, n)) * np.exp(abm)

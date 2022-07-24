import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from hypothesis import given, settings
from hypothesis import strategies as st

from qml.tools.monte_carlo import brownian_motion


@given(
    mean_1=st.floats(min_value=-10, max_value=-10),
    mean_2=st.floats(min_value=-10, max_value=-10),
    corr=st.floats(min_value=-0.9, max_value=0.9),
    num_steps=st.integers(min_value=1, max_value=10),
    final_time=st.floats(min_value=0.1, max_value=1.0),
    var_1=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=5, deadline=None)
def test_mean_invariant_to_steps(
    mean_1: float,
    mean_2: float,
    corr: float,
    num_steps: int,
    final_time: float,
    var_1: float,
):
    """Example test.

    Verify that mean of brownian motion is independent of number of
    steps computed. Same for correlation and variance
    """
    expected_mean = np.array([mean_1, mean_2]) * final_time
    seed = 4
    key = jax.random.PRNGKey(seed)
    matrix_corr = jnp.array(
        [
            [
                1.0,
                corr,
            ],
            [corr, 1.0],
        ]
    )
    C = jnp.array(
        scipy.linalg.cholesky(
            matrix_corr,
            lower=True,
        )
    )
    paths = 10000
    brownian_output = brownian_motion.vector_arithmetic_brownian_motion(
        num_paths=paths,
        mu=jnp.array([mean_1, mean_2]),
        sigma=jnp.array([math.sqrt(var_1), 1.0]),
        final_time=final_time * jnp.ones((paths,)),
        steps=num_steps,
        key=key,
        C=C,
    )  # paths by steps by n
    sample_mean = brownian_output[:, -1].mean(axis=0)
    diff = (brownian_output[:, -1] - sample_mean).reshape(paths, 2, 1)
    sample_matrix_cov = 1 / paths * (diff * diff.reshape(paths, 1, 2)).sum(axis=0)
    sample_corr = sample_matrix_cov[1, 0] / final_time / math.sqrt(var_1)
    sample_var = sample_matrix_cov[0, 0] / final_time
    assert jnp.linalg.norm(sample_mean - expected_mean, ord=float("inf")) < 0.1
    assert abs(sample_corr - corr) < 0.1
    assert abs(sample_var - var_1) < 0.1

from qml.tools.monte_carlo.brownian_motion_util import add_risk_free_rate_to_params


def test_shapes_are_correct():
    mu, sigma, C, start = add_risk_free_rate_to_params([1.0], [2.0], [[1.0]], [3.0], risk_free_rate=0.01)
    assert len(mu) == len(sigma) == C.shape[0] == C.shape[1] == len(start)

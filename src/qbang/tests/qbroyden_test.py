import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from qbang.optimizer import (
    BangOptimizer,
    QBroydenOptimizer,
    QITEOptimizer,
    QNGInvOptimizer,
)
from qbang.tests.utils import circuit


@pytest.mark.parametrize(
    "A, u, epsilon, solution",
    [
        (
            np.identity(3),
            np.array([1, 2, 3]),
            0.0,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        (
            np.identity(3),
            np.array([1, 1, 1]),
            0.5,
            np.array([[1.5, -0.5, -0.5], [-0.5, 1.5, -0.5], [-0.5, -0.5, 1.5]]),
        ),
    ],
)
def test_sherman_morrison_update(A, u, epsilon, solution):
    optimizer = QBroydenOptimizer(stepsize=0.01)
    A_sherman_morrison = optimizer.sherman_morrison_update(A, u, epsilon)
    np.testing.assert_allclose(A_sherman_morrison, solution)


@pytest.mark.parametrize(
    "approx",
    [
        ("diag"),
        ("block-diag"),
        ("identity"),
        (None),
    ],
)
def test_optimizer_approx_methods(approx):
    optimizer = QBroydenOptimizer(stepsize=0.01, approx=approx)
    params = pnp.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    _ = optimizer.step(circuit, params)

    assert optimizer.approx == approx
    assert optimizer.F_t is not None


if __name__ == "__main__":
    test_optimizer_approx_methods(approx="block-diag")
    test_optimizer_approx_methods(approx=None)

    pytest.main()
    print("All tests passed!")

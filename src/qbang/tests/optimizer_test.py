import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from pennylane import AdamOptimizer, QNGOptimizer
from qbang.optimizer import (
    BangOptimizer,
    QBroydenOptimizer,
    QITEOptimizer,
    QNGInvOptimizer,
)
from qbang.tests.utils import circuit_2


@pytest.mark.parametrize(
    "Optimizer",
    [
        (QBroydenOptimizer),
        (BangOptimizer),
        (QITEOptimizer),
        (QNGInvOptimizer),
    ],
)
def test_run_optimizer(Optimizer):
    # TODO extend test to verify that the circuit is actually minimized.

    optim = Optimizer(stepsize=0.05)
    params = pnp.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)

    for _ in range(1000):
        params, cost = optim.step_and_cost(circuit_2, params)

        if cost < -0.61:
            break


if __name__ == "__main__":
    # pytest.main()
    test_run_optimizer(BangOptimizer)
    # test_run_optimizer(QNGOptimizer)
    # test_run_optimizer(AdamOptimizer)
    print("All tests passed!")

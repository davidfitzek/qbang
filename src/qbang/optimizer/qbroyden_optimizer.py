from typing import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from scipy.linalg import lstsq

from qbang.abstract_broyden_optimizer import AbstractBroydenOptimizer
from qbang.utils.utils import is_invertible, make_invertible


class QBroydenOptimizer(AbstractBroydenOptimizer):
    def __init__(
        self,
        stepsize: float = 0.01,
        discount: float = 0.2,
        approx: str = None,
        lam: float = 1e-7,
        recalculate_F_steps: int = np.inf,
    ):
        super().__init__(discount, approx, stepsize, lam)  # for parent class Optimizer
        # approx can be diag, block-diag, None or Identity
        self.F_t = None
        self.F_inv_t = None
        self.grad_fun = None
        self.recalculate_F_steps = recalculate_F_steps

    def step(
        self,
        objective_fn: Callable,
        params: pnp.ndarray,
        grad_fn: Callable = None,
        *args,
        **kwargs,
    ):
        params_shape = params.shape
        params = params.reshape((-1,))

        if self.counter > 0:
            assert (
                self.F_inv_t is not None
            ), "After the first iteration F_inv_t should not be None"

        if grad_fn is None and self.grad_fun is None:
            self.grad_fun = qml.grad(objective_fn)

        self.counter += 1

        if self.F_inv_t is None or self.counter % self.recalculate_F_steps == 0:
            self.nat_grad = self._qng_update(params, objective_fn)
        else:
            self.grad = self.grad_fun(params)

            self.nat_grad = self.F_inv_t @ self.grad

            self.F_inv_t = self.sherman_morrison_update(
                self.F_inv_t, self.grad, self.epsilon
            )

        params -= self.stepsize * self.nat_grad

        params = params.reshape(params_shape)
        return params

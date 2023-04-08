from typing import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from scipy.linalg import lstsq

from qbang.abstract_broyden_optimizer import AbstractBroydenOptimizer
from qbang.utils.utils import is_invertible, make_invertible


class QBangOptimizer(AbstractBroydenOptimizer):
    def __init__(
        self,
        stepsize: float = 0.01,
        discount: float = 0.2,
        approx: str = None,
        b_1: float = 0.9,
        b_2: float = 0.999,
        kappa: float = 1e-8,
        lam: float = 1e-7,
        recalculate_F_steps: int = np.inf,
    ):
        super().__init__(discount, approx, stepsize, lam)  # for parent class Optimizer
        self.F_t = None
        self.F_inv_t = None
        self.b_1 = b_1
        self.b_2 = b_2
        self.kappa = kappa
        self.recalculate_F_steps = recalculate_F_steps

        self.m_prev = None
        self.v_prev = None
        self.counter = 0

        self.grad = None
        self.grad_fun = None

    def step(
        self,
        objective_fn: Callable,
        params: pnp.ndarray,
        grad_fn: Callable = None,
        *args,
        **kwargs,
    ):
        if self.counter > 0:
            assert (
                self.F_inv_t is not None
            ), "After the first iteration F_inv_t should not be None"

        # flatten params and reshape after update
        params_shape = params.shape
        params = params.reshape((-1,))
        # Get gradient function
        if grad_fn is None and self.grad_fun is None:
            self.grad_fun = qml.grad(objective_fn)
            self.grad = self.grad_fun(params)

        # Initialize m and v for Adam
        if self.counter == 0:
            self.m_prev = np.zeros(self.grad.shape)
            self.v_prev = np.zeros(self.grad.shape)

        self.counter += 1  # increase counter to avoid division by zero

        if self.F_inv_t is None or self.counter % self.recalculate_F_steps == 0:
            self.nat_grad = self._qng_update(params, objective_fn)
            self.nat_grad = self.nat_grad.flatten()
            self.counter = 1  # try setting counter to 1
        else:
            self.grad = self.grad_fun(params)
            self._adam_update()

            # TODO Note to improve efficiency at the expense of clarity
            # we can replace some lines of the Adam update, see
            # https://arxiv.org/abs/1412.6980 Sec. 2 last paragraph for details.
            self.nat_grad = self.F_inv_t @ self.m_hat

            self.F_inv_t = self.sherman_morrison_update(
                self.F_inv_t, self.grad, self.epsilon
            )

        params -= self.stepsize * self.nat_grad
        params = params.reshape(params_shape)
        return params

    def _adam_update(self):
        """The Adam update. https://arxiv.org/abs/1412.6980.

        Args:
            params (pnp.ndarray): The current parameters.
        """

        self.m = self._update_m(self.grad)
        self.v = self._update_v(self.grad)

        self.m_hat = self._get_m_hat(self.m)
        v_hat = self._get_v_hat(self.v)

        self.m_hat = self.m_hat / (np.sqrt(v_hat) + self.kappa)

        self.m_prev = self.m
        self.v_prev = self.v

    def _update_m(self, grad):
        return self.b_1 * self.m_prev + (1 - self.b_1) * grad

    def _update_v(self, grad):
        grad_squared = np.einsum("i,i -> i", grad, grad)
        return self.b_2 * self.v_prev + (1 - self.b_2) * grad_squared

    def _get_m_hat(self, m):
        return m / (1 - self.b_1**self.counter)

    def _get_v_hat(self, v):
        return v / (1 - self.b_2**self.counter)

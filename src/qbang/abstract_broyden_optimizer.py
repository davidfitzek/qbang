import abc
from typing import Callable

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from scipy.linalg import lstsq

from qbang.abstract_optimizer import AbstractOptimizer
from qbang.utils.utils import is_invertible, make_invertible


class AbstractBroydenOptimizer(AbstractOptimizer, abc.ABC):
    """
    Abstract class for an optimizer using natural gradient and the Sherman-Morrison update rule.
    """

    @abc.abstractmethod
    def __init__(
        self, discount: float, approx: str, stepsize: float, lam: float = 1e-7
    ):
        super().__init__(stepsize)  # for parent class AbstractOptimizer
        self.discount = discount
        self.approx = approx
        self.counter = 0
        self.lam = lam

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def step_and_cost(
        self,
        objective_fn: Callable,
        params: pnp.ndarray,
        grad_fn: Callable = None,
        *args,
        **kwargs,
    ):
        cost = objective_fn(params)
        params = self.step(objective_fn, params, grad_fn, *args, **kwargs)

        return params, cost

    @property
    def epsilon(self):
        self._epsilon = self.discount / self.counter
        return self._epsilon

    def sherman_morrison_update(
        self, F: pnp.ndarray, grad: pnp.ndarray, epsilon: float
    ):
        """The Sherman-Morrison update rule.

        Args:
            F (pnp.tensor): [num_params^2, num_params^2]
            grad (pnp.tensor): [num_params^2]
            epsilon (float): Filter parameter.

        Returns:
            F_new (pnp.tensor): [num_params^2, num_params^2]
        """
        left = 1 / (1 - epsilon) * F
        numerator = F @ np.outer(grad, grad) @ F
        denominator = (1 - epsilon) + epsilon * (grad @ F @ grad)
        right = epsilon / (1 - epsilon) * (numerator / denominator)
        F_new = left - right
        return F_new

    def _qng_update(self, params: pnp.ndarray, objective_fn: Callable):
        """A single step of the QNG algorithm. The algorithm is equivalent to
            QITE, with the difference that we approximate the metric tensor in QNG and
            the quantum Fisher information in QITE.

            https://quantum-journal.org/papers/q-2020-05-25-269/pdf/
            https://arxiv.org/pdf/1912.08660.pdf

            1. Compute the metric tensor F_0.
            2. Check invertability of F_0.
            3. Compute the natural gradient.

            We also calculate the inverse of the F_0 matrix to use it later
            for the update.

        Args:
            params (pnp.ndarray): The current parameters.
            objective_fn (Callable): the quantum circuit as a callable function.

        Returns:
            nat_grad (pnp.ndarray): The natural gradient.
        """
        self.grad = self.grad_fun(params)
        self.F_t = self.get_F_0(objective_fn, params, self.approx)

        # check if Matrix is invertible
        if not is_invertible(self.F_t):
            self.F_t = make_invertible(self.F_t, self.lam)
        assert is_invertible(self.F_t), "F_t is singular, cannot invert"

        # crucial for Sherman-Morrison update rule
        self.F_inv_t = np.linalg.inv(self.F_t)

        nat_grad, _, _, _ = lstsq(self.F_t, self.grad.flatten(), cond=self.lam)
        return nat_grad

    def get_F_0(self, objective_fn: Callable, params: pnp.ndarray, approx: str):
        match approx:
            case "diag":
                return qml.metric_tensor(objective_fn, approx=self.approx)(params)
            case "block-diag":
                return qml.metric_tensor(objective_fn, approx=self.approx)(params)
            case "identity":
                F_0 = qml.metric_tensor(objective_fn, approx="block-diag")(params)
                return np.identity(F_0.shape[0])
            case "QFI":
                print("todo")
                raise NotImplementedError
            case "None" | None:
                return qml.metric_tensor(objective_fn, approx=None)(params)
            case _:
                print("\n Approx method invalid \n ")
                raise ValueError

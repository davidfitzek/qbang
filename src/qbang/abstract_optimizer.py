import abc
from typing import Callable

import numpy as np


# TODO Should there be a function step_and_cost? It returns the params and
# the value of the objective function.
class AbstractOptimizer(abc.ABC):
    """
    Abstract class for an optimizer.
    """

    @abc.abstractmethod
    def __init__(self, stepsize: float):
        self.stepsize = stepsize

    @abc.abstractmethod
    def step(self, fun: Callable, x0: np.ndarray):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

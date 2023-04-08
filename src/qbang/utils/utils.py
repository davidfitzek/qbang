import numpy as np
import pennylane as qml


# https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
def is_invertible(a: np.ndarray) -> bool:
    """Check if a matrix is invertible.

    Args:
        a (np.ndarray): A matrix to check for invertibility.

    Returns:
        bool: Boolean indicating if the matrix is invertible.
    """
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def make_invertible(a: np.ndarray, min_value: float = 1e-7) -> np.ndarray:
    """Make a matrix invertible by adding a small value to the diagonal.

    Args:
        a (np.ndarray): Matrix to check for invertibility.
        min_value (float, optional): _description_. Defaults to 1e-7.

    Returns:
        a (np.ndarray): A matrix that is invertible.
    """
    shape = qml.math.shape(a)
    size = qml.math.prod(shape[: len(shape) // 2])
    a = qml.math.reshape(a, (size, size))
    a = a + min_value * qml.math.eye(size, like=a)
    return a

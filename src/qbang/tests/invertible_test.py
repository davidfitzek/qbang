import numpy as np
import pytest
from qbang.utils.utils import is_invertible, make_invertible


@pytest.mark.parametrize(
    "matrix, solution",
    [
        (np.array([[1, 0], [0, 0]]), False),
        (np.array([[1, 0], [0, 1]]), True),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), True),
        (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]), False),
    ],
)
def test_is_invertible(matrix, solution):
    _bool = is_invertible(matrix)
    assert _bool == solution


@pytest.mark.parametrize(
    "matrix",
    [
        (np.array([[1e-17, 0], [0, 0]])),
        (np.array([[0, 0], [0, 0]], dtype=np.float64)),
    ],
)
def test_make_invertible(matrix):
    mat = make_invertible(matrix, min_value=1e-7)
    diag = np.diag(mat)

    assert np.allclose(diag, np.ones(diag.shape) * 1e-7)
    assert is_invertible(mat)


if __name__ == "__main__":
    test_is_invertible(np.array([[1, 0], [0, 0]]), False)
    test_make_invertible(np.array([[1e-17, 0], [0, 0]]))
    test_make_invertible(np.array([[0, 0], [0, 0]], dtype=np.float64))
    print("All tests passed")

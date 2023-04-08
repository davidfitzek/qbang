from typing import Callable, Dict, List, Optional

import autograd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.operation import Operation


class BarrenPlateauCircuit:
    """Class for building Barren Plateau Circuits.
        https://arxiv.org/pdf/1803.11173.pdf

    The class implements the methods and properties required for building a
    Barren Plateau Circuit.

    Attributes:
        num_layers (int): Number of layers in the circuit.
        num_qubits (int): Number of qubits in the circuit.
        list_gate_set (list): List of gate sequences for each layer.
        params (numpy.ndarray): Parameters for the circuit.
    """

    def __init__(self, num_layers: int, num_qubits: int = 5):
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.params_shape = (self.num_layers * self.num_qubits,)
        self.list_gate_set = []  # (n_qubits, layers)

    @property
    def wires(self):
        """Property that returns the list of wires in the circuit.

        Returns:
            list: List of wires in the circuit.
        """
        return range(self.num_qubits)

    def init(self, seed: int = None):
        """Initializes the gate sequences and circuit parameters.

        Args:
            seed (int, optional): Seed for the random number generator.

        Returns:
            numpy.ndarray: Circuit parameters.
        """
        if seed is not None:
            np.random.seed(seed)
            pnp.random.seed(seed)

        self.list_gate_set = self._pauli_gates(self.num_layers, self.wires, seed)
        return self._params(seed)

    def _pauli_gates(
        self, layer: int, wires: List[int], seed: Optional[int] = None
    ) -> List[Dict[int, Callable[[float], Operation]]]:
        """Generates a random sequence of Pauli gates for each layer.

        Args:
            layer (int): Number of layers in the circuit.
            wires (List): List of wires in the circuit.
            rng_key (int, optional): Seed for the random number generator.

        Returns:
            List: List of gate sequences for each layer.
        """
        list_gate_set = []
        for _ in range(layer):
            gate_set = [qml.RX, qml.RY, qml.RZ]
            random_gate_sequence = {i: np.random.choice(gate_set) for i in wires}
            list_gate_set.append(random_gate_sequence)
        return list_gate_set

    def _params(self, seed: int):
        """Initialize the parameters for the circuit evaluation.

        Args:
            seed (int, optional): Seed for the random number generator.

        Returns:
            np.ndarray: The parameters for the circuit evaluation.

        """
        return pnp.array(
            pnp.random.uniform(low=-np.pi, high=np.pi, size=self.params_shape),
            requires_grad=True,
        )

    def __call__(self, params):
        """Execute the circuit with given parameters.

        Args:
            params (np.ndarray): The parameters to be used for executing the circuit.

        Returns:
            Operation: The evaluation of the circuit.
        """
        return self._circuit_ansatz(params)

    def _circuit_ansatz(self, params) -> Operation:
        """Perform the actual circuit execution with given parameters.

        Args:
            params (np.ndarray): The parameters to be used for evaluating the circuit.

        Returns:
            Operation: The evaluation of the circuit.

        Raises:
            AssertionError: If `self.list_gate_set` is not None or if the shape of
                the 'params' does not match.
        """
        assert self.list_gate_set is not None
        assert (
            len(params.flatten()) == self.num_qubits * self.num_layers
        ), f"{len(params.flatten())} != {self.num_qubits * self.num_layers}"

        assert isinstance(params, pnp.tensor) or isinstance(
            params, autograd.numpy.numpy_boxes.ArrayBox
        ), f"params must be either a pnp.tensor or autograd.numpy.numpy_boxes.ArrayBox but got {type(params)}"

        params = np.reshape(params, (self.num_layers, self.num_qubits))

        for i in self.wires:
            qml.RY(np.pi / 4, wires=i)

        for layer in range(self.num_layers):
            for wire in self.wires:
                self.list_gate_set[layer][wire](params[layer][wire], wires=wire)

            qml.Barrier(wires=self.wires)

            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

            qml.Barrier(wires=self.wires)

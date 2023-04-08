import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=4)  # one additional wire for the ancilla qubit


@qml.qnode(dev, interface="autograd")
def circuit(weights):
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=1)
    qml.RZ(weights[3], wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


dev = qml.device("default.qubit", wires=4)  # one additional wire for the ancilla qubit


@qml.qnode(dev)
def circuit_2(params):
    # |psi_0>: state preparation
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 7, wires=2)

    # V0(theta0, theta1): Parametrized layer 0
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)

    # W1: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    # V_1(theta2, theta3): Parametrized layer 1
    qml.RY(params[2], wires=1)
    qml.RX(params[3], wires=2)

    # W2: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliY(0))

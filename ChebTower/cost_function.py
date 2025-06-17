from pennylane import numpy as np
import pennylane as qml

from feature_map import U
from ansatz import HEA


def total_magnetization(num_qubits):
    return sum(qml.PauliZ(i) for i in range(num_qubits))

def create_circuit(x, num_qubits, d, theta, label):
    U(x, num_qubits, label)
    qml.Barrier()
    HEA(theta, num_qubits, d)
    cost_op = total_magnetization(num_qubits) 

    return qml.expval(cost_op)  


def f(x, num_qubits, d, theta):
    label = 0
    dev = qml.device('default.qubit', wires=list(range(num_qubits))) #qml.device("default.tensor", wires=list(range(num_qubits)), method="mps", max_bond_dim=4, contract="auto-mps")
    circuit = qml.QNode(create_circuit, dev)
    result = circuit(x, num_qubits, d, theta, label) 
    
    return result

def dphi(x):
    return -2/np.sqrt(1-(x**2)) 

def df(x, num_qubits, d, theta):
    C_plus = 0 
    C_minus = 0
    dev = qml.device('default.qubit', wires=list(range(num_qubits))) #qml.device("default.tensor", wires=list(range(num_qubits)), method="mps", max_bond_dim=4, contract="auto-mps")
    circuit = qml.QNode(create_circuit, dev)
    for i in range(1, 2*num_qubits+1):
        if i <= num_qubits:
            C_plus += (i)*circuit(x, num_qubits, d, theta, i) 
            
        else:
            C_minus += (i-num_qubits)*circuit(x, num_qubits, d, theta, i) 
            
    return 1/4*dphi(x)*(C_plus-C_minus) 


def func_and_deriv(x, num_qubits, d, theta):
    functions = []
    derivatives = []
    for i in range(len(x)):

        func = f(x[i], num_qubits, d, theta)
        f0 = f(0, num_qubits, d, theta)
        functions.append(func + (1 - f0)) #FLOATING BOUNDARY HANDLING!!! 
        
        deriv = df(x[i], num_qubits, d, theta)
        derivatives.append(deriv)

    return np.array(functions), np.array(derivatives)
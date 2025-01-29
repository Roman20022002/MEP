from pennylane import numpy as np
import pennylane as qml
import torch


from feature_map import U
from ansatz import HEA

def H(num_qubits):
    ops = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(num_qubits)])
    return ops

def total_magnetization(num_qubits):
    return sum(qml.PauliZ(i) for i in range(num_qubits))

def create_circuit(x, num_qubits, d, theta, label):

    U(x, num_qubits, label)
    qml.Barrier()
    HEA(theta, num_qubits, d)
    cost_op = total_magnetization(num_qubits) #H(num_qubits)

    return qml.expval(cost_op)  


def f(x, num_qubits, d, theta):

    label = 0
    dev = qml.device('default.qubit', wires=list(range(num_qubits)))
    circuit = qml.QNode(create_circuit, dev)
    result = circuit(x, num_qubits, d, theta, label) 

    '''
    qml.drawer.use_style('black_white')
    fig, ax = qml.draw_mpl(circuit)(x, num_qubits, d, theta, label)
    plt.show()
    '''
    return result 

def dphi(x):
    return -2/np.sqrt(1-x**2) 

def df(x, num_qubits, d, theta):

    C_plus = 0 
    C_minus = 0
    for i in range(1, 2*num_qubits+1):
        if i <= num_qubits:
            dev = qml.device('default.qubit', wires=list(range(num_qubits)))
            circuit = qml.QNode(create_circuit, dev)
            C_plus += (i)*circuit(x, num_qubits, d, theta, i) 
            
        else:
            dev = qml.device('default.qubit', wires=list(range(num_qubits)))
            circuit = qml.QNode(create_circuit, dev) 
            C_minus += (i-num_qubits)*circuit(x, num_qubits, d, theta, i) 
            

    return 1/4*dphi(x)*(C_plus-C_minus) 


def func_and_deriv(x, num_qubits, d, theta):
    
    functions = []
    derivatives = []
    for i in x:
        functions.append(f(i, num_qubits, d, theta)) #+ (1 - f(0, num_qubits, d, theta))) #FLOATING BOUNDARY HANDLING!!!
        derivatives.append(df(i, num_qubits, d, theta))

    return np.array(functions), np.array(derivatives)
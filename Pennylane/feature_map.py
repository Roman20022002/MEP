from pennylane import numpy as np
import pennylane as qml

def phi(x, qubit): 
    return 2*qubit*np.arccos(x)

def phi_plus(x, qubit): 
    return 2*qubit*np.arccos(x) + np.pi/2

def phi_min(x, qubit):
    return 2*qubit*np.arccos(x) - np.pi/2


def U(x, num_qubits, label):

    N = num_qubits
    assert 0 <= label <= 2*N, "label must be in [0, 2*N]!"

    # Return U if label==0
    if label == 0:
        for i in range(N):
            qml.RY(phi(x, i), i) 

    # Return U+ if label in [1, N]
    elif label >= 1 and label <= N:
        for i in range(N):
            if i == label-1:
                qml.RY(phi_plus(x, i), i) 
            else:
                qml.RY(phi(x, i), i) 

    # Return U- if label in [N+1,2*N]
    else:
        for i in range(N):
            if i == label-N-1:
                qml.RY(phi_min(x, i), i) 
            else:
                qml.RY(phi(x, i), i) 
    
    return 0
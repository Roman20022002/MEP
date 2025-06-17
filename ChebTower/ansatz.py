from pennylane import numpy as np
import pennylane as qml

def rotations_layers(theta, i):
    assert len(theta) == 3, "theta size incorrect!"
    
    qml.RZ(theta[0], i) 
    qml.RX(theta[1], i) 
    qml.RZ(theta[2], i) 

    return 0

def entangling_layers(num_qubits):
    N = num_qubits
    #pair qubits
    for i in range(0, N-1, 2):
            qml.CNOT([i, i+1])
    #impair qubits
    for i in range(1, N-1, 2):
            qml.CNOT([i, i+1])

    return 0

def HEA(theta, num_qubits, d):
    N = num_qubits
    Theta = np.reshape(theta, (N,d,3))
  
    for i in range(d):
        for j in range(N):

            rotations_layers(Theta[j,i,:], j) 
        entangling_layers(N)
        
    return 0
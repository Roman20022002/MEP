from pennylane import numpy as np

from cost_function import func_and_deriv

def squared_diff(f, df, x, lamb=8, k=0.1):
    a = df + lamb*f*(k + np.tan(lamb*x))
    loss = (a-0)**2
    return loss

def u(x, x0=0, u0=1, lamb=8, k=0.1):
    u_tilde = np.exp(-k*lamb*x0)*np.cos(lamb*x0)
    const = u0 - u_tilde
    return np.exp(-k*lamb*x)*np.cos(lamb*x) + const

'''
def lin_reg(i, epochs):
    return 1 - i/epochs
def reg_loss(x, num_qubits, d, theta, n_iter, epochs):
    sigm = lin_reg(n_iter, epochs)
    
    x_reg = np.array([0.0409, 0.0882, 0.1356, 0.1830, 0.2303, 0.2777, 0.3251, 0.3725, 0.4198, 0.4672, 0.5146, 0.5619, 0.6093, 0.6567, 0.7040, 0.7514, 0.7988, 0.8461, 0.8935])

    
    f_reg, _ = func_and_deriv(x_reg, num_qubits, d, theta)
    u_reg = u(x_reg)

    loss_array = sigm*(f_reg - u_reg)**2

    return np.sum(loss_array)
'''

def diff_loss(f, df, x):
    loss = np.sum(squared_diff(f, df, x)) / len(x)
    return loss

'''
def boundary_loss(f0, nabla, u0=1):
    loss = nabla*(f0-u0)**2
    return loss
'''

def loss_function(x, num_qubits, d, theta, n_iter, epochs, nabla): 
    f, df = func_and_deriv(x, num_qubits, d, theta)
    loss = diff_loss(f, df, x) #+ reg_loss(x, num_qubits, d, theta, n_iter, epochs) #+ boundary_loss(f[0], nabla)
    return loss
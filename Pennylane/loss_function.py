from pennylane import numpy as np

from cost_function import func_and_deriv

def MSE(f, df, x, lamb=8, k=0.1):
    a = df + lamb*f*(k + np.tan(lamb*x))
    loss = (a-0)**2
    return loss

def u(x, x0=0, u0=1, lamb=8, k=0.1):
    u_tilde = np.exp(-k*lamb*x0)*np.cos(lamb*x0)
    const = u0 - u_tilde
    return np.exp(-k*lamb*x)*np.cos(lamb*x) + const
def lin_reg(i, epochs):
    return 1 - i/epochs
def reg_loss(x, num_qubits, d, theta, n_iter, epochs, step=4):
    sigm = lin_reg(n_iter, epochs)
    x_reg = x[::step]
    
    f_reg, df_reg = func_and_deriv(x_reg, num_qubits, d, theta)
    u_reg = u(x_reg)

    loss_array = sigm*(f_reg - u_reg)**2

    return np.sum(loss_array)

def diff_loss(f, df, x):
    loss = np.sum(MSE(f, df, x)) / len(x)
    return loss

def boundary_loss(f0, nabla, u0=1):
    loss = nabla*(f0-u0)**2
    return loss

def loss_function(x, num_qubits, d, theta, n_iter, epochs, nabla): 
    f, df = func_and_deriv(x, num_qubits, d, theta)
    loss = diff_loss(f, df, x) + reg_loss(x, num_qubits, d, theta, n_iter, epochs) + boundary_loss(f[0], nabla) 
    return loss
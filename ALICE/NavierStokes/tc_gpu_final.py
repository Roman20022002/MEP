# --- GPU & Environment Check ---
import os
import time
import matplotlib.pyplot as plt
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
import numpy as np
import argparse


# --- Argument parser for job array index as bond dimension selector ---
parser = argparse.ArgumentParser()
parser.add_argument("--bond_dimension", type=int, required=True)
args = parser.parse_args()
bond_dimension = args.bond_dimension


# --- Backend Configuration ---
tc.set_backend("jax")


# --- Get init index from environment ---
init_index = int(os.environ["INIT_INDEX"])


# --- Feature Map ---
def phi(x):
    return jnp.arccos(x)

def feature_map(c, x, num_qubits):
    for i in range(num_qubits):
        c.ry(i, theta=phi(x))
    return c


# --- Ansatz ---
def ansatz(c, theta, num_qubits, d):
    theta = jnp.reshape(theta, [num_qubits, d, 3])
    for i in range(d):
        for q in range(num_qubits):
            c.rz(q, theta=theta[q, i, 0])
            c.rx(q, theta=theta[q, i, 1])
            c.rz(q, theta=theta[q, i, 2])
        for j in range(0, num_qubits - 1, 2):
            c.cnot(j, j + 1)
        for j in range(1, num_qubits - 1, 2):
            c.cnot(j, j + 1)
    return c


# --- MPS Circuit ---
def create_mps_circuit(x, theta, num_qubits, d, bond_dimension):
    c = tc.MPSCircuit(num_qubits)
    c.set_split_rules({"max_singular_values": bond_dimension})
    c = feature_map(c, x, num_qubits)
    c = ansatz(c, theta, num_qubits, d)
    return c

def f_x(x, theta, num_qubits, d, bond_dimension):
    c = create_mps_circuit(x, theta, num_qubits, d, bond_dimension)
    return jnp.real(sum(c.expectation_ps(z=[i]) for i in range(num_qubits)))

def f_x2(x, theta2, num_qubits, d, bond_dimension):
    c = create_mps_circuit(x, theta2, num_qubits, d, bond_dimension)
    return jnp.real(sum(c.expectation_ps(z=[i]) for i in range(num_qubits)))

def f_x3(x, theta3, num_qubits, d, bond_dimension):
    c = create_mps_circuit(x, theta3, num_qubits, d, bond_dimension)
    return jnp.real(sum(c.expectation_ps(z=[i]) for i in range(num_qubits)))

# --- Loss ---   
def loss_function(params, x_values, num_qubits, d, bond_dimension, b1=1.0, b2=1.0, b3=0.1):
    theta, theta2, theta3 = params

    def f_single(x): return f_x(x, theta, num_qubits, d, bond_dimension)
    def f_single2(x): return f_x2(x, theta2, num_qubits, d, bond_dimension)
    def f_single3(x): return f_x3(x, theta3, num_qubits, d, bond_dimension)

    f0_1 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, theta2, num_qubits, d, bond_dimension)
    f0_3 = f_x3(0.0, theta3, num_qubits, d, bond_dimension)

    f_vg1 = jax.value_and_grad(f_single)
    f_vg2 = jax.value_and_grad(f_single2)
    f_vg3 = jax.value_and_grad(f_single3)

    f_vals1, df_dx1 = jax.vmap(f_vg1)(x_values)
    f_vals2, df_dx2 = jax.vmap(f_vg2)(x_values)
    f_vals3, df_dx3 = jax.vmap(f_vg3)(x_values)

    f1_shifted = f_vals1 + (b1 - f0_1)
    f2_shifted = f_vals2 + (b2 - f0_2)
    f3_shifted = f_vals3 + (b3 - f0_3)
    
    A = 1 + 4.95*((2*x_values - 1)**2)
    dA = 19.8*(2*x_values - 1) / A
    gamma = 1.4
    a1 = -f1_shifted*df_dx3 -f1_shifted*f3_shifted*dA -f3_shifted*df_dx1
    a2 = -f3_shifted*df_dx2 -(gamma-1)*f2_shifted*(df_dx3 + f3_shifted*dA)
    a3 = -f3_shifted*df_dx3 -(1/gamma)*(df_dx2 + (f2_shifted/f1_shifted)*df_dx1)	
	
    loss1 = jnp.mean(a1 ** 2)
    loss2 = jnp.mean(a2 ** 2)
    loss3 = jnp.mean(a3 ** 2)

    return loss1 + loss2 + loss3
    
def lin_reg(epoch, epochs):
    return 1 - jnp.tanh((epoch-150)/epochs) 
def reg(epoch, epochs, reg1, reg2, reg3, f1_rshifted, f2_rshifted, f3_rshifted):
    sigm = lin_reg(epoch, epochs)
    reg_loss1 = sigm*((f1_rshifted - reg1)**2)
    reg_loss2 = sigm*((f2_rshifted - reg2)**2)
    reg_loss3 = sigm*((f3_rshifted - reg3)**2)
    return jnp.sum(reg_loss1) + jnp.sum(reg_loss2) + jnp.sum(reg_loss3)
def loss_function2(params, x_values, x_reg, num_qubits, d, bond_dimension, epoch, epochs, reg1, reg2, reg3, b1=1.0, b2=1.0, b3=0.1):
    theta, theta2, theta3 = params

    def f_single(x): return f_x(x, theta, num_qubits, d, bond_dimension)
    def f_single2(x): return f_x2(x, theta2, num_qubits, d, bond_dimension)
    def f_single3(x): return f_x3(x, theta3, num_qubits, d, bond_dimension)

    f0_1 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, theta2, num_qubits, d, bond_dimension)
    f0_3 = f_x3(0.0, theta3, num_qubits, d, bond_dimension)

    f_vg1 = jax.value_and_grad(f_single)
    f_vg2 = jax.value_and_grad(f_single2)
    f_vg3 = jax.value_and_grad(f_single3)

    f_vals1, df_dx1 = jax.vmap(f_vg1)(x_values)
    f_vals2, df_dx2 = jax.vmap(f_vg2)(x_values)
    f_vals3, df_dx3 = jax.vmap(f_vg3)(x_values)
    f_reg1, _ = jax.vmap(f_vg1)(x_reg) 
    f_reg2, _ = jax.vmap(f_vg2)(x_reg)
    f_reg3, _ = jax.vmap(f_vg3)(x_reg)

    f1_shifted = f_vals1 + (b1 - f0_1)
    f2_shifted = f_vals2 + (b2 - f0_2)
    f3_shifted = f_vals3 + (b3 - f0_3)
    f1_rshifted = f_reg1 + (b1 - f0_1)
    f2_rshifted = f_reg2 + (b2 - f0_2)
    f3_rshifted = f_reg3 + (b3 - f0_3)
    
    A = 1 + 4.95*((2*x_values - 1)**2)
    dA = 19.8*(2*x_values - 1) / A
    gamma = 1.4
    a1 = -f1_shifted*df_dx3 -f1_shifted*f3_shifted*dA -f3_shifted*df_dx1
    a2 = -f3_shifted*df_dx2 -(gamma-1)*f2_shifted*(df_dx3 + f3_shifted*dA)
    a3 = -f3_shifted*df_dx3 -(1/gamma)*(df_dx2 + (f2_shifted/f1_shifted)*df_dx1)	
	
    loss1 = jnp.mean(a1 ** 2)
    loss2 = jnp.mean(a2 ** 2)
    loss3 = jnp.mean(a3 ** 2)
    reg_loss = reg(epoch, epochs, reg1, reg2, reg3, f1_rshifted, f2_rshifted, f3_rshifted)

    return loss1 + loss2 + loss3 + reg_loss
    

# --- Training ---
def train():
    print(f"\n[INFO] Running train with bond dimension: {bond_dimension}")
    print(f"[INFO] Using init index: {init_index}")
    num_qubits, d, epochs, epochs2, lr, lr2 = 6, 6, 200, 600, 0.01, 0.005
    
    def generate_chebyshev_grid(n, a, b):
            k = jnp.arange(n)
            chebyshev_nodes = jnp.cos(jnp.pi * (2 * k + 1) / (2 * n))
            scaled_nodes = ((chebyshev_nodes + 1) * (b - a) / 2) + a
            return scaled_nodes[::-1]
    x_train = generate_chebyshev_grid(20, a=0.0, b=0.4)
    tmp = generate_chebyshev_grid(20, a=0.0, b=0.4)
    tmp2 = generate_chebyshev_grid(20, a=0.6, b=0.9)
    x_train2 = jnp.concatenate([tmp, tmp2])
    tmp3 = [0.8284, 0.8074, 0.7618, 0.7382, 0.6005, 0.3994, 0.3945, 0.3848, 0.3705,
       0.3521, 0.3299, 0.3045, 0.2765, 0.2467, 0.2157, 0.1843, 0.1533, 0.1235,
       0.0955, 0.0701, 0.0479, 0.0295, 0.0152, 0.0055, 0.0006]
    x_reg = jnp.array(tmp3)
    x_test = generate_chebyshev_grid(100, a=0.0, b=0.9)
    
    init_path = f"init_thetas_NS/init_{init_index}.npy"
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"[FATAL] Missing init file: {init_path}. Terminating script.")
    theta = jnp.array(np.load(init_path), dtype=jnp.float32)
    theta2 = jnp.array(np.load(init_path), dtype=jnp.float32)
    theta3 = jnp.array(np.load(init_path), dtype=jnp.float32)
    params = (theta, theta2, theta3)
    
    output_dir = f"plotsfinal_bond_{bond_dimension}/index_{init_index}"
    os.makedirs(output_dir, exist_ok=True)

    opt = optax.adam(lr)
    opt_state = opt.init(params)
    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        loss, grads = jax.value_and_grad(loss_function)(params, x_train, num_qubits, d, bond_dimension)
        if jnp.isnan(loss) or jnp.isnan(jnp.sum(grads[0]) + jnp.sum(grads[1]) + jnp.sum(grads[2])):
            raise ValueError(f"[FATAL] NaN encountered at epoch {epoch} for bond={bond_dimension}, init={init_index}. Terminating script.")
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: l_f = {loss:.6f} (epoch time: {time.time() - t1:.2f}s)")
    print(f"SubTraining time: {time.time() - t0:.2f}s")
    
    np.save(os.path.join(output_dir, "inter_losses.npy"), losses)
    f0_1 = f_x(0.0, params[0], num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, params[1], num_qubits, d, bond_dimension)
    f0_3 = f_x3(0.0, params[2], num_qubits, d, bond_dimension)
    f_vals1 = jax.vmap(lambda x: f_x(x, params[0], num_qubits, d, bond_dimension))(x_test)
    f_vals2 = jax.vmap(lambda x: f_x2(x, params[1], num_qubits, d, bond_dimension))(x_test)
    f_vals3 = jax.vmap(lambda x: f_x3(x, params[2], num_qubits, d, bond_dimension))(x_test)
    pred1 = f_vals1 + (1.0 - f0_1)
    pred2 = f_vals2 + (1.0 - f0_2)
    pred3 = f_vals3 + (0.1 - f0_3)
    L_f = losses[-1]
    np.save(os.path.join(output_dir, "inter_pred1.npy"), pred1)
    np.save(os.path.join(output_dir, "inter_pred2.npy"), pred2)
    np.save(os.path.join(output_dir, "inter_pred3.npy"), pred3)
    np.save(os.path.join(output_dir, "inter_L_f.npy"), L_f)

    # Plot loss
    plt.figure(figsize=(8, 8))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(f"{output_dir}/inter_losses.png", bbox_inches="tight")
    plt.close()

    # Plot pred
    plt.figure(figsize=(8, 8))
    plt.plot(x_test, pred1, label="p(x)")
    plt.plot(x_test, pred2, label="T(x)")
    plt.plot(x_test, pred3, label="V(x)")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(f"{output_dir}/inter_f_x.png", bbox_inches="tight")
    plt.close()

    print(f"L_f = {L_f:.6f}")
    
    f_reg1 = jax.vmap(lambda x: f_x(x, params[0], num_qubits, d, bond_dimension))(x_reg)
    f_reg2 = jax.vmap(lambda x: f_x2(x, params[1], num_qubits, d, bond_dimension))(x_reg)
    f_reg3 = jax.vmap(lambda x: f_x3(x, params[2], num_qubits, d, bond_dimension))(x_reg)
    reg1 = f_reg1 + (1.0 - f0_1)
    reg2 = f_reg2 + (1.0 - f0_2)
    reg3 = f_reg3 + (0.1 - f0_3)
    opt = optax.adam(lr2)
    opt_state = opt.init(params)
    losses = []
    t0 = time.time()
    for epoch in range(epochs2):
        t1 = time.time()
        loss, grads = jax.value_and_grad(loss_function2)(params, x_train2, x_reg, num_qubits, d, bond_dimension, epoch, epochs2, reg1, reg2, reg3)
        if jnp.isnan(loss) or jnp.isnan(jnp.sum(grads[0]) + jnp.sum(grads[1]) + jnp.sum(grads[2])):
            raise ValueError(f"[FATAL] NaN encountered at epoch {epoch} for bond={bond_dimension}, init={init_index}. Terminating script.")
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: l_f = {loss:.6f} (epoch time: {time.time() - t1:.2f}s)")
    print(f"SubTraining time: {time.time() - t0:.2f}s")
    
    np.save(os.path.join(output_dir, "losses.npy"), losses)
    f0_1 = f_x(0.0, params[0], num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, params[1], num_qubits, d, bond_dimension)
    f0_3 = f_x3(0.0, params[2], num_qubits, d, bond_dimension)
    f_vals1 = jax.vmap(lambda x: f_x(x, params[0], num_qubits, d, bond_dimension))(x_test)
    f_vals2 = jax.vmap(lambda x: f_x2(x, params[1], num_qubits, d, bond_dimension))(x_test)
    f_vals3 = jax.vmap(lambda x: f_x3(x, params[2], num_qubits, d, bond_dimension))(x_test)
    pred1 = f_vals1 + (1.0 - f0_1)
    pred2 = f_vals2 + (1.0 - f0_2)
    pred3 = f_vals3 + (0.1 - f0_3)
    L_f = losses[-1]
    np.save(os.path.join(output_dir, "pred1.npy"), pred1)
    np.save(os.path.join(output_dir, "pred2.npy"), pred2)
    np.save(os.path.join(output_dir, "pred3.npy"), pred3)
    np.save(os.path.join(output_dir, "L_f.npy"), L_f)

    # Plot loss
    plt.figure(figsize=(8, 8))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(f"{output_dir}/losses.png", bbox_inches="tight")
    plt.close()

    # Plot pred
    plt.figure(figsize=(8, 8))
    plt.plot(x_test, pred1, label="p(x)")
    plt.plot(x_test, pred2, label="T(x)")
    plt.plot(x_test, pred3, label="V(x)")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(f"{output_dir}/f_x.png", bbox_inches="tight")
    plt.close()

    print(f"L_f = {L_f:.6f}")
    
# --- Run Main ---
if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"[SUMMARY] Full script time: {time.time() - start_time:.2f}s")

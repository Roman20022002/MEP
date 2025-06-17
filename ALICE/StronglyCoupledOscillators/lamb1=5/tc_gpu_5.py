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
def phi(x, q):
    return q * jnp.arccos(x)

def feature_map(c, x, num_qubits):
    for i in range(num_qubits):
        c.ry(i, theta=phi(x, i + 1))
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


# --- Loss ---
def u_target1(x, lamb1):
    return 0.375 * jnp.sin(4*x) + 0.5 * jnp.cos(4*x)
def u_target2(x, lamb2):
    return -0.625 * jnp.sin(4*x) 
    
def loss_function(params, x_values, num_qubits, d, bond_dimension, lamb1, lamb2, b1=0.5, b2=0.0):
    theta, theta2 = params

    def f_single(x): return f_x(x, theta, num_qubits, d, bond_dimension)
    def f_single2(x): return f_x2(x, theta2, num_qubits, d, bond_dimension)

    f0_1 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, theta2, num_qubits, d, bond_dimension)

    f_vg1 = jax.value_and_grad(f_single)
    f_vg2 = jax.value_and_grad(f_single2)

    f_vals1, df_dx1 = jax.vmap(f_vg1)(x_values)
    f_vals2, df_dx2 = jax.vmap(f_vg2)(x_values)

    f1_shifted = f_vals1 + (b1 - f0_1)
    f2_shifted = f_vals2 + (b2 - f0_2)

    a1 = df_dx1 -lamb1 * f2_shifted -lamb2 * f1_shifted
    a2 = df_dx2 + lamb2 * f2_shifted + lamb1 * f1_shifted

    loss1 = jnp.mean(a1 ** 2)
    loss2 = jnp.mean(a2 ** 2)

    return loss1 + loss2



# --- Training ---
def train():
    print(f"\n[INFO] Running train with bond dimension: {bond_dimension}")
    print(f"[INFO] Using init index: {init_index}")
    num_qubits, d, epochs, lr, lamb1, lamb2 = 6, 5, 250, 0.02, 5, 3
    
    def generate_chebyshev_grid(n, a=0.0, b=0.9):
            k = jnp.arange(n)
            chebyshev_nodes = jnp.cos(jnp.pi * (2 * k + 1) / (2 * n))
            scaled_nodes = ((chebyshev_nodes + 1) * (b - a) / 2) + a
            return scaled_nodes[::-1]
    x_train = generate_chebyshev_grid(20)
    x_test = generate_chebyshev_grid(100)
    
    init_path = f"init_thetas_5/init_{init_index}.npy"
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"[FATAL] Missing init file: {init_path}. Terminating script.")
    theta = jnp.array(np.load(init_path), dtype=jnp.float32)
    theta2 = jnp.array(np.load(init_path), dtype=jnp.float32)
    params = (theta, theta2)
    
    output_dir = f"plots5_bond_{bond_dimension}/index_{init_index}"
    os.makedirs(output_dir, exist_ok=True)

    opt = optax.adam(lr)
    opt_state = opt.init(params)
    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        loss, grads = jax.value_and_grad(loss_function)(params, x_train, num_qubits, d, bond_dimension, lamb1, lamb2)
        if jnp.isnan(loss) or jnp.isnan(jnp.sum(grads[0]) + jnp.sum(grads[1])):
            raise ValueError(f"[FATAL] NaN encountered at epoch {epoch} for bond={bond_dimension}, init={init_index}. Terminating script.")
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        if epoch % 10 == 0:
            f0_1 = f_x(0.0, params[0], num_qubits, d, bond_dimension)
            f0_2 = f_x2(0.0, params[1], num_qubits, d, bond_dimension)
            f_vals1 = jax.vmap(lambda x: f_x(x, params[0], num_qubits, d, bond_dimension))(x_train)
            f_vals2 = jax.vmap(lambda x: f_x2(x, params[1], num_qubits, d, bond_dimension))(x_train)
            f1 = f_vals1 + (0.5 - f0_1)
            f2 = f_vals2 + (0.0 - f0_2)
            u1 = u_target1(x_train, lamb1)
            u2 = u_target2(x_train, lamb2)
            l_q1 = jnp.sum((u1-f1)**2)/20
            l_q2 = jnp.sum((u2-f2)**2)/20
            print(f"Epoch {epoch}: l_f = {loss:.6f}, l_q1 = {l_q1:.6f}, l_q2 = {l_q2:.6f} (epoch time: {time.time() - t1:.2f}s)")
    print(f"SubTraining time: {time.time() - t0:.2f}s")
    
    np.save(os.path.join(output_dir, "losses.npy"), losses)
    f0_1 = f_x(0.0, params[0], num_qubits, d, bond_dimension)
    f0_2 = f_x2(0.0, params[1], num_qubits, d, bond_dimension)
    f_vals1 = jax.vmap(lambda x: f_x(x, params[0], num_qubits, d, bond_dimension))(x_test)
    f_vals2 = jax.vmap(lambda x: f_x2(x, params[1], num_qubits, d, bond_dimension))(x_test)
    pred1 = f_vals1 + (0.5 - f0_1)
    pred2 = f_vals2 + (0.0 - f0_2)
    u_true1 = u_target1(x_test, lamb1)
    u_true2 = u_target2(x_test, lamb2)
    L_q1 = jnp.sum((u_true1-pred1)**2)/100
    L_q2 = jnp.sum((u_true2-pred2)**2)/100
    L_f = losses[-1]
    np.save(os.path.join(output_dir, "pred1.npy"), pred1)
    np.save(os.path.join(output_dir, "pred2.npy"), pred2)
    np.save(os.path.join(output_dir, "L_f.npy"), L_f)
    np.save(os.path.join(output_dir, "L_q1.npy"), L_q1)
    np.save(os.path.join(output_dir, "L_q2.npy"), L_q2)

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
    plt.plot(x_test, u_true1, label="u1(x)")
    plt.plot(x_test, pred1, label="f1(x)")
    plt.plot(x_test, u_true2, label="u2(x)")
    plt.plot(x_test, pred2, label="f2(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig(f"{output_dir}/f_x.png", bbox_inches="tight")
    plt.close()

    print(f"L_f = {L_f:.6f}, L_q1 = {L_q1:.6f}, L_q2 = {L_q2:.6f}")
    

# --- Run Main ---
if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"[SUMMARY] Full script time: {time.time() - start_time:.2f}s")

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


# --- Circuit ---
def create_circuit(x, theta, num_qubits, d):
    c = tc.Circuit(num_qubits)
    c = feature_map(c, x, num_qubits)
    c = ansatz(c, theta, num_qubits, d)
    return c

def f_x(x, theta, num_qubits, d):
    c = create_circuit(x, theta, num_qubits, d)
    return jnp.real(sum(c.expectation_ps(z=[i]) for i in range(num_qubits)))
 
def func_and_deriv(x_values, num_qubits, d, theta, b=1.0): #deriv later
    f0 = f_x(0.0, theta, num_qubits, d)
    def f_shifted(x):
        return f_x(x, theta, num_qubits, d) + (b-f0)
    f_vals = jax.vmap(f_shifted)(x_values)
    return f_vals
    

# --- Loss ---
def u_target(x, k, lamb):
    return jnp.exp(-k * lamb * x) * jnp.cos(lamb * x)

def loss_function(theta, x_values, num_qubits, d, k, lamb, b=1.0):
    def f_single(x):
        return f_x(x, theta, num_qubits, d)
    f0 = f_x(0.0, theta, num_qubits, d)
    f_vg = jax.value_and_grad(f_single)
    f_vals, df_dx = jax.vmap(f_vg)(x_values)
    f_shifted_vals = f_vals + (b - f0)
    a = df_dx + lamb * f_shifted_vals * (k + jnp.tan(lamb * x_values))
    loss = jnp.mean(a ** 2)
    return loss


# --- Training ---
def train():
    print(f"[INFO] Using init index: {init_index}")
    num_qubits, d, epochs, lr, k, lamb = 6, 5, 250, 0.1, 0.1, 8
    
    def generate_chebyshev_grid(n, a=0.0, b=0.9):
            k = jnp.arange(n)
            chebyshev_nodes = jnp.cos(jnp.pi * (2 * k + 1) / (2 * n))
            scaled_nodes = ((chebyshev_nodes + 1) * (b - a) / 2) + a
            return scaled_nodes[::-1]
    x_train = generate_chebyshev_grid(20)
    x_test = generate_chebyshev_grid(100)
    
    init_path = f"init_thetas_A/init_{init_index}.npy"
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"[FATAL] Missing init file: {init_path}. Terminating script.")
    theta = jnp.array(np.load(init_path), dtype=jnp.float32)

    output_dir = f"plots8/index_{init_index}"
    os.makedirs(output_dir, exist_ok=True)

    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    losses = []
    
    t0 = time.time()
    for epoch in range(epochs):
    	t1 = time.time()
    	loss, grads = jax.value_and_grad(loss_function)(theta, x_train, num_qubits, d, k, lamb)
    	if jnp.isnan(loss) or jnp.isnan(jnp.sum(grads)):
    		raise ValueError(f"[FATAL] NaN encountered at epoch {epoch}, init={init_index}. Terminating script.")
    	updates, opt_state = opt.update(grads, opt_state, theta)
    	theta = optax.apply_updates(theta, updates)
    	losses.append(loss)
    	if epoch % 10 == 0:
    		f0 = f_x(0.0, theta, num_qubits, d)
    		f_vals, _ = jax.vmap(jax.value_and_grad(lambda x: f_x(x, theta, num_qubits, d)))(x_train)
    		f = f_vals + (1.0 - f0)
    		u = u_target(x_train, k, lamb)
    		l_q = jnp.sum((u-f)**2)/20
    		print(f"Epoch {epoch}: l_f = {loss:.6f}, l_q = {l_q:.6f} (epoch time: {time.time() - t1:.2f}s)")
    print(f"SubTraining time: {time.time() - t0:.2f}s")
    
    np.save(os.path.join(output_dir, "losses.npy"), losses)
    f0 = f_x(0.0, theta, num_qubits, d)
    f_vals = jax.vmap(lambda x: f_x(x, theta, num_qubits, d))(x_test)
    pred = f_vals + (1.0 - f0)
    u_true = u_target(x_test, k, lamb)
    L_q = jnp.sum((u_true-pred)**2)/100
    L_f = losses[-1]
    np.save(os.path.join(output_dir, "pred.npy"), pred)
    np.save(os.path.join(output_dir, "L_f.npy"), L_f)
    np.save(os.path.join(output_dir, "L_q.npy"), L_q)

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
    plt.plot(x_test, u_true, label="u(x)")
    plt.plot(x_test, pred, label="f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig(f"{output_dir}/f_x.png", bbox_inches="tight")
    plt.close()

    print(f"L_f = {L_f:.6f}, L_q = {L_q:.6f}")
    

# --- Run Main ---
if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"[SUMMARY] Full script time: {time.time() - start_time:.2f}s")

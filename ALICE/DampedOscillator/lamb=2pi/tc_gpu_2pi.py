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
 
def func_and_deriv(x_values, num_qubits, d, theta, bond_dimension, b=1.0): #deriv later
    f0 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    def f_shifted(x):
        return f_x(x, theta, num_qubits, d, bond_dimension) + (b-f0)
    f_vals = jax.vmap(f_shifted)(x_values)
    return f_vals

def get_svd_vector_mps(circuit, bond_index):
    tensors = circuit.get_tensors()
    assert 0 < bond_index < len(tensors), "Invalid bond index"
    A = tensors[bond_index - 1]
    B = tensors[bond_index]
    
    A_shape = A.shape
    B_shape = B.shape
    
    A_matrix = jnp.reshape(A, (A_shape[0] * A_shape[1], A_shape[2]))
    B_matrix = jnp.reshape(B, (B_shape[0], B_shape[1] * B_shape[2]))
    
    bond_matrix = jnp.dot(A_matrix, B_matrix)
    _, s, _ = jnp.linalg.svd(bond_matrix, full_matrices=False)
    return s
def compute_avg_max_entropy(theta, x_values, num_qubits, d, bond_dimension):
    avg_entropies = []
    max_entropies = []
    for x in x_values:
        circuit = create_mps_circuit(x, theta, num_qubits, d, bond_dimension)

        entropies = []
        for i in range(1, num_qubits):
            s = get_svd_vector_mps(circuit, i)
            s_squared = jnp.clip(s**2, 1e-12, 1.0)
            entropy = -jnp.sum(s_squared * jnp.log(s_squared))
            entropies.append(entropy)
        avg_entropies.append(jnp.mean(jnp.array(entropies)))
        max_entropies.append(jnp.max(jnp.array(entropies)))
    return jnp.mean(jnp.array(avg_entropies)), jnp.mean(jnp.array(max_entropies))


# --- Loss ---
def u_target(x, k, lamb):
    return jnp.exp(-k * x)*jnp.cos(lamb * x)

def loss_function(theta, x_values, num_qubits, d, bond_dimension, k, lamb, b=1.0):
    def f_single(x):
        return f_x(x, theta, num_qubits, d, bond_dimension)
    f0 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    f_vg = jax.value_and_grad(f_single)
    f_vals, df_dx = jax.vmap(f_vg)(x_values)
    f_shifted_vals = f_vals + (b - f0)
    a = df_dx + k * f_shifted_vals + lamb*jnp.exp(-k * x_values)*jnp.sin(lamb * x_values)
    loss = jnp.mean(a ** 2)
    return loss


# --- Training ---
def train():
    print(f"\n[INFO] Running train with bond dimension: {bond_dimension}")
    print(f"[INFO] Using init index: {init_index}")
    num_qubits, d, epochs, lr, k, lamb = 4, 3, 100, 0.1, 1, 2*jnp.pi
    
    def generate_chebyshev_grid(n, a=0.0, b=0.9):
            k = jnp.arange(n)
            chebyshev_nodes = jnp.cos(jnp.pi * (2 * k + 1) / (2 * n))
            scaled_nodes = ((chebyshev_nodes + 1) * (b - a) / 2) + a
            return scaled_nodes[::-1]
    x_train = generate_chebyshev_grid(20)
    x_test = generate_chebyshev_grid(100)
    
    init_path = f"init_thetas_2pi/init_{init_index}.npy"
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"[FATAL] Missing init file: {init_path}. Terminating script.")
    theta = jnp.array(np.load(init_path), dtype=jnp.float32)

    output_dir = f"plots2pi_bond_{bond_dimension}/index_{init_index}"
    os.makedirs(output_dir, exist_ok=True)

    opt = optax.adam(lr)
    opt_state = opt.init(theta)
    losses = []
    avg_entropy_hist = []
    max_entropy_hist = []
    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        loss, grads = jax.value_and_grad(loss_function)(theta, x_train, num_qubits, d, bond_dimension, k, lamb)
        if jnp.isnan(loss) or jnp.isnan(jnp.sum(grads)):
            raise ValueError(f"[FATAL] NaN encountered at epoch {epoch} for bond={bond_dimension}, init={init_index}. Terminating script.")
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        losses.append(loss)

        avg_entropy, max_entropy = compute_avg_max_entropy(theta, x_train, num_qubits, d, bond_dimension)
        avg_entropy_hist.append(avg_entropy)
        max_entropy_hist.append(max_entropy)
        if epoch % 10 == 0:
            f0 = f_x(0.0, theta, num_qubits, d, bond_dimension)
            f_vals, _ = jax.vmap(jax.value_and_grad(lambda x: f_x(x, theta, num_qubits, d, bond_dimension)))(x_train)
            f = f_vals + (1.0 - f0)
            u = u_target(x_train, k, lamb)
            l_q = jnp.sum((u-f)**2)/20
            maxi = jnp.log(bond_dimension)
            print(f"Epoch {epoch}: l_f = {loss:.6f}, l_q = {l_q:.6f}, avg S = {avg_entropy:.4f}/{maxi:.4f}, max S = {max_entropy:.4f}/{maxi:.4f} (epoch time: {time.time() - t1:.2f}s)")
    print(f"SubTraining time: {time.time() - t0:.2f}s")
    
    np.save(os.path.join(output_dir, "losses.npy"), losses)
    f0 = f_x(0.0, theta, num_qubits, d, bond_dimension)
    f_vals = jax.vmap(lambda x: f_x(x, theta, num_qubits, d, bond_dimension))(x_test)
    pred = f_vals + (1.0 - f0)
    u_true = u_target(x_test, k, lamb)
    L_q = jnp.sum((u_true-pred)**2)/100
    L_f = losses[-1]
    np.save(os.path.join(output_dir, "pred.npy"), pred)
    np.save(os.path.join(output_dir, "L_f.npy"), L_f)
    np.save(os.path.join(output_dir, "L_q.npy"), L_q)
    np.save(os.path.join(output_dir, "avg_entropy_hist.npy"), avg_entropy_hist)
    np.save(os.path.join(output_dir, "max_entropy_hist.npy"), max_entropy_hist)

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

    max_entropy_bound = jnp.log(bond_dimension)
    normalized_avg_entropy = [s / max_entropy_bound for s in avg_entropy_hist]
    normalized_max_entropy = [s / max_entropy_bound for s in max_entropy_hist]
    # Plot normalized entropies
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_avg_entropy, label="Normalized Avg Entropy")
    plt.plot(normalized_max_entropy, label="Normalized Max Entropy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy /log(X)")
    plt.legend()
    plt.savefig(f"{output_dir}/normalized_entropies.png", bbox_inches="tight")
    plt.close()
    

# --- Run Main ---
if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"[SUMMARY] Full script time: {time.time() - start_time:.2f}s")

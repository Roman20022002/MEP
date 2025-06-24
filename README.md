# The Role of Entanglement in Solving Differential Equations

## JAX-Based Quantum & ML Toolkit (ALICE HPC)

This project is intended to run on the **ALICE GPU cluster** at Leiden University (see [ALICE](https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/pages/37519361/ALICE) documentation). It uses Python 3.12, [JAX](https://github.com/google/jax) with CUDA 12 support, and scientific libraries such as `tensorcircuit`.

---

### Step 1 – Environment Setup (ALICE)

Follow this single block of commands to set up your environment on ALICE:

```bash
# Load Python 3.12 module
module load python/3.12

# Create and activate a virtual environment
python -m venv jax_venv
source jax_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX with CUDA 12 (GPU support)
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install additional dependencies
pip install -r requirements.txt
```

See also ALICE [tutorial](https://pubappslu.atlassian.net/wiki/spaces/HPCWIKI/pages/37027959/Your+first+Python+job) to install Python modules.


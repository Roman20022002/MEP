#!/bin/bash
#SBATCH --job-name=tc_gpu_A
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user="r.a.r.ilic@student.tudelft.nl"
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"
#SBATCH --ntasks=1
#SBATCH --array=0-79   # 8 bond dimensions * 10 init indices = 80 jobs

mkdir -p logs

module load Python/3.12.3-GCCcore-13.3.0
source $HOME/Documents/MPS/cuda-samples/jax_venv/bin/activate

bond_dimensions=(8 7 6 5 4 3 2 1)

# Total number of init files per bond dimension
num_inits=10

# Derive bond dimension and init index from task ID
bond_index=$((SLURM_ARRAY_TASK_ID / num_inits))
init_index=$((SLURM_ARRAY_TASK_ID % num_inits))
bond_dimension=${bond_dimensions[$bond_index]}

# Export init index for the Python script to read
export INIT_INDEX=$init_index

echo "[$SHELL] #### Starting GPU TensorCircuit array job"
echo "[$SHELL] This is $SLURM_JOB_USER and my MPS job has the ID $SLURM_JOB_ID, Task ID $SLURM_ARRAY_TASK_ID"
echo "[$SHELL] Using bond dimension: $bond_dimension and init index: $init_index"

# Get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: $CWD"

# Which GPU has been assigned
echo "[$SHELL] Using GPU: $CUDA_VISIBLE_DEVICES"

# Set the path to the python file
export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

# Set name of the python file
export PYFILE=$CWD/tc_gpu_A.py

# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH/tc_gpu_A_bond_${bond_dimension}_init_${init_index}_${SLURM_JOB_ID} 
mkdir -p $RUNDIR
echo "[$SHELL] Run directory"$RUNDIR

# Create directory for plots
export PLOTDIR=$RUNDIR/plotsA_bond_${bond_dimension}/index_${init_index}
mkdir -p $PLOTDIR

# Copy script and init_thetas to local scratch directory and change into it
cp $PYFILE $RUNDIR/
cp -r $CWD/init_thetas_A $RUNDIR/
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script with bond dimension $bond_dimension and init index $init_index"
python3 tc_gpu_A.py --bond_dimension $bond_dimension
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
mkdir -p $CWD/plotsA_bond_${bond_dimension}
if cp -r $RUNDIR/plotsA_bond_${bond_dimension}/index_${init_index} $CWD/plotsA_bond_${bond_dimension}/; then
    echo "[$SHELL] Successfully copied results to $CWD/plotsA_bond_${bond_dimension}/index_${init_index}"
else
    echo "[$SHELL] ERROR: Failed to copy results"
fi

echo "[$SHELL] #### Finished GPU TensorCircuit array job. Have a nice day"

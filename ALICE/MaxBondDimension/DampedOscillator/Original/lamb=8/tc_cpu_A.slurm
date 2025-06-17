#!/bin/bash
#SBATCH --job-name=tc_cpu_8
#SBATCH --output=logs8/%x_%A_%a.out
#SBATCH --error=logs8/%x_%A_%a.err
#SBATCH --mail-user="r.a.r.ilic@student.tudelft.nl"
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"
#SBATCH --ntasks=1
#SBATCH --array=0-9   # 8 bond dimensions * 10 init indices = 80 jobs

mkdir -p logs8

module load Python/3.12.3-GCCcore-13.3.0
source $HOME/Documents/MPS/cuda-samples/jax_venv/bin/activate


# Total number of init files per bond dimension
num_inits=10

# Derive bond dimension and init index from task ID
init_index=$((SLURM_ARRAY_TASK_ID % num_inits))

# Export init index for the Python script to read
export INIT_INDEX=$init_index

echo "[$SHELL] #### Starting GPU TensorCircuit array job"
echo "[$SHELL] This is $SLURM_JOB_USER and my MPS job has the ID $SLURM_JOB_ID, Task ID $SLURM_ARRAY_TASK_ID"
echo "[$SHELL] Using bond dimension: init index: $init_index"

# Get the current working directory
export CWD=$(pwd)
echo "[$SHELL] CWD: $CWD"

# Which GPU has been assigned
echo "[$SHELL] Using GPU: $CUDA_VISIBLE_DEVICES"

# Set the path to the python file
export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

# Set name of the python file
export PYFILE=$CWD/tc_cpu_8.py

# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH/tc_cpu_8_init_${init_index}_${SLURM_JOB_ID} 
mkdir -p $RUNDIR
echo "[$SHELL] Run directory"$RUNDIR

# Create directory for plots
export PLOTDIR=$RUNDIR/plots8/index_${init_index}
mkdir -p $PLOTDIR

# Copy script and init_thetas to local scratch directory and change into it
cp $PYFILE $RUNDIR/
cp -r $CWD/init_thetas_A $RUNDIR/
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script with bond dimension init index $init_index"
python3 tc_cpu_8.py 
echo "[$SHELL] Script finished"

# Move stat directory back to CWD
echo "[$SHELL] Copy files back to cwd"
mkdir -p $CWD/plots8
if cp -r $RUNDIR/plots8/index_${init_index} $CWD/plots8/; then
    echo "[$SHELL] Successfully copied results to $CWD/plots8/index_${init_index}"
else
    echo "[$SHELL] ERROR: Failed to copy results"
fi

echo "[$SHELL] #### Finished GPU TensorCircuit array job. Have a nice day"

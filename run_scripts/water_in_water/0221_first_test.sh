#!/bin/sh

PROJECT_NAME="fab-torch"

# Original code folder is here
BASE_DIR="${SCRATCH:-$HOME}"
MAIN_DIR="${BASE_DIR}/${PROJECT_NAME}"
CONDA_ENV_DIR=${BASE_DIR}/anaconda3/envs/bgsol

CONDA_ENV="bgsol"

JOB_NAME="h2o_test"

# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch/
mkdir -p "${LAUNCH_DIR}"

# Create dir for specific experiment run
dt=$(date '+%F_%H-%M-%S.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"

# Copy code to experiment folder
rsync -arm "${MAIN_DIR}/" --stats --exclude-from="${MAIN_DIR}/SYNC_EXCLUDE" "${LOGS_DIR}/${PROJECT_NAME}/"
cd "${LOGS_DIR}/${PROJECT_NAME}"


# Make SLURM file
SLURM="${LOGS_DIR}/run.sh"
cat > "${SLURM}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm-h2o-%j.out
#SBATCH --error=logs/slurm-h2o-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:05:00

module purge
module load 2025
module load Anaconda3/2025.06-1

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}:\$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

nvidia-smi

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \\
  --config-name h2oinh2o_forwardkl.yaml \\
  flow.blocks=12 flow.hidden_units=256 flow.num_bins=9 \\
  training.n_iterations=500 evaluation.n_eval=50 evaluation.n_plots=10 evaluation.n_checkpoints=1
EOF

sbatch ${SLURM}


# TODO:
#  1. Install OpenMMtools with conda: conda config --add channels omnia --add channels conda-forge; conda install openmmtools
#  1b. If this does not work, rebuild the env, doing pip install after the openmm install.
#  2. Try running the experiment test script again.
#  2b. Debug packages as they come up.
#  3. If it works, run the hyperparam experiment script.

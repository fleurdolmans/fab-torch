#!/bin/bash
#SBATCH --job-name=fab-torch-md
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=0:50:00

module purge
module load 2025
module load Anaconda3/2025.06-1

PROJECT_NAME="fab-torch"

# Original code folder is here
BASE_DIR="${SCRATCH:-$HOME}"
MAIN_DIR="${BASE_DIR}/${PROJECT_NAME}"
CONDA_ENV_DIR=${BASE_DIR}/anaconda3/envs/bgsol

CONDA_ENV="bgsol"

# Working directory in scratch
WORKDIR="/scratch-shared/$USER/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Conda activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Diagnostics
srun nvidia-smi

# Environment variables (no need to force CUDA_VISIBLE_DEVICES; Slurm sets GPU visibility)
export PYTHONPATH="${MAIN_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

srun python "${MAIN_DIR}/experiments/solvation/create_md_data.py" \
  --config-name make_md_data.yaml solute_name=water solute_xml_path=null

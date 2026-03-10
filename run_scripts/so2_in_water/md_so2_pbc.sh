#!/bin/bash
#SBATCH --job-name=md-so2-in-h2o
#SBATCH --output=logs/slurm-so2-%j.out
#SBATCH --error=logs/slurm-so2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:40:00

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
export MAIN_DIR="${MAIN_DIR}"


# Create MD data for water in water (PBC) and plot diagnostics
# srun python "${MAIN_DIR}/data_generation/md_data.py" \
#   --config-name make_md_data \
#   solute_name=so2 \
#   simulation_version="v7" \
#   create_md=true \
#   validate_md=true \
#   boundary_condition=pbc \
#   nonbonded_cutoff_nm=0.8 \
#   femtoseconds_per_timestep=2.0 \
#   solvent_density=1.0 \
#   box_length_nm=1.8 \
#   internal_constraints=hbonds \
#   rigid_water=true \
#   report_interval=1e4 \
#   save_interval=100 \

srun python "${MAIN_DIR}/data_generation/md_data.py" \
  --config-name make_md_data \
  solute_name=so2 \
  simulation_version="v10" \
  create_md=true \
  validate_md=true \
  boundary_condition=pbc \
  nonbonded_cutoff_nm=0.4 \
  femtoseconds_per_timestep=2.0 \
  solvent_density=1.0 \
  box_length_nm=0.9 \
  internal_constraints=hbonds \
  rigid_water=true \
  report_interval=1e4 \
  save_interval=100 \

  # Speed up version for testing:
  # equi_steps=1e4 \
  # burnin_steps=1e5 \
  # num_steps=1e7 \
  # report_interval=1e4 \
  # save_interval=1000 \

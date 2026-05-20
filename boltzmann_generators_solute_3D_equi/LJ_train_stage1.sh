#!/bin/bash
#SBATCH --job-name=slurm_lj_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=staging
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

PROJECT_NAME="fab-torch"

# Original code folder is here
BASE_DIR="${SCRATCH:-$HOME}"
MAIN_DIR="${BASE_DIR}/${PROJECT_NAME}"

CONDA_ENV="bgsol"
JOB_NAME="LJ_train"

# Launch dir
LAUNCH_DIR="${MAIN_DIR}/launch"
mkdir -p "${LAUNCH_DIR}"

# Create dir for specific experiment run
dt=$(date '+%F_%H-%M-%S.%3N')
LOGS_DIR="${LAUNCH_DIR}/${dt}"
mkdir -p "${LOGS_DIR}"

# Copy code to experiment folder
rsync -arm "${MAIN_DIR}/" \
  --stats \
  --exclude-from="${MAIN_DIR}/SYNC_EXCLUDE" \
  "${LOGS_DIR}/${PROJECT_NAME}/"

cd "${LOGS_DIR}/${PROJECT_NAME}"

# Make SLURM file
SLURM="${LOGS_DIR}/run.sh"
mkdir -p "${LOGS_DIR}/logs"

cat > "${SLURM}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOGS_DIR}/logs/slurm-LJ-%j.out
#SBATCH --error=${LOGS_DIR}/logs/slurm-LJ-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

source \$(conda info --base)/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER=\${MKL_INTERFACE_LAYER:-LP64}

set +u
conda activate ${CONDA_ENV}
set -u

export PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}\${PYTHONPATH:+:\$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MAIN_DIR="${MAIN_DIR}"

# Use job-local scratch if available, otherwise fallback to /tmp
export SAVE_DIR="\${TMPDIR:-/tmp}/equivariant_3d"
mkdir -p "\$SAVE_DIR/md"

echo "Running on host: \$(hostname)"
echo "TMPDIR=\${TMPDIR:-unset}"
echo "SAVE_DIR=\$SAVE_DIR"
ls -ld "\$SAVE_DIR" "\$SAVE_DIR/md"

python -c "import sys, hydra; print('PY', sys.executable, 'hydra', hydra.__version__)"
python -c "import numpy; from openmm import app; print('numpy', numpy.__version__, 'openmm app OK')"

nvidia-smi

# stage 0: MD
# python ${LOGS_DIR}/${PROJECT_NAME}/boltzmann_generators_solute_3D_equi/Notebooks/train_equivariant_3d.py \\
#     --run_md \\
#     --mc_cache "\$SAVE_DIR/md/xtraj.npy" \\
#     --save_dir "\$SAVE_DIR"

# Stage 1: ML

MD_CACHE="/home/fdolmans/HDD/data/LJ2/md/xtraj.npy"
python ${LOGS_DIR}/${PROJECT_NAME}/boltzmann_generators_solute_3D_equi/Notebooks/train_equivariant_3d.py \\
    --run_stage1 \\
    --mc_cache "\$MD_CACHE" \\
    --save_dir "\$SAVE_DIR" \\
    --metric_every 10 \\
    --plot_every 10 \\
    --checkpoint_every 10 \\
    --metric_samples 2048 \\
    --eval_samples 2000 \\
    --wandb_group equivariant_3d \\
    --wandb_name stage1_ml

# Copy results back to persistent launch directory before TMPDIR disappears
mkdir -p "${LOGS_DIR}/results"
rsync -a "\$SAVE_DIR/" "${LOGS_DIR}/results/"
echo "Results copied to ${LOGS_DIR}/results"
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"
sbatch "${SLURM}"
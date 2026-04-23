#!/bin/bash
#SBATCH --job-name=slurm_lj_base_transport
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
JOB_NAME="LJ_test"

# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch
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
mkdir -p "${LOGS_DIR}/logs"   # ensure GPU job logs dir exists
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
#SBATCH --time=00:10:00

module purge
module load 2025
module load Anaconda3/2025.06-1

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python -c "import sys, hydra; print('PY', sys.executable, 'hydra', hydra.__version__)"
python -c "import numpy; from openmm import app; print('numpy', numpy.__version__, 'openmm app OK')"

export PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}:\$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MAIN_DIR="${MAIN_DIR}"

nvidia-smi

# We are essentially just using the loss_type and use_ais arguments when doing forward KL training.

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run_LJ.py \\
  --config-name LJ \\
  target.base_state=v4 target.target_state=v5 target.solute_sigma_nm=0.4\\
  fab.loss_type=base_transport fab.use_ais=false \\
  flow.hidden_units=128 flow.base.type=gauss-uni flow.base.learn_mean_var=false flow.type=lj_coupling_torus\\
  flow.n_layers=12 flow.layer_nodes_per_dim=4 \\
  training.lr=7e-5 training.wd=1e-6 training.batch_size=502 evaluation.eval_batch_size=128\\
  training.max_grad_norm=10 training.warmup_iter=100 \\
  training.overlap_penalty=0.0 training.mixing=0.0 training.energy_mode=full target.transform_version=v1\\
  training.n_iterations=300 training.buffer.use=false training.buffer.prioritised=false training.lr_scheduler.decay_iter=300\\
  evaluation.n_eval=6 evaluation.n_plots=6 evaluation.n_checkpoints=1 
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}
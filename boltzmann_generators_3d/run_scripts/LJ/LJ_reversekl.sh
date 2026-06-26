#!/bin/bash
#SBATCH --job-name=slurm_lj_reversekl
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
#SBATCH --partition=gpu_h100
#SBATCH --time=01:40:00

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

# We are essentially just using the loss_type and use_ais arguments when doing reverse KL training.

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run_LJ.py \\
  --config-name LJ \\
  target.base_state=v10 target.target_state=v10 target.solid=false\\
  target.solute_sigma_nm=0.38 target.solvent_sigma_nm=0.34 target.n_solvent=32\\
  target.energy_cut=1.e+20 target.energy_max=1.e+30 target.temperature=100\\
  fab.loss_type=flow_reverse_kl fab.use_ais=false \\
  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/LJ/MD_training/2026-04-18/14-37-42_995342 \\
  flow.hidden_units=128 flow.base.type=multishell-torus flow.base.learn_mean_var=false flow.type=lj_torus_spline\\
  flow.n_layers=6 flow.layer_nodes_per_dim=4 flow.hidden_units=128 flow.num_bins=8 \\
  training.lr=1e-5 training.wd=1e-6 training.batch_size=64 evaluation.eval_batch_size=32\\
  training.max_grad_norm=10 training.warmup_iter=50 \\
  training.overlap.penalty=1.0 training.overlap.dist_ssolv=0.32 training.overlap.dist_solute=0.35 training.mixing=0.2 training.energy_mode=full target.transform_version=v4\\
  training.n_iterations=200 training.buffer.use=false training.buffer.prioritised=false training.lr_scheduler.decay_iter=200\\
  evaluation.n_eval=4 evaluation.n_plots=4 evaluation.n_checkpoints=1 
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"


sbatch ${SLURM}

# training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/LJ/MD_training/2026-04-20/12-31-20_576657 \\
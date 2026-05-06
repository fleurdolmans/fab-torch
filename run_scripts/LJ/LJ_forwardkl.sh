#!/bin/bash
#SBATCH --job-name=slurm_lj_forwardkl
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
#SBATCH --time=00:20:00

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
  target.base_state=v10 target.target_state=v10 target.solid=false\\
  target.solute_sigma_nm=0.38 target.solvent_sigma_nm=0.34 target.n_solvent=32\\
  target.energy_cut=1.e+20 target.energy_max=1.e+30 target.temperature=100\\
  fab.loss_type=forward_kl fab.use_ais=false \\
  flow.hidden_units=128 flow.base.type=shell-torus flow.base.learn_mean_var=false flow.type=lj_torus_spline\\
  flow.n_layers=6 flow.layer_nodes_per_dim=4 flow.hidden_units=128 flow.num_bins=8\\
  training.lr=7e-5 training.wd=1e-6 training.batch_size=64 evaluation.eval_batch_size=32\\
  training.max_grad_norm=1 training.warmup_iter=100 \\
  training.overlap.penalty=100.0 training.overlap.dist_ssolv=0.31 training.overlap.dist_solute=0.31 training.mixing=0.0 training.energy_mode=full target.transform_version=v4\\
  training.n_iterations=600 training.buffer.use=false training.buffer.prioritised=false training.lr_scheduler.decay_iter=600\\
  evaluation.n_eval=12 evaluation.n_plots=12 evaluation.n_checkpoints=1 
EOF

chmod +x "${SLURM}"

# target.solute_sigma_nm=0.32 target.solvent_sigma_nm=0.30 target.n_solvent=32\\
#   target.box_length_nm=1.16 target.nonbonded_cutoff_nm=0.52 target.grid_spacing_nm=0.20 \\

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}
#!/bin/bash
#SBATCH --job-name=prep_water_water
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=staging
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail

mkdir -p logs   # ensure slurm logs dir exists

PROJECT_NAME="fab-torch"
BASE_DIR="${SCRATCH:-$HOME}"
MAIN_DIR="${BASE_DIR}/${PROJECT_NAME}"

CONDA_ENV="bgsol"
SOLUTE="water"
SOLVENT="water"
JOB_NAME="hyper_${SOLUTE}_in_${SOLVENT}"

LOSS_TYPE="forward_kl"

LAUNCH_DIR="${MAIN_DIR}/launch"
mkdir -p "${LAUNCH_DIR}"

TRAIN_ITERS=10000
NUM_EVAL=1000
NUM_PLOTS=100
NUM_CKPTS=10

BLOCKS=(12 12 16 16 12 12 16 16)
HIDDEN_UNITS=(256 256 512 512 256 256 512 512)
NUM_BINS=(9 9 13 13 9 9 13 13)
LR=(5e-4 5e-4 5e-4 5e-4 1e-4 1e-4 1e-4 1e-4)
WD=(0 1e-5 0 1e-5 0 1e-5 0 1e-5)

for index in "${!BLOCKS[@]}"; do
  dt=$(date '+%F_%H-%M-%S.%3N')
  LOGS_DIR="${LAUNCH_DIR}/${dt}"
  mkdir -p "${LOGS_DIR}"

  rsync -arm "${MAIN_DIR}/" --exclude-from="${MAIN_DIR}/SYNC_EXCLUDE" "${LOGS_DIR}/${PROJECT_NAME}/"
  cd "${LOGS_DIR}/${PROJECT_NAME}"

  SLURM="${LOGS_DIR}/run.sh"
  mkdir -p "${LOGS_DIR}/logs"   # ensure GPU job logs dir exists

  cat > "${SLURM}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.out
#SBATCH --error=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00

module purge
module load 2025
module load Anaconda3/2025.06-1

source \$(conda info --base)/etc/profile.d/conda.sh
conda deactivate || true
conda activate ${CONDA_ENV}

export PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}:\$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MAIN_DIR="${MAIN_DIR}"

nvidia-smi

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \\
  --config-name SoluteInSolvent \\
  target.solute_name=${SOLUTE} target.solvent_name=${SOLVENT} \\
  target.solute_xml_path=null target.simulation_version=v1 \\
  fab.loss_type=forward_kl fab.use_ais=false \\
  training.n_iterations=${TRAIN_ITERS} training.buffer.use=false training.buffer.prioritised=false \\
  evaluation.n_eval=${NUM_EVAL} evaluation.n_plots=${NUM_PLOTS} evaluation.n_checkpoints=${NUM_CKPTS} \\
  flow.blocks=${BLOCKS[$index]} flow.hidden_units=${HIDDEN_UNITS[$index]} flow.num_bins=${NUM_BINS[$index]} \\
  training.lr=${LR[$index]} training.wd=${WD[$index]} \\
  logger.wandb.name=${BLOCKS[$index]}bl_${HIDDEN_UNITS[$index]}hd_${NUM_BINS[$index]}bi \\
  logger.wandb.project=${LOSS_TYPE}_${JOB_NAME}
EOF

  chmod +x "${SLURM}"
  echo "Submitting GPU job: ${SLURM}"
  sbatch "${SLURM}"
done

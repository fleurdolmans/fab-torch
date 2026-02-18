#!/bin/bash
#SBATCH --job-name=slurm_so2_water
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

SOLUTE="so2"
SOLVENT="water"
JOB_NAME="fab_${SOLUTE}_in_${SOLVENT}_test"

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
mkdir -p "${LOGS_DIR}/logs"   # ensure GPU job logs dir exists
cat > "${SLURM}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.out
#SBATCH --error=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:05:00

module purge
module load 2025
module load Anaconda3/2025.06-1

source \$(conda info --base)/etc/profile.d/conda.sh
conda deactivate
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
  target.boundary_condition=pbc target.simulation_version=v4\\
  target.box_length_nm=2.5 target.num_solvent_molecules=522 target.internal_constraints=hbonds target.rigid_water=true \\
  target.energy_cut=1e4 target.energy_max=1e5 \\
  flow.blocks=12 flow.hidden_units=512 flow.num_bins=8 \\
  fab.n_intermediate_distributions=2 fab.transition_operator.n_inner_steps=1 fab.transition_operator.init_step_size=0.01 \\
  training.batch_size=64 evaluation.eval_batch_size=256 \\
  training.buffer.maximum_length=32768 training.buffer.min_length=4096\\
  training.lr=5e-5 training.max_grad_norm=0.5 training.buffer.n_batches_sampling=4 training.buffer.w_adjust_max_clip=3\\
  training.n_iterations=1000 evaluation.n_eval=100 evaluation.n_plots=10 evaluation.n_checkpoints=1
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}

# fab.n_intermediate_distributions=32 fab.transition_operator.n_inner_steps=16 fab.transition_operator.target_p_accept=0.8 \\
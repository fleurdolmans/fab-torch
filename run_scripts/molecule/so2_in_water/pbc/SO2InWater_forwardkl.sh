#!/bin/bash
#SBATCH --job-name=slurm_so2_water
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=rome
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
JOB_NAME="${SOLUTE}_in_${SOLVENT}_test"

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
#SBATCH --output=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.out
#SBATCH --error=${LOGS_DIR}/logs/slurm-${SOLUTE}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00

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

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \\
  --config-name SoluteInSolvent \\
  target.solute_name=${SOLUTE} target.solvent_name=${SOLVENT} max_n_train_samples=50000\\
  target.simulation_version=v8 target.box_length_nm=2.0 target.nonbonded_cutoff_nm=0.9 target.num_solvent_molecules=267 \\
  target.internal_constraints=hbonds target.rigid_water=true \\
  target.energy_cut=1e6 target.energy_max=1e16 \\
  target.boundary_condition=pbc fab.loss_type=forward_kl fab.use_ais=false \\
  flow.hidden_units=128 flow.base.type=gauss flow.base.learn_mean_var=false flow.type=perm-equi-spline-nf\\
  flow.layers=8 flow.blocks_per_layer=2 flow.group_size=6 flow.tail_bound=4 \\
  training.lr=1e-4 training.wd=1e-6 training.batch_size=32 evaluation.eval_batch_size=32 \\
  training.max_grad_norm=10 training.warmup_iter=500 \\
  training.overlap_penalty=100 training.mixing=0.0 training.energy_mode=full target.transform.version=GPR \\
  training.n_iterations=10000 training.buffer.use=false training.buffer.prioritised=false training.lr_scheduler.decay_iter=10000\\
  evaluation.n_eval=10 evaluation.n_plots=10 evaluation.n_checkpoints=1 
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}
# target.simulation_version=v1 target.box_length_nm=3.105 target.nonbonded_cutoff_nm=1.0 target.num_solvent_molecules=1000 \\

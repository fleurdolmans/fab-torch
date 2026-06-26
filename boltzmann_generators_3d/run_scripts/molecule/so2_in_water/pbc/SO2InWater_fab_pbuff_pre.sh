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
#SBATCH --partition=gpu_a100
#SBATCH --time=04:10:00

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
  target.solute_name=${SOLUTE} target.solvent_name=${SOLVENT} target.simulation_version=v10 \\
  target.box_length_nm=0.9 target.nonbonded_cutoff_nm=0.4 target.num_solvent_molecules=22 \\
  target.internal_constraints=hbonds target.rigid_water=true \\
  target.energy_cut=1e10 target.energy_max=1e16 target.curriculum_type=temperature target.curriculum_lambda=0.6\\
  target.boundary_condition=pbc fab.loss_type=fab_alpha_div fab.use_ais=true \\
  flow.hidden_units=128 flow.layers=12 flow.blocks_per_layer=4 flow.group_size=6 flow.tail_bound=3\\
  flow.base.type=structured-gauss flow.base.learn_mean_var=true flow.type=perm-equi-spline-nf\\
  fab.n_intermediate_distributions=8 fab.transition_operator.n_inner_steps=6 fab.transition_operator.init_step_size=0.05 \\
  training.energy_mode=full target.transform.version=GPR\\
  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/13-43-46_592586 \\
  training.lr=5e-5 training.wd=1e-6 training.batch_size=128 evaluation.eval_batch_size=128\\
  training.buffer.maximum_length=32768 training.buffer.min_length=4096 \\
  training.max_grad_norm=10 training.buffer.n_batches_sampling=4 training.buffer.w_adjust_max_clip=1\\
  training.warmup_iter=50 training.n_iterations=400 training.lr_scheduler.decay_iter=400\\
  evaluation.n_eval=20 evaluation.n_plots=20 evaluation.n_checkpoints=1
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}

# PRE100 FIXED + REVERSE KL
# 
# training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-24/15-23-41_999900 \\
 # PRE100 FIXED
#  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-24/13-26-08_780710 \\
 # PRE100 STRUCGAUSS
#  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-23/10-54-21_712292 \\
# PRE100 
  # training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-23/10-54-21_712292 \\

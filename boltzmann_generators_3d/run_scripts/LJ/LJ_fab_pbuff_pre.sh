#!/bin/bash
#SBATCH --job-name=slurm_LJ_fab
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

JOB_NAME="fab_LJ_fab_test"

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
#SBATCH --output=${LOGS_DIR}/logs/slurm-LJ-fab-%j.out
#SBATCH --error=${LOGS_DIR}/logs/slurm-LJ-fab-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

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

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run_LJ.py \\
  --config-name LJ \\
  target.base_state=v10 target.target_state=v10 target.solid=false\\
  target.solute_sigma_nm=0.38 target.solvent_sigma_nm=0.34 target.n_solvent=32\\
  fab.loss_type=fab_alpha_div fab.use_ais=true \\
  target.energy_cut=1.e+10 target.energy_max=1.e+20 target.temperature=100\\
  flow.hidden_units=128 flow.base.type=multishell-torus flow.base.learn_mean_var=false flow.type=lj_torus_spline \\
  flow.n_layers=6 flow.layer_nodes_per_dim=4 flow.hidden_units=128 flow.num_bins=8 \\
  flow.group_size=6 flow.tail_bound=3 target.transform_version=v4\\
  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/LJ/MD_training/2026-04-18/14-37-42_995342 \\
  fab.n_intermediate_distributions=8 fab.transition_operator.n_inner_steps=2 fab.transition_operator.init_step_size=0.01 \\
  training.lr=5e-5 training.wd=1e-6 training.batch_size=32 evaluation.eval_batch_size=128\\
  training.buffer.maximum_length=8192 training.buffer.min_length=1024 \\
  training.max_grad_norm=10 training.buffer.n_batches_sampling=1 training.buffer.w_adjust_max_clip=1\\
  training.warmup_iter=50 training.n_iterations=200 training.lr_scheduler.decay_iter=200\\
  evaluation.n_eval=20 evaluation.n_plots=20 evaluation.n_checkpoints=1
EOF

chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}

# target.curriculum_type=temperature target.curriculum_lambda=0.6\\

# PRE100 FIXED + REVERSE KL
# 
# training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-24/15-23-41_999900 \\
 # PRE100 FIXED
#  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-24/13-26-08_780710 \\
 # PRE100 STRUCGAUSS
#  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-23/10-54-21_712292 \\
# PRE100 
  # training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-03-23/10-54-21_712292 \\

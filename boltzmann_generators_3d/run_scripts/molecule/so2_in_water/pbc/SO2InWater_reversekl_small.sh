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
#SBATCH --time=03:00:00

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

python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \\
  --config-name SoluteInSolvent \\
  target.solute_name=${SOLUTE} target.solvent_name=${SOLVENT} target.simulation_version=v10\\
  target.box_length_nm=0.9 target.nonbonded_cutoff_nm=0.4 target.num_solvent_molecules=22 \\
  target.internal_constraints=hbonds target.rigid_water=true \\
  target.energy_cut=1e6 target.energy_max=1e16 \\
  target.boundary_condition=pbc fab.loss_type=flow_reverse_kl fab.use_ais=false \\
  flow.hidden_units=128 flow.base.type=gauss flow.base.learn_mean_var=false flow.type=perm-equi-spline-nf \\
  flow.layers=12 flow.blocks_per_layer=4 flow.group_size=6 flow.tail_bound=3 \\
  training.checkpoint_load_dir=/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-19/08-39-30_632046\\
  training.lr=5e-5 training.wd=1e-6 training.batch_size=256 evaluation.eval_batch_size=64 \\
  training.max_grad_norm=1 training.warmup_iter=250\\
  target.transform.canonical_sorting=true target.transform.version=GPR\\
  training.overlap_penalty=5 training.mixing=0.0 training.energy_mode=full\\
  training.n_iterations=3000 training.buffer.use=false training.buffer.prioritised=false training.lr_scheduler.decay_iter=3000 \\
  evaluation.n_eval=30 evaluation.n_plots=30 evaluation.n_checkpoints=1
EOF



chmod +x "${SLURM}"

echo "Submitting GPU job: ${SLURM}"

sbatch ${SLURM}

# LGT stage 1
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/21-50-48_669540
# LGT stage 1- structure gauss
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/09-34-30_089429
# LGT stage 2
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-19/08-37-35_506211
# LGT stage 2 - structured gauss
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-19/08-37-25_892744
#LGT stage 1 - structured gauss - learnable
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-14/22-02-32_007312

#LGT longer
# 

# GPR stage 3 - structured
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-14/22-35-24_624343

# GPR stage 1
#/home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/17-30-46_607608
# GPR - stage 1 -structured gauss
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/13-43-46_592586
#GPR - stage 2
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-14/10-52-27_144440
#GPR -stage 2 - structure gauss
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-18/21-54-45_621932
#GPR stage 1 structured gauss learnable
# /home/fdolmans/HDD/results/fab/SoluteInwater/so2_in_water/MD_training/2026-06-14/22-02-42_727953

# 

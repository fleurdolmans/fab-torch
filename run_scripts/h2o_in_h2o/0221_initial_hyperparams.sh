#!/bin/sh

# Name of project folder
PROJECT_NAME=fab-torch
# Original code folder is here
MAIN_DIR=/home/tbbakke/${PROJECT_NAME}
# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch/
mkdir -p "${LAUNCH_DIR}"

TRAIN_ITERS=10000
NUM_EVAL=1000
NUM_PLOTS=100
NUM_CKPTS=10

BLOCKS=(12 12 16 16 12 12 16 16)
HIDDEN_UNITS=(256 256 512 512 256 256 512 512)
NUM_BINS=(9 9 13 13 9 9 13 13)
LR=('5e-4' '5e-4' '5e-4' '5e-4' '1e-4' '1e-4' '1e-4' '1e-4')
WD=(0 '1e-5' 0 '1e-5' 0 '1e-5' 0 '1e-5')

for index in "${!BLOCKS[@]}"; do
  JOB_NAME=0221_hyperparams

  # Create dir for specific experiment run
  dt=$(date '+%F_%H-%M-%S.%3N')
  LOGS_DIR=${LAUNCH_DIR}/${dt}
  mkdir -p "${LOGS_DIR}"
  # Copy code to experiment folder
  rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
  cd ${LOGS_DIR}/${PROJECT_NAME}

  # Make SLURM file
  JOB_NAME=${JOB_NAME}
  SLURM=${LOGS_DIR}/run.slrm
  echo "${SLURM}"
  echo "#!/bin/bash" > ${SLURM}
  echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
  echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
  echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
  echo "#SBATCH --gres=gpu:1" >> ${SLURM}
  echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
  echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
  echo "#SBATCH --mem=8G" >> ${SLURM}
  echo "#SBATCH --time=1-0:00:00" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
  {
    echo nvidia-smi
    echo PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}" HYDRA_FULL_ERROR=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
      /home/tbbakke/anaconda3/envs/bgsol/bin/python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \
      --config-name h2oinh2o_forwardkl.yaml node=ivicl \
      training.n_iterations=${TRAIN_ITERS} evaluation.n_eval=${NUM_EVAL} evaluation.n_plots=${NUM_PLOTS} evaluation.n_checkpoints=${NUM_CKPTS} \
      flow.blocks=${BLOCKS[index]} flow.hidden_units=${HIDDEN_UNITS[index]} flow.num_bins=${NUM_BINS[index]} \
      training.lr=${LR[index]} training.wd=${WD[index]}
  } >> ${SLURM}

  sbatch ${SLURM}
  sleep .5
done

#!/bin/sh

# Name of project folder
PROJECT_NAME=fab-torch

# Home dir
HOME_DIR=/home/tbbakke
# Original code folder is here
MAIN_DIR=${HOME_DIR}/${PROJECT_NAME}
# Conda environment folder
CONDA_ENV_DIR=${HOME_DIR}/anaconda3/envs/bgsol
# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch/
mkdir -p "${LAUNCH_DIR}"


#### --------------- ####
#### HYPERPARAMETERS ####
#### --------------- ####

TRAIN_ITERS=10000
NUM_EVAL=500
NUM_PLOTS=20
NUM_CKPTS=1

SCHEDULER=("step" "cosine" "step" "cosine" "step" "cosine" "step" "cosine")
RATE_DECAY=(0.1 1 0.1 1 0.1 1 0.1 1)
DECAY_ITER=(2000 1 2000 1 2000 1 2000 1)
GRAD_NORM=(1 1 1 1 1 1 1 1)

BLOCKS=(16 16 16 16 24 24 16 16)
HIDDEN_UNITS=(512 512 1024 1024 512 512 512 512)
NUM_BINS=(11 11 11 11 11 11 15 15)

JOB_NAME=0417_hyperparams

#### --------------- ####
#### HYPERPARAMETERS ####
#### --------------- ####


for index in "${!SCHEDULER[@]}"; do
  # Create dir for specific experiment run
  dt=$(date '+%F_%H-%M-%S.%3N')
  LOGS_DIR=${LAUNCH_DIR}/${dt}
  mkdir -p "${LOGS_DIR}"
  # Copy code to experiment folder
  rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
  cd ${LOGS_DIR}/${PROJECT_NAME}

  # Make SLURM file
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
  echo "#SBATCH --time=0-8:00:00" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
  {
    echo nvidia-smi
    echo PYTHONPATH="${LOGS_DIR}/${PROJECT_NAME}" LD_LIBRARY_PATH=${CONDA_ENV_DIR}/lib \
      HYDRA_FULL_ERROR=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
      ${CONDA_ENV_DIR}/bin/python ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \
      --config-name so2inh2o_forwardkl.yaml node=ivicl \
      training.n_iterations=${TRAIN_ITERS} evaluation.n_eval=${NUM_EVAL} evaluation.n_plots=${NUM_PLOTS} evaluation.n_checkpoints=${NUM_CKPTS} \
      training.max_grad_norm=${GRAD_NORM[$index]} training.lr_scheduler.type=${SCHEDULER[$index]} \
      training.lr_scheduler.rate_decay=${RATE_DECAY[$index]} training.lr_scheduler.decay_iter=${DECAY_ITER[$index]} \
      flow.blocks=${BLOCKS[index]} flow.hidden_units=${HIDDEN_UNITS[index]} flow.num_bins=${NUM_BINS[index]}
  } >> ${SLURM}

  sbatch ${SLURM}
  sleep .1
done

# TODO:
# Looks like we need bigger models! 24 blocks, 1024 hidden units, 15 bins works much better for overfitting on 1000 MD
#  samples with 2 molecules than 16,512,11, which works much better than the default of 12,256,8.
# Cosine ends up better than step scheduler, but only because it ends up with a lower learning rate at the right moment.
# Grad norm seems to not matter much. Just set to 1?
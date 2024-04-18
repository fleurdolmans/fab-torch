#!/bin/sh

# Name of project folder
PROJECT_NAME=fab-torch
# Original code folder is here
HOME_DIR=/home/tbbakke
MAIN_DIR=${HOME_DIR}/${PROJECT_NAME}
CONDA_ENV_DIR=/ivi/zfs/s0/original_homes/tbbakke/anaconda3/envs/bgsol
# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch/
mkdir -p "${LAUNCH_DIR}"


#### --------------- ####
#### HYPERPARAMETERS ####
#### --------------- ####

#TRAIN_ITERS=500
#NUM_EVAL=50
#NUM_PLOTS=5
#NUM_CKPTS=1
#
#SCHEDULER=("exponential")
#RATE_DECAY=(1)
#DECAY_ITER=(1)
#GRAD_NORM=(1)

TRAIN_ITERS=5000
NUM_EVAL=500
NUM_PLOTS=50
NUM_CKPTS=5

SCHEDULER=("exponential" "step" "cosine" "cosine_restart" "exponential" "step" "cosine" "cosine_restart" "exponential" "step" "cosine" "cosine_restart" "exponential" "step" "cosine" "cosine_restart")
RATE_DECAY=(0.99 0.1 1 1 0.99 0.1 1 1 0.99 0.1 1 1 0.99 0.1 1 1)
DECAY_ITER=(10 2500 1 1000 10 2500 1 1000 10 2500 1 1000 10 2500 1 1000)
GRAD_NORM=(0.1 0.1 0.1 0.1 1 1 1 1 10 10 10 10 1000 1000 1000 1000)

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
      training.lr_scheduler.rate_decay=${RATE_DECAY[$index]} training.lr_scheduler.decay_iter=${DECAY_ITER[$index]}
  } >> ${SLURM}

  sbatch ${SLURM}
  sleep .1
done

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

TRAIN_ITERS=50000
NUM_EVAL=500
NUM_PLOTS=50
NUM_CKPTS=1


# Scheduler args
SCHEDULER=("cosine" "cosine" "cosine" "cosine" "cosine" "cosine" "cosine" "cosine")
RATE_DECAY=(1 1 1 1 1 1 1 1)
DECAY_ITER=(1 1 1 1 1 1 1 1)

# Stability args
GRAD_NORM=(1 1 1 1 1 1 1 1)

# Architecture args
BLOCKS=(24 24 36 36 24 24 36 36)
HIDDEN_UNITS=(512 1024 512 1024 512 1024 512 1024)
NUM_BINS=(8 8 8 8 8 8 8 8)
BLOCKS_PER_LAYER=(1 1 1 1 1 1 1 1)

# Target args
CONSTRAINT_RADIUS=(0.3 0.3 0.3 0.3 0.4 0.4 0.4 0.4)

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
  echo "#SBATCH --mem=11G" >> ${SLURM}
  echo "#SBATCH --time=1-12:00:00" >> ${SLURM}
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
      flow.blocks=${BLOCKS[index]} flow.blocks_per_layer=${BLOCKS_PER_LAYER[index]} \
      flow.hidden_units=${HIDDEN_UNITS[index]} flow.num_bins=${NUM_BINS[index]} \
      training.constraint_radius=${CONSTRAINT_RADIUS[index]}
  } >> ${SLURM}

  sbatch ${SLURM}
  sleep .1
done

# Memory usage (blocks, hidden units, bins): 1blockperlayer unless otherwise specified (2 solvent).
# 24-1024-15: ~4.7GB
# 36-1024-15: ~6.7GB
# 36-1024-8: ~7GB (weird but true, actually more than with 15 bins)
# 36-1536-15: ~10.6GB
# 36-1024-15-2blocksperlayer: ~10.2GB
# 48-512-8: ~4.2GB

# 5 solvent
# 24-1024-8: ~ 5.4GB
# 36-512-8: ~4.7GB
# 36-1024-8: ~7.6GB
# 36-1024-16: ~10.1GB
# 60-512-8: ~7.1GB
# 96-512-8: ~10.7GB

# 13 solvent:
# 36-512-8: ~8.2GB

# Run time (2 solvent):
# 512 to 1024 hidden --> about 2x run time.
# 1 to 2 blocks per layer --> about 1.5x run time.
# 36p2-1024-15 takes about 12 hours on IvI for 10000 iters (with 500 evals and 20 plotting steps).

# Num params (2 solvent):
# 12-512-8: 7,993,656
# 12-512-16: 9,547,992
# 12-1024-8: 28,563,768
# 120-512-8: 79,936,560 (~8.7GB)

# Looks like we need bigger models! 24 blocks, 1024 hidden units, 15 bins works much better for overfitting on 1000 MD
#  samples with 2 molecules than 16-512-11, which works much better than the default of 12-256-8.
# Cosine ends up better than step scheduler, but only because it ends up with a lower learning rate at the right moment.
# Grad norm seems to not matter much. Just set to 1?

# 24-1024-15 for 5K iters performs very similarly in loss to 24-512-11 and 16-1024-11 for 10K iter (cosine)!
#  So assuming bins is not too important, we see than 2x iters = 2x hidden units = 1.5x blocks.
#  Pretty stark difference in KLs though.

# Compare 71 and 59 to see largest difference with model size (cosine scheduler).
# How much does hidden dim (512 vs 1024) matter with 36 layers? More is better for loss, worse for KL.
# Does using 2 blocks per layer improve performance? Somewhat, but mostly in loss; worse for KL mostly.
# How much do spline bins matter (8 vs 15)?  Matters little, but more is slightly higher loss, worse KL in general.
# Cosine vs designed step scheduler: prefer the former (fewer hyperparams), but latter may be better? Probably cosine.
#   Would be best if we can stretch the last 10-20% of training time; e.g., run more training with lr < 0.1 * initial_lr.
#   Perhaps running for more iters also achieves this indirectly, although this might waste computation in early/mid steps?

# Another run analysis for 2 solvent:
# 10K MD samples, 50K iters; 36 48 60 72 layers with 512 hidden (and one with 1024 for 36 layers), 8 bins. Runs are numbered 80-82 with duplicates.
# These all work very well, except 72 layers, which gives very high KL.
# 36p1 gets lowest KL, 36p2 second lowest. 36p1 gets highest loss, but 36p2 is third highest loss: potentially p2 is usefull.
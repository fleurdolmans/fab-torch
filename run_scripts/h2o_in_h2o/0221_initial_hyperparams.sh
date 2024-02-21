#!/bin/sh

# Original code folder is here
MAIN_DIR=/home/tbbakke/fab-torch
# Launch dir
LAUNCH_DIR=/home/tbbakke/fab-torch/launch/
mkdir -p "${LAUNCH_DIR}"


# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
JOB_NAME=dc_weighted_mnist
SLURM=${LOGS_DIR}/run.slrm
# Make SLURM file
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=7-0:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
{
    echo CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/ml/bin/python ${LOGS_DIR}/alrl/run.py \
        --data_type ${dataset} --wandb True --wandb_entity timsey --wandb_project al_imba \
        --out_dir ${OUT_DIR} --abs_path_to_data_store /home/tbbakke/data/alrl/ \
        --data_split_seed ${seed} --seed ${seed} --normalise_features True --stratify True \
        --eval_split test --num_val_points 0 --num_reward_points 0 --num_annot_points ${num_annot} \
        --imbalance_factors 1 10 1 10 1 10 1 10 1 10 --ignore_existing_imbalance False --class_weight_type balanced --use_class_weights_for_fit ${ucw} \
        --classifier_type ${classifier} --acquisition_strategy ${method} --budget ${budget} --al_steps 10
    echo
} >> ${SLURM}

sbatch ${SLURM}


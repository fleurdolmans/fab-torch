#!/usr/bin/env bash

source /home/fdolmans/anaconda3/bin/activate fab

nvidia-smi

HOME_DIR=/home/fdolmans
CONDA_ENV_DIR=${HOME_DIR}/anaconda3/envs/fab
PROJECT_DIR=${HOME_DIR}/Projects/fab-torch

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PROJECT_DIR} LD_LIBRARY_PATH=${CONDA_ENV_DIR}/lib \
  ${CONDA_ENV_DIR}/bin/python ${PROJECT_DIR}/experiments/solvation/run.py --config-name so2inh2o_forwardkl.yaml
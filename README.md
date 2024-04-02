# Boltzmann Generator for solute-solvent systems.

Code is based on Flow Annealed Importance Sampling Bootstrap for PyTorch [Git repo](https://github.com/lollcat/fab-torch). See their README for useful tips and more information.

## Steps for creating the environment
Have conda installed, clone this repository, and `cd` into it. Then run:
> $ conda create --name bgsol python=3.7.16 \
$ conda activate bgsol \
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ conda install -c conda-forge openmm cudatoolkit=10.1 \
$ conda config --add channels omnia --add channels conda-forge \
$ conda install openmmtools \
$ pip install -r requirements.txt

### Conda speedups
If conda in taking very long solving the environment, you may want to try using the libmamba solver, which is part of newer conda versions: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community


## Experiments
Solvation experiments are run using `experiments/solvation/run.py`. Experiments are run using Hydra configs, which can be found in `experiments/solvation/config/`.

Currently, we support two methods of training the Boltzmann Generator/Normalising Flow; likelihood (forward KL-divergence) training using MD samples and FAB training without MD samples (although MD samples are still required for evaluation). FAB is much slower, so is not recommended for initial tests; as a general rule, use it only once likelihood training with MD samples works. Configuration files for each are provided, ending in either `_forwardkl.yaml` or `_fab_pbuff.yaml` (FAB with prioritised buffer). 

MD data can be created using `experiments/solvation/create_md_data.py`. This script uses the OpenMM library to run MD simulations.

## Good to know
This is research code that is not fully documented. Many comments are interspersed throughout the code. Some things that are good to know when going through it:

- You will often see comments like "I --> X". Here I means internal coordinates, and X means Cartesian coordinates. The arrow denotes a transformation from one to the other. The transform is a 3D Spherical Coordinate transform, implemented in `fab/transforms/global_2point_spherical_transform.py`. The chemical system definition for the solute-in-water system is in `fab/target_distributions/solute_in_water.py`. Similar systems can be created for non-water solvents.
- log_p refers to the log of the distribution p, which often refers to the Boltzmann distribution. log_q refers to the log of q, which is often the Normalising Flow / Boltzmann Generator. This may not always be consistent however, since the original codebase does not always use this convention.
- We use a Circular Coupled Rational Quadratic Spline flow which uses circular coupling for bond angle dimensions. The model is constructed in `make_wrapped_normflow_solvent_flow` in `experiments/make_flow/make_normflow_model.py`.
- Trainer objects we use are in `fab/trainer.py` and `fab/train_with_prioritised_buffer.py`. We use the first for likelihood training and the second for FAB (or AIS: Annealed Importance Sampling) training.

## Logging

We recommend using Weights&Biases for logging Boltzmann Generator experiments. This can be done by uncommenting the `wandb` entry in the relevant Hydra config and providing your own entity and project name.

Data is also stored on disk in the `hydra.run.dir` directory. This directory is created by Hydra and contains all the necessary information for a run. The structure of this directory is as follows:

Example structure of `hydra.run.dir` for an experiment run:
```
- plots: Directory containing any plots saved on disk (typically not used when already sending images to Wandb).
- metrics: Directory containing any metrics saved on disk.
- model_checkpoints: Directory containing any model checkpoints saved on disk.
- wandb: Wandb logging files, see below.
```

Example of structure of `wandb/run-YYYMMDD_HHMMSS-RUN_ID/files/` dir created by wandb inside the `hydra.run.dir`.
```
- config.yaml: Contains Hydra config.
- config.txt: Contains Hydra config in plaintext.
- output.log: Contains stdout of run.
- wandb-summary.json: JSON file containing logged metrics.
- wandb-metadata.json: JSON file containing metadata about the run.
- requirements.txt: Plaintext file of pip packages used.
- media: Directory containing any media files logged to Wandb, such as images.
- ```
# Boltzmann Generator for solute-solvent systems.

Code is based on Flow Annealed Importance Sampling Bootstrap for PyTorch [Git repo](https://github.com/lollcat/fab-torch). See their README for useful tips and more information.

## Steps for creating the environment
Have conda installed, clone this repository, and `cd` into it. Then run the following to create a conda environment named `bgsol` with all the necessary packages:
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

MD data can be created using `experiments/solvation/create_md_data.py`. This script uses the OpenMM library to run MD simulations. These scripts create data files that can be used as `train_samples_path` (or `val_samples_path` or `test_samples_path`) in the Hydra configs of the Boltzmann Generators for training and/or evaluating on MD samples.

### Example commands:
All the below commands use a Hydra config file that specifies all the arguments for the run. These are located in `experiments/solvation/config/`. 

The configs take a `node` argument that sets the machine paths. It is specified by the files in `experiments/solvation/config/node/`: the default `node` is `desktop` (see `defaults` at the top of every config file). You can create your own node by creating a new file in `experiments/solvation/config/node/` and specifying the paths on your machine. Then you can chance the default node of the configs to your own node to automatically run everything with the paths on your machine.

Commands should be run after activating the conda environment (here assumed to be `bgsol`).

Create MD data:
> $ python ./experiments/solvation/create_md_data.py --config-name make_md_data.yaml

H2O in water, likelihood training:
> $ python ./experiments/solvation/run.py --config-name h2oinh2o_forwardkl.yaml

SO2 in water, likelihood training: 
> $ python ./experiments/solvation/run.py --config-name so2inh2o_forwardkl.yaml

H2O in water, FAB training:
> $ python ./experiments/solvation/run.py --config-name h2oinh2o_fab_pbuff.yaml

You may need to set the `PYTHONPATH` and `LD_LIBRARY_PATH` environment variables to properly run everything in some special cases, e.g. when running on a cluster. Example:
> $ PYTHONPATH="{PATH_TO_fab-torch_DIR}" LD_LIBRARY_PATH="{PATH_TO_CONDA_DIR}/envs/bgsol/lib" ${PATH_TO_CONDA_DIR}/envs/bgsol/bin/python ./experiments/solvation/run.py --config-name so2inh2o_forwardkl.yaml


## Good to know
This is research code that is not fully documented. Many comments are interspersed throughout the code. Some things that are good to know when going through it:

- You will often see comments like "I --> X". Here I means internal coordinates, and X means Cartesian coordinates. The arrow denotes a transformation from one to the other. The transform is a 3D Spherical Coordinate transform, implemented in `fab/transforms/global_2point_spherical_transform.py`. The chemical system definition for the solute-in-water system is in `fab/target_distributions/solute_in_water.py`. Similar systems can be created for non-water solvents.
- log_p refers to the log of the distribution p, which often refers to the Boltzmann distribution. log_q refers to the log of q, which is often the Normalising Flow / Boltzmann Generator. This may not always be consistent however, since the original codebase does not always use this convention.
- It can be useful to set `target.n_threads` in the Hydra config to 1 when debugging.
- We use a Circular Coupled Rational Quadratic Spline flow which uses circular coupling for bond angle dimensions. The model is constructed in `make_wrapped_normflow_solvent_flow` in `experiments/make_flow/make_normflow_model.py`.
- Trainer objects we use are in `fab/trainer.py` and `fab/train_with_prioritised_buffer.py`. We use the first for likelihood training and the second for FAB (or AIS: Annealed Importance Sampling) training.
- A decent starting point for likelihood training is running 5000 iterations of batch size 1024, learning rate of 5e-4, and weight decay 1e-5. This is with a Flow of 12 blocks with hidden dimension 256 and 8 bins per spline. This setting was mostly used to see if the model could overfit 100-1000 MD data points (which it mostly can).

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
# Steps for creating the environment

## Using conda
Have conda installed, clone this repository, and `cd` into it. Then run:
> $ conda create --name bgsol python=3.7.16 \
$ conda activate bgsol \
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ conda install -c conda-forge openmm cudatoolkit=10.1 \
$ conda config --add channels omnia --add channels conda-forge \
$ conda install openmmtools \
$ pip install -r requirements.txt

## Conda speedups
If conda in taking very long solving the environment, you may want to try using the libmamba solver, which is part of newer conda versions: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community
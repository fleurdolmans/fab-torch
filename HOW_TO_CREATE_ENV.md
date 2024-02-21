# Steps for creating conda environment

> $ conda create --name NAME python=3.7.16 \
$ conda activate NAME \
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ conda install -c conda-forge openmm cudatoolkit=10.1 \
$ pip install -r requirements.txt
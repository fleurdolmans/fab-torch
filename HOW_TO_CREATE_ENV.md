# Steps for creating conda environment

> $ conda create --name ENV_NAME python=3.7.16 \
$ conda activate ENV_NAME \
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ conda install -c conda-forge openmm cudatoolkit=10.1 \
$ conda config --add channels omnia --add channels conda-forge \
$ conda install openmmtools \
$ pip install -r requirements.txt
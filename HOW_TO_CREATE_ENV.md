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


## Using micromamba (faster than conda)
Install micromamba using instructions here: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Have micromamba installed, clone this repository, and `cd` into it. Then run:
> $ micromamba create --name bgsol python=3.7.16 \
$ micromamba activate bgsol \
$ micromamba install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ micromamba install -c conda-forge openmm cudatoolkit=10.1 \
$ micromamba config --add channels omnia --add channels conda-forge \
$ micromamba install openmmtools \
$ pip install -r requirements.txt


## Output of ```pip freeze``` (for debugging)
absl-py==2.1.0
aiofiles==22.1.0
aiosqlite==0.19.0
antlr4-python3-runtime==4.9.3
anyio==3.7.1
appdirs==1.4.4
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
astunparse @ file:///home/conda/feedstock_root/build_artifacts/astunparse_1610696312422/work
attrs==23.2.0
Babel==2.14.0
backcall==0.2.0
beautifulsoup4==4.12.3
bleach==6.0.0
boltzgen @ git+https://github.com/VincentStimper/boltzmann-generators.git@2b177fc155f533933489b8fce8d6483ebad250d3
Bottleneck @ file:///home/conda/feedstock_root/build_artifacts/bottleneck_1656803757560/work
cached-property==1.5.2
cachetools==5.3.2
certifi==2024.2.2
cffi==1.15.1
cftime @ file:///home/conda/feedstock_root/build_artifacts/cftime_1663606412550/work
charset-normalizer==3.3.2
click==8.1.7
cycler==0.11.0
debugpy==1.7.0
decorator==5.1.1
defusedxml==0.7.1
docker-pycreds==0.4.0
entrypoints==0.4
exceptiongroup==1.2.0
fastjsonschema==2.19.1
fonttools==4.38.0
fqdn==1.5.1
gitdb==4.0.11
GitPython==3.1.42
google-auth==2.28.1
google-auth-oauthlib==0.4.6
grpcio==1.62.0
h5py==3.8.0
hydra-core==1.3.2
hydra-joblib-launcher==1.2.0
idna==3.6
importlib-metadata==6.7.0
importlib-resources==5.12.0
ipykernel==6.16.2
ipython==7.34.0
ipython-genutils==0.2.0
isoduration==20.11.0
jedi==0.19.1
Jinja2==3.1.3
joblib==1.2.0
json5==0.9.16
jsonpointer==2.4
jsonschema==4.17.3
jupyter-events==0.6.3
jupyter-server==1.24.0
jupyter-ydoc==0.2.5
jupyter_client==7.4.9
jupyter_core==4.12.0
jupyter_server_fileid==0.9.1
jupyter_server_ydoc==0.8.0
jupyterlab==3.6.7
jupyterlab-pygments==0.2.2
jupyterlab_server==2.24.0
kiwisolver==1.4.5
larsflow @ git+https://github.com/VincentStimper/resampled-base-flows.git@18db5bf28ffa1d5ab9ef2b63856e186affee604b
llvmlite==0.38.1
Markdown==3.4.4
MarkupSafe==2.1.5
matplotlib==3.5.3
matplotlib-inline==0.1.6
mdtraj @ file:///home/conda/feedstock_root/build_artifacts/mdtraj_1663069972907/work
mistune==3.0.2
mkl-fft==1.3.1
mkl-random==1.2.2
mkl-service==2.4.0
mpiplus @ file:///home/conda/feedstock_root/build_artifacts/mpiplus_1682694361894/work
nbclassic==1.0.0
nbclient==0.7.4
nbconvert==7.6.0
nbformat==5.8.0
nest-asyncio==1.6.0
netCDF4 @ file:///home/conda/feedstock_root/build_artifacts/netcdf4_1656505588132/work
nflows==0.14
normflows==1.6.1
nose @ file:///home/conda/feedstock_root/build_artifacts/nose_1602434998960/work
notebook==6.5.6
notebook_shim==0.2.4
numba @ file:///home/conda/feedstock_root/build_artifacts/numba_1655473306076/work
numexpr @ file:///home/conda/feedstock_root/build_artifacts/numexpr_1636286753250/work
numpy @ file:///opt/conda/conda-bld/numpy_and_numpy_base_1653915516269/work
oauthlib==3.2.2
olefile @ file:///home/conda/feedstock_root/build_artifacts/olefile_1701735466804/work
omegaconf==2.3.0
OpenMM==7.3.1
openmmtools @ file:///home/conda/feedstock_root/build_artifacts/openmmtools_1682714428367/work
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1696202382185/work
pandas==1.3.5
pandocfilters==1.5.1
parso==0.8.3
pathtools==0.1.2
pdbfixer==1.5
pexpect==4.9.0
pickleshare==0.7.5
Pillow==9.5.0
pkgutil_resolve_name==1.3.10
prometheus-client==0.17.1
prompt-toolkit==3.0.43
protobuf==3.20.3
psutil==5.9.8
ptyprocess==0.7.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
pycparser==2.21
Pygments==2.17.2
pymbar==3.0.5
pyparsing @ file:///home/conda/feedstock_root/build_artifacts/pyparsing_1690737849915/work
pyrsistent==0.19.3
python-dateutil @ file:///home/conda/feedstock_root/build_artifacts/python-dateutil_1626286286081/work
python-json-logger==2.0.7
pytz @ file:///home/conda/feedstock_root/build_artifacts/pytz_1706886791323/work
PyYAML @ file:///home/conda/feedstock_root/build_artifacts/pyyaml_1648757092905/work
pyzmq==24.0.1
requests==2.31.0
requests-oauthlib==1.3.1
residual-flows @ git+https://github.com/VincentStimper/residual-flows.git@2bbbf70570f1ce4ec8de21da3423fcc773f48f98
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rsa==4.9
scipy @ file:///opt/conda/conda-bld/scipy_1661390393401/work
Send2Trash==1.8.2
sentry-sdk==1.40.5
setproctitle==1.3.3
six @ file:///home/conda/feedstock_root/build_artifacts/six_1620240208055/work
smmap==5.0.1
sniffio==1.3.0
soupsieve==2.4.1
tables @ file:///home/conda/feedstock_root/build_artifacts/pytables_1643135942042/work
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
terminado==0.17.1
tinycss2==1.2.1
tomli==2.0.1
torch==1.8.1
torchaudio==0.8.0a0+e4e171a
torchvision==0.9.1
tornado==6.2
tqdm==4.64.1
traitlets==5.9.0
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/typing_extensions_1688315532570/work
uri-template==1.3.0
urllib3==2.0.7
wandb==0.13.10
wcwidth==0.2.13
webcolors==1.13
webencodings==0.5.1
websocket-client==1.6.1
Werkzeug==2.2.3
y-py==0.6.2
ypy-websocket==0.8.4
zipp==3.15.0
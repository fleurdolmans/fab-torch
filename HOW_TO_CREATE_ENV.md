# Steps for creating conda environment

> $ conda create --name NAME python=3.7.16 \
$ conda activate NAME \
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch \
$ pip install -r requirements.txt

# What my pip freeze looks like after:

> absl-py==2.1.0 \
aiofiles==22.1.0 \
aiosqlite==0.19.0 \
antlr4-python3-runtime==4.9.3 \
anyio==3.7.1 \
appdirs==1.4.4 \
argon2-cffi==23.1.0 \
argon2-cffi-bindings==21.2.0 \
arrow==1.2.3 \
astunparse==1.6.3 \
attrs==23.2.0 \
Babel==2.14.0 \
backcall==0.2.0 \
beautifulsoup4==4.12.3 \
bleach==6.0.0 \
boltzgen @ git+https://github.com/VincentStimper/boltzmann-generators.git@2b177fc155f533933489b8fce8d6483ebad250d3 \
cached-property==1.5.2 \
cachetools==5.3.2 \
certifi @ file:///croot/certifi_1671487769961/work/certifi \
cffi==1.15.1 \
charset-normalizer==3.3.2 \
click==8.1.7 \
cycler==0.11.0 \
debugpy==1.7.0 \
decorator==5.1.1 \
defusedxml==0.7.1 \
docker-pycreds==0.4.0 \
entrypoints==0.4 \
exceptiongroup==1.2.0 \
fastjsonschema==2.19.1 \
flit_core @ file:///opt/conda/conda-bld/flit-core_1644941570762/work/source/flit_core \
fonttools==4.38.0 \
fqdn==1.5.1 \
gitdb==4.0.11 \
GitPython==3.1.42 \
google-auth==2.28.0 \
google-auth-oauthlib==0.4.6 \
grpcio==1.60.1 \
h5py==3.8.0 \
hydra-core==1.3.2 \
hydra-joblib-launcher==1.2.0 \
idna==3.6 \
importlib-metadata==6.7.0 \
importlib-resources==5.12.0 \
ipykernel==6.16.2 \
ipython==7.34.0 \
ipython-genutils==0.2.0 \
isoduration==20.11.0 \
jedi==0.19.1 \
Jinja2==3.1.3 \
joblib==1.2.0 \
json5==0.9.16 \
jsonpointer==2.4 \
jsonschema==4.17.3 \
jupyter-events==0.6.3 \
jupyter-server==1.24.0 \
jupyter-ydoc==0.2.5 \
jupyter_client==7.4.9 \
jupyter_core==4.12.0 \
jupyter_server_fileid==0.9.1 \
jupyter_server_ydoc==0.8.0 \
jupyterlab==3.6.7 \
jupyterlab-pygments==0.2.2 \
jupyterlab_server==2.24.0 \
kiwisolver==1.4.5 \
larsflow @ git+https://github.com/VincentStimper/resampled-base-flows.git@18db5bf28ffa1d5ab9ef2b63856e186affee604b \
Markdown==3.4.4 \
MarkupSafe==2.1.5 \
matplotlib==3.5.3 \
matplotlib-inline==0.1.6 \
mdtraj==1.9.9 \
mistune==3.0.2 \
mkl-fft==1.3.1 \
mkl-random @ file:///tmp/build/80754af9/mkl_random_1626179032232/work \
mkl-service==2.4.0 \
nbclassic==1.0.0 \
nbclient==0.7.4 \
nbconvert==7.6.0 \
nbformat==5.8.0 \
nest-asyncio==1.6.0 \
nflows==0.14 \
normflows==1.6.1 \
notebook==6.5.6 \
notebook_shim==0.2.4 \
numpy @ file:///opt/conda/conda-bld/numpy_and_numpy_base_1653915516269/work \
oauthlib==3.2.2 \
omegaconf==2.3.0 \
packaging==23.2 \
pandas==1.3.5 \
pandocfilters==1.5.1 \
parso==0.8.3 \
pathtools==0.1.2 \
pexpect==4.9.0 \
pickleshare==0.7.5 \
Pillow==9.3.0 \
pkgutil_resolve_name==1.3.10 \
prometheus-client==0.17.1 \
prompt-toolkit==3.0.43 \
protobuf==3.20.3 \
psutil==5.9.8 \
ptyprocess==0.7.0 \
pyasn1==0.5.1 \
pyasn1-modules==0.3.0 \
pycparser==2.21 \
Pygments==2.17.2 \
pyparsing==3.1.1 \
pyrsistent==0.19.3 \
python-dateutil==2.8.2 \
python-json-logger==2.0.7 \
pytz==2024.1 \
PyYAML==6.0.1 \
pyzmq==24.0.1 \
requests==2.31.0 \
requests-oauthlib==1.3.1 \
residual-flows @ git+https://github.com/VincentStimper/residual-flows.git@2bbbf70570f1ce4ec8de21da3423fcc773f48f98 \
rfc3339-validator==0.1.4 \
rfc3986-validator==0.1.1 \
rsa==4.9 \
scipy==1.7.3 \
Send2Trash==1.8.2 \
sentry-sdk==1.40.5 \
setproctitle==1.3.3 \
six @ file:///tmp/build/80754af9/six_1644875935023/work \
smmap==5.0.1 \
sniffio==1.3.0 \
soupsieve==2.4.1 \
tensorboard==2.11.2 \
tensorboard-data-server==0.6.1 \
tensorboard-plugin-wit==1.8.1 \
terminado==0.17.1 \
tinycss2==1.2.1 \
tomli==2.0.1 \
torch==1.8.1 \
torchaudio==0.8.0a0+e4e171a \
torchvision==0.9.1 \
tornado==6.2 \
tqdm==4.64.1 \
traitlets==5.9.0 \
typing_extensions @ file:///croot/typing_extensions_1669924550328/work \
uri-template==1.3.0 \
urllib3==2.0.7 \
wandb==0.13.10 \
wcwidth==0.2.13 \
webcolors==1.13 \
webencodings==0.5.1 \
websocket-client==1.6.1 \
Werkzeug==2.2.3 \
y-py==0.6.2 \
ypy-websocket==0.8.4 \
zipp==3.15.0
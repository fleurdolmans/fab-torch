# Boltzmann Generators for Solute-Solvent Systems

This codebase trains Boltzmann Generators (normalising flows) to sample from the Boltzmann distribution of solute-solvent molecular systems. It contains two independent implementations:

- **2D** (`boltzmann_generators_2d/`) — a 2D purely repulsive Lennard-Jones (LJ) system, based on [BG LJ Dimer](https://github.com/weitse-hsu/boltzmann_generators/tree/master).
- **3D** (`boltzmann_generators_3d/`) — a full 3D framework built on top of [FAB-torch](https://github.com/lollcat/fab-torch) (Flow Annealed importance sampling Bootstrap), supporting real molecular systems (OpenMM) and LJ particle systems.

---

## Repository structure

```
fab-torch/
├── boltzmann_generators_2d/
│   ├── config.py                        # Central config (system params, flow, training)
│   ├── factories.py                     # Builders: build_system(), build_flow(), build_trainer()
│   ├── Library/                         # Core physics, flows, training
│   │   ├── potentials.py                # 2D LJ potential with soft harmonic walls
│   │   ├── sampling.py                  # MetropolisSampler for MC data generation
│   │   ├── boltzmann.py                 # ML / KL / overlap training loop
│   │   ├── build_system.py              # Flow construction, RSA initialisation
│   │   ├── evaluate.py                  # Evaluation metrics
│   │   ├── visual.py                    # Visualisation helpers
│   │   └── flow/
│   │       ├── realnvp.py               # RealNVP affine coupling flow
│   │       └── spline.py                # Rational-quadratic spline flow
│   ├── fab/                             # FAB-specific wrappers
│   │   ├── target.py                    # SoluteTarget2D / SoluteFullTarget2D
│   │   ├── flow_adapter.py              # FAB-compatible flow interface
│   │   └── train_fab_spline.py          # FAB training with spline flow + temperature curriculum
│   ├── Notebooks/
│   │   ├── Solute-LJ-Bath-2D.ipynb      # MC generation, pretraining (ML+KL), analysis
│   │   └── MC_data/                     # Pre-generated MC trajectories (.npz)
│   └── Trained_models/
│       ├── FAB/                         # FAB-trained checkpoints (spline, per stage)
│       ├── realnvp/                     # Pretrained RealNVP checkpoints
│       └── spline/                      # Pretrained spline checkpoints
│
└── boltzmann_generators_3d/
    ├── fab/                             # Core FAB library
    │   ├── core.py                      # FABModel: flow + AIS + loss functions
    │   ├── types_.py                    # Abstract base types
    │   ├── train.py                     # Trainer (real molecular systems)
    │   ├── train_LJ.py                  # TrainerLJ (LJ systems, forward KL + mixing)
    │   ├── train_with_buffer.py         # BufferTrainer (non-prioritised replay)
    │   ├── train_with_prioritised_buffer.py     # PrioritisedBufferTrainer (FAB + buffer)
    │   ├── train_with_prioritised_buffer_LJ.py  # PrioritisedBufferTrainerLJ (LJ variant)
    │   ├── target_distributions/
    │   │   ├── solute_in_water.py       # Real molecular system (OpenMM, AMBER14/TIP3P)
    │   │   ├── solute_in_water_LJ.py    # 3D LJ solute-solvent system (OpenMM)
    │   │   ├── boltzmann.py             # TransformedBoltzmann energy wrapper
    │   │   └── gaussian.py              # Gaussian target (for testing)
    │   ├── transforms/
    │   │   ├── global_3point_spherical_transform.py     # GPS (droplet + PBC)
    │   │   ├── global3point_radial_rotvec_transform.py  # GPR (droplet)
    │   │   ├── lab_frame_torus_transform.py             # LFT (PBC)
    │   │   ├── lab_frame_geometric_torus_transform.py   # LGT (PBC)
    │   │   └── transform_LJ.py                          # Coordinate transforms for LJ systems
    │   ├── flow/
    │   │   ├── water_coupling.py        # Permutation-equivariant water coupling layers
    │   │   └── solute_coupling.py       # Solute sub-flow layers
    │   ├── sampling_methods/
    │   │   ├── ais.py                   # Annealed Importance Sampling
    │   │   └── transition_operators/
    │   │       ├── hmc.py               # Hamiltonian Monte Carlo
    │   │       └── metropolis.py        # Metropolis-Hastings
    │   └── utils/
    │       ├── evaluate.py              # Evaluation metrics
    │       ├── visuals.py               # Plotting utilities
    │       ├── logging.py               # ListLogger, WandbLogger
    │       ├── numerical.py             # ESS, expectation utilities
    │       ├── replay_buffer.py
    │       └── prioritised_replay_buffer.py
    ├── experiments/
    │   ├── setup_run.py                 # Model + trainer setup (molecular systems)
    │   ├── setup_run_LJ.py              # Model + trainer setup (LJ systems)
    │   ├── logger_setup.py
    │   ├── load_model_for_eval.py
    │   ├── make_flow/
    │   │   ├── make_normflow_model_equi.py      # Permutation-equivariant droplet flows
    │   │   ├── make_normflow_model_droplet.py   # Non-equivariant droplet flows
    │   │   ├── make_normflow_model_torus.py     # PBC / torus flows
    │   │   ├── make_normflow_model_gps.py       # GPS flow (droplet or PBC)
    │   │   └── make_normflow_model_LJ.py        # Flow builder for LJ systems
    │   ├── make_base/
    │   │   ├── base.py                  # Base distribution factory (molecular)
    │   │   └── base_LJ.py               # Base distribution factory (LJ)
    │   └── solvation/
    │       ├── run.py                   # Entry point: real molecular system
    │       ├── run_LJ.py                # Entry point: LJ system
    │       ├── run_pbc.py               # Entry point: PBC variant
    │       └── config/
    │           ├── SoluteInSolvent.yaml
    │           └── LJ.yaml
    ├── data_generation/
    │   ├── md_data.py                   # MD data generation (molecular systems)
    │   ├── md_data_LJ.py                # MD data generation (LJ systems)
    │   └── config/
    │       ├── make_md_data.yaml
    │       └── make_md_data_LJ.yaml
    ├── config/
    │   └── node/                        # Machine-specific path configs
    └── visualize_models.ipynb           # Model evaluation notebook
```

---

## Installation

Requires conda. Clone this repository, `cd` into it, then:

```bash
conda create --name bgsol python=3.11 -y
conda activate bgsol

# PyTorch (GPU)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# PyTorch (CPU / Apple Silicon)
# conda install -y pytorch torchvision torchaudio -c pytorch

# OpenMM and scientific packages
conda install -y -c conda-forge numpy openmm openmmtools
conda install -y -c conda-forge mdtraj pandas h5py

# Python packages
python -m pip install -U pip
pip install -r requirements.txt
```

**Sanity checks:**
```bash
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'cuda ver', torch.version.cuda)"
python -c "import openmm, openmm.testInstallation; openmm.testInstallation.main()"
```

If conda is slow resolving the environment, use the libmamba solver.

---

## 2D system (`boltzmann_generators_2d/`)

A 2D implementation of a single replusive LJ solute surrounded by repulsive LJ solvent particles in a non-periodic box. Energy is computed directly in PyTorch (no OpenMM), making it fast to iterate on and useful for debugging flow architectures.

**System**: 1 fixed LJ solute at the origin + 36 LJ solvent particles in 2D. Input dimension to the flow is 72 (36 × 2D Cartesian).

### Data generation

Data is generated via **Metropolis Monte Carlo** using `MetropolisSampler` in `Library/sampling.py`. No MD or OpenMM required.

The sampler proposes single-particle displacements and accepts/rejects based on the Boltzmann weight. The system is initialised using **Random Sequential Addition** (RSA) from `Library/build_system.py` to avoid initial overlaps.

The notebook drives data generation:

```python
# from Notebooks/Solute-LJ-Bath-2D.ipynb
sampler = MetropolisSampler(system, T=1.0, sigma_MC=0.1, stride=10)
xtraj = sampler.run(n_steps=500_000)
np.savez("Notebooks/MC_data/traj.npz", xtraj=xtraj)
```

Pre-generated trajectories are stored in `Notebooks/MC_data/` as `.npz` files and used to seed the replay buffer during FAB training.

### Available flows

Two flow architectures are implemented in `Library/flow/`:

| Flow | File | Architecture | Input dim |
|---|---|---|---|
| RealNVP | `flow/realnvp.py` | Affine coupling layers, alternating binary masks, MLP conditioners | 72D (solvent) or 74D (all particles) |
| Spline flow | `flow/spline.py` | Rational-quadratic spline coupling, odd/even particle partitioning, MLP or DeepSets conditioner | 72D (solvent only) |

Both use a diagonal Gaussian base distribution (σ=1.0).

**Default hyperparameters** (set in `config.py`):
- 8 coupling blocks (16 layers total)
- 256-dim hidden layers, 3-layer MLP conditioners
- 8 spline bins, tail bound 5.5 (spline only)

### Training

Training is done using stage training:

**Stage 1 — Pretraining** (`Library/boltzmann.py`):
Trains the flow on MC data.


**Stage 2 — ML + KL** (`Library/boltzmann.py`):
Trains the flow on 20% MC data and on 80% energies of the flow genereted samples, with an overlap penlaty of 2. 


**Stage 3 — KL only** (`Library/boltzmann.py`):
Trains fully on energies of the flow genereted samples, with an overlap penlaty of 10.
- after this we do some finetuning by running it for a small number of epoch with overlap penalty 20 and the 30. 


**Stage 4 — (optional) FAB fine-tuning** (`fab/train_fab.py`, `fab/train_fab_spline.py`):
Fine-tunes the pretrained flow with FAB using a **temperature curriculum**:

| Stage | Temperature | Iterations |
|---|---|---|
| 1 | 5.0 | 30 |
| 2 | 3.0 | 40 |
| 3 | 2.0 | 50 |
| 4 | 1.0 | 600 |

Each stage uses AIS with 16 intermediate distributions and Metropolis transitions (15 steps, adaptive step size). Samples are stored in a prioritised replay buffer and reused for multiple gradient updates per AIS call.

FAB checkpoints are saved per stage to `Trained_models/FAB/`.

### Notebooks

**`Notebooks/Solute-LJ-Bath-2D.ipynb`** — the main experimental notebook:
- Monte Carlo data generation for different system sizes
- RealNVP and spline flow pretraining (ML + KL + overlap)
- Hungarian alignment preprocessing for particle permutation invariance
- Checkpoint saving for FAB fine-tuning
- Post-training analysis: radial distribution functions, position densities, energy distributions

---

## 3D system (`boltzmann_generators_3d/`)

The full 3D framework for training Boltzmann Generators on realistic molecular systems using OpenMM or LJ potentials. Built on the FAB algorithm with AIS-based bootstrapping.

### Target distributions

The goal is to sample from the Boltzmann distribution $p(x) \propto \exp(-U(x) / k_B T)$ where:

- **`SoluteInWater`** — triatomic solute (e.g. SO2, H2O) in explicit TIP3P water. $U(x)$ evaluated with OpenMM using AMBER14 force fields. Supports **droplet** and **PBC** boundary conditions.
- **`SoluteInWaterLJ`** — one or more LJ solutes in a 3D periodic box of LJ solvent particles. $U(x)$ evaluated with OpenMM LJ potential and switching function.

### Data generation

MD simulations are run with OpenMM. Run from the `boltzmann_generators_3d/` directory:

**Real molecular systems:**
```bash
python data_generation/md_data.py --config-name make_md_data.yaml
```

**LJ systems:**
```bash
python data_generation/md_data_LJ.py --config-name make_md_data_LJ.yaml
```

For LJ systems, initialisation follows: L-BFGS energy minimisation → short NVE equilibration → Nosé-Hoover thermostat equilibration → burn-in → production run.

Output is saved as `.h5` files with a `coordinates` dataset of shape `(n_frames, 3N)` in nanometres.

### Coordinate transforms (X ↔ I)

The flow operates in *internal coordinates* $I$ rather than Cartesian coordinates $X$ to remove translational/rotational symmetry and handle periodicity. `transform.forward(i)` maps $I \to X$; `transform.inverse(x)` maps $X \to I$.

**For real molecular systems** (`target.transform.version` in config):

| Key | Class | Boundary | Description |
|---|---|---|---|
| `GPT` | `Global3PointSphericalTransform` | Droplet or PBC | 3-point spherical; removes 6 global DOF. Mixed radial + periodic angles. |
| `GPR` | `Global3PointRadialRotvecTransform` | Droplet | Radial + rotation vector per water relative to 3 reference atoms. All in ℝ. |
| `SFIC` | `SFICTransform` | Droplet | Solute-frame internal coordinates. |
| `SFIC-T` | `SFICTorusTransform` | PBC | SFIC with torus parametrisation. |
| `LGT` | `LabFrameGeometricTorusTransform` | PBC | Lab-frame torus coords; encodes solvent relative to solute. |

**For LJ systems** (`target.transform_version` in config):

| Key | Class | Description |
|---|---|---|
| `v1` | `SolventOnlyTransform` | Identity on solvent; fixed solute. `internal_dim = 3 * n_solvent` |
| `v2` | `SoluteCenteredSolventTransform` | Solvent coords relative to solute centroid |
| `v3` | `TorusCartesianTransform` | Solvent on torus in Cartesian space |
| `v4` | `FixedSoluteUnitTorusTransform` | Solvent mapped to unit torus `[0, 1)³` |

### Available flows

Set via `flow.type` in config:

| Type | Equivariant | Boundary | Description |
|---|---|---|---|
| `shared-water-spline-nf` | No | Droplet | Shared-weight spline coupling over water blocks |
| `spherical-circ-rqs-nf` | No | Droplet | Circular RQS on mixed radial + spherical coordinates (GPT transform) |
| `coupled-rqs-nf` | No | Droplet | Standard flat RQS coupling on unconstrained internal coordinates |
| `realnvp-nf` | No | Droplet | Affine coupling (RealNVP) on flat internal coordinates |
| `circ-rqs-torus-nf` | No | **PBC only** | Circular RQS on torus coordinates |
| `perm-equi-torus-nf` | **Yes** | **PBC only** | Permutation-equivariant spline flow on torus. Periodic τ angles with circular RQS. Water mean-pool conditioning. |
| `perm-equi-joint-spline-nf` | **Yes** | Droplet | Jointly updates solute and water each layer. Water conditioned on mean-pool. |
| `perm-equi-spline-nf` | **Yes** | Droplet | Water-only flow conditioned on pairwise O-O distances (RBF geometry). Solute used as context only. |
| `perm-equi-gps-nf` | **Yes** | Both | 9D spherical water blocks (O + H1 + H2). Mean-pool + solute shape conditioning. |

**Key flow hyperparameters** (in config):

| Parameter | Description |
|---|---|
| `flow.layers` | Number of coupling layers |
| `flow.hidden_units` | Conditioner network hidden width |
| `flow.num_bins` | Number of RQS spline bins |
| `flow.tail_bound` | Spline tail bound |
| `flow.dropout` | Dropout rate in conditioner networks |
| `flow.base.type` | Base distribution (`gauss`, `structured-gauss`) |

### Training

#### Loss types (`fab.loss_type` in config)

| Loss | Requires MD data | Description |
|---|---|---|
| `forward_kl` | Yes | Maximum likelihood on MD samples. Stable, requires a pre-generated dataset. |
| `base_transport` | Yes | Base transport on MD samples. |
| `fab_alpha_div` | No | FAB loss. AIS generates importance-weighted samples targeting $p^\alpha / q^{\alpha-1}$. |
| `fab_ub_alpha_2_div` | No | Upper bound on alpha-2 divergence. Conservative FAB variant. |
| `flow_reverse_kl` | No | Mode-seeking reverse KL from flow samples. |
| `flow_alpha_2_div` | No | Alpha-2 divergence estimated from flow samples. |

#### Trainers

| Class | File | When to use |
|---|---|---|
| `Trainer` | `train.py` | Real molecular systems (`SoluteInWater`). Forward KL, base transport, and reverse KL. SO2/water-specific overlap penalty. |
| `TrainerLJ` | `train_LJ.py` | LJ systems. Same loss modes plus optional mixing of MD and reverse-KL objectives. Generic LJ overlap penalty. Saves flow samples as `.h5` and `.pdb` at end of training. |
| `PrioritisedBufferTrainer` | `train_with_prioritised_buffer.py` | FAB with prioritised experience replay (molecular systems). Targets $p^\alpha / q^{\alpha-1}$. |
| `PrioritisedBufferTrainerLJ` | `train_with_prioritised_buffer_LJ.py` | Same, LJ variant. |
| `BufferTrainer` | `train_with_buffer.py` | FAB with non-prioritised replay buffer. |

The trainer is selected automatically in `setup_run.py` / `setup_run_LJ.py` based on `fab.loss_type` and `training.buffer` settings.

#### FAB mechanism

When using a FAB loss with a replay buffer:
1. AIS runs from $q_\theta$ toward $p^\alpha / q^{\alpha-1}$ using HMC or Metropolis transitions.
2. Samples and log-weights are stored in a prioritised replay buffer.
3. Multiple gradient steps are taken per AIS call using buffered samples.
4. Log-weights are adjusted in the buffer to account for flow parameter updates.

#### Optional: overlap penalty

`Trainer` and `TrainerLJ` support an auxiliary clash penalty on flow samples to discourage atom overlaps (`overlap_penalty` in config). For the real molecular system this is a SO2/water-specific pairwise penalty; for LJ systems it is a generic solute-solvent and solvent-solvent penalty.

#### Optional: MD mixing

When using a reverse-KL or FAB loss, a fraction `mixing` of each batch can be drawn from MD data and trained with likelihood loss simultaneously.

### Running experiments

All experiments use [Hydra](https://hydra.cc/). Run from `boltzmann_generators_3d/`:

**Real molecular system (SO2 or H2O in water):**
```bash
python experiments/solvation/run.py --config-name SoluteInSolvent.yaml
```

**LJ system:**
```bash
python experiments/solvation/run_LJ.py --config-name LJ.yaml
```

**On a cluster (e.g. Snellius):**
```bash
PYTHONPATH="/path/to/fab-torch/boltzmann_generators_3d" \
LD_LIBRARY_PATH="/path/to/conda/envs/bgsol/lib" \
/path/to/conda/envs/bgsol/bin/python \
  experiments/solvation/run_LJ.py --config-name LJ.yaml node=snellius
```

Machine-specific paths live in `config/node/`. Override on the command line with `node=<name>`.

### Notebooks

**`visualize_models.ipynb`** — loads saved checkpoints and generates evaluation plots: radial distribution functions, energy histograms, position densities, and marginal distributions.

---

## Evaluation metrics (3D)

At each evaluation step (controlled by `n_eval` in config):

| Metric | Description |
|---|---|
| `flow_test_log_prob` | Mean log probability of MD validation data under the flow |
| `flow_test_log_prob_per_dim` | Same, normalised per internal coordinate dimension |
| `eval_ess_flow` | Effective sample size of flow samples reweighted to target |
| `flow_frac_clipped` | Fraction of flow samples with energy above `energy_cut` |
| `mean_forward_kl_marginals` | Mean per-dimension forward KL from marginal histograms |
| `mean_reverse_kl_marginals` | Mean per-dimension reverse KL from marginal histograms |

---

## Logging (3D)

Configure [Weights & Biases](https://wandb.ai) in the config:

```yaml
logger:
  wandb:
    name: my_run_name
    project: my_project
    entity: my_wandb_entity
```

Output saved to `{hydra.run.dir}/`:

```
{hydra.run.dir}/
├── plots/              # Saved figures
├── model_checkpoints/  # Checkpoints (model + optimizer state)
└── wandb/              # W&B run files
```

---

## Code conventions (3D)

- `log_p` — log probability of the Boltzmann target $p$.
- `log_q` — log probability of the flow $q_\theta$.
- `I` = internal coordinates (flow space); `X` = Cartesian coordinates.
- `transform.forward(i)` maps $I \to X$; `transform.inverse(x)` maps $X \to I$.
- All coordinates are in **nanometres**.
- `train_data_i` / `train_logdet_xi` — MD data transformed to internal coordinates, stored on the target distribution object.
- `energy_cut` / `energy_max` — energies above `energy_cut` are regularised with a soft log; above `energy_max` they are clamped.

---

## Good starting hyperparameters

**2D system — FAB spline:**
- Pretrain with ML + KL for 200 epochs before FAB
- FAB: temperature curriculum T = 5 → 3 → 2 → 1, ~600 iterations at T=1
- Adam, LR `5e-5`, gradient clipping norm 5.0

**3D system — forward KL on LJ:**
- 50 000 iterations, batch size 1024
- Learning rate `7e-5`, cosine schedule
- 36 flow layers, hidden dim 256, 8 spline bins, tail bound 4.0

**3D system — forward KL on real molecular system:**
- 5 000 iterations to detect overfitting on 100–1000 MD samples
- Learning rate `5e-4`, weight decay `1e-5`
- 12 flow layers, hidden dim 256, 8 spline bins

**3D system — FAB:**
- Start with forward KL pre-training until the flow has reasonable coverage.
- Then switch to FAB (`fab_alpha_div`, α=2) with a prioritised buffer.
- FAB is significantly slower per iteration than likelihood training.

# Boltzmann Generators for Solute-Solvent Systems

This codebase trains Boltzmann Generators (normalising flows) to sample from the Boltzmann distribution of solute-solvent systems. It is built on top of [FAB-torch](https://github.com/lollcat/fab-torch) (Flow Annealed importance sampling Bootstrap), which provides the core FAB training infrastructure.

---

## Supported systems

Three types of solute-solvent system are implemented:

| System | Target class | Run script | Config |
|---|---|---|---|
| Triatomic solute (e.g. H2O, SO2) in explicit water | `SoluteInWater` | `experiments/solvation/run.py` | `experiments/solvation/config/SoluteInSolvent.yaml` |
| LJ solute(s) + LJ solvent in 3D periodic box | `LJParticles` | `experiments/solvation/run_LJ.py` | `experiments/solvation/config/LJ.yaml` |
| LJ solute + LJ solvent in 2D periodic box | `LJParticles2D` | `experiments/solvation/run_LJ_2D.py` | *(derived from `LJ.yaml`)* |

---

## Repository structure

```
fab-torch/
├── fab/
│   ├── target_distributions/
│   │   ├── solute_in_water.py          # Real molecular system (OpenMM)
│   │   ├── solute_in_water_LJ.py       # 3D LJ solute-solvent system
│   │   ├── solute_in_water_LJ_2D.py    # 2D LJ solute-solvent system
│   │   └── boltzmann.py                # TransformedBoltzmann energy wrapper
│   ├── transforms/
│   │   ├── transform_LJ.py             # Coordinate transforms for LJ systems (v1-v4)
│   │   ├── transform_LJ_2D.py          # Coordinate transforms for 2D LJ
│   │   ├── global_3point_spherical_transform.py   # GPT transform for molecular systems
│   │   ├── global3point_radial_rotvec_transform.py  # GPR transform
│   │   ├── lab_frame_geometric_torus_transform.py   # LGT transform
│   │   ├── sfic_transform.py           # SFIC transform
│   │   └── sfic_torus_transform.py     # SFIC-T transform
│   ├── flow/
│   │   ├── water_coupling.py           # Permutation-equivariant water spline coupling
│   │   └── solute_coupling.py          # Solute sub-flow layers
│   ├── sampling_methods/
│   │   ├── ais.py                      # Annealed Importance Sampling
│   │   └── transition_operators/
│   │       ├── hmc.py                  # Hamiltonian Monte Carlo
│   │       └── metropolis.py           # Metropolis-Hastings
│   ├── train_LJ.py                     # Trainer for LJ systems (forward KL)
│   ├── train_with_prioritised_buffer_LJ.py  # Trainer with FAB + prioritised buffer
│   └── core.py                         # FABModel core
├── experiments/
│   ├── solvation/
│   │   ├── run.py                      # Entry point: real molecular system
│   │   ├── run_LJ.py                   # Entry point: 3D LJ system
│   │   ├── run_LJ_2D.py                # Entry point: 2D LJ system
│   │   └── config/
│   │       ├── SoluteInSolvent.yaml    # Config for real molecular system
│   │       └── LJ.yaml                 # Config for LJ system
│   ├── make_flow/
│   │   ├── make_normflow_model.py      # Flow builder for molecular systems
│   │   └── make_normflow_model_LJ.py   # Flow builder for LJ systems
│   ├── setup_run_LJ.py                 # Trainer + model setup for LJ experiments
│   └── setup_run.py                    # Trainer + model setup for molecular experiments
├── data_generation/
│   ├── md_data.py                      # MD data generation for molecular systems
│   ├── md_data_LJ.py                   # MD data generation for LJ systems
│   └── config/
│       ├── make_md_data.yaml
│       └── make_md_data_LJ.yaml
└── config/
    └── node/                           # Machine-specific path configs
```

---

## Installation

Requires conda. Clone this repository, `cd` into it, then:

```bash
conda create --name bgsol python=3.7.16
conda activate bgsol
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge openmm cudatoolkit=10.1
conda config --add channels omnia --add channels conda-forge
conda install openmmtools
pip install -r requirements.txt
```

If conda is slow resolving the environment, use the libmamba solver (available in newer conda versions).

---

## How it works

### 1. Physical system and energy

The target distribution is the Boltzmann distribution at temperature $T$:

$$p(x) \propto \exp\!\left(-\frac{U(x)}{k_B T}\right)$$

where $U(x)$ is the potential energy of configuration $x$.

- **Real molecular systems** (`SoluteInWater`): $U(x)$ is evaluated using OpenMM with AMBER14/TIP3P force fields. Supports droplet and PBC boundary conditions.
- **LJ systems** (`LJParticles`, `LJParticles2D`): $U(x)$ is a Lennard-Jones potential evaluated in PyTorch (2D) or via OpenMM (3D), with a switching function and periodic boundary conditions.

### 2. Coordinate transforms (X ↔ I)

The flow does not operate directly on Cartesian coordinates $X$ (e.g. due to translation/rotation symmetry and periodicity). Instead, configurations are mapped to *internal coordinates* $I$ via a fixed, invertible coordinate transform. The flow is trained in internal coordinate space.

Notation used throughout the code: **I → X** means a forward transform from internal to Cartesian; **X → I** means the inverse (used to transform MD data before feeding it to the flow).

#### Transforms for real molecular systems (`transform_version` in config)

| Key | Class | Description |
|---|---|---|
| `GPT` | `Global3PointSphericalTransform` | 3-point spherical transform; removes 6 global DOF (3 translation + 3 rotation), dim = `3N - 6` |
| `GPR` | `Global3PointRadialRotvecTransform` | Radial + rotation vector representation for each water molecule relative to 3 reference atoms |
| `SFIC` | `SFICTransform` | Solute-frame internal coordinates |
| `LGT` | `LabFrameGeometricTorusTransform` | Lab-frame torus coordinates; encodes solvent positions relative to solute center |
| `SFIC-T` | `SFICTorusTransform` | SFIC transform with torus parametrisation |

The default for molecular systems is `LGT`.

#### Transforms for LJ systems (`transform_version` in config)

| Key | Class | Description |
|---|---|---|
| `v1` | `SolventOnlyTransform` | Identity on solvent coords; inserts fixed solute coordinates. `internal_dim = 3 * n_solvent` |
| `v2` | `SoluteCenteredSolventTransform` | Solvent coords relative to solute centroid |
| `v3` | `TorusCartesianTransform` | Solvent coords on torus in Cartesian space |
| `v4` | `FixedSoluteUnitTorusTransform` | Solvent coords mapped to unit torus `[0,1)` |

For 2D LJ systems the only supported transform is `v4` (`FixedSoluteUnitTorusTransform2D`).

### 3. Normalising flow

A normalising flow $q_\theta(i)$ is trained to approximate $p(i)$, the Boltzmann distribution in internal coordinates. The flow is a stack of neural spline coupling layers (Rational Quadratic Splines) built using the `normflows` library.

- **Molecular systems**: Supports circular coupling for bond angle dimensions. Flow builder: `make_normflow_model.py`.
- **LJ systems**: Coupling layers operate on groups of 3 (xyz per particle); optionally permutation-equivariant over solvent molecules using `PermEquiWaterSplineCoupling` in `fab/flow/water_coupling.py`. Flow builder: `make_normflow_model_LJ.py`.

Key flow hyperparameters (set in the config):

| Parameter | Description |
|---|---|
| `flow.n_layers` / `flow.blocks` | Number of coupling layers |
| `flow.hidden_units` | Hidden layer width of conditioner networks |
| `flow.num_bins` | Number of spline bins |
| `flow.tail_bound` | Spline tail bound (assumes ~unit-variance data) |
| `flow.group_size` | Feature group size for coupling splits (6 for molecular systems: xyz per water) |
| `flow.base.type` | Base distribution type (e.g. `gauss-uni`, shell-based) |

### 4. Training

Two training modes are supported:

#### Forward KL (likelihood training)
Minimises the forward KL divergence $D_\text{KL}(p \| q)$ using MD samples. This requires pre-generated MD data. Corresponds to maximum likelihood training on the MD trajectory.

Trainer: `fab/train_LJ.py` (`TrainerLJ`)

#### FAB (Flow Annealed importance sampling Bootstrap)
Minimises an alpha-divergence objective without requiring MD samples for training (only for evaluation). Uses Annealed Importance Sampling (AIS) with HMC or Metropolis transitions to generate improved samples, optionally stored in a prioritised replay buffer.

Trainer: `fab/train_with_prioritised_buffer_LJ.py` (`PrioritisedBufferTrainer`)

The training mode is controlled by `fab.loss_type` and `training.buffer` in the config.

---

## Generating MD data

MD simulations to generate training/validation data are run using OpenMM.

For real molecular systems:
```bash
python ./data_generation/md_data.py --config-name make_md_data.yaml
```

For LJ systems:
```bash
python ./data_generation/md_data_LJ.py --config-name make_md_data_LJ.yaml
```

Output is saved as `.h5` files containing a `coordinates` dataset of shape `(n_frames, dim)`, in nanometres.

---

## Running experiments

All experiments use [Hydra](https://hydra.cc/) configs. Commands should be run from the repository root with the conda environment active.

### Node configuration

Machine-specific paths (data directories, output directories) are defined in `config/node/`. The default node is `desktop`. Create your own node file and update the `defaults` in the relevant config to use it automatically. Alternatively, override on the command line:

```bash
python experiments/solvation/run_LJ.py --config-name LJ.yaml node=snellius
```

You can also set the `MAIN_DIR` environment variable to the root of the repository so that Hydra can locate the `config/` directory:

```bash
export MAIN_DIR=/path/to/fab-torch
```

### Real molecular system (H2O or SO2 in water)

```bash
python experiments/solvation/run.py --config-name SoluteInSolvent.yaml
```

Key config options in `SoluteInSolvent.yaml`:

| Option | Description |
|---|---|
| `target.solute_name` | Name of the solute (e.g. `water`, `so2`). Must match a `.pdb` (and optionally `.xml`) file in the data directory. |
| `target.num_solvent_molecules` | Number of water molecules |
| `target.boundary_condition` | `droplet` or `pbc` |
| `target.transform.version` | Coordinate transform: `LGT` (default), `GPT`, `GPR`, `SFIC`, `SFIC-T` |
| `target.temperature` | Temperature in Kelvin |
| `target.train_samples_path` | Path to MD training data (`.h5` or `.pt`) |
| `target.val_samples_path` | Path to MD validation data |

### 3D LJ system

```bash
python experiments/solvation/run_LJ.py --config-name LJ.yaml
```

Key config options in `LJ.yaml`:

| Option | Description |
|---|---|
| `target.n_solvent` | Number of LJ solvent particles |
| `target.solute_positions_nm` | List of fixed solute positions `[[x,y,z], ...]` |
| `target.solvent_sigma_nm` | Solvent LJ sigma (nm) |
| `target.solvent_epsilon_kjmol` | Solvent LJ epsilon (kJ/mol) |
| `target.solute_sigma_nm` | Solute LJ sigma (nm) |
| `target.box_length_nm` | Cubic box side length (nm) |
| `target.transform_version` | Coordinate transform: `v1` (default), `v2`, `v3`, `v4` |
| `target.temperature` | Temperature (reduced units: K) |
| `target.base_state` / `target.target_state` | Labels for base/target MD simulation versions |

### 2D LJ system

```bash
python experiments/solvation/run_LJ_2D.py --config-name LJ.yaml
```

Operates in 2D: particles have $(x, y)$ coordinates. The energy is computed directly in PyTorch (no OpenMM), making this lightweight for debugging. Only `transform_version: v4` is supported.

### On a cluster (e.g. Snellius)

Set `PYTHONPATH` and `LD_LIBRARY_PATH` if needed:

```bash
PYTHONPATH="/path/to/fab-torch" \
LD_LIBRARY_PATH="/path/to/conda/envs/bgsol/lib" \
/path/to/conda/envs/bgsol/bin/python \
  experiments/solvation/run_LJ.py --config-name LJ.yaml node=snellius
```

---

## Evaluation metrics

At each evaluation step (controlled by `evaluation.n_eval` in config), the following metrics are computed and logged:

| Metric | Description |
|---|---|
| `flow_test_log_prob` | Mean log probability of MD validation data under the flow |
| `flow_test_log_prob_per_dim` | Same, normalised per internal coordinate dimension |
| `eval_ess_flow` | Effective Sample Size of flow samples under the target |
| `flow_frac_clipped` | Fraction of flow samples with clipped (high) energies |
| `mean_forward_kl_marginals` | Mean per-dimension forward KL estimated from marginal histograms |
| `mean_reverse_kl_marginals` | Mean per-dimension reverse KL estimated from marginal histograms |

Metrics are saved to `{hydra.run.dir}/metrics/metrics_{iteration}.json`.

---

## Logging

We recommend [Weights & Biases](https://wandb.ai) for logging. Configure it in the config:

```yaml
logger:
  wandb:
    name: my_run_name
    project: my_project
    entity: my_wandb_entity
```

Output is also stored on disk in the `hydra.run.dir` directory:

```
{hydra.run.dir}/
├── plots/                  # Saved figures
├── metrics/                # JSON files with evaluation metrics
├── model_checkpoints/      # Model checkpoint files
└── wandb/                  # W&B run files (config, logs, media)
```

---

## Code conventions

- **log_p** refers to the log probability of the Boltzmann target $p$.
- **log_q** refers to the log probability of the flow $q_\theta$.
- **I** = internal coordinates (flow space); **X** = Cartesian coordinates.
- `transform.forward(i)` maps $I \to X$; `transform.inverse(x)` maps $X \to I$.
- All coordinates are in **nanometres**.
- `energy_cut` / `energy_max`: energies above `energy_cut` are handled with a soft log regularisation; above `energy_max` they are clamped. Set `target.n_threads = 1` to simplify debugging.

---

## Good starting hyperparameters

For likelihood training on the LJ system:
- 50 000 iterations, batch size 1024
- Learning rate `7e-5`, weight decay `1e-5`, cosine LR schedule
- 36 flow layers, hidden dimension 256, 8 spline bins

For likelihood training on the molecular system:
- 5 000 iterations is enough to check overfitting on 100–1000 MD samples
- Learning rate `5e-4`, weight decay `1e-5`
- 12 flow layers, hidden dimension 256, 8 spline bins

FAB training is significantly slower than likelihood training. Use FAB only once likelihood training is working.

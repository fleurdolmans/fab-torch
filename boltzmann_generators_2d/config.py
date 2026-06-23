"""
Central configuration for the 2D Boltzmann generator.

This package models a 2D repulsive Lennard-Jones solute-in-bath system.

All downstream parameters (hyperparameters, flow architecture, box size, etc.)
are stored in _CONFIGS['2d'].
"""

# ============================================================
# SYSTEM TYPE (fixed to '2d' for this package)
# ============================================================
SYSTEM_TYPE: str = '2d'

# ============================================================
# Per-system parameter dictionary
# ============================================================

_CONFIGS = {
    '2d': dict(
        # Physical system
        dim=2,
        n_particles=33,
        n_solvent=32,
        l_box=4.5,
        sigma=1.1,
        epsilon=1.0,
        k_box=100.0,
        center_solute=True,
        k_center=20.0,
        # Flow
        flow_dim=64,           # 32 solvent × 2
        flow_arch='spline', # 'spline' | 'realnvp'
        prior_sigma=1.0,
        num_bins=8,
        tail_bound=7.0,
        # Training
        n_blocks=8,
        n_nodes=100,
        n_layers=3,
        n_epochs=200,
        batch_size=2048,
        lr=1e-3,
        w_overlap=0.0,
        patience=50,
        min_delta=1e-4,
        # Potential selector
        potential='lj_2d',
    ),
}

if SYSTEM_TYPE not in _CONFIGS:
    raise ValueError(
        f"SYSTEM_TYPE must be one of {list(_CONFIGS.keys())}, got '{SYSTEM_TYPE}'"
    )

# ============================================================
# Flatten keys into module-level names so callers can write:
#   from boltzmann_generators_2d.config import l_box, batch_size
# or
#   import boltzmann_generators_2d.config as cfg; cfg.l_box
# ============================================================
_C = _CONFIGS[SYSTEM_TYPE]

dim           = _C['dim']
n_particles   = _C['n_particles']
n_solvent     = _C['n_solvent']
l_box         = _C['l_box']
sigma         = _C['sigma']
epsilon       = _C['epsilon']
k_box         = _C['k_box']
center_solute = _C['center_solute']
k_center      = _C['k_center']
flow_dim      = _C['flow_dim']
flow_arch     = _C['flow_arch']
prior_sigma   = _C['prior_sigma']
num_bins      = _C['num_bins']
tail_bound    = _C['tail_bound']
n_blocks      = _C['n_blocks']
n_nodes       = _C['n_nodes']
n_layers      = _C['n_layers']
n_epochs      = _C['n_epochs']
batch_size    = _C['batch_size']
lr            = _C['lr']
w_overlap     = _C['w_overlap']
patience      = _C['patience']
min_delta     = _C['min_delta']
potential     = _C['potential']

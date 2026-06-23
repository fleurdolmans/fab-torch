import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from .flow.spline import SoluteSplineFlow, SoluteSplineCoupling, SplineConditioner2D, DeepSetsConditioner2D, GaussianPrior
from .flow.realnvp import RealNVP
from .flow.equivariant import EquivariantPolarFlow, EquivariantPolarCouplingLayer
from .flow.equivariant import GaussianPrior as EquivGaussianPrior

def build_solute_coords_2d_rsa(N=37, l_box=5.0, sigma=1.1, seed=42):
    np.random.seed(seed)
    coords = np.zeros((N, 2), dtype=np.float32)
    coords[0] = [0.0, 0.0]   # solute fixed at origin
    placed = 1
    attempts = 0
    while placed < N:
        candidate = np.random.uniform(-l_box, l_box, size=2).astype(np.float32)
        dists = np.linalg.norm(coords[:placed] - candidate, axis=1)
        if dists.min() >= sigma:
            coords[placed] = candidate
            placed += 1
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("RSA failed — density too high")
    return coords



def build_spline_flow(
    system,
    n_solvent=32,
    dim=2,
    n_blocks=8,
    n_nodes=256,
    n_hidden=3,
    num_bins=8,
    tail_bound=6.0,
    hidden=128,
    conditioner_type='mlp',
):
    """
    Build a SoluteSplineFlow for the 2-D solute-LJ-bath system.

    Models only the 36 solvent particles (72 dimensions). The solute is fixed
    at the origin; loss_KL prepends it via _add_fixed_solute before computing energy.

    Parameters
    ----------
    system      : system object with .get_energy() and .sigma
    n_particles : number of solvent particles to model (default 36)
    n_blocks    : number of A→B + B→A coupling block pairs (total layers = 2*n_blocks)
    n_nodes     : hidden layer width in each conditioner MLP (only used when conditioner='mlp')
    n_hidden    : number of hidden layers in each conditioner MLP (only used when conditioner='mlp')
    num_bins    : number of spline bins
    tail_bound  : spline domain (-tail_bound, tail_bound); default 7.0 suits
                  Cartesian data in [-5, 5] with 40% margin.
    hidden      : hidden dimension for PairwiseInvariantConditioner (only used when
                  conditioner='pairwise'; default 128).
    """

    all_idx = torch.arange(n_solvent)          # 0..31 (0-based solvent indices)
    group_A = all_idx[all_idx % 2 == 0]          # 16 particles: [0, 2, ..., 30]
    group_B = all_idx[all_idx % 2 == 1]          # 16 particles: [1, 3, ..., 31]
    n_A, n_B = len(group_A), len(group_B)

    CondClass = DeepSetsConditioner2D if conditioner_type == 'deepsets' else SplineConditioner2D

    layers = []
    for _ in range(n_blocks):
        cond_AB = CondClass(n_B * 2, n_A * 2, num_bins, n_nodes, n_hidden)
        cond_BA = CondClass(n_A * 2, n_B * 2, num_bins, n_nodes, n_hidden)
        layers.append(SoluteSplineCoupling(cond_AB, group_B, group_A, num_bins, tail_bound))
        layers.append(SoluteSplineCoupling(cond_BA, group_A, group_B, num_bins, tail_bound))

    prior = GaussianPrior(dim=n_solvent * dim)
    return SoluteSplineFlow(
        layers=layers, prior=prior, system=system,
        n_particles=n_solvent,
    )


def build_realnvp_flow(
    system,
    n_blocks=8,
    n_solvent=32,
    dim=2, 
    n_nodes=100,
    n_layers=3,
    prior_sigma=1.0,
):
    """
    Build an untrained RealNVP for the 2-D solute-LJ-bath system.

    Parameters
    ----------
    system      : system object with .get_energy() and .sigma
    n_blocks    : number of A→B + B→A coupling block pairs (total masks = 2*n_blocks)
    n_solvent   : number of solvent particles to model (default 33)
    dim         : dimensionality of the particle positions (default 2)
    n_nodes     : hidden layer width in each s/t network
    n_layers    : total depth of each s/t network (including input + output layers)
    prior_sigma : std-dev of the isotropic Gaussian prior
    """
    dimension = n_solvent * dim
    s_net = lambda: nn.Sequential(
        nn.Linear(dimension, n_nodes),
        *[l for _ in range(n_layers - 2)
          for l in (nn.ReLU(), nn.Linear(n_nodes, n_nodes))],
        nn.ReLU(), nn.Linear(n_nodes, dimension), nn.Tanh(),
    )
    t_net = lambda: nn.Sequential(
        nn.Linear(dimension, n_nodes),
        *[l for _ in range(n_layers - 2)
          for l in (nn.ReLU(), nn.Linear(n_nodes, n_nodes))],
        nn.ReLU(), nn.Linear(n_nodes, dimension),
    )
    affine = np.concatenate((np.ones(dimension // 2), np.zeros(dimension // 2)))
    affine = np.array([affine, np.flip(affine)] * n_blocks)
    mask   = torch.from_numpy(affine.astype(np.float32))
    prior  = distributions.MultivariateNormal(
        torch.zeros(dimension),
        torch.eye(dimension) * prior_sigma,
    )
    return RealNVP(s_net, t_net, mask, prior, system, (n_solvent, dim))


def build_equivariant_flow(
    system,
    n_solvent=28,
    n_blocks=8,
    hidden_dim=128,
    num_bins=8,
    tail_bound=6.0,
    n_layers=3,
    prior_sigma=1.0,
):
    """
    Build an EquivariantPolarFlow for the 2-D solute-LJ-bath system.

    Models only the n_solvent solvent particles; the solute is fixed at the
    origin and prepended in loss_KL before energy evaluation.

    Architecture
    ------------
    Each of the 2*n_blocks coupling layers transforms half the particles using
    an equivariant polar coupling:
      - Radial transform: rational-quadratic spline shared across all active
        particles, parameterised by a DeepSets context from the frozen half.
      - Angular transform: constant shift from the same context.

    Symmetry
    --------
    SO(2) equivariant, S_{N_solvent} equivariant.
    No Hungarian pre-processing of training data is required.

    Parameters
    ----------
    system     : SoluteSimulation2D
    n_solvent  : int     number of solvent particles to model
    n_blocks   : int     number of A→B / B→A coupling block pairs
    hidden_dim : int     width of DeepSets encoder and parameter MLPs
    num_bins   : int     rational-quadratic spline bins for radial transform
    tail_bound : float   spline domain (linear tails outside ±tail_bound)
    n_layers   : int     depth of all internal MLPs
    prior_sigma: float   std-dev of the isotropic Gaussian prior
    """
    all_idx = torch.arange(n_solvent)
    group_A = all_idx[all_idx % 2 == 0]
    group_B = all_idx[all_idx % 2 == 1]

    layers = []
    for _ in range(n_blocks):
        layers.append(EquivariantPolarCouplingLayer(
            mask_A=group_A, mask_B=group_B,
            hidden_dim=hidden_dim, num_bins=num_bins,
            tail_bound=tail_bound, n_layers=n_layers,
        ))
        layers.append(EquivariantPolarCouplingLayer(
            mask_A=group_B, mask_B=group_A,
            hidden_dim=hidden_dim, num_bins=num_bins,
            tail_bound=tail_bound, n_layers=n_layers,
        ))

    prior = EquivGaussianPrior(dim=n_solvent * 2, sigma=prior_sigma)
    return EquivariantPolarFlow(
        layers=layers, prior=prior, system=system, n_particles=n_solvent,
    )

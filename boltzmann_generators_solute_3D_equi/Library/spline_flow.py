"""
CartesianSplineFlow: RQ-spline coupling flow on raw Cartesian coordinates.

More expressive than RealNVP (affine coupling) while working directly on
[-L/2, L/2]^3 Cartesian coordinates as produced by the MC sampler.

Key differences from RealNVP:
  - Coupling:  rational-quadratic spline instead of affine (s, t)
  - Masking:   particle-group alternating (freeze 19, update 19 particles)
               instead of checkerboard on the flat vector
  - Interface: same loss_ML / loss_KL / generator / inverse_generator API
               → drop-in replacement for RealNVP in training.py
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Make nflows importable (it lives in the fab-torch repo)
_REPO = Path(__file__).resolve().parents[3]   # …/fab-torch/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


# ---------------------------------------------------------------------------
# Conditioner MLP
# ---------------------------------------------------------------------------

class SplineConditioner(nn.Module):
    """MLP mapping frozen particle coordinates to spline parameters.

    Parameters
    ----------
    n_frozen_coords : int
        Number of frozen coordinates fed as input (e.g. 19 * 3 = 57).
    n_active_coords : int
        Number of active coordinates to parameterise (e.g. 19 * 3 = 57).
    num_bins : int
        Number of spline bins per coordinate.
    n_nodes : int
        Width of hidden layers.
    n_hidden : int
        Number of hidden layers.
    """

    def __init__(self, n_frozen_coords: int, n_active_coords: int,
                 num_bins: int, n_nodes: int, n_hidden: int):
        super().__init__()
        params_per_coord = 3 * num_bins - 1
        layers = []
        in_dim = n_frozen_coords
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_nodes), nn.ReLU()]
            in_dim = n_nodes
        final = nn.Linear(in_dim, n_active_coords * params_per_coord)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.n_active_coords = n_active_coords
        self.params_per_coord = params_per_coord
        self.num_bins = num_bins

    def forward(self, x_frozen: torch.Tensor):
        # x_frozen: (B, n_frozen_coords)
        # returns:  (B, n_active_coords, params_per_coord)
        B = x_frozen.shape[0]
        out = self.net(x_frozen)                          # (B, n_active * ppc)
        return out.view(B, self.n_active_coords, self.params_per_coord)


# ---------------------------------------------------------------------------
# One coupling layer
# ---------------------------------------------------------------------------

class CartesianSplineCoupling(nn.Module):
    """
    One RQ-spline coupling layer for a system of N particles in 3D.

    Frozen group: particles at indices `frozen_idx` (shape (N_F,))
    Active group: particles at indices `active_idx` (shape (N_A,))

    The 3 coordinates of each active particle are transformed by an
    independent RQ spline whose parameters come from an MLP conditioned
    on the flat coordinates of the frozen group.

    Parameters
    ----------
    conditioner : SplineConditioner
    frozen_idx, active_idx : 1-D integer tensors
    num_bins : int
    tail_bound : float
        Spline covers (-tail_bound, tail_bound); linear extrapolation outside.
        Set slightly above the data range.  MC data lives in [-1.5, 1.5] nm,
        so tail_bound=2.0 is appropriate.
    """

    def __init__(self, conditioner: SplineConditioner,
                 frozen_idx: torch.Tensor, active_idx: torch.Tensor,
                 num_bins: int, tail_bound: float = 2.0,
                 min_bin_width: float = 1e-3, min_bin_height: float = 1e-3,
                 min_derivative: float = 1e-3):
        super().__init__()
        self.conditioner = conditioner
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    def _get_params(self, x: torch.Tensor):
        # x: (B, N, 3)
        x_frozen_flat = x[:, self.frozen_idx, :].reshape(x.shape[0], -1)
        params = self.conditioner(x_frozen_flat)    # (B, N_A*3, ppc)
        B, N_A3, ppc = params.shape
        params = params.view(B, -1, 3, ppc)         # (B, N_A, 3, ppc)
        widths      = params[..., :self.num_bins]
        heights     = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]
        return widths, heights, derivatives

    def forward(self, x: torch.Tensor):
        """x: (B, N, 3) → y: (B, N, 3),  logdet: (B,)"""
        widths, heights, derivatives = self._get_params(x)
        x_active = x[:, self.active_idx, :]         # (B, N_A, 3)

        y_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = x.clone()
        y[:, self.active_idx, :] = y_active
        return y, logabsdet.sum(dim=(-1, -2))       # sum over N_A particles × 3 dims

    def inverse(self, y: torch.Tensor):
        """y: (B, N, 3) → x: (B, N, 3),  logdet: (B,)"""
        widths, heights, derivatives = self._get_params(y)
        y_active = y[:, self.active_idx, :]

        x_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        x = y.clone()
        x[:, self.active_idx, :] = x_active
        return x, logabsdet.sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Gaussian prior (for solute system with soft walls)
# ---------------------------------------------------------------------------

class GaussianPrior:
    """Isotropic Gaussian prior N(0, sigma^2 I).

    Used by the solute spline flow where coordinates are not uniform across
    the box but concentrated near the origin (soft-wall system).
    """

    def __init__(self, dim: int, sigma: float = 1.0):
        self.dim   = int(dim)
        self.sigma = float(sigma)
        import math
        self._log_norm = -0.5 * self.dim * (math.log(2 * math.pi) + 2 * math.log(self.sigma))

    def sample(self, shape, device=None, dtype=torch.float32):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.randn(*shape, self.dim, device=device, dtype=dtype) * self.sigma

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z / self.sigma).pow(2).sum(-1) + self._log_norm


class SphericalPrior:
    """Prior for the equivariant flow: log-normal radii, uniform directions.

    For each particle i, samples r_i = exp(log_r_i) where
    log_r_i ~ N(mu_r, sigma_r²), and u_i ~ Uniform(S²) independently.
    The resulting Cartesian position is x_i = r_i * u_i.

    The Cartesian density has a mode at r_mode = exp(mu_r - 3*sigma_r²).
    Setting mu_r = log(sigma_LJ) + 3*sigma_r² places the mode at the LJ
    contact distance, which aligns the prior with the MC distribution.

    For PBC systems with l_box=2.0, use sigma_r=0.40 so that the 95%
    sample range [exp(mu_r - 2*sigma_r), exp(mu_r + 2*sigma_r)] covers
    the full Cartesian distance range [sigma_LJ, sqrt(3)*l_box] ≈ [1.1, 3.46].

    Parameters
    ----------
    n_particles : int   number of particles (36 for the solute system).
    sigma_r     : float std of log(r_i).  Default 0.40 for PBC l_box=2.0.
    mu_r        : float mean of log(r_i).  Set via
                        mu_r = log(sigma_LJ) + 3*sigma_r**2 in the factory.
                        Default 0.0 (backwards compatible).
    """

    def __init__(self, n_particles: int, sigma_r: float = 1.0,
                 mu_r: float = 0.0):
        self.n_particles = int(n_particles)
        self.dim         = self.n_particles * 3
        self.sigma_r     = float(sigma_r)
        self.mu_r        = float(mu_r)
        # Normalisation constant for log p(x_i):
        # log p(x_i) = -0.5*((log_r - mu_r)/sigma_r)^2 - 3*log_r - _log_norm_particle
        # mu_r shifts the distribution but does not change the normalisation.
        self._log_norm_particle = (math.log(self.sigma_r)
                                   + 0.5 * math.log(2.0 * math.pi)
                                   + math.log(4.0 * math.pi))

    def sample(self, shape, device=None, dtype=torch.float32):
        if isinstance(shape, int):
            shape = (shape,)
        n = shape[0]
        # Radial: log_r ~ N(mu_r, sigma_r^2), so r = exp(log_r) > 0 always.
        log_r = (torch.randn(n, self.n_particles, 1, device=device, dtype=dtype)
                 * self.sigma_r + self.mu_r)                        # (n, N, 1)
        r = torch.exp(log_r)                                       # (n, N, 1)
        # Angular: u ~ Uniform(S^2) via normalised i.i.d. N(0,1) vectors.
        gauss = torch.randn(n, self.n_particles, 3, device=device, dtype=dtype)
        u = gauss / gauss.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (n, N, 3)
        return (r * u).reshape(n, -1)                              # (n, N*3)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Log-density of z under the spherical prior in Cartesian coordinates.

        Derivation (change of variables (log_r, u) → x = exp(log_r)*u):
          |det J_{(log_r,u)→x}| = exp(3*log_r) = r^3
          log p(x_i) = log p(log_r_i) + log p(u_i) - 3*log_r_i
                     = -0.5*((log_r - mu_r)/sigma_r)^2 - 3*log_r - _log_norm_particle

        z : (B, N*3)
        Returns (B,)
        """
        B = z.shape[0]
        x = z.reshape(B, self.n_particles, 3)          # (B, N, 3)
        log_r = torch.log(x.norm(dim=-1).clamp(min=1e-8))  # (B, N)
        log_p = (-0.5 * ((log_r - self.mu_r) / self.sigma_r).pow(2)
                 - 3.0 * log_r
                 - self._log_norm_particle)             # (B, N)
        return log_p.sum(dim=-1)                        # (B,)


# ---------------------------------------------------------------------------
# Uniform prior on [-bound, bound]^dim
# ---------------------------------------------------------------------------

class BoxUniformPrior:
    """Uniform distribution on [-bound, bound]^dim.

    More appropriate than a Gaussian for soft-wall MC data where particles
    are distributed roughly uniformly across the box rather than concentrated
    at the origin.

    Not a `torch.distributions` object; provides only the methods needed:
      - prior.sample(shape) → Tensor  (uniform in [-bound, bound]^dim)
      - prior.log_prob(z)   → Tensor  (constant: -dim * log(2*bound))
    """

    def __init__(self, dim: int, bound: float):
        self.dim = int(dim)
        self.bound = float(bound)
        self._log_prob_val = -self.dim * math.log(2.0 * self.bound)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.rand(*shape, self.dim) * (2 * self.bound) - self.bound

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.full((z.shape[0],), self._log_prob_val,
                          device=z.device, dtype=z.dtype)


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------

class CartesianSplineFlow(nn.Module):
    """
    Stack of CartesianSplineCoupling layers with a BoxUniform prior.

    Drop-in replacement for RealNVP: provides loss_ML / loss_KL /
    generator / inverse_generator with the same call signatures.

    Parameters
    ----------
    layers : list[CartesianSplineCoupling]
    prior  : BoxUniformPrior  (uniform on [-tail_bound, tail_bound]^dim)
    system : DimerSimulation3D  (needed for loss_KL energy computation)
    n_particles : int  (38 for the dimer-in-bath system)
    """

    def __init__(self, layers, prior, system, n_particles: int,
                 fixed_solute: bool = False):
        super().__init__()
        self.coupling_layers = nn.ModuleList(layers)
        self.prior = prior
        self.system = system
        self.n_particles = int(n_particles)
        self.fixed_solute = bool(fixed_solute)
        # sys_dim used by overlap_penalty: (n_particles, 3) for solvent-only flow,
        # or (n_particles+1, 3) when a fixed solute is prepended for energy evaluation.
        self.sys_dim = (n_particles + 1, 3) if fixed_solute else (n_particles, 3)

    # ------------------------------------------------------------------
    # Core transforms  (particle view: (B, N, 3))
    # ------------------------------------------------------------------

    def forward_map(self, x: torch.Tensor):
        """x (B, N, 3)  →  z (B, N, 3),  logdet (B,)"""
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.coupling_layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse_map(self, z: torch.Tensor):
        """z (B, N, 3)  →  x (B, N, 3),  logdet (B,)"""
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.coupling_layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    # ------------------------------------------------------------------
    # Flat helpers (interface expected by training.py)
    # ------------------------------------------------------------------

    def _to_particle(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.n_particles, 3)

    def _to_flat(self, particle: torch.Tensor) -> torch.Tensor:
        return particle.reshape(particle.shape[0], -1)

    def _add_fixed_solute(self, solvent_flat: torch.Tensor) -> torch.Tensor:
        """Prepend a fixed solute at (0,0,0) → (B, (N+1)*3) for energy evaluation."""
        B = solvent_flat.shape[0]
        solute = torch.zeros(B, 3, device=solvent_flat.device, dtype=solvent_flat.dtype)
        return torch.cat([solute, solvent_flat], dim=1)

    # ------------------------------------------------------------------
    # Loss functions  (same signatures as RealNVP)
    # ------------------------------------------------------------------

    def loss_ML(self, batch_x, weighted=False):
        """Forward KL  –E_data[log q(x)].

        batch_x : (B, N*3) flat Cartesian nm.
        """
        x = self._to_particle(batch_x)
        z, logdet_fwd = self.forward_map(x)
        log_pz = self.prior.log_prob(self._to_flat(z))
        return -(log_pz + logdet_fwd).mean()

    def loss_KL(self, batch_z, weighted=False, energy_cap=None, w_overlap=0.0):
        """Reverse KL  E_z[U(G(z)) – log|det J_zx|].

        batch_z : (B, N*3) samples from prior.
        energy_cap : float or None — soft-cap to prevent explosion.
        w_overlap : float — weight for the overlap penalty (solute system).
        """
        z = self._to_particle(batch_z)
        x, logdet_inv = self.inverse_map(z)
        x_flat = self._to_flat(x)
        # For the solute system, prepend fixed solute before computing energy
        x_for_energy = self._add_fixed_solute(x_flat) if self.fixed_solute else x_flat
        u_x = self.system.get_energy_batch(x_for_energy)
        if energy_cap is not None:
            u_x = torch.where(
                u_x < energy_cap,
                u_x,
                energy_cap + torch.log1p(u_x - energy_cap),
            )
        loss = (u_x - logdet_inv).mean()
        if w_overlap > 0.0:
            loss = loss + w_overlap * overlap_penalty(
                x_for_energy, self.sys_dim, self.system.sigma
            )
        return loss

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def generator(self, z):
        """z (B, N*3 flat) → x (B, N*3 flat),  logdet (B,)."""
        x_particle, logdet_inv = self.inverse_map(self._to_particle(z))
        return self._to_flat(x_particle), logdet_inv

    def inverse_generator(self, x):
        """x (B, N*3 flat) → z (B, N*3 flat),  logdet (B,)."""
        z_particle, logdet_fwd = self.forward_map(self._to_particle(x))
        return self._to_flat(z_particle), logdet_fwd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cartesian_spline_flow(
    system,
    n_particles: int = 38,
    n_blocks: int = 8,
    n_nodes: int = 256,
    n_hidden: int = 3,
    num_bins: int = 8,
    tail_bound: float = 2.0,
) -> CartesianSplineFlow:
    """
    Build a CartesianSplineFlow for the dimer-in-bath system.

    Prior: BoxUniform on [-tail_bound, tail_bound]^dim.  This matches the
    roughly uniform distribution of soft-wall MC data (PERIODIC=False) better
    than a Gaussian, which concentrates mass at the origin.

    Parameters
    ----------
    system : DimerSimulation3D
    n_particles : int
        Total particles (38 for the standard dimer-in-LJ-bath).
    n_blocks : int
        Number of A→B / B→A cycles; total coupling layers = n_blocks * 2.
    n_nodes, n_hidden : int
        MLP conditioner size.
    num_bins : int
        RQ spline bins per coordinate.
    tail_bound : float
        Spline covers (–tail_bound, tail_bound) and the prior samples from
        [–tail_bound, tail_bound]^dim.  Use 2.0 for data in [–1.5, 1.5] nm.
    """
    # Particle groups
    all_idx = torch.arange(n_particles)
    group_A = all_idx[all_idx % 2 == 0]   # even: 0,2,4,…,36  (19 particles)
    group_B = all_idx[all_idx % 2 == 1]   # odd:  1,3,5,…,37  (19 particles)

    n_A = len(group_A)
    n_B = len(group_B)
    dim = n_particles * 3

    layers = []
    for block in range(n_blocks):
        # Layer 1 of block: freeze B, update A
        cond_AB = SplineConditioner(
            n_frozen_coords=n_B * 3,
            n_active_coords=n_A * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(CartesianSplineCoupling(
            conditioner=cond_AB,
            frozen_idx=group_B,
            active_idx=group_A,
            num_bins=num_bins,
            tail_bound=tail_bound,
        ))

        # Layer 2 of block: freeze A, update B
        cond_BA = SplineConditioner(
            n_frozen_coords=n_A * 3,
            n_active_coords=n_B * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(CartesianSplineCoupling(
            conditioner=cond_BA,
            frozen_idx=group_A,
            active_idx=group_B,
            num_bins=num_bins,
            tail_bound=tail_bound,
        ))

    prior = BoxUniformPrior(dim, bound=tail_bound)

    return CartesianSplineFlow(
        layers=layers,
        prior=prior,
        system=system,
        n_particles=n_particles,
    )


def build_solute_spline_flow_3d(
    system,
    n_particles: int = 36,
    n_blocks: int = 8,
    n_nodes: int = 256,
    n_hidden: int = 3,
    num_bins: int = 8,
    tail_bound: float = 7.0,
) -> CartesianSplineFlow:
    """
    Build a CartesianSplineFlow for the 3D solute-in-LJ-bath system.

    Models the 36 **solvent** particles only (108D = 36×3).
    The solute (particle 0) is fixed at the origin and prepended inside
    ``loss_KL`` before calling ``system.get_energy_batch``.

    Prior: GaussianPrior(dim=108) — appropriate for soft-wall data where
    coordinates are concentrated near the origin rather than uniformly spread.

    Parameters
    ----------
    system : SoluteSimulation3D
    n_particles : int
        Number of SOLVENT particles (default 36); dim = n_particles * 3.
    n_blocks : int
        Number of A→B / B→A coupling cycles; total layers = n_blocks * 2.
    n_nodes, n_hidden : int
        MLP conditioner width and depth.
    num_bins : int
        RQ spline bins per coordinate.
    tail_bound : float
        Spline linear-tail boundary; set larger than the typical coordinate range.
    """
    dim = n_particles * 3

    all_idx = torch.arange(n_particles)
    group_A = all_idx[all_idx % 2 == 0]   # 18 even-indexed particles
    group_B = all_idx[all_idx % 2 == 1]   # 18 odd-indexed particles
    n_A, n_B = len(group_A), len(group_B)

    layers = []
    for _ in range(n_blocks):
        # Layer 1: freeze B, update A
        cond_AB = SplineConditioner(
            n_frozen_coords=n_B * 3,
            n_active_coords=n_A * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(CartesianSplineCoupling(
            conditioner=cond_AB,
            frozen_idx=group_B,
            active_idx=group_A,
            num_bins=num_bins,
            tail_bound=tail_bound,
        ))

        # Layer 2: freeze A, update B
        cond_BA = SplineConditioner(
            n_frozen_coords=n_A * 3,
            n_active_coords=n_B * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(CartesianSplineCoupling(
            conditioner=cond_BA,
            frozen_idx=group_A,
            active_idx=group_B,
            num_bins=num_bins,
            tail_bound=tail_bound,
        ))

    prior = GaussianPrior(dim=dim, sigma=1.0)

    return CartesianSplineFlow(
        layers=layers,
        prior=prior,
        system=system,
        n_particles=n_particles,
        fixed_solute=True,
    )

def overlap_penalty(x_flat, sys_dim, sigma):
    """
    Differentiable soft overlap penalty.

    Parameters
    ----------
    x_flat : (B, N*D) tensor
    sys_dim : tuple (N, D)
    sigma : float — minimum allowed distance

    Returns
    -------
    scalar — mean over batch of sum_{i<j} relu(sigma^2 - r_ij^2)^2
    """
    N, D = sys_dim
    B = x_flat.shape[0]
    coords = x_flat.reshape(B, N, D)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, D)
    r2 = (diff * diff).sum(-1)                          # (B, N, N)
    idx = torch.triu_indices(N, N, offset=1, device=x_flat.device)
    r2_pairs = r2[:, idx[0], idx[1]]                   # (B, N*(N-1)/2)
    return torch.relu(sigma ** 2 - r2_pairs).pow(2).sum(-1).mean()


# ===========================================================================
# Torus Spline Flow  (for PBC data)
# ===========================================================================
#
# Coordinate convention:
#   Cartesian  x  ∈ [-L/2, L/2]^3   (MC sampler output with PERIODIC=True)
#   Unit torus u  ∈ [0, 1)^3         u = (x + L/2) / L
#
# The coupling layers operate on u.  The seam of the torus (u=0/1) sits at
# the box boundary x = ±L/2, where particle density is lowest — so the
# discontinuity in the representation affects the distribution minimally.
#
# Prior: Uniform on [0,1)^dim  → log_prob = 0 everywhere.
#
# Conditioner input: sin/cos embedding of frozen torus coords so the MLP
# sees no discontinuity at the 0/1 boundary.
# ===========================================================================


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cart_to_torus(x_flat: torch.Tensor, L: float) -> torch.Tensor:
    """Cartesian [-L/2, L/2]^d → unit torus [0, 1)^d."""
    return (x_flat + L / 2.0) / L


def torus_to_cart(u_flat: torch.Tensor, L: float) -> torch.Tensor:
    """Unit torus [0, 1)^d → Cartesian [-L/2, L/2]^d."""
    return u_flat * L - L / 2.0


# ---------------------------------------------------------------------------
# Uniform prior on [0,1)^dim
# ---------------------------------------------------------------------------

class UniformTorusPrior:
    """Uniform distribution on [0, 1)^dim — natural prior for a torus flow.

    Not a `torch.distributions` object; provides only the methods that
    `BoltzmannGenerator3D.train()` and the loss functions need:
      - prior.sample(shape) → Tensor
      - prior.log_prob(z)   → Tensor  (all zeros)
    """

    def __init__(self, dim: int):
        self.dim = int(dim)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.rand(*shape, self.dim)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)


# ---------------------------------------------------------------------------
# Conditioner with sin/cos input (torus-aware MLP)
# ---------------------------------------------------------------------------

class TorusSplineConditioner(nn.Module):
    """MLP mapping sin/cos-embedded frozen torus coords to spline parameters.

    Input dim  : n_frozen_coords * 2  (sin and cos of 2π·u for each coord)
    Output dim : n_active_coords × (3·num_bins − 1)
    """

    def __init__(self, n_frozen_coords: int, n_active_coords: int,
                 num_bins: int, n_nodes: int, n_hidden: int):
        super().__init__()
        params_per_coord = 3 * num_bins - 1
        in_dim = n_frozen_coords * 2   # sin + cos embedding
        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_nodes), nn.ReLU()]
            in_dim = n_nodes
        final = nn.Linear(in_dim, n_active_coords * params_per_coord)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.n_active_coords = n_active_coords
        self.params_per_coord = params_per_coord
        self.num_bins = num_bins

    def forward(self, u_frozen: torch.Tensor):
        # u_frozen: (B, n_frozen_coords) in [0, 1)
        # embed as sin/cos to respect torus topology
        emb = torch.cat([
            torch.sin(2.0 * math.pi * u_frozen),
            torch.cos(2.0 * math.pi * u_frozen),
        ], dim=-1)                                    # (B, n_frozen*2)
        B = u_frozen.shape[0]
        out = self.net(emb)                           # (B, n_active * ppc)
        return out.view(B, self.n_active_coords, self.params_per_coord)


# ---------------------------------------------------------------------------
# One torus coupling layer
# ---------------------------------------------------------------------------

class TorusSplineCoupling(nn.Module):
    """RQ-spline coupling layer on unit-torus coordinates [0, 1)^3.

    For each active particle:
      1. Centre:   y = u − 0.5  so y ∈ [−0.5, 0.5)
      2. Spline:   y → y′  (RQ spline, tail_bound = 0.5)
      3. Uncentre: u′ = y′ + 0.5  (wraps naturally back to [0,1))

    The conditioner receives sin/cos(2πu) of the frozen group so it is
    continuous across the torus boundary at u = 0/1.
    """

    def __init__(self, conditioner: TorusSplineConditioner,
                 frozen_idx: torch.Tensor, active_idx: torch.Tensor,
                 num_bins: int,
                 min_bin_width: float = 1e-3,
                 min_bin_height: float = 1e-3,
                 min_derivative: float = 1e-3):
        super().__init__()
        self.conditioner = conditioner
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        # Torus spline always uses tail_bound = 0.5 (half-unit interval)
        self._tail_bound = 0.5

    def _get_params(self, u: torch.Tensor):
        """u: (B, N, 3) → widths, heights, derivatives for active particles."""
        u_frozen_flat = u[:, self.frozen_idx, :].reshape(u.shape[0], -1)
        params = self.conditioner(u_frozen_flat)    # (B, N_A*3, ppc)
        B, N_A3, ppc = params.shape
        params = params.view(B, -1, 3, ppc)         # (B, N_A, 3, ppc)
        widths      = params[..., :self.num_bins]
        heights     = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]
        return widths, heights, derivatives

    def forward(self, u: torch.Tensor):
        """u: (B, N, 3) in [0,1) → v: (B, N, 3) in [0,1),  logdet: (B,)"""
        widths, heights, derivatives = self._get_params(u)
        u_active = u[:, self.active_idx, :]         # (B, N_A, 3)

        # Centre to [-0.5, 0.5) for the spline
        y_active = u_active - 0.5

        y_out, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self._tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        # Uncentre back to [0,1)
        v = u.clone()
        v[:, self.active_idx, :] = y_out + 0.5
        return v, logabsdet.sum(dim=(-1, -2))

    def inverse(self, v: torch.Tensor):
        """v: (B, N, 3) in [0,1) → u: (B, N, 3) in [0,1),  logdet: (B,)"""
        widths, heights, derivatives = self._get_params(v)
        v_active = v[:, self.active_idx, :]

        y_active = v_active - 0.5

        y_out, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=self._tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        u = v.clone()
        u[:, self.active_idx, :] = y_out + 0.5
        return u, logabsdet.sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Full torus flow
# ---------------------------------------------------------------------------

class TorusSplineFlow(nn.Module):
    """
    Stack of TorusSplineCoupling layers with a Uniform prior on [0,1)^dim.

    Handles the Cartesian ↔ torus coordinate conversion internally so the
    external interface is identical to CartesianSplineFlow:
      loss_ML(batch_x)  — batch_x in flat Cartesian nm
      loss_KL(batch_z)  — batch_z sampled from prior.sample()
      generator(z)      — z → x flat Cartesian
      inverse_generator(x) — x flat Cartesian → z

    Parameters
    ----------
    layers : list[TorusSplineCoupling]
    prior  : UniformTorusPrior
    system : DimerSimulation3D
    n_particles : int
    L : float — box side length (Cartesian range is [-L/2, L/2])
    """

    def __init__(self, layers, prior: UniformTorusPrior, system,
                 n_particles: int, L: float):
        super().__init__()
        self.coupling_layers = nn.ModuleList(layers)
        self.prior = prior
        self.system = system
        self.n_particles = int(n_particles)
        self.L = float(L)

    # ------------------------------------------------------------------
    # Core torus-space transforms  (B, N, 3) in [0,1)
    # ------------------------------------------------------------------

    def forward_map(self, u: torch.Tensor):
        """u (B,N,3) [0,1) → z (B,N,3) [0,1),  logdet (B,)"""
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for layer in self.coupling_layers:
            u, ld = layer(u)
            logdet = logdet + ld
        return u, logdet

    def inverse_map(self, z: torch.Tensor):
        """z (B,N,3) [0,1) → u (B,N,3) [0,1),  logdet (B,)"""
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.coupling_layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    # ------------------------------------------------------------------
    # Flat helpers
    # ------------------------------------------------------------------

    def _to_particle(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.n_particles, 3)

    def _to_flat(self, particle: torch.Tensor) -> torch.Tensor:
        return particle.reshape(particle.shape[0], -1)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def loss_ML(self, batch_x, weighted=False):
        """Forward KL  –E_data[log q(x)].

        batch_x : (B, N*3) flat Cartesian nm  (periodic, in [-L/2, L/2]).
        Converts to torus internally before passing through the flow.
        """
        if not isinstance(batch_x, torch.Tensor):
            batch_x = torch.as_tensor(batch_x, dtype=torch.float32)
        u_flat = cart_to_torus(batch_x, self.L)          # → [0,1)^dim
        u = self._to_particle(u_flat)
        z, logdet_fwd = self.forward_map(u)
        # prior.log_prob = 0 everywhere, so loss = -logdet_fwd
        return -logdet_fwd.mean()

    def loss_KL(self, batch_z, weighted=False, energy_cap=None):
        """Reverse KL  E_z[U(G(z)) – log|det J|].

        batch_z : (B, N*3) uniform samples from prior (in [0,1)).
        """
        if not isinstance(batch_z, torch.Tensor):
            batch_z = torch.as_tensor(batch_z, dtype=torch.float32)
        z = self._to_particle(batch_z)
        u, logdet_inv = self.inverse_map(z)
        x_flat = torus_to_cart(self._to_flat(u), self.L)  # → Cartesian nm
        u_x = self.system.get_energy_batch(x_flat)
        if energy_cap is not None:
            u_x = torch.where(
                u_x < energy_cap,
                u_x,
                energy_cap + torch.log1p(u_x - energy_cap),
            )
        return (u_x - logdet_inv).mean()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def generator(self, z):
        """z (B, N*3) uniform [0,1) → x (B, N*3) flat Cartesian,  logdet (B,)."""
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, dtype=torch.float32)
        u, logdet_inv = self.inverse_map(self._to_particle(z))
        x_flat = torus_to_cart(self._to_flat(u), self.L)
        return x_flat, logdet_inv

    def inverse_generator(self, x):
        """x (B, N*3) flat Cartesian → z (B, N*3) uniform [0,1),  logdet (B,)."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        u_flat = cart_to_torus(x, self.L)
        z, logdet_fwd = self.forward_map(self._to_particle(u_flat))
        return self._to_flat(z), logdet_fwd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_torus_spline_flow(
    system,
    n_particles: int = 38,
    n_blocks: int = 8,
    n_nodes: int = 256,
    n_hidden: int = 3,
    num_bins: int = 8,
) -> TorusSplineFlow:
    """
    Build a TorusSplineFlow for the dimer-in-bath system with PBC.

    Coordinate convention: Cartesian x ∈ [−L/2, L/2]^3, where L = system.l_box.
    The flow maps internally to unit-torus u ∈ [0,1)^3 via u = (x + L/2)/L.

    Parameters
    ----------
    system : DimerSimulation3D  (must have .periodic=True and .l_box set)
    n_particles : int
    n_blocks : int   — number of A→B / B→A cycles; total layers = n_blocks * 2
    n_nodes, n_hidden : int   — MLP conditioner size
    num_bins : int   — RQ spline bins (tail_bound is fixed at 0.5 for torus)
    """
    L = float(system.l_box)
    dim = n_particles * 3

    all_idx = torch.arange(n_particles)
    group_A = all_idx[all_idx % 2 == 0]   # even: 0,2,…,36  (19 particles)
    group_B = all_idx[all_idx % 2 == 1]   # odd:  1,3,…,37  (19 particles)
    n_A, n_B = len(group_A), len(group_B)

    layers = []
    for _ in range(n_blocks):
        # Layer 1: freeze B, update A
        cond_AB = TorusSplineConditioner(
            n_frozen_coords=n_B * 3,
            n_active_coords=n_A * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(TorusSplineCoupling(
            conditioner=cond_AB,
            frozen_idx=group_B,
            active_idx=group_A,
            num_bins=num_bins,
        ))

        # Layer 2: freeze A, update B
        cond_BA = TorusSplineConditioner(
            n_frozen_coords=n_A * 3,
            n_active_coords=n_B * 3,
            num_bins=num_bins,
            n_nodes=n_nodes,
            n_hidden=n_hidden,
        )
        layers.append(TorusSplineCoupling(
            conditioner=cond_BA,
            frozen_idx=group_A,
            active_idx=group_B,
            num_bins=num_bins,
        ))

    prior = UniformTorusPrior(dim)

    return TorusSplineFlow(
        layers=layers,
        prior=prior,
        system=system,
        n_particles=n_particles,
        L=L,
    )

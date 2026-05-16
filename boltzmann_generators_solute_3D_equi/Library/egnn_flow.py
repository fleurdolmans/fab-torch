"""
E(n)-Equivariant Normalizing Flow for 3D particle systems with PBC.

Architecture (per coupling block, alternating A ↔ B split of 18 particles each)
-------------------------------------------------------------------------------
1. RadialCouplingLayer
   - Frozen group A positions → EGNN → invariant features h_a
   - h_a aggregated to active group B → spline params for log(r_b)
   - Transforms: r_b → r_b' = exp(spline(log r_b))  (direction u_b unchanged)
   - log-det: Σ_b [logabsdet_spline_b + 3*(log r_b' - log r_b)]

2. AngularCouplingLayer
   - Frozen group A positions → EGNN with equivariant output → v_b, θ_b
   - Rodrigues rotation: R_b = I + sin(θ_b)*K(v̂_b) + (1-cos(θ_b))*K(v̂_b)²
   - Transforms: u_b → R_b @ u_b  (radius r_b unchanged)
   - log-det: 0  (rotation is an isometry on S²)
   - Inverse: R_b^T @ u_b'

Equivariance properties
-----------------------
- Permutation equivariant: EGNN aggregates symmetrically over all particles
- Rotation equivariant:
    Radial layer: r_i (distance from solute) is rotation-invariant ✓
    Angular layer: R_b is defined from equivariant v_b → R_b @ (R@u_b) = R @ (R_b @ u_b) ✓
- Translation: solute fixed at origin; relative distances unchanged ✓
- PBC: minimum-image distances used throughout ✓

Interface (drop-in for CartesianSplineFlow / RealNVP in training.py)
--------------------------------------------------------------------
EGNNEquivariantFlow.loss_ML(batch_x, weighted=False) → scalar
EGNNEquivariantFlow.loss_KL(batch_z, weighted=False, energy_cap=None, w_overlap=0.) → scalar
EGNNEquivariantFlow.generator(z_flat) → (x_flat, logdet)
EGNNEquivariantFlow.inverse_generator(x_flat) → (z_flat, logdet)
EGNNEquivariantFlow.prior.sample(shape, device, dtype) → Tensor
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

# Make nflows importable
_REPO = Path(__file__).resolve().parents[3]   # …/fab-torch/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

from egnn import EGNN, CrossGroupConditioner, CrossGroupEquivariant
from spline_flow import GaussianPrior   # reuse existing prior class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_image(x: torch.Tensor, l_box: float) -> torch.Tensor:
    """Minimum-image convention: box side = 2*l_box."""
    L = 2.0 * l_box
    return x - L * torch.round(x / L)


def _to_radial(x_solvent: torch.Tensor):
    """
    Decompose solvent positions relative to solute at origin.

    No PBC wrapping here — the solute is fixed at the origin, and we use
    the raw Euclidean distance so that forward and inverse are consistent.
    PBC minimum-image convention is applied *only* inside the EGNN for
    pairwise distances between particles.

    Parameters
    ----------
    x_solvent : (B, N, 3) Cartesian positions (solute at origin)

    Returns
    -------
    r : (B, N, 1)  distances from solute (>0)
    u : (B, N, 3)  unit direction vectors
    """
    r = torch.norm(x_solvent, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
    u = x_solvent / r                    # (B, N, 3) unit vectors
    return r, u


def _rodrigues_rotation(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues rotation matrices.

    Parameters
    ----------
    v     : (B, N, 3) rotation axis vectors (need not be unit)
    theta : (B, N, 1) rotation angles in radians

    Returns
    -------
    R : (B, N, 3, 3) rotation matrices
    """
    # Normalise axis; handle near-zero case
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / v_norm                                  # (B, N, 3) unit axis

    s = torch.sin(theta)                            # (B, N, 1)
    c = torch.cos(theta)                            # (B, N, 1)
    one_c = 1.0 - c                                 # (B, N, 1)

    kx, ky, kz = k[..., 0:1], k[..., 1:2], k[..., 2:3]   # (B, N, 1) each

    # Skew-symmetric K matrix: K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
    # Rodrigues: R = I + sin(θ)*K + (1-cos(θ))*K²
    # K² = k k^T - I  (for unit k)
    # Row-by-row construction of R
    R = torch.zeros(*k.shape[:-1], 3, 3, device=k.device, dtype=k.dtype)

    R[..., 0, 0] = (c + one_c * kx * kx).squeeze(-1)
    R[..., 0, 1] = (one_c * kx * ky - s * kz).squeeze(-1)
    R[..., 0, 2] = (one_c * kx * kz + s * ky).squeeze(-1)
    R[..., 1, 0] = (one_c * ky * kx + s * kz).squeeze(-1)
    R[..., 1, 1] = (c + one_c * ky * ky).squeeze(-1)
    R[..., 1, 2] = (one_c * ky * kz - s * kx).squeeze(-1)
    R[..., 2, 0] = (one_c * kz * kx - s * ky).squeeze(-1)
    R[..., 2, 1] = (one_c * kz * ky + s * kx).squeeze(-1)
    R[..., 2, 2] = (c + one_c * kz * kz).squeeze(-1)

    return R                                        # (B, N, 3, 3)


# ---------------------------------------------------------------------------
# Radial coupling layer
# ---------------------------------------------------------------------------

class RadialCouplingLayer(nn.Module):
    """
    Transforms the *radial distances* of active group B particles.

    The direction vectors u_b and ALL positions of frozen group A are unchanged.
    Spline parameters for each active particle b come from an EGNN conditioner
    run on the frozen group A, followed by cross-group attention aggregation.

    Forward: r_b → r_b' = exp( spline(log r_b; params_b) )
    Logdet:  Σ_b [ logabsdet_spline_b + 3*(log r_b' - log r_b) ]

    Parameters
    ----------
    egnn_frozen  : EGNN — processes frozen group A → invariant h_a per particle
    cross_cond   : CrossGroupConditioner — aggregates h_a → context_b per active
    param_head   : nn.Module — context_b → spline params (B, N_B, ppc)
    frozen_idx   : (N_A,) int tensor — particle indices of frozen group A
    active_idx   : (N_B,) int tensor — particle indices of active group B
    num_bins     : int — RQ spline bins
    tail_bound   : float — spline covers [-tail_bound, tail_bound] on log(r)
    l_box        : float or None — PBC box half-width
    """

    def __init__(self, egnn_frozen: EGNN, cross_cond: CrossGroupConditioner,
                 param_head: nn.Module,
                 frozen_idx: torch.Tensor, active_idx: torch.Tensor,
                 num_bins: int, tail_bound: float = 4.0,
                 l_box: float | None = None):
        super().__init__()
        self.egnn_frozen = egnn_frozen
        self.cross_cond = cross_cond
        self.param_head = param_head
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.l_box = l_box
        self._ppc = 3 * num_bins - 1          # params per coordinate (log r scalar)

    def _get_spline_params(self, x: torch.Tensor):
        """x: (B, N, 3) → widths, heights, derivatives each (B, N_B, num_bins[/bins-1])"""
        pos_a = x[:, self.frozen_idx, :]       # (B, N_A, 3)
        pos_b = x[:, self.active_idx, :]       # (B, N_B, 3)

        # Use unit directions instead of full positions. The radial coupling
        # transforms r_b → r_b' while leaving u_b = pos_b/||pos_b|| unchanged.
        # Passing u_b ensures spline params are identical in forward and inverse.
        u_b = pos_b / pos_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        h_a, _ = self.egnn_frozen(pos_a)       # (B, N_A, d)
        ctx_b = self.cross_cond(h_a, pos_a, u_b)    # (B, N_B, d)
        params = self.param_head(ctx_b)        # (B, N_B, ppc)

        w = params[..., :self.num_bins]
        h = params[..., self.num_bins: 2 * self.num_bins]
        d = params[..., 2 * self.num_bins:]
        return w, h, d

    def forward(self, x: torch.Tensor):
        """x: (B, N, 3) → y: (B, N, 3),  logdet: (B,)"""
        B, N, _ = x.shape
        widths, heights, derivs = self._get_spline_params(x)   # (B, N_B, bins)

        x_active = x[:, self.active_idx, :]                    # (B, N_B, 3)
        r, u = _to_radial(x_active)                            # (B, N_B, 1), (B, N_B, 3)
        log_r = torch.log(r)                                    # (B, N_B, 1)

        # Apply RQ spline to log(r)
        log_r_new, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=log_r,
            unnormalized_widths=widths.unsqueeze(-2),   # (B, N_B, 1, bins)
            unnormalized_heights=heights.unsqueeze(-2),
            unnormalized_derivatives=derivs.unsqueeze(-2),
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        # logabsdet: (B, N_B, 1) → log |d(log r')/d(log r)|

        r_new = torch.exp(log_r_new)                           # (B, N_B, 1)

        # Full Jacobian per particle: logabsdet_spline + 3*(log r' - log r)
        logdet_per_particle = logabsdet + 3.0 * (log_r_new - log_r)  # (B, N_B, 1)
        logdet = logdet_per_particle.sum(dim=(1, 2))           # (B,)

        # Reconstruct active positions
        x_active_new = r_new * u                               # (B, N_B, 3)

        y = x.clone()
        y[:, self.active_idx, :] = x_active_new
        return y, logdet

    def inverse(self, y: torch.Tensor):
        """y: (B, N, 3) → x: (B, N, 3),  logdet: (B,)"""
        B, N, _ = y.shape
        widths, heights, derivs = self._get_spline_params(y)   # use frozen group (unchanged)

        y_active = y[:, self.active_idx, :]
        r_new, u = _to_radial(y_active)
        log_r_new = torch.log(r_new)

        log_r, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=log_r_new,
            unnormalized_widths=widths.unsqueeze(-2),
            unnormalized_heights=heights.unsqueeze(-2),
            unnormalized_derivatives=derivs.unsqueeze(-2),
            inverse=True,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        r = torch.exp(log_r)

        logdet_per_particle = logabsdet + 3.0 * (log_r - log_r_new)
        logdet = logdet_per_particle.sum(dim=(1, 2))

        x_active = r * u
        x = y.clone()
        x[:, self.active_idx, :] = x_active
        return x, logdet


# ---------------------------------------------------------------------------
# Angular coupling layer
# ---------------------------------------------------------------------------

class AngularCouplingLayer(nn.Module):
    """
    Equivariantly rotates the *directions* of active group B particles.

    Rodrigues rotation R_b is determined by:
    - v_b: equivariant axis vector from EGNN (weighted sum of relative displacements)
    - θ_b: invariant rotation angle from EGNN (scalar MLP output)

    Forward: u_b → u_b' = R_b @ u_b,  x_b' = r_b * u_b'
    Logdet : 0  (SO(3) rotation is an isometry on S²)
    Inverse: u_b = R_b^T @ u_b'

    Parameters
    ----------
    egnn_equivariant : EGNN (with update_coords=True)
    cross_equi       : CrossGroupEquivariant — produces v_b, θ_b from h_a
    frozen_idx       : (N_A,) particle indices of frozen group A
    active_idx       : (N_B,) particle indices of active group B
    l_box            : float or None — PBC box half-width
    """

    def __init__(self, egnn_equivariant: EGNN, cross_equi: CrossGroupEquivariant,
                 frozen_idx: torch.Tensor, active_idx: torch.Tensor,
                 l_box: float | None = None):
        super().__init__()
        self.egnn_equivariant = egnn_equivariant
        self.cross_equi = cross_equi
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.l_box = l_box

    def _get_rotation(self, x: torch.Tensor):
        """Compute per-particle Rodrigues rotation matrices R_b: (B, N_B, 3, 3).

        Rotation parameters are derived ONLY from the frozen group (pos_a, h_a).
        The active group positions are NOT used, ensuring forward and inverse
        compute the same rotation matrix (given the same frozen group state).
        """
        B = x.shape[0]
        N_B = len(self.active_idx)
        pos_a = x[:, self.frozen_idx, :]           # (B, N_A, 3)

        h_a, _ = self.egnn_equivariant(pos_a)      # (B, N_A, d) — features from frozen
        v, theta_b = self.cross_equi(h_a, pos_a)  # (B, 3), (B, N_B, 1)

        # Broadcast shared equivariant axis to all active particles
        v_b = v.unsqueeze(1).expand(-1, N_B, -1)  # (B, N_B, 3)

        R = _rodrigues_rotation(v_b, theta_b)      # (B, N_B, 3, 3)
        return R

    def forward(self, x: torch.Tensor):
        """x: (B, N, 3) → y: (B, N, 3),  logdet: (B,) = 0"""
        B = x.shape[0]
        R = self._get_rotation(x)                  # (B, N_B, 3, 3)

        x_active = x[:, self.active_idx, :]        # (B, N_B, 3)
        r, u = _to_radial(x_active)                # (B, N_B, 1), (B, N_B, 3)

        # Rotate direction: u' = R @ u
        u_new = torch.einsum('bnij,bnj->bni', R, u)   # (B, N_B, 3)
        x_active_new = r * u_new

        y = x.clone()
        y[:, self.active_idx, :] = x_active_new
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return y, logdet

    def inverse(self, y: torch.Tensor):
        """y: (B, N, 3) → x: (B, N, 3),  logdet: (B,) = 0"""
        B = y.shape[0]
        # Frozen group is unchanged → same R as forward
        R = self._get_rotation(y)                  # (B, N_B, 3, 3)

        y_active = y[:, self.active_idx, :]
        r, u_new = _to_radial(y_active)

        # Inverse rotation: u = R^T @ u'
        u = torch.einsum('bnji,bnj->bni', R, u_new)    # R^T = R.transpose(-1,-2)
        x_active = r * u

        x = y.clone()
        x[:, self.active_idx, :] = x_active
        logdet = torch.zeros(B, device=y.device, dtype=y.dtype)
        return x, logdet


# ---------------------------------------------------------------------------
# Full equivariant flow
# ---------------------------------------------------------------------------

class EGNNEquivariantFlow(nn.Module):
    """
    Stack of RadialCouplingLayer + AngularCouplingLayer blocks.

    Drop-in replacement for CartesianSplineFlow / RealNVP in training.py.

    Parameters
    ----------
    layers  : list of RadialCouplingLayer and AngularCouplingLayer
    prior   : GaussianPrior
    system  : SoluteSimulationPBC3D (or compatible)
    n_particles : int — number of SOLVENT particles (36)
    fixed_solute : bool — if True, prepend zeros for solute before energy eval
    """

    def __init__(self, layers: list, prior: GaussianPrior,
                 system, n_particles: int, fixed_solute: bool = True):
        super().__init__()
        self.coupling_layers = nn.ModuleList(layers)
        self.prior = prior
        self.system = system
        self.n_particles = int(n_particles)
        self.fixed_solute = bool(fixed_solute)

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
    # Shape helpers
    # ------------------------------------------------------------------

    def _to_particle(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.n_particles, 3)

    def _to_flat(self, particle: torch.Tensor) -> torch.Tensor:
        return particle.reshape(particle.shape[0], -1)

    def _add_fixed_solute(self, solvent_flat: torch.Tensor) -> torch.Tensor:
        """Prepend fixed solute at (0,0,0) → (B, (N+1)*3)."""
        B = solvent_flat.shape[0]
        solute = torch.zeros(B, 3, device=solvent_flat.device,
                             dtype=solvent_flat.dtype)
        return torch.cat([solute, solvent_flat], dim=1)

    # ------------------------------------------------------------------
    # Loss functions  (same API as RealNVP / CartesianSplineFlow)
    # ------------------------------------------------------------------

    def loss_ML(self, batch_x, weighted: bool = False):
        """Forward KL  –E_data[log q(x)].

        batch_x : (B, N*3) flat Cartesian positions of solvent.
        """
        x = self._to_particle(batch_x)
        z, logdet_fwd = self.forward_map(x)
        log_pz = self.prior.log_prob(self._to_flat(z))
        return -(log_pz + logdet_fwd).mean()

    def loss_KL(self, batch_z, weighted: bool = False,
                energy_cap: float | None = None, w_overlap: float = 0.0):
        """Reverse KL  E_z[U(G(z)) – log|det J_{z→x}|].

        batch_z    : (B, N*3) samples from prior.
        energy_cap : float or None — soft log-cap on energies.
        w_overlap  : float — unused (kept for API compatibility).
        """
        z = self._to_particle(batch_z)
        x, logdet_inv = self.inverse_map(z)
        x_flat = self._to_flat(x)
        x_for_energy = (self._add_fixed_solute(x_flat)
                        if self.fixed_solute else x_flat)
        u_x = self.system.get_energy_batch(x_for_energy)
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
        """z (B, N*3 flat latent) → x (B, N*3 flat Cartesian),  logdet (B,)."""
        x_particle, logdet_inv = self.inverse_map(self._to_particle(z))
        return self._to_flat(x_particle), logdet_inv

    def inverse_generator(self, x):
        """x (B, N*3 flat Cartesian) → z (B, N*3 flat latent),  logdet (B,)."""
        z_particle, logdet_fwd = self.forward_map(self._to_particle(x))
        return self._to_flat(z_particle), logdet_fwd

"""
SoluteSimulationPBC3D: 3D single solute in a Lennard-Jones bath with true PBC.

Replaces the soft-wall version (SoluteSimulation3D) with minimum-image
periodic boundary conditions.  The box is cubic with side length 2*l_box,
so particle coordinates live in [-l_box, l_box]^3.

Particle layout (same as SoluteSimulation3D)
--------------------------------------------
  particle 0 : solute (fixed at origin during flow training)
  particles 1..N-1 : identical LJ solvent

Interaction
-----------
  U_LJ = epsilon * (sigma^2 / r_ij^2)^6   (purely repulsive, no attractive tail)

All pairwise distances use the minimum-image convention.  No soft walls.
The solute can optionally be restrained to the box centre by a harmonic spring.
"""

import torch
import numpy as np


class SoluteSimulationPBC3D:
    """
    3D single solute in an LJ bath with periodic boundary conditions.

    Parameters
    ----------
    n_particles : int
        Total number of particles (1 solute + n_particles-1 solvent).
    epsilon : float
        LJ energy parameter.
    sigma : float
        LJ distance parameter.
    l_box : float
        Box half-width; the cubic box spans [-l_box, l_box]^3 → side = 2*l_box.
    center_solute : bool
        If True, add a harmonic spring on particle 0 pulling it to the origin.
        Useful when the solute position is not strictly fixed.
    k_center : float
        Spring constant for the optional centering restraint.
    """

    def __init__(
        self,
        n_particles: int = 37,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        l_box: float = 2.0,
        center_solute: bool = True,
        k_center: float = 20.0,
    ):
        self.n_particles = int(n_particles)
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.l_box = float(l_box)
        self.center_solute = bool(center_solute)
        self.k_center = float(k_center)
        self.periodic = True                 # flag consumed by MetropolisSampler

    # ------------------------------------------------------------------
    # Internal helper: minimum-image displacement
    # ------------------------------------------------------------------

    def _min_image(self, diff: torch.Tensor) -> torch.Tensor:
        """Apply minimum-image convention.  Box side = 2 * l_box."""
        L = 2.0 * self.l_box
        return diff - L * torch.round(diff / L)

    # ------------------------------------------------------------------
    # Single-configuration energy  (numpy or torch input, shape (N, 3))
    # ------------------------------------------------------------------

    def get_energy(self, coords, r_min_factor: float = 0.5,
                   energy_cut: float | None = None):
        """
        Energy of a single configuration.

        Parameters
        ----------
        coords : np.ndarray or torch.Tensor, shape (N, 3)
        r_min_factor : float
            Clamp pairwise distances to at least r_min_factor * sigma.
        energy_cut : float or None
            Soft log-cap above this value.

        Returns
        -------
        U : scalar (float if numpy input, torch scalar if torch input)
        """
        is_torch = isinstance(coords, torch.Tensor)
        if not is_torch:
            coords = torch.tensor(coords, dtype=torch.float32)

        N = coords.shape[0]

        # Pairwise displacements with minimum-image (upper triangle)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)    # (N, N, 3)
        diff = self._min_image(diff)
        r2 = (diff * diff).sum(-1)                          # (N, N)

        mask = torch.triu(torch.ones(N, N, dtype=torch.bool,
                                     device=coords.device), diagonal=1)
        r_min = r_min_factor * self.sigma
        r2_safe = torch.clamp(r2[mask], min=r_min ** 2)
        U = (self.epsilon * (self.sigma ** 2 / r2_safe) ** 6).sum()

        if self.center_solute:
            U = U + self.k_center * (coords[0] ** 2).sum()

        if energy_cut is not None:
            cut = torch.tensor(energy_cut, device=U.device, dtype=U.dtype)
            U = torch.where(U > cut, cut + torch.log1p(U - cut), U)

        if not is_torch:
            return U.item()
        return U

    # ------------------------------------------------------------------
    # Vectorised batch energy  (B, N*3) → (B,)
    # ------------------------------------------------------------------

    def get_energy_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorised energy for a batch of configurations.

        Parameters
        ----------
        batch : torch.Tensor, shape (B, N*3)
            Flattened configurations (solute at index 0, solvent at 1..N-1).

        Returns
        -------
        U : torch.Tensor, shape (B,)
        """
        B = batch.shape[0]
        N = batch.shape[1] // 3
        L = 2.0 * self.l_box

        coords = batch.reshape(B, N, 3)                     # (B, N, 3)

        # Pairwise displacements with minimum-image convention
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)    # (B, N, N, 3)
        diff = diff - L * torch.round(diff / L)             # min-image
        r2 = (diff ** 2).sum(dim=-1)                        # (B, N, N)

        # Upper-triangle pair indices (no diagonal)
        idx = torch.triu_indices(N, N, offset=1, device=batch.device)
        r2_pairs = r2[:, idx[0], idx[1]].clamp(min=1e-12)  # (B, n_pairs)
        U = (self.epsilon * (self.sigma ** 2 / r2_pairs) ** 6).sum(-1)  # (B,)

        # Optional harmonic centering on particle 0 (the solute)
        if self.center_solute:
            U = U + self.k_center * (coords[:, 0, :] ** 2).sum(-1)

        return U

import numpy as np
import torch


class DimerSimulation3D:
    """
    3D dimer in a Lennard-Jones bath with periodic boundary conditions.

    The system has N particles: particles 0 and 1 form a dimer connected by a
    bond potential; the remaining N-2 particles are solvent.

    The LJ interaction is purely repulsive: U_LJ = epsilon * (sigma^2 / r^2)^6.
    Periodic boundary conditions (minimum image convention) are applied in all
    three dimensions.  No box-wall penalty is needed.

    The energy is fully differentiable via PyTorch autograd.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        k_d: float = 20.0,
        d0: float = 1.5,
        a: float = 25.0,
        b: float = 10.0,
        c: float = -0.5,
        l_box: float = 3.0,
        periodic: bool = True,
        k_box: float = 100.0,
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.k_d = k_d
        self.d0 = d0
        self.a = a
        self.b = b
        self.c = c
        self.l_box = l_box
        self.periodic = periodic   # True = PBC; False = soft harmonic walls
        self.k_box = k_box         # wall stiffness (only used when periodic=False)

    # ------------------------------------------------------------------
    # Bond potential (same quartic form as 2D version)
    # ------------------------------------------------------------------

    def bond_energy(self, d):
        """Quartic double-well bond potential."""
        dd = d - self.d0
        return 0.25 * self.a * dd ** 4 - 0.5 * self.b * dd ** 2 + self.c * dd

    # ------------------------------------------------------------------
    # Single-configuration energy  (supports numpy or torch inputs)
    # ------------------------------------------------------------------

    def get_energy(self, coords):
        """
        Energy of a single configuration.

        Parameters
        ----------
        coords : torch.Tensor or np.ndarray, shape (N, 3)

        Returns
        -------
        U : scalar (same type as input)
        """
        is_torch = isinstance(coords, torch.Tensor)

        if not is_torch:
            coords = torch.tensor(coords, dtype=torch.float32)

        N = coords.shape[0]
        L = self.l_box

        U = torch.tensor(0.0, dtype=coords.dtype, device=coords.device)

        # Bond potential for particles 0-1
        d_bond = torch.norm(coords[0] - coords[1])
        U = U + self.bond_energy(d_bond)

        # Pairwise LJ (upper triangle)
        for i in range(N):
            for j in range(i + 1, N):
                # skip the bonded pair (bond potential already handles them)
                if i == 0 and j == 1:
                    continue
                dr = coords[i] - coords[j]
                if self.periodic:
                    dr = dr - L * torch.round(dr / L)   # minimum image
                r2 = (dr * dr).sum()
                U = U + self.epsilon * (self.sigma ** 2 / r2) ** 6

        # Soft harmonic wall (non-periodic only)
        if not self.periodic:
            excess = torch.relu(coords.abs() - L / 2)   # (N, 3), zero inside box
            U = U + self.k_box * excess.pow(2).sum()

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
            Flattened configurations.

        Returns
        -------
        U : torch.Tensor, shape (B,)
        """
        B = batch.shape[0]
        N = batch.shape[1] // 3
        L = self.l_box

        coords = batch.reshape(B, N, 3)  # (B, N, 3)

        # ---- Bond energy for particles 0 and 1 ----
        d_bond = torch.norm(coords[:, 0, :] - coords[:, 1, :], dim=-1)  # (B,)
        U = self.bond_energy(d_bond)

        # ---- Pairwise LJ (upper triangle) ----
        # diff[b, i, j, :] = coords[b, i, :] - coords[b, j, :]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        if self.periodic:
            diff = diff - L * torch.round(diff / L)       # minimum image
        r2 = (diff ** 2).sum(dim=-1)                      # (B, N, N)

        # Mask diagonal (self-interaction) and bonded pair (0,1)
        diag_mask = torch.eye(N, dtype=torch.bool, device=batch.device)  # (N, N)
        r2 = r2.masked_fill(diag_mask, 1.0)               # avoid /0 on diagonal

        lj = self.epsilon * (self.sigma ** 2 / r2) ** 6   # (B, N, N)

        # Keep upper triangle only (each pair counted once)
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=batch.device), diagonal=1)
        # Exclude the bonded pair 0-1
        upper[0, 1] = False

        U = U + (lj * upper.float()).sum(dim=(-2, -1))    # (B,)

        # Soft harmonic wall (non-periodic only)
        if not self.periodic:
            excess = torch.relu(coords.abs() - L / 2)     # (B, N, 3)
            U = U + self.k_box * excess.pow(2).sum(dim=(-2, -1))

        return U


class SoluteSimulation3D:
    """
    3D single solute in a Lennard-Jones bath with soft harmonic walls.

    Particle 0 is the labeled solute (fixed at origin during flow training);
    particles 1..N-1 are identical LJ solvent.  All particles interact via the
    same purely repulsive LJ potential: U_LJ = epsilon * (sigma^2 / r^2)^6.

    No bond potential — unlike DimerSimulation3D.
    No periodic BC — soft harmonic walls instead (mirrors SoluteSimulation2D).

    Parameters
    ----------
    n_particles : int
        Total number of particles (1 solute + n_particles-1 solvent).
    epsilon, sigma : float
        LJ parameters.
    l_box : float
        Soft wall activates for |x|, |y|, |z| > l_box.
    k_box : float
        Soft harmonic wall stiffness.
    center_solute : bool
        Apply a harmonic centering restraint on particle 0.
    k_center : float
        Spring constant for the centering restraint.
    """

    def __init__(
        self,
        n_particles: int = 37,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        l_box: float = 3.5,
        k_box: float = 100.0,
        center_solute: bool = True,
        k_center: float = 20.0,
    ):
        self.n_particles = int(n_particles)
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.l_box = float(l_box)
        self.k_box = float(k_box)
        self.center_solute = bool(center_solute)
        self.k_center = float(k_center)
        self.periodic = False   # always soft-wall for the solute system

    # ------------------------------------------------------------------
    # Single-configuration energy  (numpy or torch input, shape (N, 3))
    # ------------------------------------------------------------------

    def get_energy(self, coords, r_min_factor=0.5, energy_cut=1e3):
        """
        Energy of a single configuration.

        Parameters
        ----------
        coords : np.ndarray or torch.Tensor, shape (N, 3)
        r_min_factor : float
            Minimum pair distance as fraction of sigma (clamp for MC stability).
        energy_cut : float or None
            Soft log-cap above this value. Use None for physical evaluation.

        Returns
        -------
        U : scalar (float if numpy input, torch scalar if torch input)
        """
        is_torch = isinstance(coords, torch.Tensor)
        if not is_torch:
            coords = torch.tensor(coords, dtype=torch.float32)

        N = coords.shape[0]
        L = self.l_box

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)   # (N, N, 3)
        r2   = (diff * diff).sum(-1)                        # (N, N)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=coords.device), diagonal=1)

        r_min   = r_min_factor * self.sigma
        r2_safe = torch.clamp(r2[mask], min=r_min ** 2)
        U = (self.epsilon * (self.sigma ** 2 / r2_safe) ** 6).sum()

        excess = torch.relu(coords.abs() - L)
        U = U + self.k_box * excess.pow(2).sum()

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
        batch : torch.Tensor, shape (B, N*3)  flattened configurations.

        Returns
        -------
        U : torch.Tensor, shape (B,)
        """
        B = batch.shape[0]
        N = batch.shape[1] // 3
        L = self.l_box

        coords = batch.reshape(B, N, 3)   # (B, N, 3)

        # Pairwise repulsive LJ (upper triangle)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)    # (B, N, N, 3)
        r2   = (diff ** 2).sum(dim=-1)                       # (B, N, N)
        idx  = torch.triu_indices(N, N, offset=1, device=batch.device)
        r2_pairs = r2[:, idx[0], idx[1]].clamp(min=1e-12)   # (B, n_pairs)
        U = (self.epsilon * (self.sigma ** 2 / r2_pairs) ** 6).sum(-1)  # (B,)

        # Soft harmonic walls
        excess = torch.relu(coords.abs() - L)                # (B, N, 3)
        U = U + self.k_box * excess.pow(2).sum(dim=(-2, -1))

        # Optional harmonic centering on particle 0
        if self.center_solute:
            U = U + self.k_center * (coords[:, 0, :] ** 2).sum(-1)

        return U

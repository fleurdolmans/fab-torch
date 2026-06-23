import numpy as np
import torch


class SoluteSimulation2D:
    """
    2D single solute in a repulsive Lennard-Jones bath with soft harmonic walls.

    Particle 0 is the labeled solute; particles 1..N-1 are identical LJ solvent.
    All particles interact via the same purely repulsive LJ potential.
    No bond potential — unlike DimerSimulation.

    Parameters
    ----------
    n_particles : int
        Total number of particles (1 solute + n_particles-1 solvent).
    epsilon, sigma : float
        LJ parameters.  Default sigma=1.1 matches DimerSimulation.
    l_box : float
        Soft wall kicks in at |x| > l_box and |y| > l_box (square walls),
        or at r > r_box (circular walls). Coordinates nominally live in
        [-l_box, l_box]^2 or the disk of radius r_box.
    k_box : float
        Soft harmonic wall stiffness.
    circular_walls : bool
        If True, use a circular (radial) wall instead of a square wall.
        The circular wall is SO(2)-invariant and matches the symmetry of the
        equivariant flow. Penalty: k_box * relu(‖x_i‖ - r_box)^2.
    r_box : float or None
        Radius for the circular wall. If None, defaults to l_box * sqrt(4/pi)
        so that the disk has the same area as the square [-l_box, l_box]^2.

    Compatible with
    ---------------
    boltzmann_generators/Library/sampling.py   (MetropolisSampler)
    boltzmann_generators/Library/generator.py  (RealNVP, sys_dim=(n_particles, 2))
    boltzmann_generators/Library/training.py   (BoltzmannGenerator)
    """

    def __init__(self, n_particles: int = 37, epsilon: float = 1.0,
                 sigma: float = 1.1, l_box: float = 3.0, k_box: float = 100.0,
                 center_solute: bool = False, k_center: float = 20.0,
                 circular_walls: bool = False, r_box: float = None):
        self.n_particles = int(n_particles)
        self.epsilon = epsilon
        self.sigma = sigma
        self.l_box = l_box
        self.k_box = k_box
        self.center_solute = center_solute  # harmonic centering restraint on solute (particle 0)
        self.k_center = k_center            # spring constant for centering
        self.circular_walls = circular_walls
        # equal-area default: pi * r_box^2 = (2*l_box)^2  =>  r_box = 2*l_box / sqrt(pi)
        self.r_box = r_box if r_box is not None else 2.0 * l_box / (np.pi ** 0.5)

    def get_energy(self, coords, r_min_factor=0.5, energy_cut=None):
        """
        Energy of a single configuration.

        r_min_factor:
            Minimum pair distance as a fraction of sigma.
            Use 0.5 for FAB warmup, maybe 0.3 later.

        energy_cut:
            Optional soft cap. Use 1e3 or 1e4 for FAB warmup.
            Use None for physical/evaluation energy.
        """
        is_torch = isinstance(coords, torch.Tensor)
        if not is_torch:
            coords = torch.tensor(coords, dtype=torch.float32)

        N = coords.shape[0]
        L = self.l_box

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        r2 = (diff * diff).sum(-1)

        mask = torch.triu(
            torch.ones(
                N,
                N,
                dtype=torch.bool,
                device=coords.device,
            ),
            diagonal=1,
        )

        r_min = r_min_factor * self.sigma
        r2_safe = torch.clamp(r2[mask], min=r_min ** 2)

        U = (4 * self.epsilon * (self.sigma ** 2 / r2_safe) ** 6).sum()

        if self.circular_walls:
            r = coords.norm(dim=-1)                          # (N,)
            excess = torch.relu(r - self.r_box)
            U = U + self.k_box * excess.pow(2).sum()
        else:
            excess = torch.relu(coords.abs() - L)
            U = U + self.k_box * excess.pow(2).sum()

        if self.center_solute:
            U = U + self.k_center * (coords[0] ** 2).sum()

        if energy_cut is not None:
            cut = torch.tensor(
                energy_cut,
                device=U.device,
                dtype=U.dtype,
            )
            U = torch.where(
                U > cut,
                cut + torch.log1p(U - cut),
                U,
            )

        if not is_torch:
            return U.item()

        return U

    def log_prob(self, x_solvent: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log probability log p(x) = -U(x), for compatibility with
        the SoluteTarget2D interface used in evaluate_ablation / eval_energies.

        Parameters
        ----------
        x_solvent : (B, N_solvent * 2)
            Flat solvent-only coordinates. The solute is fixed at the origin.

        Returns
        -------
        log_p : (B,)
        """
        B = x_solvent.shape[0]
        D = 2
        N_solvent = x_solvent.shape[1] // D
        solvent = x_solvent.view(B, N_solvent, D)
        solute = torch.zeros(B, 1, D, device=x_solvent.device, dtype=x_solvent.dtype)
        coords_full = torch.cat([solute, solvent], dim=1).reshape(B, -1)
        energies = self.get_energy_batch(coords_full)
        return -energies

    def get_energy_batch(self, coords_flat, **kwargs):
        """
        Energy for a batch of flat configurations.

        Parameters
        ----------
        coords_flat : torch.Tensor or np.ndarray, shape (batch, n_particles * dim)
            Each row is a flattened configuration of shape (n_particles, dim).

        Returns
        -------
        energies : torch.Tensor, shape (batch,)
        """
        is_torch = isinstance(coords_flat, torch.Tensor)
        if not is_torch:
            coords_flat = torch.tensor(coords_flat, dtype=torch.float32)

        batch = coords_flat.shape[0]
        dim = coords_flat.shape[1] // self.n_particles
        configs = coords_flat.reshape(batch, self.n_particles, dim)

        energies = torch.stack([
            self.get_energy(configs[i], **kwargs) for i in range(batch)
        ])
        return energies

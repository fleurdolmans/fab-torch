import numpy as np
import torch


class SoluteSimulation2D:
    """
    2D single solute in a Lennard-Jones bath with soft harmonic square walls.

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
        Soft wall kicks in at |x| > l_box and |y| > l_box.
        Coordinates nominally live in [-l_box, l_box]^2.
    k_box : float
        Soft harmonic wall stiffness.

    Compatible with
    ---------------
    boltzmann_generators/Library/sampling.py   (MetropolisSampler)
    boltzmann_generators/Library/generator.py  (RealNVP, sys_dim=(n_particles, 2))
    boltzmann_generators/Library/training.py   (BoltzmannGenerator)
    """

    def __init__(self, n_particles: int = 37, epsilon: float = 1.0,
                 sigma: float = 1.1, l_box: float = 3.0, k_box: float = 100.0,
                 center_solute: bool = False, k_center: float = 20.0):
        self.n_particles = int(n_particles)
        self.epsilon = epsilon
        self.sigma = sigma
        self.l_box = l_box
        self.k_box = k_box
        self.center_solute = center_solute  # harmonic centering restraint on solute (particle 0)
        self.k_center = k_center            # spring constant for centering

    # ------------------------------------------------------------------
    # Single-configuration energy  (numpy or torch input, shape (N, 2))
    # ------------------------------------------------------------------

    # def get_energy(self, coords):
    #     """
    #     Energy of a single configuration.

    #     Parameters
    #     ----------
    #     coords : np.ndarray or torch.Tensor, shape (N, 2)

    #     Returns
    #     -------
    #     U : scalar (float if numpy input, torch scalar if torch input)
    #     """
    #     is_torch = isinstance(coords, torch.Tensor)
    #     if not is_torch:
    #         coords = torch.tensor(coords, dtype=torch.float32)

    #     N = coords.shape[0]
    #     L = self.l_box

    #     # Pairwise repulsive LJ — vectorised upper triangle, no Python loop
    #     # diff[i,j] = coords[i] - coords[j],  shape (N, N, 2)
    #     diff = coords.unsqueeze(1) - coords.unsqueeze(0)          # (N, N, 2)
    #     r2   = (diff * diff).sum(-1)                               # (N, N)
    #     mask = torch.triu(torch.ones(N, N, dtype=torch.bool,
    #                                  device=coords.device), diagonal=1)
        

    #     # U = (self.epsilon * (self.sigma ** 2 / r2[mask]) ** 6).sum()

    #     r2_safe = torch.clamp(r2[mask], min=1e-8)
    #     U = (self.epsilon * (self.sigma ** 2 / r2_safe) ** 6).sum()

    #     # Soft harmonic square wall: penalty for |coord| > l_box
    #     excess = torch.relu(coords.abs() - L)                      # (N, 2)
    #     U = U + self.k_box * excess.pow(2).sum()

    #     # Optional harmonic centering restraint on solute (particle 0)
    #     if self.center_solute:
    #         U = U + self.k_center * (coords[0] ** 2).sum()

    #     if not is_torch:
    #         return U.item()
    #     return U
    def get_energy(self, coords, r_min_factor=0.5, energy_cut=1e3):
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

        U = (self.epsilon * (self.sigma ** 2 / r2_safe) ** 6).sum()

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

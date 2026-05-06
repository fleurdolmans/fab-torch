import numpy as np
import itertools


def build_dimer_coords_3d(N: int = 38, box: float = 6.0, noise: float = 0.1,
                          bond: list = None) -> np.ndarray:
    """
    Place N particles on a simple-cubic lattice inside a cubic box [-box/2, box/2]^3.

    Particles ``bond[0]`` and ``bond[1]`` are moved to index 0 and 1 so that the
    dimer occupies the first two rows of the returned array, matching the
    convention in ``potentials.DimerSimulation3D``.

    Parameters
    ----------
    N : int
        Total number of particles (dimer + solvent).
    box : float
        Side length of the cubic box.
    noise : float
        Standard deviation of Gaussian noise added to lattice positions.
    bond : list of two ints
        Lattice indices of the two dimer particles before re-ordering.
        Defaults to [0, 1].

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
    """
    if bond is None:
        bond = [0, 1]

    r_min = -box / 2.0
    r_max = box / 2.0

    # spacing along one axis so that N^{1/3} cells fit
    n_side = int(np.ceil(N ** (1.0 / 3.0)))
    d = box / n_side

    pos_1d = np.linspace(r_min + 0.5 * d, r_max - 0.5 * d, n_side)
    lattice = list(itertools.product(pos_1d, repeat=3))  # (n_side^3,) list of 3-tuples
    lattice = lattice[:N]

    # Move dimer particles to the front
    lattice.insert(0, lattice.pop(bond[0]))
    # after the first pop/insert, bond[1] index may have shifted
    idx1 = bond[1] - 1 if bond[1] > bond[0] else bond[1]
    lattice.insert(1, lattice.pop(idx1 + 1))  # +1 because we already inserted at 0

    coords = np.array(lattice, dtype=np.float64)

    if noise:
        coords += noise * np.random.randn(*coords.shape)

    return coords


def build_solute_coords_3d_rsa(
    N: int = 37,
    l_box: float = 2.3,
    sigma: float = 1.1,
    max_attempts: int = 100_000,
    seed: int = None,
) -> np.ndarray:
    """
    Place N particles in [-l_box, l_box]^3 using Random Sequential Addition (RSA).

    Particle 0 (solute) is fixed at the origin. Each subsequent solvent particle is
    placed at a random position that does not overlap with any already-placed particle
    (minimum separation = sigma). Works reliably up to reduced density rho* ~ 0.55.

    Parameters
    ----------
    N : int
        Total number of particles (1 solute + N-1 solvent). Default 37.
    l_box : float
        Half-width of the cubic box; particles are drawn from [-l_box, l_box]^3.
    sigma : float
        Minimum allowed centre-to-centre distance (LJ diameter).
    max_attempts : int
        Maximum insertion attempts per particle before raising RuntimeError.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
        coords[0] = (0, 0, 0)  (solute at origin)
        coords[1:] = RSA-placed solvent particles
    """
    rng = np.random.default_rng(seed)
    coords = [np.zeros(3)]   # solute at origin

    for i in range(N - 1):
        placed = False
        for _ in range(max_attempts):
            pos = rng.uniform(-l_box, l_box, 3)
            dists = np.linalg.norm(np.array(coords) - pos, axis=1)
            if np.all(dists >= sigma):
                coords.append(pos)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed to place particle {i + 1} after {max_attempts} attempts. "
                f"System may be too dense (l_box={l_box}, sigma={sigma}, N={N}). "
                f"Try increasing l_box or reducing N."
            )

    return np.array(coords, dtype=np.float64)


def build_solute_coords_3d(N: int = 37, box: float = 3.5,
                           noise: float = 0.1) -> np.ndarray:
    """
    Place N particles on a simple-cubic lattice inside a cubic box [-box/2, box/2]^3
    for the solute-in-LJ-bath system.

    Particle 0 (the solute) is fixed at the origin.
    Particles 1..N-1 are placed on lattice sites.

    Parameters
    ----------
    N : int
        Total number of particles (1 solute + N-1 solvent).
    box : float
        Side length of the cubic box.
    noise : float
        Standard deviation of Gaussian noise added to solvent positions.

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
        coords[0] = (0, 0, 0)  (fixed solute at origin)
        coords[1:] = solvent on lattice + noise
    """
    n_solvent = N - 1
    n_side = int(np.ceil(n_solvent ** (1.0 / 3.0)))
    d = box / n_side
    r_min = -box / 2.0

    pos_1d = np.linspace(r_min + 0.5 * d, r_min + (n_side - 0.5) * d, n_side)
    lattice = list(itertools.product(pos_1d, repeat=3))[:n_solvent]

    solvent = np.array(lattice, dtype=np.float64)
    if noise:
        solvent += noise * np.random.randn(*solvent.shape)

    solute = np.zeros((1, 3), dtype=np.float64)
    return np.concatenate([solute, solvent], axis=0)

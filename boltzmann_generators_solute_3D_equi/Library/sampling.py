import numpy as np
import copy
from tqdm.auto import tqdm


class MetropolisSampler:
    """
    Metropolis–Hastings sampler for an N-particle system in any spatial dimension,
    with periodic boundary conditions.

    After each proposed move the coordinates are wrapped back into
    [-L/2, L/2]^3 so the simulation stays in the primary unit cell.

    Parameters
    ----------
    model : potentials object
        Must expose ``get_energy(coords)`` where coords is (N, 3).
    temp : float
        Reduced temperature.
    sigma : float
        Standard deviation of the isotropic Gaussian displacement.
    stride : int
        Save a frame every ``stride`` steps.
    l_box : float
        Side length of the cubic periodic box.
    """

    def __init__(self, model, temp=1.0, sigma=0.1, stride=1, l_box=None, periodic=None):
        self.model = model
        self.sigma = sigma
        self.temp = temp
        self.stride = stride
        # Infer box length from the system object if not given explicitly
        if l_box is not None:
            self.l_box = l_box
        elif hasattr(model, 'l_box'):
            self.l_box = model.l_box
        else:
            raise ValueError("Provide l_box or use a model that has a .l_box attribute.")
        # Infer periodicity from the model if not specified explicitly
        if periodic is not None:
            self.periodic = periodic
        elif hasattr(model, 'periodic'):
            self.periodic = model.periodic
        else:
            self.periodic = True

    def _wrap(self, coords):
        """Wrap coordinates into [-l_box, l_box]^3 (full box side = 2*l_box)."""
        L = 2.0 * self.l_box      # full box side length
        return coords - L * np.round(coords / L)

    def run(self, x0, nsteps, diff=False):
        """
        Run the Metropolis simulation.

        Parameters
        ----------
        x0 : np.ndarray, shape (N, 3)
            Initial configuration.
        nsteps : int
            Number of MC steps.
        diff : bool
            If True, skip saving duplicate frames (non-Boltzmann-weighted).

        Attributes set
        --------------
        xtraj : np.ndarray, shape (n_saved, N, 3)
        etraj : np.ndarray, shape (n_saved,)
        """
        x = self._wrap(x0.copy()) if self.periodic else x0.copy()
        E = self.model.get_energy(x)

        self.xtraj = [copy.deepcopy(x)]
        self.etraj = [E if not hasattr(E, 'item') else E.item()]

        beta = 1.0 / self.temp

        for step in tqdm(range(nsteps)):
            dx = self.sigma * np.random.randn(*x.shape)
            x_proposed = self._wrap(x + dx) if self.periodic else (x + dx)
            E_proposed = self.model.get_energy(x_proposed)

            E_val = E if not hasattr(E, 'item') else E.item()
            Ep_val = E_proposed if not hasattr(E_proposed, 'item') else E_proposed.item()
            delta_E = Ep_val - E_val

            if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                x = x_proposed
                E = E_proposed

            if step % self.stride == self.stride - 1:
                E_save = E if not hasattr(E, 'item') else E.item()
                if diff and len(self.xtraj) > 0:
                    if not np.array_equal(x, self.xtraj[-1]):
                        self.xtraj.append(copy.deepcopy(x))
                        self.etraj.append(E_save)
                else:
                    self.xtraj.append(copy.deepcopy(x))
                    self.etraj.append(E_save)

        self.xtraj = np.array(self.xtraj)
        self.etraj = np.array(self.etraj)

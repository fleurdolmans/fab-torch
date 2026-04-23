import torch
from torch import nn, Tensor
from typing import Optional, Literal, Callable
import pathlib
import h5py
import numpy as np
import warnings

from fab.target_distributions.base import TargetDistribution
from fab.transforms.transform_LJ_2D import SoluteCenteredUnitTorusTransform2D


class LJParticles2D(nn.Module, TargetDistribution):
    """
    2D LJ Boltzmann target for one solute + LJ solvent in a periodic square box.
    """

    def __init__(
        self,
        dim: int,
        n_solvent: int,
        temperature: float,
        energy_cut: float = 1.0e8,
        energy_max: float = 1.0e20,
        n_threads: int = 1,
        train_samples_path: Optional[str] = None,
        val_samples_path: Optional[str] = None,
        test_samples_path: Optional[str] = None,
        eval_mode: Literal["val", "test"] = "val",
        device: str = "cpu",
        logger=None,
        save_dir: Optional[str] = None,
        box_length_nm: float = 2.14,
        solvent_sigma_nm: float = 0.34,
        solvent_epsilon_kjmol: float = 0.996,
        solute_sigma_nm: float = 0.38,
        solute_epsilon_kjmol: float = 0.996,
        n_solute: int = 1,
        transform_version: str = "v4",
    ):
        nn.Module.__init__(self)
        TargetDistribution.__init__(self)

        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.n_particles = self.n_solute + self.n_solvent
        self.temperature = float(temperature)
        self.device = device
        self.eval_mode = eval_mode
        self.logger = logger
        self.save_dir = save_dir
        self.energy_cut = float(energy_cut)
        self.energy_max = float(energy_max)
        self.n_threads = int(n_threads)

        self.box_length_nm = float(box_length_nm)
        self.solvent_sigma_nm = float(solvent_sigma_nm)
        self.solvent_epsilon_kjmol = float(solvent_epsilon_kjmol)
        self.solute_sigma_nm = float(solute_sigma_nm)
        self.solute_epsilon_kjmol = float(solute_epsilon_kjmol)

        self.cartesian_dim = int(dim)
        expected_dim = 2 * self.n_particles
        if self.cartesian_dim != expected_dim:
            raise ValueError(f"dim={dim}, expected {expected_dim}")

        if transform_version != "v4":
            raise ValueError("For now use transform_version='v4' for 2D.")

        self.coordinate_transform = SoluteCenteredUnitTorusTransform2D(
            n_solvent=self.n_solvent,
            box_length_nm=self.box_length_nm,
            n_solute=self.n_solute,
        )

        self.internal_dim = self.coordinate_transform.internal_dim

        self.train_data_x = self.load_target_data(train_samples_path, self.cartesian_dim) if train_samples_path else None
        self.val_data_x = self.load_target_data(val_samples_path, self.cartesian_dim) if val_samples_path else None
        self.test_data_x = self.load_target_data(test_samples_path, self.cartesian_dim) if test_samples_path else None

        self.train_data_i, self.train_logdet_xi = (
            self.coordinate_transform.inverse(self.train_data_x) if self.train_data_x is not None else (None, None)
        )
        self.val_data_i, self.val_logdet_xi = (
            self.coordinate_transform.inverse(self.val_data_x) if self.val_data_x is not None else (None, None)
        )
        self.test_data_i, self.test_logdet_xi = (
            self.coordinate_transform.inverse(self.test_data_x) if self.test_data_x is not None else (None, None)
        )

    def load_target_data(self, data_path: pathlib.Path | str, dim: int):
        data_path = pathlib.Path(data_path)

        if data_path.suffix == ".h5":
            with h5py.File(str(data_path), "r") as f:
                coords = torch.from_numpy(f["coordinates"][()]).reshape(-1, dim)

        elif data_path.suffix == ".pt":
            coords = torch.load(str(data_path))
            if coords.ndim != 2:
                raise ValueError("PT data must have shape (num_frames, dim)")

        elif data_path.suffix == ".npz":
            arr = np.load(str(data_path))
            coords = torch.from_numpy(arr["coordinates"]).reshape(-1, dim)

        else:
            raise ValueError(f"Unsupported suffix: {data_path.suffix}")

        if coords.shape[1] != dim:
            raise ValueError(f"Loaded dim {coords.shape[1]} != expected {dim}")
        return coords.float()

    def _reshape_x(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], self.n_particles, 2)

    def _minimum_image(self, dr: Tensor) -> Tensor:
        L = torch.as_tensor(self.box_length_nm, device=dr.device, dtype=dr.dtype)
        return dr - L * torch.round(dr / L)

    def energy_x(self, x: Tensor) -> Tensor:
        X = self._reshape_x(x)

        sigmas = torch.tensor(
            [self.solute_sigma_nm] * self.n_solute + [self.solvent_sigma_nm] * self.n_solvent,
            device=X.device,
            dtype=X.dtype,
        )
        epsilons = torch.tensor(
            [self.solute_epsilon_kjmol] * self.n_solute + [self.solvent_epsilon_kjmol] * self.n_solvent,
            device=X.device,
            dtype=X.dtype,
        )

        Xi = X[:, :, None, :]
        Xj = X[:, None, :, :]
        dr = self._minimum_image(Xi - Xj)
        r2 = torch.sum(dr * dr, dim=-1)

        sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
        epsilon_ij = torch.sqrt(epsilons[:, None] * epsilons[None, :])

        sigma_ij = sigma_ij[None, :, :]
        epsilon_ij = epsilon_ij[None, :, :]

        eye = torch.eye(self.n_particles, device=X.device, dtype=torch.bool)[None, :, :]
        r2 = torch.where(eye, torch.ones_like(r2), r2)

        sr2 = (sigma_ij * sigma_ij) / r2
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        lj = 4.0 * epsilon_ij * (sr12 - sr6)
        lj = torch.where(eye, torch.zeros_like(lj), lj)

        E = 0.5 * torch.sum(lj, dim=(1, 2))
        return E

    def log_prob_x(self, x: Tensor):
        return -self.energy_x(x) / self.temperature

    def log_prob(self, i: Tensor):
        x, logdet = self.coordinate_transform.forward(i)
        return self.log_prob_x(x) + logdet

    def log_prob_and_jac(self, i: Tensor):
        x, logdet = self.coordinate_transform.forward(i)
        return self.log_prob_x(x) + logdet, logdet
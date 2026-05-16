"""
Target distributions for the 3D solute-in-LJ-bath system (FAB).

SoluteTarget3D     — original soft-wall version (kept for reference)
SoluteTargetPBC3D  — PBC version (minimum-image distances, no soft walls)

The flow models 36 solvent particles in 3D Cartesian space (108D). The solute
is fixed at the origin. Both targets:
  1. Prepend the fixed solute at (0, 0, 0).
  2. Evaluate the batched, differentiable energy of the full 37-particle system.
  3. Return -energy / temperature as the unnormalised log probability.
"""

import abc
from typing import Callable, Dict, Optional, Tuple

import torch


LogProbFunc = Callable[[torch.Tensor], torch.Tensor]


class TargetDistribution(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def performance_metrics(self, samples, log_w, log_q_fn=None, batch_size=None) -> Dict:
        raise NotImplementedError

    def sample(self, shape):
        raise NotImplementedError


class SoluteTarget3D(TargetDistribution):
    """
    FAB target for the 3D solute-in-LJ-bath system.

    Parameters
    ----------
    n_solvent    : int   — number of solvent particles (default 36)
    epsilon      : float — LJ energy scale
    sigma        : float — LJ length scale; particles repel below ~sigma
    l_box        : float — soft wall activates for |x|, |y|, |z| > l_box
    k_box        : float — soft-wall spring constant
    center_solute: bool  — whether to apply harmonic centering on particle 0
    k_center     : float — centering spring constant
    temperature  : float — temperature in reduced units (default 1.0)
    energy_cut   : float — energies above this are log-compressed
    """

    def __init__(
        self,
        n_solvent: int = 36,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        l_box: float = 3.5,
        k_box: float = 100.0,
        center_solute: bool = True,
        k_center: float = 20.0,
        temperature: float = 1.0,
        energy_cut: float = 1.0e4,
    ):
        super().__init__()
        self.n_solvent    = int(n_solvent)
        self.n_particles  = self.n_solvent + 1   # 1 fixed solute + n_solvent
        self.epsilon      = float(epsilon)
        self.sigma        = float(sigma)
        self.l_box        = float(l_box)
        self.k_box        = float(k_box)
        self.center_solute = bool(center_solute)
        self.k_center     = float(k_center)
        self.temperature  = float(temperature)
        self.energy_cut   = float(energy_cut)

    # ------------------------------------------------------------------
    # Batched, differentiable energy (mirrors SoluteSimulation3D exactly)
    # ------------------------------------------------------------------

    def _batched_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (B, N, 3)  full configuration, particle 0 is the solute

        Returns
        -------
        energy : (B,)
        """
        B, N, _ = coords.shape

        # Pairwise repulsive LJ  (epsilon * (sigma^2 / r^2)^6)
        diff     = coords.unsqueeze(2) - coords.unsqueeze(1)    # (B, N, N, 3)
        r2       = (diff * diff).sum(-1)                         # (B, N, N)
        idx      = torch.triu_indices(N, N, offset=1, device=coords.device)
        r2_pairs = r2[:, idx[0], idx[1]].clamp(min=1e-12)       # (B, n_pairs)
        U_lj     = (self.epsilon * (self.sigma ** 2 / r2_pairs) ** 6).sum(-1)  # (B,)

        # Soft harmonic walls:  k_box * sum relu(|x| - l_box)^2
        excess = torch.relu(coords.abs() - self.l_box)           # (B, N, 3)
        U_wall = self.k_box * excess.pow(2).sum(dim=(1, 2))      # (B,)

        energy = U_lj + U_wall

        # Optional harmonic centering on the solute (particle index 0)
        if self.center_solute:
            energy = energy + self.k_center * (coords[:, 0, :] ** 2).sum(-1)

        return energy

    def _clip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Soft-cap large energies to prevent -inf log probs during AIS."""
        cut = torch.tensor(self.energy_cut, device=energy.device, dtype=energy.dtype)
        return torch.where(
            energy > cut,
            cut + torch.log1p((energy - cut).clamp(min=0.0)),
            energy,
        )

    # ------------------------------------------------------------------
    # TargetDistribution interface
    # ------------------------------------------------------------------

    def log_prob(self, x_solvent: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log probability  log p(x) = -U(x) / T.

        Parameters
        ----------
        x_solvent : (B, 108)  Cartesian coordinates of the 36 solvent particles.
                               The solute is implicitly fixed at (0, 0, 0).

        Returns
        -------
        log_p : (B,)
        """
        B      = x_solvent.shape[0]
        device = x_solvent.device
        dtype  = x_solvent.dtype

        # Prepend fixed solute at origin → (B, 37, 3)
        solute  = torch.zeros(B, 1, 3, device=device, dtype=dtype)
        solvent = x_solvent.view(B, self.n_solvent, 3)
        coords  = torch.cat([solute, solvent], dim=1)

        energy = self._batched_energy(coords)
        energy = self._clip_energy(energy)
        return -energy / self.temperature

    def performance_metrics(
        self,
        samples: torch.Tensor,
        log_w: torch.Tensor,
        log_q_fn=None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        """Returns ESS, mean log-weight, log_Z, and mean energy."""
        if samples is None or log_w is None:
            return {}

        valid    = torch.isfinite(log_w)
        log_w_v  = log_w[valid]
        samples_v = samples[valid]

        if len(log_w_v) == 0:
            return {"ess": 0.0, "mean_log_w": float("nan"), "log_Z": float("nan")}

        n      = float(len(log_w_v))
        log_Z  = torch.logsumexp(log_w_v, dim=0) - torch.log(
            torch.tensor(n, device=log_w_v.device))
        w_norm = torch.softmax(log_w_v, dim=0)
        ess    = (1.0 / (w_norm ** 2).sum()).item()

        with torch.no_grad():
            log_p = self.log_prob(samples_v)
        mean_energy = (-log_p * self.temperature).mean().item()

        return {
            "ess":          ess,
            "mean_log_w":   log_w_v.mean().item(),
            "log_Z":        log_Z.item(),
            "mean_energy":  mean_energy,
        }


# ---------------------------------------------------------------------------
# PBC target  (minimum-image distances, no soft walls)
# ---------------------------------------------------------------------------

class SoluteTargetPBC3D(TargetDistribution):
    """
    FAB target for the 3D PBC solute-in-LJ-bath system.

    Mirrors SoluteSimulationPBC3D: purely repulsive LJ with minimum-image
    convention; no soft harmonic walls.

    Parameters
    ----------
    n_solvent    : int   — number of solvent particles (default 36)
    epsilon      : float — LJ energy scale
    sigma        : float — LJ length scale
    l_box        : float — box half-width (box side = 2*l_box)
    center_solute: bool  — harmonic centering restraint on particle 0
    k_center     : float — centering spring constant
    temperature  : float — reduced temperature (default 1.0)
    energy_cut   : float — soft log-cap threshold
    """

    def __init__(
        self,
        n_solvent: int = 36,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        l_box: float = 2.0,
        center_solute: bool = True,
        k_center: float = 20.0,
        temperature: float = 1.0,
        energy_cut: float = 1.0e4,
    ):
        super().__init__()
        self.n_solvent     = int(n_solvent)
        self.n_particles   = self.n_solvent + 1
        self.epsilon       = float(epsilon)
        self.sigma         = float(sigma)
        self.l_box         = float(l_box)
        self.center_solute = bool(center_solute)
        self.k_center      = float(k_center)
        self.temperature   = float(temperature)
        self.energy_cut    = float(energy_cut)

    def _batched_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : (B, N, 3) — full system (particle 0 = solute)
        returns : (B,)
        """
        B, N, _ = coords.shape
        L = 2.0 * self.l_box

        # Pairwise displacements with minimum-image convention
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)    # (B, N, N, 3)
        diff = diff - L * torch.round(diff / L)
        r2   = (diff ** 2).sum(-1)                          # (B, N, N)

        idx      = torch.triu_indices(N, N, offset=1, device=coords.device)
        r2_pairs = r2[:, idx[0], idx[1]].clamp(min=1e-12)  # (B, n_pairs)
        U = (self.epsilon * (self.sigma ** 2 / r2_pairs) ** 6).sum(-1)  # (B,)

        if self.center_solute:
            U = U + self.k_center * (coords[:, 0, :] ** 2).sum(-1)

        return U

    def _clip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        cut = torch.tensor(self.energy_cut, device=energy.device, dtype=energy.dtype)
        return torch.where(
            energy > cut,
            cut + torch.log1p((energy - cut).clamp(min=0.0)),
            energy,
        )

    def log_prob(self, x_solvent: torch.Tensor) -> torch.Tensor:
        """
        x_solvent : (B, 108)  Cartesian coordinates of 36 solvent particles.
        Returns   : (B,)  unnormalised log p(x) = -U(x)/T
        """
        B      = x_solvent.shape[0]
        device = x_solvent.device
        dtype  = x_solvent.dtype

        solute  = torch.zeros(B, 1, 3, device=device, dtype=dtype)
        solvent = x_solvent.view(B, self.n_solvent, 3)
        coords  = torch.cat([solute, solvent], dim=1)           # (B, 37, 3)

        energy = self._batched_energy(coords)
        energy = self._clip_energy(energy)
        return -energy / self.temperature

    def performance_metrics(
        self,
        samples: torch.Tensor,
        log_w: torch.Tensor,
        log_q_fn=None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        if samples is None or log_w is None:
            return {}
        valid     = torch.isfinite(log_w)
        log_w_v   = log_w[valid]
        samples_v = samples[valid]
        if len(log_w_v) == 0:
            return {"ess": 0.0, "mean_log_w": float("nan"), "log_Z": float("nan")}
        n      = float(len(log_w_v))
        log_Z  = torch.logsumexp(log_w_v, dim=0) - torch.log(
            torch.tensor(n, device=log_w_v.device))
        w_norm = torch.softmax(log_w_v, dim=0)
        ess    = (1.0 / (w_norm ** 2).sum()).item()
        with torch.no_grad():
            log_p = self.log_prob(samples_v)
        mean_energy = (-log_p * self.temperature).mean().item()
        return {
            "ess":         ess,
            "mean_log_w":  log_w_v.mean().item(),
            "log_Z":       log_Z.item(),
            "mean_energy": mean_energy,
        }

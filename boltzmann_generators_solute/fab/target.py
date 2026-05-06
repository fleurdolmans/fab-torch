"""
SoluteTarget2D — FAB TargetDistribution for the 2D solute-in-LJ-bath system.

The flow models 36 solvent particles in Cartesian space (72D). The solute is
fixed at the origin. This target:
  1. Prepends the fixed solute at (0, 0).
  2. Evaluates the batched, differentiable energy of the full 37-particle system.
  3. Returns -energy / temperature as the unnormalised log probability.

Energy function exactly mirrors SoluteSimulation2D (purely repulsive LJ +
soft harmonic walls + optional harmonic centering on the solute), re-implemented
in vectorised PyTorch so it supports autograd and batched evaluation.
"""

from typing import Dict, Optional

import torch


import abc


from typing import Callable, Tuple, Mapping, Any, Iterator


LogProbFunc = Callable[[torch.Tensor], torch.Tensor]


class Distribution(abc.ABC):
    """Used for distributions that have a defined sampling and log probability function."""

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, shape: Tuple) -> torch.Tensor:
        """Returns samples from the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of a single sample."""
        raise NotImplementedError


class Model(object):
    @abc.abstractmethod
    def loss(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def get_iter_info(self) -> Mapping[str, Any]:
        """Return information from latest loss iteration, for use in logging."""
        raise NotImplementedError

    def get_eval_info(self, outer_batch_size: int, inner_batch_size: int) -> Mapping[str, Any]:
        """Evaluate the model at the current point in training. This is useful for more expensive
        evaluation metrics than what is computed in get_iter_info."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Returns the tunable parameters of the model for use inside the train loop. This is
        required for gradient norm clipping."""

    def save(self, file_path) -> None:
        """Save model to file_path."""
        raise NotImplementedError

    def load(self, file_path, map_location) -> None:
        """Load model from file_path."""
        raise NotImplementedError

# from fab.target_distributions.base import TargetDistribution

class TargetDistribution(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(
        self,
        samples: torch.Tensor,
        log_w: torch.Tensor,
        log_q_fn: Optional[LogProbFunc] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.
            batch_size: If performance metrics are aggregated over many points that require network
                forward passes, batch_size ensures that the forward passes don't overload GPU
                memory by doing all the points together.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError

    def sample(self, shape):
        raise NotImplementedError


class SoluteTarget2D(TargetDistribution):
    """
    FAB target for the 2D solute-in-LJ-bath system.

    Parameters
    ----------
    n_solvent    : int   — number of solvent particles (default 36)
    epsilon      : float — LJ energy scale
    sigma        : float — LJ length scale (nm); particles repel below ~sigma
    l_box        : float — soft wall activates for |x|, |y| > l_box
    k_box        : float — soft-wall spring constant
    center_solute: bool  — whether to apply harmonic centering on particle 0
    k_center     : float — centering spring constant
    temperature  : float — temperature in reduced units (default 1.0)
    energy_cut   : float — energies above this are log-compressed
    energy_max   : float — hard cap before log-compression
    """

    def __init__(
        self,
        n_solvent: int = 36,
        epsilon: float = 1.0,
        sigma: float = 1.1,
        l_box: float = 5.0,
        k_box: float = 100.0,
        center_solute: bool = True,
        k_center: float = 20.0,
        temperature: float = 1.0,
        energy_cut: float = 1.0e4,
        energy_max: float = 1.0e10,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.n_particles = self.n_solvent + 1  # 1 fixed solute + n_solvent
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.l_box = float(l_box)
        self.k_box = float(k_box)
        self.center_solute = bool(center_solute)
        self.k_center = float(k_center)
        self.temperature = float(temperature)
        self.energy_cut = float(energy_cut)
        self.energy_max = float(energy_max)

    # ------------------------------------------------------------------
    # Batched, differentiable energy (mirrors SoluteSimulation2D exactly)
    # ------------------------------------------------------------------

    def _batched_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : (B, N, 2)  full configuration, particle 0 is the solute

        Returns
        -------
        energy : (B,)
        """
        B, N, _ = coords.shape

        # Pairwise repulsive LJ  (epsilon * (sigma^2 / r^2)^6)
        # Only upper-triangle pairs to avoid double counting.
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 2)
        r2   = (diff * diff).sum(-1)                        # (B, N, N)
        idx  = torch.triu_indices(N, N, offset=1, device=coords.device)
        r2_pairs = r2[:, idx[0], idx[1]]                   # (B, n_pairs)
        # Clamp to avoid division by zero; high energy handles the singularity.
        r2_pairs = r2_pairs.clamp(min=1e-12)
        U_lj = (self.epsilon * (self.sigma ** 2 / r2_pairs) ** 6).sum(-1)  # (B,)

        # Soft harmonic walls:  k_box * sum relu(|x| - l_box)^2
        excess = torch.relu(coords.abs() - self.l_box)      # (B, N, 2)
        U_wall = self.k_box * excess.pow(2).sum(dim=(1, 2))  # (B,)

        energy = U_lj + U_wall

        # Optional harmonic centering on the solute (particle index 0)
        if self.center_solute:
            energy = energy + self.k_center * (coords[:, 0, :] ** 2).sum(-1)

        return energy

    def _clip_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Soft-cap large energies to prevent -inf log probs during AIS.
        Values above energy_cut are replaced by energy_cut + log1p(excess).
        """
        cut = torch.tensor(self.energy_cut, device=energy.device, dtype=energy.dtype)
        clipped = torch.where(
            energy > cut,
            cut + torch.log1p((energy - cut).clamp(min=0.0)),
            energy,
        )
        return clipped

    # ------------------------------------------------------------------
    # TargetDistribution interface
    # ------------------------------------------------------------------

    def log_prob(self, x_solvent: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log probability  log p(x) = -U(x) / T.

        Parameters
        ----------
        x_solvent : (B, 72)  Cartesian coordinates of the 36 solvent particles.
                              The solute is implicitly fixed at (0, 0).

        Returns
        -------
        log_p : (B,)
        """
        B = x_solvent.shape[0]
        device, dtype = x_solvent.device, x_solvent.dtype

        # Prepend fixed solute at origin → (B, 37, 2)
        solute  = torch.zeros(B, 1, 2, device=device, dtype=dtype)
        solvent = x_solvent.view(B, self.n_solvent, 2)
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
        """
        Returns ESS, mean log-weight, and mean energy from weighted samples.
        """
        if samples is None or log_w is None:
            return {}

        valid = torch.isfinite(log_w)
        log_w_v = log_w[valid]
        samples_v = samples[valid]

        if len(log_w_v) == 0:
            return {"ess": 0.0, "mean_log_w": float("nan"), "log_Z": float("nan")}

        n = float(len(log_w_v))
        log_Z = torch.logsumexp(log_w_v, dim=0) - torch.log(torch.tensor(n, device=log_w_v.device))
        w_norm = torch.softmax(log_w_v, dim=0)
        ess = (1.0 / (w_norm ** 2).sum()).item()

        # Mean energy (unnormalised, -log_p * T)
        with torch.no_grad():
            log_p = self.log_prob(samples_v)
        mean_energy = (-log_p * self.temperature).mean().item()

        return {
            "ess": ess,
            "mean_log_w": log_w_v.mean().item(),
            "log_Z": log_Z.item(),
            "mean_energy": mean_energy,
        }


class SoluteFullTarget2D(SoluteTarget2D):
    """
    FAB target for RealNVP, which models all 37 particles in 74D.

    Unlike SoluteTarget2D (which fixes the solute at the origin and accepts
    72D solvent-only input), this class accepts the full 74D vector where
    particle 0 is the solute. The centering penalty in the energy function
    naturally pushes the solute toward the origin during training.

    Parameters
    ----------
    Same as SoluteTarget2D.
    """

    def log_prob(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        Unnormalised log probability  log p(x) = -U(x) / T.

        Parameters
        ----------
        x_full : (B, 74)  Cartesian coordinates of all 37 particles;
                          particle 0 is the solute.

        Returns
        -------
        log_p : (B,)
        """
        B = x_full.shape[0]
        coords = x_full.view(B, 37, 2)
        energy = self._batched_energy(coords)
        energy = self._clip_energy(energy)
        return -energy / self.temperature

    def performance_metrics(
        self,
        samples: torch.Tensor,
        log_w: torch.Tensor,
        log_q_fn=None,
        batch_size=None,
        **kwargs,
    ):
        if samples is None or log_w is None:
            return {}
        valid = torch.isfinite(log_w)
        log_w_v, samples_v = log_w[valid], samples[valid]
        if len(log_w_v) == 0:
            return {"ess": 0.0, "mean_log_w": float("nan"), "log_Z": float("nan")}
        n = float(len(log_w_v))
        log_Z = torch.logsumexp(log_w_v, 0) - torch.log(
            torch.tensor(n, device=log_w_v.device)
        )
        w_norm = torch.softmax(log_w_v, 0)
        ess = (1.0 / (w_norm ** 2).sum()).item()
        with torch.no_grad():
            log_p = self.log_prob(samples_v)
        mean_energy = (-log_p * self.temperature).mean().item()
        return {
            "ess": ess,
            "mean_log_w": log_w_v.mean().item(),
            "log_Z": log_Z.item(),
            "mean_energy": mean_energy,
        }

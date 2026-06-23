import numpy as np
import torch
import torch.nn as nn
from .. import utils as utils


# ===========================================================================
# Spline coupling flow for the 2-D solute system
# ===========================================================================
import math as _math

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


class BoxUniformPrior:
    """Uniform prior on [-bound, bound]^dim."""

    def __init__(self, dim, bound):
        self.dim = int(dim)
        self.bound = float(bound)
        self._log_prob_val = -self.dim * _math.log(2.0 * self.bound)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.rand(*shape, self.dim) * (2 * self.bound) - self.bound

    def log_prob(self, z):
        return torch.full(
            (z.shape[0],), self._log_prob_val, device=z.device, dtype=z.dtype
        )


class GaussianPrior:
    """Isotropic Gaussian prior N(0, sigma^2 * I_dim)."""

    def __init__(self, dim, sigma=1.0):
        self.dim = int(dim)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * self.dim * (_math.log(2 * _math.pi) + 2 * _math.log(sigma))

    def sample(self, shape, device=None):
        if isinstance(shape, int):
            shape = (shape,)
        t = torch.randn(*shape, self.dim) * self.sigma
        return t.to(device) if device is not None else t

    def log_prob(self, z):
        return -0.5 * (z / self.sigma).pow(2).sum(-1) + self._log_norm


class SplineConditioner2D(nn.Module):
    """MLP that maps frozen-particle coordinates to spline parameters for active particles."""

    def __init__(self, n_frozen_coords, n_active_coords, num_bins, n_nodes, n_hidden):
        super().__init__()
        self.params_per_coord = 3 * num_bins - 1
        self.n_active_coords = n_active_coords
        self.num_bins = num_bins

        layers = []
        in_dim = n_frozen_coords
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_nodes), nn.ReLU()]
            in_dim = n_nodes
        final = nn.Linear(in_dim, n_active_coords * self.params_per_coord)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x_frozen):
        B = x_frozen.shape[0]
        return self.net(x_frozen).view(B, self.n_active_coords, self.params_per_coord)



class DeepSetsConditioner2D(nn.Module):
    """
    DeepSets conditioner: permutation-invariant over frozen particles.

    Architecture
    ------------
    phi (encoder) : shared MLP mapping each frozen particle's 2D position to a
                    phi_dim-dimensional feature vector.
    Aggregation   : sum over all frozen particles → permutation-invariant context h.
    rho (decoder) : MLP mapping h to spline parameters for all active coordinates.

    Same external interface as SplineConditioner2D so it is a drop-in replacement
    inside SoluteSplineCoupling.
    """

    def __init__(self, _n_frozen_coords, n_active_coords, num_bins, n_nodes, n_hidden):
        super().__init__()
        self.params_per_coord = 3 * num_bins - 1
        self.n_active_coords = n_active_coords
        self.num_bins = num_bins
        phi_dim = n_nodes

        # Per-particle encoder phi: (2,) -> (phi_dim,)
        phi_layers = [nn.Linear(2, n_nodes), nn.ReLU()]
        for _ in range(n_hidden - 1):
            phi_layers += [nn.Linear(n_nodes, n_nodes), nn.ReLU()]
        phi_layers.append(nn.Linear(n_nodes, phi_dim))
        self.phi = nn.Sequential(*phi_layers)

        # Output MLP rho: (phi_dim,) -> (n_active_coords * params_per_coord,)
        rho_layers = [nn.Linear(phi_dim, n_nodes), nn.ReLU()]
        for _ in range(n_hidden - 1):
            rho_layers += [nn.Linear(n_nodes, n_nodes), nn.ReLU()]
        final = nn.Linear(n_nodes, n_active_coords * self.params_per_coord)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        rho_layers.append(final)
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x_frozen_flat):
        B = x_frozen_flat.shape[0]
        n_frozen = x_frozen_flat.shape[1] // 2
        x_frozen = x_frozen_flat.view(B, n_frozen, 2)                    # (B, N_f, 2)
        phi_out = self.phi(x_frozen.reshape(B * n_frozen, 2))            # (B*N_f, phi_dim)
        h = phi_out.view(B, n_frozen, -1).sum(dim=1)                     # (B, phi_dim) — perm-invariant
        out = self.rho(h)                                                 # (B, n_active_coords * ppc)
        return out.view(B, self.n_active_coords, self.params_per_coord)


class SoluteSplineCoupling(nn.Module):
    """Rational-quadratic spline coupling layer for 2-D particle systems."""

    def __init__(self, conditioner, frozen_idx, active_idx, num_bins,
                 tail_bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3,
                 min_derivative=1e-3):
        super().__init__()
        self.conditioner = conditioner
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    def _get_params(self, x):
        x_frozen_flat = x[:, self.frozen_idx, :].reshape(x.shape[0], -1)
        params = self.conditioner(x_frozen_flat)   # (B, N_A*2, ppc)
        B2, _N_A2, ppc = params.shape
        params = params.view(B2, -1, 2, ppc)       # (B, N_A, 2, ppc)
        nb = self.num_bins
        return params[..., :nb], params[..., nb:2*nb], params[..., 2*nb:]

    def forward(self, x):
        """x: (B, N, 2) — returns (y, logabsdet) where logabsdet is (B,)."""
        widths, heights, derivatives = self._get_params(x)
        x_active = x[:, self.active_idx, :]        # (B, N_A, 2)
        y_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        y = x.clone()
        y[:, self.active_idx, :] = y_active
        return y, logabsdet.sum(dim=(-1, -2))

    def inverse(self, y):
        """y: (B, N, 2) — returns (x, logabsdet) where logabsdet is (B,)."""
        widths, heights, derivatives = self._get_params(y)
        y_active = y[:, self.active_idx, :]
        x_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        x = y.clone()
        x[:, self.active_idx, :] = x_active
        return x, logabsdet.sum(dim=(-1, -2))


class SoluteSplineFlow(nn.Module):
    """
    Neural spline flow for the 2-D solute-LJ-bath system.

    Same external API as RealNVP: loss_ML, loss_KL, generator, inverse_generator.
    Additionally exposes loss_KL(w_overlap=...) to penalise particle overlaps.
    """

    def __init__(self, layers, prior, system, n_particles):
        super().__init__()
        self.coupling_layers = nn.ModuleList(layers)
        self.prior = prior
        self.system = system
        self.n_particles = int(n_particles)
        self.sys_dim = (n_particles, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_particle(self, flat):
        return flat.view(flat.shape[0], self.n_particles, 2)

    def _to_flat(self, particle):
        return particle.reshape(particle.shape[0], -1)

    def forward_map(self, x):
        """x: (B, N, 2) — returns (z, sum_logdet)."""
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.coupling_layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse_map(self, z):
        """z: (B, N, 2) — returns (x, sum_logdet)."""
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.coupling_layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def loss_ML(self, batch_x, weighted=False):
        """Maximum-likelihood loss. batch_x: (B, N*2) in Cartesian coordinates."""
        x = self._to_particle(batch_x)
        z, logdet_fwd = self.forward_map(x)
        log_pz = self.prior.log_prob(self._to_flat(z))
        return -(log_pz + logdet_fwd).mean()
    
    def _add_fixed_solute(self, solvent_flat):
        B = solvent_flat.shape[0]
        solvent = solvent_flat.view(B, self.n_particles, 2)

        solute = torch.zeros(
            B, 1, 2,
            device=solvent.device,
            dtype=solvent.dtype,
        )

        full = torch.cat([solute, solvent], dim=1)  # (B, n_particles+1, 2)
        return full.reshape(B, -1)

    def loss_KL(self, batch_z, weighted=False, energy_cap=1e3, w_overlap=0.0, cutoff_factor=0.9):
        """
        Reverse-KL loss. batch_z: (B, N*2) samples from prior.

        Parameters
        ----------
        w_overlap : float
            Weight for the overlap penalty term (default 0 = disabled).
        cutoff_factor : float
            Factor for determining the cutoff distance for overlap penalty.
        """
        z = self._to_particle(batch_z)
        x, logdet_inv = self.inverse_map(z)
        x_cart = self._to_flat(x)

        x_full_flat = self._add_fixed_solute(x_cart)
        energies = torch.stack([
            self.system.get_energy(x_full_flat[i].reshape(self.n_particles + 1, 2))
            for i in range(x_full_flat.shape[0])
        ])

        def soft_cap_with_floor(U, cap, alpha=0.05):
            excess = torch.clamp(U - cap, min=0.0)
            return torch.where(
                U < cap,
                U,
                cap + alpha * excess + (1.0 - alpha) * torch.log1p(excess)
            )
        u_x = energies if energy_cap is None else soft_cap_with_floor(energies, energy_cap, alpha=0.05)
        loss = (u_x - logdet_inv).mean()

        if w_overlap > 0.0:
            penalty = utils.overlap_penalty(
                x_full_flat, (self.n_particles + 1, 2), self.system.sigma, cutoff_factor=cutoff_factor)
            loss = loss + w_overlap * penalty
        return loss

    def loss_overlap(self, batch_z, cutoff_factor=0.9):
        """
        Overlap penalty on flow-generated samples, decoupled from the KL loss.

        Parameters
        ----------
        batch_z : (B, N*2) tensor — prior samples
        cutoff_factor : float

        Returns
        -------
        penalty : scalar tensor
        """
        z = self._to_particle(batch_z)
        x, _ = self.inverse_map(z)
        x_full_flat = self._add_fixed_solute(self._to_flat(x))
        return utils.overlap_penalty(
            x_full_flat, (self.n_particles + 1, 2), self.system.sigma,
            cutoff_factor=cutoff_factor,
        )

    # ------------------------------------------------------------------
    # Generator interface (same as RealNVP)
    # ------------------------------------------------------------------
    def generator(self, z):
        """z: (B, N*2) latent → x: (B, N*2) Cartesian."""
        x_particle, logdet_inv = self.inverse_map(self._to_particle(z))
        return self._to_flat(x_particle), logdet_inv

    def inverse_generator(self, x):
        """x: (B, N*2) Cartesian → z: (B, N*2) latent."""
        z_particle, logdet_fwd = self.forward_map(self._to_particle(x))
        return self._to_flat(z_particle), logdet_fwd




import math
from typing import Optional, Sequence

import numpy as np
import torch
import normflows as nf
from torch import nn

from fab.wrappers.normflows import WrappedNormFlowModel
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

from ..make_base.base_LJ import UnitTorusPeriodicBase, HardCoreRandomPlacementBase, ExactTorusSoluteMixtureBase, GaussianUnitTorusBase, ShellTorusSoluteBase, MultiShellAutoregressiveTorusBase

import math
import torch
from torch import nn


def wrap_centered_box(x: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    return torch.remainder(x + 0.5 * L, L) - 0.5 * L


def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


class WrappedNormalCentered1D:
    """
    1D wrapped normal on centered interval [-L/2, L/2).
    """

    def __init__(self, sigma: float, box_length: float, image_range: int = 1):
        self.sigma = float(sigma)
        self.box_length = float(box_length)
        self.image_range = int(image_range)

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(self.sigma, device=x.device, dtype=x.dtype)
        L = torch.as_tensor(self.box_length, device=x.device, dtype=x.dtype)

        offsets = torch.arange(
            -self.image_range,
            self.image_range + 1,
            device=x.device,
            dtype=x.dtype,
        )

        norm_const = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma)
        z = x.unsqueeze(-1) - mu.unsqueeze(-1) + offsets * L
        log_terms = norm_const - 0.5 * (z / sigma) ** 2
        return torch.logsumexp(log_terms, dim=-1)

    def sample(self, mu: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(self.sigma, device=mu.device, dtype=mu.dtype)
        eps = sigma * torch.randn_like(mu)
        return wrap_centered_box(mu + eps, self.box_length)


class FCCGaussianCenteredPeriodicBase(nn.Module):
    """
    Exact factorized wrapped-Gaussian base on centered periodic coordinates
    z in [-L/2, L/2)^(3*n_solvent).

    Intended for:
      - SoluteCenteredSolventTransform
      - lj_torus_spline_noF_v2

    Each solvent particle is centered around an assigned FCC site in the same
    solute-centered relative coordinate chart.
    """

    def __init__(
        self,
        solvent_fcc_rel_nm,   # shape (n_solvent, 3), centered relative coords
        box_length_nm: float,
        sigma_nm: float = 0.03,
        image_range: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        solvent_fcc_rel_nm = torch.as_tensor(solvent_fcc_rel_nm, device=device, dtype=dtype)
        if solvent_fcc_rel_nm.ndim != 2 or solvent_fcc_rel_nm.shape[1] != 3:
            raise ValueError("solvent_fcc_rel_nm must have shape (n_solvent, 3)")

        solvent_fcc_rel_nm = wrap_centered_box(solvent_fcc_rel_nm, box_length_nm)

        self.n_solvent = int(solvent_fcc_rel_nm.shape[0])
        self.dim = 3 * self.n_solvent
        self.shape = (self.dim,)
        self.box_length_nm = float(box_length_nm)
        self.sigma_nm = float(sigma_nm)
        self.image_range = int(image_range)

        self.register_buffer("centers", solvent_fcc_rel_nm.reshape(1, self.dim))
        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))

        self._wn = WrappedNormalCentered1D(
            sigma=self.sigma_nm,
            box_length=self.box_length_nm,
            image_range=self.image_range,
        )

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def sample(self, n: int) -> torch.Tensor:
        mu = self.centers.to(device=self.device, dtype=self.dtype).expand(n, -1)
        z = self._wn.sample(mu)
        z = wrap_centered_box(z, self.box_length_nm)
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z = wrap_centered_box(z, self.box_length_nm)
        mu = self.centers.to(device=z.device, dtype=z.dtype).expand_as(z)
        lp = self._wn.log_prob(z, mu)
        return lp.sum(dim=-1)

    def forward(self, n: int):
        z = self.sample(n)
        log_q = self.log_prob(z)
        return z, log_q


# ============================================================
# Basic helpers
# ============================================================

def _cartesian_partition_cycle():
    return [
        [0],
        [1],
        [2],
        [0, 1],
        [1, 2],
        [0, 2],
    ]


def _cartesian_partition_mask_bool(dim: int, axes_to_transform):
    if dim % 3 != 0:
        raise ValueError(f"Expected dim divisible by 3, got dim={dim}")
    mask = torch.zeros(dim, dtype=torch.bool)
    for ax in axes_to_transform:
        mask[ax::3] = True
    return mask


def _cartesian_partition_mask_float(dim: int, axes_to_transform):
    return _cartesian_partition_mask_bool(dim, axes_to_transform).float()


def torus_project(x: torch.Tensor, box_length: float) -> torch.Tensor:
    two_pi = torch.as_tensor(2.0 * math.pi, device=x.device, dtype=x.dtype)
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    angle = two_pi * x / L
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def mic_unit(du: torch.Tensor) -> torch.Tensor:
    return du - torch.round(du)


def mic(dx: torch.Tensor, box_length: torch.Tensor | float) -> torch.Tensor:
    return dx - box_length * torch.round(dx / box_length)


def build_fcc_sites_unit(n_total_sites: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    FCC lattice sites in unit-cube coordinates [0,1)^3.

    Requires:
        n_total_sites = 4 * n_cells^3
    """
    n_total_sites = int(n_total_sites)
    n_cells_float = (n_total_sites / 4.0) ** (1.0 / 3.0)
    n_cells = round(n_cells_float)
    if 4 * n_cells**3 != n_total_sites:
        raise ValueError(
            f"FCC requires n_total_sites = 4 * n_cells^3, got {n_total_sites}"
        )

    basis = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        device=device,
        dtype=dtype,
    )

    pts = []
    inv_n = 1.0 / n_cells
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell = torch.tensor([i, j, k], device=device, dtype=dtype)
                pts.append((cell[None, :] + basis) * inv_n)

    return wrap_unit(torch.cat(pts, dim=0))  # (n_total_sites, 3)



def assign_solvent_sites_from_solute(
    solute_positions_unit: torch.Tensor,
    n_solvent: int,
) -> torch.Tensor:
    """
    Build a full FCC lattice for n_solute + n_solvent sites, assign the FCC sites
    closest to the provided solute positions to the solute, and return the remaining
    sites for solvent.

    Parameters
    ----------
    solute_positions_unit : (n_solute, 3)
        Fixed solute positions in unit coordinates [0,1).
    n_solvent : int

    Returns
    -------
    solvent_sites_unit : (n_solvent, 3)
        Remaining FCC sites assigned to solvent.
    """
    solute_positions_unit = wrap_unit(solute_positions_unit)
    n_solute = int(solute_positions_unit.shape[0])
    n_total = n_solute + int(n_solvent)

    fcc_sites = build_fcc_sites_unit(
        n_total_sites=n_total,
        device=solute_positions_unit.device,
        dtype=solute_positions_unit.dtype,
    )  # (n_total, 3)

    remaining_mask = torch.ones(n_total, device=fcc_sites.device, dtype=torch.bool)
    assigned_solute_site_indices = []

    # greedy nearest-site assignment for solutes
    for s in range(n_solute):
        candidates = fcc_sites[remaining_mask]  # (M,3)
        cand_idx_global = torch.arange(n_total, device=fcc_sites.device)[remaining_mask]

        d = mic_unit(candidates - solute_positions_unit[s].view(1, 3))
        d = torch.linalg.norm(d, dim=-1)
        j_local = torch.argmin(d)
        j_global = cand_idx_global[j_local]

        assigned_solute_site_indices.append(j_global.item())
        remaining_mask[j_global] = False

    solvent_sites = fcc_sites[remaining_mask]
    if solvent_sites.shape[0] != n_solvent:
        raise RuntimeError(
            f"Expected {n_solvent} solvent sites, got {solvent_sites.shape[0]}"
        )
    return solvent_sites


class SoluteAwareFCCGaussianUnitTorusBase(nn.Module):
    """
    Exact wrapped-Gaussian torus base for v4, using FCC solvent sites that are
    consistent with an embedded fixed solute.

    Coordinates:
        flattened solvent unit-torus coordinates in [0,1)^(3*n_solvent)

    Construction:
      1. Build a full FCC lattice for (n_solute + n_solvent) total sites
      2. Assign FCC sites nearest to the provided solute positions to the solute
      3. Use the remaining FCC sites as solvent reference sites
      4. Put an independent wrapped Gaussian around each solvent site
    """

    def __init__(
        self,
        solute_positions_nm,
        box_length_nm: float,
        n_solvent: int,
        sigma_unit: float = 0.03,
        image_range: int = 1,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.shape = (self.dim,)
        self.sigma_unit = float(sigma_unit)
        self.image_range = int(image_range)
        self.box_length_nm = float(box_length_nm)

        if self.sigma_unit <= 0.0:
            raise ValueError("sigma_unit must be > 0")

        solute_positions_nm = torch.as_tensor(
            solute_positions_nm, device=device, dtype=dtype
        )
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")

        solute_positions_unit = wrap_unit(solute_positions_nm / self.box_length_nm)

        solvent_sites_unit = assign_solvent_sites_from_solute(
            solute_positions_unit=solute_positions_unit,
            n_solvent=self.n_solvent,
        )  # (n_solvent, 3)

        self.register_buffer("solvent_ref_unit", solvent_sites_unit.reshape(-1))
        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))

    def sample(self, n: int) -> torch.Tensor:
        mu = self.solvent_ref_unit.unsqueeze(0).expand(n, -1)
        eps = self.sigma_unit * torch.randn(
            n,
            self.dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        return wrap_unit(mu + eps)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z = wrap_unit(z)
        mu = self.solvent_ref_unit.unsqueeze(0).to(device=z.device, dtype=z.dtype)

        shifts = torch.arange(
            -self.image_range,
            self.image_range + 1,
            device=z.device,
            dtype=z.dtype,
        )  # (K,)

        diff = z.unsqueeze(-1) - (mu.unsqueeze(-1) + shifts.view(1, 1, -1))  # (B,D,K)

        var = self.sigma_unit ** 2
        log_comp = -0.5 * (diff ** 2) / var - 0.5 * math.log(2.0 * math.pi * var)

        logp_dim = torch.logsumexp(log_comp, dim=-1)  # (B,D)
        return logp_dim.sum(dim=-1)

    def forward(self, n: int):
        z = self.sample(n)
        log_q = self.log_prob(z)
        return z, log_q

# ============================================================
# Base for torus flow
# ============================================================

class UniformUnitTorusBase(nn.Module):
    """
    Uniform base on [0,1)^dim.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.shape = (self.dim,)
        self.register_buffer("_dummy", torch.zeros(1))

    def sample(self, n: int):
        return torch.rand(
            n,
            self.dim,
            device=self._dummy.device,
            dtype=self._dummy.dtype,
        )

    def log_prob(self, z: torch.Tensor):
        z = wrap_unit(z)
        return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    def forward(self, n: int):
        z = self.sample(n)
        log_q = self.log_prob(z)
        return z, log_q

    __call__ = forward


# ============================================================
# lj_coupling_spline
# ============================================================

class ParticleConditioner(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, n_hidden=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        final = nn.Linear(d, out_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParticleTorusSplineCoupling(nn.Module):
    def __init__(
        self,
        n_solvent: int,
        box_length: float,
        update_axes,
        num_bins: int = 8,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)
        self.update_axes = list(update_axes)
        self.keep_axes = [a for a in [0, 1, 2] if a not in self.update_axes]
        self.num_bins = int(num_bins)

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        d_keep = len(self.keep_axes)
        d_update = len(self.update_axes)
        params_per_dim = 3 * self.num_bins - 1

        self.conditioner = ParticleConditioner(
            in_dim=2 * d_keep,
            out_dim=d_update * params_per_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        )

    def wrap(self, x):
        return torch.remainder(x, self.box_length)

    def center(self, x):
        return self.wrap(x) - self.tail_bound

    def uncenter(self, x):
        return self.wrap(x + self.tail_bound)

    def forward(self, x):
        param = next(self.conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)

        x_a = x[..., self.update_axes]
        x_b = x[..., self.keep_axes]

        cond = torus_project(x_b, self.box_length)
        params = self.conditioner(cond)

        B, N, _ = params.shape
        d_update = len(self.update_axes)
        params = params.view(B, N, d_update, 3 * self.num_bins - 1)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        x_a_c = self.center(x_a)

        y_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_a_c,
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

        y_a = self.uncenter(y_a_c)
        y = x.clone()
        y[..., self.update_axes] = y_a
        y = self.wrap(y)

        return y, logabsdet.sum(dim=(-1, -2))

    def inverse(self, y):
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)

        y_a = y[..., self.update_axes]
        y_b = y[..., self.keep_axes]

        cond = torus_project(y_b, self.box_length)
        params = self.conditioner(cond)

        B, N, _ = params.shape
        d_update = len(self.update_axes)
        params = params.view(B, N, d_update, 3 * self.num_bins - 1)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        y_a_c = self.center(y_a)

        x_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a_c,
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

        x_a = self.uncenter(x_a_c)
        x = y.clone()
        x[..., self.update_axes] = x_a
        x = self.wrap(x)

        return x, logabsdet.sum(dim=(-1, -2))


class WrappedCustomFlow(nn.Module):
    """
    Flow over flattened solvent coordinates in [0, L).
    Base is expected on centered interval coordinates.
    """

    def __init__(self, layers, base, box_length: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base = base
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)
        self.event_shape = (base.shape[0],) if hasattr(base, "shape") else (base.dim,)

    def wrap_to_box(self, x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x, self.box_length)

    def center_to_interval(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wrap_to_box(x)
        return x - self.tail_bound

    def uncenter_from_interval(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrap_to_box(x + self.tail_bound)

    def _flat_to_particle(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] % 3 != 0:
            raise ValueError(f"Expected (B, 3N), got {tuple(x.shape)}")
        return x.view(x.shape[0], -1, 3)

    def _particle_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B, N, 3), got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1)

    def forward_map(self, x: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)

        z = self._flat_to_particle(x)
        logdet = torch.zeros(z.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld

        z = self._particle_to_flat(z)
        return z, logdet

    def inverse_map(self, z: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        z = z.to(device=param.device, dtype=param.dtype)

        x = self._flat_to_particle(z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld

        x = self._particle_to_flat(x)
        return x, logdet

    def forward_and_log_prob(self, x):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)

        z, logdet = self.forward_map(x)
        z_centered = self.center_to_interval(z)
        log_q = self.base.log_prob(z_centered) + logdet
        return z, log_q

    def log_prob(self, x):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)

        z, logdet = self.forward_map(x)
        z_centered = self.center_to_interval(z)
        return self.base.log_prob(z_centered) + logdet

    def sample_and_log_prob(self, shape):
        n = shape[0]
        z_centered, log_q = self.base(n)
        x = self.uncenter_from_interval(z_centered)
        x, inv_logdet = self.inverse_map(x)
        return x, log_q - inv_logdet

    def sample(self, shape):
        x, _ = self.sample_and_log_prob(shape)
        return x


# ============================================================
# lj_coupling_perm_equi
# ============================================================

class RBFFeatures(nn.Module):
    def __init__(self, n_rbf: int = 16, r_max: float = 2.0):
        super().__init__()
        centers = torch.linspace(0.0, r_max, n_rbf)
        widths = torch.full((n_rbf,), r_max / max(n_rbf - 1, 1))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, d: torch.Tensor):
        centers = self.centers.to(device=d.device, dtype=d.dtype)
        widths = self.widths.to(device=d.device, dtype=d.dtype).clamp_min(1e-6)

        c = centers.view(*([1] * d.ndim), -1)
        w = widths.view(*([1] * d.ndim), -1)
        return torch.exp(-0.5 * ((d.unsqueeze(-1) - c) / w) ** 2)


class PermEquiParticleConditioner(nn.Module):
    """
    Permutation-equivariant per-particle conditioner.
    """

    def __init__(
        self,
        d_keep: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        n_rbf: int = 16,
        rbf_max_dist: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_keep = int(d_keep)
        self.out_dim = int(out_dim)

        self.rbf = RBFFeatures(n_rbf=n_rbf, r_max=rbf_max_dist)

        local_layers = []
        d = d_keep
        for _ in range(n_hidden):
            local_layers.append(nn.Linear(d, hidden_dim))
            local_layers.append(nn.ReLU())
            if dropout > 0.0:
                local_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.local_embed = nn.Sequential(*local_layers)

        ss_layers = []
        d = n_rbf
        for _ in range(max(1, n_hidden - 1)):
            ss_layers.append(nn.Linear(d, hidden_dim))
            ss_layers.append(nn.ReLU())
            if dropout > 0.0:
                ss_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.ss_embed = nn.Sequential(*ss_layers)

        sol_layers = []
        d = n_rbf
        for _ in range(max(1, n_hidden - 1)):
            sol_layers.append(nn.Linear(d, hidden_dim))
            sol_layers.append(nn.ReLU())
            if dropout > 0.0:
                sol_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.sol_embed = nn.Sequential(*sol_layers)

        final_in = hidden_dim + hidden_dim + hidden_dim
        final_layers = []
        d = final_in
        for _ in range(n_hidden):
            final_layers.append(nn.Linear(d, hidden_dim))
            final_layers.append(nn.ReLU())
            if dropout > 0.0:
                final_layers.append(nn.Dropout(dropout))
            d = hidden_dim

        final = nn.Linear(d, out_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        final_layers.append(final)
        self.out_net = nn.Sequential(*final_layers)

    def forward(self, x_keep: torch.Tensor, pos_all: torch.Tensor, solute_ref: torch.Tensor):
        B, N, _ = pos_all.shape

        h_local = self.local_embed(x_keep)

        diff = pos_all[:, :, None, :] - pos_all[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)

        rbf_ss = self.rbf(dist)
        eye = torch.eye(N, device=pos_all.device, dtype=pos_all.dtype).view(1, N, N, 1)
        rbf_ss = rbf_ss * (1.0 - eye)
        pooled_ss = rbf_ss.sum(dim=2)
        h_ss = self.ss_embed(pooled_ss)

        d_sol = torch.linalg.norm(pos_all - solute_ref, dim=-1)
        rbf_sol = self.rbf(d_sol)
        h_sol = self.sol_embed(rbf_sol)

        h = torch.cat([h_local, h_ss, h_sol], dim=-1)
        return self.out_net(h)


class PermEquiLJParticleSplineCoupling(nn.Module):
    def __init__(
        self,
        box_length: float,
        update_axes,
        num_bins: int = 8,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        n_rbf: int = 16,
        rbf_max_dist: float = 2.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)
        self.update_axes = list(update_axes)
        self.keep_axes = [a for a in [0, 1, 2] if a not in self.update_axes]
        self.num_bins = int(num_bins)

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        d_keep = len(self.keep_axes)
        d_update = len(self.update_axes)
        params_per_dim = 3 * self.num_bins - 1

        self.conditioner = PermEquiParticleConditioner(
            d_keep=d_keep,
            out_dim=d_update * params_per_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            n_rbf=n_rbf,
            rbf_max_dist=rbf_max_dist,
            dropout=dropout,
        )

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + self.tail_bound, self.box_length) - self.tail_bound

    def forward(self, x: torch.Tensor):
        param = next(self.conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        x = self.wrap_centered(x)

        x_a = x[..., self.update_axes]
        x_b = x[..., self.keep_axes]

        B, N, _ = x.shape
        d_update = len(self.update_axes)

        solute_ref = torch.zeros((B, 1, 3), device=param.device, dtype=param.dtype)
        params = self.conditioner(x_b, x, solute_ref)
        params = params.view(B, N, d_update, 3 * self.num_bins - 1)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        y_a, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_a,
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
        y[..., self.update_axes] = y_a
        y = self.wrap_centered(y)

        logdet = logabsdet.sum(dim=(-1, -2))
        return y, logdet

    def inverse(self, y: torch.Tensor):
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)
        y = self.wrap_centered(y)

        y_a = y[..., self.update_axes]
        y_b = y[..., self.keep_axes]

        B, N, _ = y.shape
        d_update = len(self.update_axes)

        solute_ref = torch.zeros((B, 1, 3), device=param.device, dtype=param.dtype)
        params = self.conditioner(y_b, y, solute_ref)
        params = params.view(B, N, d_update, 3 * self.num_bins - 1)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        x_a, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a,
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
        x[..., self.update_axes] = x_a
        x = self.wrap_centered(x)

        logdet = logabsdet.sum(dim=(-1, -2))
        return x, logdet


class WrappedPermEquiLJFlow(nn.Module):
    """
    Flow over flattened solvent relative coordinates.
    """

    def __init__(self, layers, base, box_length: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base = base
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)
        self.event_shape = (base.shape[0],) if hasattr(base, "shape") else (base.dim,)

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + self.tail_bound, self.box_length) - self.tail_bound

    def _flat_to_particle(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] % 3 != 0:
            raise ValueError(f"Expected (B, 3N), got {tuple(x.shape)}")
        return x.view(x.shape[0], -1, 3)

    def _particle_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B, N, 3), got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1)

    def forward_map(self, x: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        x = self.wrap_centered(x)

        z = self._flat_to_particle(x)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld

        z = self._particle_to_flat(z)
        z = self.wrap_centered(z)
        return z, logdet

    def inverse_map(self, z: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        z = z.to(device=param.device, dtype=param.dtype)
        z = self.wrap_centered(z)

        x = self._flat_to_particle(z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld

        x = self._particle_to_flat(x)
        x = self.wrap_centered(x)
        return x, logdet

    def log_prob(self, x):
        z, logdet = self.forward_map(x)
        return self.base.log_prob(z) + logdet

    def forward_and_log_prob(self, x):
        z, logdet = self.forward_map(x)
        return z, self.base.log_prob(z) + logdet

    def sample_and_log_prob(self, shape):
        n = shape[0]
        z, log_q = self.base(n)
        x, inv_logdet = self.inverse_map(z)
        return x, log_q - inv_logdet

    def sample(self, shape):
        x, _ = self.sample_and_log_prob(shape)
        return x


# ============================================================
# Periodic MIC-aware feature extractor
# ============================================================

class PeriodicSolventFeatureExtractor(nn.Module):
    """
    Per-solvent features on the unit torus [0,1)^3.

    Uses:
      - sin/cos embedding of current torus coords
      - solvent-solvent MIC distances
      - solvent-solute-site MIC distances
      - nearest-neighbor and crowding summaries
    """

    def __init__(
        self,
        box_length_nm: float,
        solute_positions_nm: torch.Tensor,
        n_rbf_ss: int = 16,
        n_rbf_su: int = 16,
        rbf_ss_max_nm: Optional[float] = None,
        rbf_su_max_nm: Optional[float] = None,
        crowding_alpha: float = 12.0,
    ):
        super().__init__()
        self.box_length_nm = float(box_length_nm)
        self.crowding_alpha = float(crowding_alpha)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")
        self.register_buffer("solute_positions_nm", solute_positions_nm)

        if rbf_ss_max_nm is None:
            rbf_ss_max_nm = 0.5 * self.box_length_nm
        if rbf_su_max_nm is None:
            rbf_su_max_nm = 0.5 * self.box_length_nm

        self.rbf_ss = RBFFeatures(n_rbf=n_rbf_ss, r_max=float(rbf_ss_max_nm))
        self.rbf_su = RBFFeatures(n_rbf=n_rbf_su, r_max=float(rbf_su_max_nm))

        self.feature_dim = 6 + 2 + n_rbf_ss + n_rbf_su

    def forward(self, u: torch.Tensor, channel_mask=None):
        """
        u: (B, N, 3) in [0,1)
        channel_mask: optional bool tensor broadcastable to (3,); the same
                      dimensions are zeroed in the stored solute positions so
                      that solute-solvent distances are computed in the same
                      frozen-coordinate subspace as the solvent positions.

        returns:
            feats: (B, N, F)
            d_ss:  (B, N, N) solvent-solvent distances in nm
        """
        B, N, _ = u.shape
        dtype = u.dtype
        device = u.device

        L = torch.as_tensor(self.box_length_nm, device=device, dtype=dtype)
        solute_nm = self.solute_positions_nm.to(device=device, dtype=dtype)
        solute_u = wrap_unit(solute_nm / L)   # (K, 3)
        if channel_mask is not None:
            cm = channel_mask.view(3).to(device=device, dtype=torch.bool)
            solute_u = solute_u.masked_fill(cm.view(1, 3), 0.0)

        angle = 2.0 * math.pi * u
        periodic_embed = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

        du_ss = mic_unit(u[:, :, None, :] - u[:, None, :, :])   # (B,N,N,3)
        dx_ss = L * du_ss
        d_ss = torch.linalg.norm(dx_ss, dim=-1)                 # (B,N,N)

        eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
        d_ss_masked = d_ss.masked_fill(eye, float("inf"))

        d_nn = d_ss_masked.min(dim=-1).values
        crowding = torch.exp(-self.crowding_alpha * d_ss).masked_fill(eye, 0.0).sum(dim=-1)
        ss_rbf = self.rbf_ss(d_ss).masked_fill(eye.unsqueeze(-1), 0.0).sum(dim=2)

        du_su = mic_unit(u[:, :, None, :] - solute_u.view(1, 1, -1, 3))
        dx_su = L * du_su
        d_su = torch.linalg.norm(dx_su, dim=-1)
        su_rbf = self.rbf_su(d_su).sum(dim=2)

        scalar = torch.stack([d_nn, crowding], dim=-1)
        feats = torch.cat([periodic_embed, scalar, ss_rbf, su_rbf], dim=-1)
        return feats, d_ss


# ============================================================
# Cross-group feature extractor for particle-group split coupling
# ============================================================

class CrossGroupFeatureExtractor(nn.Module):
    """
    Per-active-particle features using own frozen coordinates AND frozen group's full 3D positions.

    Key improvement over PeriodicSolventFeatureExtractor:
      - Cross-group distances d_ab use the frozen group's ACTUAL active-dimension coordinate,
        providing richer conditioning than the 2D-projected within-group distances.

    forward(u_active, u_frozen):
        u_active: (B, N_A, 3)  — active coord zeroed, own frozen coords present
        u_frozen: (B, N_F, 3)  — ALL 3 coords available (not coord-masked)
    Returns:
        feats: (B, N_A, F)
        d_aa:  (B, N_A, N_A) within-active MIC distances in nm
    """

    def __init__(
        self,
        box_length_nm: float,
        solute_positions_nm: torch.Tensor,
        n_rbf_aa: int = 16,
        n_rbf_ab: int = 16,
        n_rbf_su: int = 16,
        rbf_aa_max_nm: Optional[float] = None,
        rbf_ab_max_nm: Optional[float] = None,
        rbf_su_max_nm: Optional[float] = None,
        crowding_alpha: float = 12.0,
    ):
        super().__init__()
        self.box_length_nm = float(box_length_nm)
        self.crowding_alpha = float(crowding_alpha)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")
        self.register_buffer("solute_positions_nm", solute_positions_nm)

        if rbf_aa_max_nm is None:
            rbf_aa_max_nm = 0.5 * self.box_length_nm
        if rbf_ab_max_nm is None:
            rbf_ab_max_nm = 0.5 * self.box_length_nm
        if rbf_su_max_nm is None:
            rbf_su_max_nm = 0.5 * self.box_length_nm

        self.rbf_aa = RBFFeatures(n_rbf=n_rbf_aa, r_max=float(rbf_aa_max_nm))
        self.rbf_ab = RBFFeatures(n_rbf=n_rbf_ab, r_max=float(rbf_ab_max_nm))
        self.rbf_su = RBFFeatures(n_rbf=n_rbf_su, r_max=float(rbf_su_max_nm))

        self.feature_dim = 6 + 2 + n_rbf_aa + n_rbf_ab + n_rbf_su

    def forward(self, u_active: torch.Tensor, u_frozen: torch.Tensor):
        """
        u_active: (B, N_A, 3) in [0,1) — active coord zeroed
        u_frozen: (B, N_F, 3) in [0,1) — ALL 3 coords, NOT coord-masked

        Returns:
            feats: (B, N_A, F)
            d_aa:  (B, N_A, N_A) within-active MIC distances in nm
        """
        B, N_A, _ = u_active.shape
        device = u_active.device
        dtype = u_active.dtype

        L = torch.as_tensor(self.box_length_nm, device=device, dtype=dtype)

        # Own periodic embedding (uses frozen coords of active particles)
        angle = 2.0 * math.pi * u_active
        periodic_embed = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B, N_A, 6)

        # Within-active-group MIC distances (active coord zeroed → 2D projected)
        du_aa = mic_unit(u_active[:, :, None, :] - u_active[:, None, :, :])   # (B,N_A,N_A,3)
        dx_aa = L * du_aa
        d_aa = torch.linalg.norm(dx_aa, dim=-1)                                # (B,N_A,N_A)

        eye_aa = torch.eye(N_A, device=device, dtype=torch.bool).view(1, N_A, N_A)
        d_aa_masked = d_aa.masked_fill(eye_aa, float("inf"))
        d_nn_aa = d_aa_masked.min(dim=-1).values                                # (B, N_A)
        crowding_aa = torch.exp(-self.crowding_alpha * d_aa).masked_fill(eye_aa, 0.0).sum(dim=-1)  # (B, N_A)
        rbf_aa = self.rbf_aa(d_aa).masked_fill(eye_aa.unsqueeze(-1), 0.0).sum(dim=2)              # (B, N_A, n_rbf_aa)

        # Cross-group MIC distances: active to frozen (frozen has FULL 3D coords — key improvement)
        du_ab = mic_unit(u_active[:, :, None, :] - u_frozen[:, None, :, :])    # (B,N_A,N_F,3)
        dx_ab = L * du_ab
        d_ab = torch.linalg.norm(dx_ab, dim=-1)                                # (B,N_A,N_F)
        rbf_ab = self.rbf_ab(d_ab).sum(dim=2)                                   # (B, N_A, n_rbf_ab)

        # Active to solute distances
        solute_nm = self.solute_positions_nm.to(device=device, dtype=dtype)
        solute_u = wrap_unit(solute_nm / L)                                     # (K, 3)
        du_as = mic_unit(u_active[:, :, None, :] - solute_u.view(1, 1, -1, 3)) # (B,N_A,K,3)
        dx_as = L * du_as
        d_as = torch.linalg.norm(dx_as, dim=-1)                                # (B,N_A,K)
        rbf_as = self.rbf_su(d_as).sum(dim=2)                                   # (B, N_A, n_rbf_su)

        scalar = torch.stack([d_nn_aa, crowding_aa], dim=-1)                    # (B, N_A, 2)
        feats = torch.cat([periodic_embed, scalar, rbf_aa, rbf_ab, rbf_as], dim=-1)  # (B, N_A, F)
        return feats, d_aa


# ============================================================
# Equivariant message-passing conditioner
# ============================================================

class EquivariantInteraction(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.psi = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, h: torch.Tensor, d_ss: torch.Tensor):
        B, N, _ = h.shape
        hi = h[:, :, None, :].expand(-1, -1, N, -1)
        hj = h[:, None, :, :].expand(-1, N, -1, -1)
        inp = torch.cat([hi, hj, d_ss.unsqueeze(-1)], dim=-1)
        m_ij = self.phi(inp)

        eye = torch.eye(N, device=h.device, dtype=torch.bool).view(1, N, N, 1)
        m_ij = m_ij.masked_fill(eye, 0.0)
        m_i = m_ij.sum(dim=2)

        return self.psi(torch.cat([h, m_i], dim=-1))


class TorusEquivariantSplineConditioner(nn.Module):
    """
    Outputs spline parameters per particle for all 3 coords.
    Final linear layer is zero-initialized for near-identity start.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_interaction_blocks: int = 2,
    ):
        super().__init__()
        self.interactions = nn.ModuleList(
            [EquivariantInteraction(feature_dim, hidden_dim) for _ in range(n_interaction_blocks)]
        )
        self.out = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, feats: torch.Tensor, d_ss: torch.Tensor) -> torch.Tensor:
        h = feats
        for block in self.interactions:
            h = h + block(h, d_ss)
        return self.out(h)


# ============================================================
# Torus-aware spline coupling with nonzero Jacobian
# ============================================================

class PeriodicParticleSplineCoupling(nn.Module):
    """
    Periodic coupling on unit-torus coordinates.

    Update selected channels with an RQ spline on centered unit interval [-0.5, 0.5),
    then wrap back into [0,1).

    This has nonzero Jacobian and can shape density, unlike pure translation coupling.
    """

    def __init__(
        self,
        feature_extractor: PeriodicSolventFeatureExtractor,
        conditioner: TorusEquivariantSplineConditioner,
        channel_mask: Sequence[int],
        num_bins: int = 8,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if len(channel_mask) != 3:
            raise ValueError("channel_mask must have length 3")

        self.feature_extractor = feature_extractor
        self.conditioner = conditioner
        self.num_bins = int(num_bins)
        self.tail_bound = 0.5

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        mask = torch.as_tensor(channel_mask, dtype=torch.bool).view(1, 1, 3)
        self.register_buffer("channel_mask", mask)

        self.active_idx = [i for i, v in enumerate(channel_mask) if v == 1]
        self.n_active = len(self.active_idx)

        params_per_dim = 3 * self.num_bins - 1
        self.params_per_dim = params_per_dim

    def center_unit(self, u: torch.Tensor) -> torch.Tensor:
        return wrap_unit(u) - 0.5

    def uncenter_unit(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_unit(x + 0.5)

    def forward(self, u: torch.Tensor):
        """
        u: (B, N, 3) in [0,1)
        """
        param = next(self.conditioner.parameters())
        u = u.to(device=param.device, dtype=param.dtype)
        u = wrap_unit(u)

        mask = self.channel_mask.to(device=u.device)
        u_frozen = u.masked_fill(mask, 0.0)

        feats, d_ss = self.feature_extractor(u_frozen, channel_mask=mask)
        raw_params = self.conditioner(feats, d_ss)  # (B,N, 3*params_per_dim)

        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 3, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=u.device, dtype=torch.long)
        params = raw_params[:, :, active, :]  # (B,N,n_active,params_per_dim)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        u_a = u[..., self.active_idx]
        u_a_c = self.center_unit(u_a)

        y_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=u_a_c,
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

        y_a = self.uncenter_unit(y_a_c)

        y = u.clone()
        y[..., self.active_idx] = y_a
        y = wrap_unit(y)

        logdet = logabsdet.sum(dim=(-1, -2))
        return y, logdet

    def inverse(self, y: torch.Tensor):
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)
        y = wrap_unit(y)

        mask = self.channel_mask.to(device=y.device)
        y_frozen = y.masked_fill(mask, 0.0)

        feats, d_ss = self.feature_extractor(y_frozen, channel_mask=mask)
        raw_params = self.conditioner(feats, d_ss)

        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 3, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=y.device, dtype=torch.long)
        params = raw_params[:, :, active, :]

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        y_a = y[..., self.active_idx]
        y_a_c = self.center_unit(y_a)

        x_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a_c,
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

        x_a = self.uncenter_unit(x_a_c)

        x = y.clone()
        x[..., self.active_idx] = x_a
        x = wrap_unit(x)

        logdet = logabsdet.sum(dim=(-1, -2))
        return x, logdet


# ============================================================
# Hybrid particle-group + coordinate-split coupling
# ============================================================

class HybridGroupCoordSplineCoupling(nn.Module):
    """
    Coupling layer combining particle-group split and coordinate-dimension split.

    Active group (group_mask=True particles): their active coordinate dimension
    is updated with an RQ spline conditioned on:
      - Their own frozen (non-active) coordinates  [per-particle differentiation]
      - The frozen group's FULL 3D positions        [richer cross-group distances]

    Key improvement over pure coordinate-split (PeriodicParticleSplineCoupling):
      Cross-group distances use the frozen group's actual value of the active-dimension
      coordinate, so d_ab includes x_B in the x-layer (not just yz-projected distances).

    Accepts and returns u: (B, N, 3) — same interface as PeriodicParticleSplineCoupling,
    so WrappedTorusSplineFlow can be reused unchanged.
    """

    def __init__(
        self,
        feature_extractor: "CrossGroupFeatureExtractor",
        conditioner: TorusEquivariantSplineConditioner,
        channel_mask: Sequence[int],
        group_mask,   # (N,) bool tensor or list: True for active particle group
        num_bins: int = 8,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if len(channel_mask) != 3:
            raise ValueError("channel_mask must have length 3")

        self.feature_extractor = feature_extractor
        self.conditioner = conditioner
        self.num_bins = int(num_bins)
        self.tail_bound = 0.5
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        coord_mask = torch.as_tensor(channel_mask, dtype=torch.bool).view(1, 1, 3)
        self.register_buffer("channel_mask", coord_mask)

        grp_mask = torch.as_tensor(group_mask, dtype=torch.bool)  # (N,)
        self.register_buffer("group_mask", grp_mask)

        self.active_idx = [i for i, v in enumerate(channel_mask) if v == 1]
        self.n_active = len(self.active_idx)
        self.params_per_dim = 3 * self.num_bins - 1

    def center_unit(self, u: torch.Tensor) -> torch.Tensor:
        return wrap_unit(u) - 0.5

    def uncenter_unit(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_unit(x + 0.5)

    def _compute_params(self, u: torch.Tensor):
        """
        Compute spline parameters from frozen inputs only.

        All inputs used for conditioning are frozen (unchanged in both forward and inverse),
        so these params are identical in the forward and inverse passes — ensuring invertibility.
        """
        group_mask = self.group_mask.to(device=u.device)
        coord_mask = self.channel_mask.to(device=u.device)

        group_idx = torch.where(group_mask)[0]    # (N_A,)
        frozen_idx = torch.where(~group_mask)[0]  # (N_F,)

        u_active = u[:, group_idx, :]   # (B, N_A, 3)
        u_frozen = u[:, frozen_idx, :]  # (B, N_F, 3) — full 3D, NOT coord-masked

        # Zero the active coordinate in the active group (frozen part of coord-split)
        u_active_masked = u_active.masked_fill(coord_mask, 0.0)  # (B, N_A, 3)

        feats, d_aa = self.feature_extractor(u_active_masked, u_frozen)  # (B, N_A, F)
        raw_params = self.conditioner(feats, d_aa)  # (B, N_A, 3 * params_per_dim)

        B, N_A, _ = raw_params.shape
        raw_params = raw_params.view(B, N_A, 3, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=u.device, dtype=torch.long)
        params = raw_params[:, :, active, :]  # (B, N_A, n_active, params_per_dim)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        return widths, heights, derivatives, group_idx, u_active

    def forward(self, u: torch.Tensor):
        """u: (B, N, 3) in [0,1)"""
        param = next(self.conditioner.parameters())
        u = u.to(device=param.device, dtype=param.dtype)
        u = wrap_unit(u)

        widths, heights, derivatives, group_idx, u_active = self._compute_params(u)

        u_a = u_active[..., self.active_idx]   # (B, N_A, n_active)
        u_a_c = self.center_unit(u_a)

        y_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=u_a_c,
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

        y_a = self.uncenter_unit(y_a_c)

        y = u.clone()
        y_active = u_active.clone()
        y_active[..., self.active_idx] = y_a
        y[:, group_idx, :] = wrap_unit(y_active)

        logdet = logabsdet.sum(dim=(-1, -2))  # sum over N_A particles × n_active dims
        return y, logdet

    def inverse(self, y: torch.Tensor):
        """y: (B, N, 3) in [0,1)"""
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)
        y = wrap_unit(y)

        # Feature computation identical to forward (uses only frozen info — unchanged in inverse)
        widths, heights, derivatives, group_idx, y_active = self._compute_params(y)

        y_a = y_active[..., self.active_idx]   # (B, N_A, n_active)
        y_a_c = self.center_unit(y_a)

        x_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a_c,
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

        x_a = self.uncenter_unit(x_a_c)

        x = y.clone()
        x_active = y_active.clone()
        x_active[..., self.active_idx] = x_a
        x[:, group_idx, :] = wrap_unit(x_active)

        logdet = logabsdet.sum(dim=(-1, -2))
        return x, logdet


class SimpleTorusSplineConditioner(nn.Module):
    """
    Shared per-particle conditioner.

    Input:
        torus embedding of frozen coordinates of shape (B, N, 2 * n_frozen)

    Output:
        spline params for all 3 coordinates, shape (B, N, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int = 2,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim

        final = nn.Linear(d, out_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PeriodicParticleSplineCouplingNoFeatures(nn.Module):
    """
    Periodic torus spline coupling without geometric feature extractor.

    The conditioner only sees the frozen channels of each particle, embedded
    with sin/cos on the torus.
    """

    def __init__(
        self,
        conditioner: SimpleTorusSplineConditioner,
        channel_mask: Sequence[int],
        num_bins: int = 8,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if len(channel_mask) != 3:
            raise ValueError("channel_mask must have length 3")

        self.conditioner = conditioner
        self.num_bins = int(num_bins)
        self.tail_bound = 0.5

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        mask = torch.as_tensor(channel_mask, dtype=torch.bool).view(1, 1, 3)
        self.register_buffer("channel_mask", mask)

        self.active_idx = [i for i, v in enumerate(channel_mask) if v == 1]
        self.frozen_idx = [i for i, v in enumerate(channel_mask) if v == 0]
        self.n_active = len(self.active_idx)

        self.params_per_dim = 3 * self.num_bins - 1

    def center_unit(self, u: torch.Tensor) -> torch.Tensor:
        return wrap_unit(u) - 0.5

    def uncenter_unit(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_unit(x + 0.5)

    def torus_embed(self, u_frozen: torch.Tensor) -> torch.Tensor:
        """
        u_frozen: (B, N, n_frozen) in [0,1)
        returns:  (B, N, 2*n_frozen)
        """
        angle = 2.0 * math.pi * u_frozen
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def _get_params(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, N, 3) in [0,1)
        returns raw spline params of shape (B, N, n_active, params_per_dim)
        """
        u_frozen = u[..., self.frozen_idx]                   # (B,N,n_frozen)
        cond_in = self.torus_embed(u_frozen)                 # (B,N,2*n_frozen)

        raw_params = self.conditioner(cond_in)               # (B,N,3*params_per_dim)
        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 3, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=u.device, dtype=torch.long)
        params = raw_params[:, :, active, :]                 # (B,N,n_active,params_per_dim)
        return params

    def forward(self, u: torch.Tensor):
        param = next(self.conditioner.parameters())
        u = u.to(device=param.device, dtype=param.dtype)
        u = wrap_unit(u)

        params = self._get_params(u)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        u_a = u[..., self.active_idx]
        u_a_c = self.center_unit(u_a)

        y_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=u_a_c,
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

        y_a = self.uncenter_unit(y_a_c)

        y = u.clone()
        y[..., self.active_idx] = y_a
        y = wrap_unit(y)

        logdet = logabsdet.sum(dim=(-1, -2))
        return y, logdet

    def inverse(self, y: torch.Tensor):
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)
        y = wrap_unit(y)

        params = self._get_params(y)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        y_a = y[..., self.active_idx]
        y_a_c = self.center_unit(y_a)

        x_a_c, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a_c,
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

        x_a = self.uncenter_unit(x_a_c)

        x = y.clone()
        x[..., self.active_idx] = x_a
        x = wrap_unit(x)

        logdet = logabsdet.sum(dim=(-1, -2))
        return x, logdet


# ============================================================
# Wrapped torus spline flow
# ============================================================

class WrappedTorusSplineFlow(nn.Module):
    """
    Flow over flattened solvent unit-torus coordinates.
    Base must live on [0,1)^dim or at least have log_prob evaluated there.
    """

    def __init__(self, layers, base):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base = base
        self.event_shape = (base.shape[0],) if hasattr(base, "shape") else (base.dim,)

    def _flat_to_particle(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] % 3 != 0:
            raise ValueError(f"Expected (B, 3N), got {tuple(x.shape)}")
        return x.view(x.shape[0], -1, 3)

    def _particle_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B, N, 3), got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1)

    def forward_map(self, x: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        x = wrap_unit(x)

        u = self._flat_to_particle(x)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            u, ld = layer(u)
            logdet = logdet + ld

        z = wrap_unit(self._particle_to_flat(u))
        return z, logdet

    def inverse_map(self, z: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        z = z.to(device=param.device, dtype=param.dtype)
        z = wrap_unit(z)

        u = self._flat_to_particle(z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            u, ld = layer.inverse(u)
            logdet = logdet + ld

        x = wrap_unit(self._particle_to_flat(u))
        return x, logdet

    def log_prob(self, x: torch.Tensor):
        z, logdet = self.forward_map(x)
        return self.base.log_prob(z) + logdet

    def forward_and_log_prob(self, x: torch.Tensor):
        z, logdet = self.forward_map(x)
        return z, self.base.log_prob(z) + logdet

    def sample_and_log_prob(self, shape):
        n = shape[0]
        z, log_q = self.base(n)
        z = wrap_unit(z)
        x, inv_logdet = self.inverse_map(z)
        x = wrap_unit(x)
        return x, log_q - inv_logdet

    def sample(self, shape):
        x, _ = self.sample_and_log_prob(shape)
        return x

import math
import torch
from torch import nn
from typing import Sequence

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


# ============================================================
# Utilities for centered periodic box coords
# ============================================================

def wrap_centered_box(x: torch.Tensor, box_length: float) -> torch.Tensor:
    """
    Wrap into [-L/2, L/2)
    """
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    return torch.remainder(x + 0.5 * L, L) - 0.5 * L


def torus_project_centered(x: torch.Tensor, box_length: float) -> torch.Tensor:
    """
    Periodic embedding of centered box coordinates.
    x can be (..., d), returns (..., 2d)
    """
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    angle = 2.0 * math.pi * x / L
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


# ============================================================
# Simple per-particle conditioner for centered periodic coords
# ============================================================

class SimpleCenteredSplineConditioner(nn.Module):
    """
    Shared per-particle conditioner.

    Input:
        periodic embedding of frozen coordinates, shape (B, N, 2*n_frozen)

    Output:
        spline params for all 3 coordinates, shape (B, N, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int = 2,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim

        final = nn.Linear(d, out_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Centered-box periodic spline coupling
# ============================================================

class CenteredPeriodicParticleSplineCoupling(nn.Module):
    """
    Coupling layer on centered periodic coordinates z in [-L/2, L/2).

    For each particle:
      - keep some channels frozen
      - periodic-embed frozen channels with sin/cos(2π z/L)
      - predict spline params for active channels
      - transform active channels with RQ spline on [-L/2, L/2)
      - wrap back into centered MIC box

    This is compatible with SoluteCenteredSolventTransform.
    """

    def __init__(
        self,
        box_length: float,
        conditioner: SimpleCenteredSplineConditioner,
        channel_mask: Sequence[int],
        num_bins: int = 8,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if len(channel_mask) != 3:
            raise ValueError("channel_mask must have length 3")

        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)

        self.conditioner = conditioner
        self.num_bins = int(num_bins)

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        mask = torch.as_tensor(channel_mask, dtype=torch.bool).view(1, 1, 3)
        self.register_buffer("channel_mask", mask)

        self.active_idx = [i for i, v in enumerate(channel_mask) if v == 1]
        self.frozen_idx = [i for i, v in enumerate(channel_mask) if v == 0]
        self.n_active = len(self.active_idx)

        self.params_per_dim = 3 * self.num_bins - 1

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_centered_box(x, self.box_length)

    def _get_params(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, 3) in [-L/2, L/2)
        returns params of shape (B, N, n_active, params_per_dim)
        """
        z = self.wrap_centered(z)
        z_frozen = z[..., self.frozen_idx]                             # (B,N,n_frozen)
        cond_in = torus_project_centered(z_frozen, self.box_length)   # (B,N,2*n_frozen)

        raw_params = self.conditioner(cond_in)                        # (B,N,3*params_per_dim)
        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 3, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=z.device, dtype=torch.long)
        params = raw_params[:, :, active, :]                          # (B,N,n_active,params_per_dim)
        return params

    def forward(self, z: torch.Tensor):
        param = next(self.conditioner.parameters())
        z = z.to(device=param.device, dtype=param.dtype)
        z = self.wrap_centered(z)

        params = self._get_params(z)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        z_a = z[..., self.active_idx]

        y_a, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=z_a,
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

        y = z.clone()
        y[..., self.active_idx] = y_a
        y = self.wrap_centered(y)

        logdet = logabsdet.sum(dim=(-1, -2))
        return y, logdet

    def inverse(self, y: torch.Tensor):
        param = next(self.conditioner.parameters())
        y = y.to(device=param.device, dtype=param.dtype)
        y = self.wrap_centered(y)

        params = self._get_params(y)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins:]

        y_a = y[..., self.active_idx]

        x_a, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_a,
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
        x[..., self.active_idx] = x_a
        x = self.wrap_centered(x)

        logdet = logabsdet.sum(dim=(-1, -2))
        return x, logdet


# ============================================================
# Wrapped flow for centered periodic solvent coordinates
# ============================================================

class WrappedCenteredPeriodicSplineFlow(nn.Module):
    """
    Flow over flattened solvent coordinates in centered periodic box:
        z in [-L/2, L/2)^(3N)

    Compatible with SoluteCenteredSolventTransform.

    Base must also live on the same centered periodic coordinates.
    """

    def __init__(self, layers, base, box_length: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base = base
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)
        self.event_shape = (base.shape[0],) if hasattr(base, "shape") else (base.dim,)

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_centered_box(x, self.box_length)

    def _flat_to_particle(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] % 3 != 0:
            raise ValueError(f"Expected (B, 3N), got {tuple(x.shape)}")
        return x.view(x.shape[0], -1, 3)

    def _particle_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B, N, 3), got {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1)

    def forward_map(self, x: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        x = self.wrap_centered(x)

        z = self._flat_to_particle(x)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld

        z = self._particle_to_flat(z)
        z = self.wrap_centered(z)
        return z, logdet

    def inverse_map(self, z: torch.Tensor):
        param = next(self.layers[0].conditioner.parameters())
        z = z.to(device=param.device, dtype=param.dtype)
        z = self.wrap_centered(z)

        x = self._flat_to_particle(z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld

        x = self._particle_to_flat(x)
        x = self.wrap_centered(x)
        return x, logdet

    def log_prob(self, x: torch.Tensor):
        z, logdet = self.forward_map(x)
        return self.base.log_prob(z) + logdet

    def forward_and_log_prob(self, x: torch.Tensor):
        z, logdet = self.forward_map(x)
        return z, self.base.log_prob(z) + logdet

    def sample_and_log_prob(self, shape):
        n = shape[0]
        z, log_q = self.base(n)         # base must already sample centered coords
        z = self.wrap_centered(z)
        x, inv_logdet = self.inverse_map(z)
        x = self.wrap_centered(x)
        return x, log_q - inv_logdet

    def sample(self, shape):
        x, _ = self.sample_and_log_prob(shape)
        return x

# ============================================================
# Factory
# ============================================================

def make_lj_flow(cfg, target):
    n_solvent = int(target.n_solvent)
    dim = int(target.internal_dim)
    L = float(target.box_length_nm)
    seed = int(cfg.training.seed)
    n_layers = int(cfg.flow.n_layers)
    hidden_dim = int(cfg.flow.hidden_units)
    n_hidden = int(cfg.flow.n_hidden_conditioner)
    num_bins = int(cfg.flow.num_bins)
    mixing = cfg.flow.mixing
    circ_shift = cfg.flow.get("circ_shift", None)

    periodic_inds = np.arange(dim)
    bound_circ = L / 2.0
    tail_bound = bound_circ * torch.ones(dim)

    base_type = cfg.flow.base.type
    flow_type = cfg.flow.type

    if flow_type == "lj_torus_spline" or flow_type == "lj_torus_spline_noF":
        if base_type == "uniform":
            base = UniformUnitTorusBase(dim=dim)
        elif base_type == "gauss-uni":
            base = GaussianUnitTorusBase(
                dim=dim,
                sigma_unit=float(getattr(cfg.flow.base, "sigma_unit", 0.08)),
                mean_unit=float(getattr(cfg.flow.base, "mean_unit", 0.5)),
                image_range=int(getattr(cfg.flow.base, "image_range", 1)),
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )
        elif base_type == "fcc-gaussian":
            base = SoluteAwareFCCGaussianUnitTorusBase(
                solute_positions_nm=target.system.solute_positions_nm,
                box_length_nm=target.box_length_nm,
                n_solvent=target.n_solvent,
                sigma_unit=float(getattr(cfg.flow.base, "sigma_unit", 0.03)),
                image_range=int(getattr(cfg.flow.base, "image_range", 1)),
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )
        elif base_type == "shell-torus":
            base = ShellTorusSoluteBase(
                n_solvent=target.n_solvent,
                box_length_nm=target.box_length_nm,
                solute_positions_nm=target.system.solute_positions_nm,
                shell_radius_nm=float(getattr(cfg.flow.base, "shell_radius_nm", 0.33)),
                shell_sigma_nm=float(getattr(cfg.flow.base, "shell_sigma_nm", 0.035)),
                n_reference_sites=getattr(cfg.flow.base, "n_reference_sites", None),
                add_uniform_component=bool(getattr(cfg.flow.base, "add_uniform_component", True)),
                uniform_weight=float(getattr(cfg.flow.base, "uniform_weight", 0.25)),
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )
        elif base_type == "multishell-torus":
            base = MultiShellAutoregressiveTorusBase(
                n_solvent=target.n_solvent,
                box_length_nm=target.box_length_nm,
                solute_positions_nm=target.system.solute_positions_nm,
                shell_radii_nm=cfg.flow.base.shell_radii_nm,
                shell_weights=cfg.flow.base.shell_weights,
                component_sigma_unit=float(getattr(cfg.flow.base, "component_sigma_unit", 0.018)),
                n_directions=float(getattr(cfg.flow.base, "n_directions", 64)),
                add_uniform_component=bool(getattr(cfg.flow.base, "add_uniform_component", False)),
                crowding_strength=float(getattr(cfg.flow.base, "crowding_strength", 20.0)),
                crowding_sigma_nm=float(getattr(cfg.flow.base, "crowding_sigma_nm", 0.08)),
                hard_core_nm=float(getattr(cfg.flow.base, "hard_core_nm", 0.24)),
                hard_core_strength=float(getattr(cfg.flow.base, "hard_core_strength", 10.0)),
                image_range=int(getattr(cfg.flow.base, "image_range", 1)),
                uniform_weight=float(getattr(cfg.flow.base, "uniform_weight", 0.10)),
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )


        elif base_type == "hardcore-random":
            def wrap_centered(x, L):
                return torch.remainder(x + 0.5 * L, L) - 0.5 * L

            solute_nm = torch.as_tensor(target.system.solute_positions_nm, dtype=torch.get_default_dtype())
            solute_centered = wrap_centered(solute_nm, target.box_length_nm)

            base_centered = HardCoreRandomPlacementBase(
                n_solvent=target.n_solvent,
                box_length=target.box_length_nm,
                solute_positions_centered=solute_centered,
                r_min_ss=cfg.flow.base.r_min_ss,
                r_min_su=cfg.flow.base.r_min_su,
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )

            base = UnitTorusPeriodicBase(
                wrapped_base=base_centered,
                dim=target.internal_dim,
                box_length=target.box_length_nm,
            )
        else:
            raise NotImplementedError(
                f"Base distribution {base_type} not implemented for flow_type={flow_type}"
            )
    else:
        if base_type == "gauss":
            base = nf.distributions.DiagGaussian(
                dim,
                trainable=bool(cfg.flow.base.learn_mean_var),
            )
        elif base_type == "gauss-uni":
            base_scale = bound_circ * torch.ones(dim)
            base = nf.distributions.UniformGaussian(dim, periodic_inds, scale=base_scale)
            base.shape = (dim,)
        elif base_type == "fcc-gaussian":
            fcc = build_fcc_positions(target.box_length_nm, n_cells=2)   # 32 sites

            solute_idx = 0
            solute_abs = fcc[solute_idx:solute_idx+1]          # (1,3)
            solvent_abs = torch.cat([fcc[:solute_idx], fcc[solute_idx+1:]], dim=0)  # (31,3)

            # match SoluteCenteredSolventTransform(reference="first_solute")
            ref = solute_abs[0]
            solvent_rel = wrap_centered_box(solvent_abs - ref, target.box_length_nm)

            base = FCCGaussianCenteredPeriodicBase(
                solvent_fcc_rel_nm=solvent_rel,
                box_length_nm=target.box_length_nm,
                sigma_nm=0.03,
                image_range=1,
                device=str(target.device),
                dtype=torch.get_default_dtype(),
            )
        else:
            raise NotImplementedError(
                f"Base distribution {base_type} not implemented for flow_type={flow_type}"
            )

    if flow_type == "lj_coupling_spline":
        cycle = _cartesian_partition_cycle()
        layers = []

        for i in range(n_layers):
            layers.append(
                ParticleTorusSplineCoupling(
                    n_solvent=n_solvent,
                    box_length=L,
                    update_axes=cycle[i % len(cycle)],
                    num_bins=num_bins,
                    hidden_dim=hidden_dim,
                    n_hidden=n_hidden,
                )
            )

        return WrappedCustomFlow(layers, base, box_length=L)

    elif flow_type == "lj_coupling_perm_equi":
        cycle = _cartesian_partition_cycle()
        layers = []

        for i in range(n_layers):
            layers.append(
                PermEquiLJParticleSplineCoupling(
                    box_length=L,
                    update_axes=cycle[i % len(cycle)],
                    num_bins=num_bins,
                    hidden_dim=hidden_dim,
                    n_hidden=n_hidden,
                    n_rbf=int(getattr(cfg.flow, "pair_rbf_dim", 16)),
                    rbf_max_dist=float(getattr(cfg.flow, "pair_rbf_max_dist", 2.0)),
                    dropout=float(getattr(cfg.flow, "dropout", 0.0)),
                )
            )

        return WrappedPermEquiLJFlow(layers, base, box_length=L)

    elif flow_type == "lj_torus_spline":
        cycle = _cartesian_partition_cycle()

        feature_extractor = PeriodicSolventFeatureExtractor(
            box_length_nm=L,
            solute_positions_nm=target.system.solute_positions_nm,
            n_rbf_ss=16,
            n_rbf_su=16,
        )

        params_per_dim = 3 * num_bins - 1
        layers = []

        for i in range(n_layers):
            conditioner = TorusEquivariantSplineConditioner(
                feature_dim=feature_extractor.feature_dim,
                hidden_dim=hidden_dim,
                out_dim=3 * params_per_dim,
                n_interaction_blocks=2,
            )

            mask = [1 if ax in cycle[i % len(cycle)] else 0 for ax in [0, 1, 2]]

            layers.append(
                PeriodicParticleSplineCoupling(
                    feature_extractor=feature_extractor,
                    conditioner=conditioner,
                    channel_mask=mask,
                    num_bins=num_bins,
                    min_bin_width=1e-3,
                    min_bin_height=1e-3,
                    min_derivative=1e-3,
                )
            )

        return WrappedTorusSplineFlow(layers=layers, base=base)
    elif flow_type == "lj_torus_spline_noF":
        cycle = _cartesian_partition_cycle()
        layers = []

        params_per_dim = 3 * num_bins - 1

        for i in range(n_layers):
            mask = [1 if ax in cycle[i % len(cycle)] else 0 for ax in [0, 1, 2]]
            n_frozen = 3 - sum(mask)

            conditioner = SimpleTorusSplineConditioner(
                in_dim=2 * n_frozen,
                hidden_dim=hidden_dim,
                out_dim=3 * params_per_dim,
                n_hidden=n_hidden,
            )

            layers.append(
                PeriodicParticleSplineCouplingNoFeatures(
                    conditioner=conditioner,
                    channel_mask=mask,
                    num_bins=num_bins,
                    min_bin_width=1e-3,
                    min_bin_height=1e-3,
                    min_derivative=1e-3,
                )
            )

        return WrappedTorusSplineFlow(layers=layers, base=base)
    
    elif flow_type == "lj_torus_spline_noF_v2":
        cycle = _cartesian_partition_cycle()
        layers = []

        params_per_dim = 3 * num_bins - 1

        for i in range(n_layers):
            mask = [1 if ax in cycle[i % len(cycle)] else 0 for ax in [0, 1, 2]]
            n_frozen = 3 - sum(mask)

            conditioner = SimpleCenteredSplineConditioner(
                in_dim=2 * n_frozen,
                hidden_dim=hidden_dim,
                out_dim=3 * params_per_dim,
                n_hidden=n_hidden,
            )

            layers.append(
                CenteredPeriodicParticleSplineCoupling(
                    box_length=L,
                    conditioner=conditioner,
                    channel_mask=mask,
                    num_bins=num_bins,
                    min_bin_width=1e-3,
                    min_bin_height=1e-3,
                    min_derivative=1e-3,
                )
            )

        return WrappedCenteredPeriodicSplineFlow(
            layers=layers,
            base=base,
            box_length=L,
        )

    elif flow_type == "lj_particle_group_spline":
        # Hybrid particle-group + coordinate-split coupling.
        # Cycles through 6 layer types: (coord=x/y/z) × (group=A/B), then repeats.
        # Group A = even solvent indices, Group B = odd solvent indices.
        # In each layer the active group's active coordinate is updated conditioned on:
        #   - own frozen (non-active) coordinates  [per-particle differentiation]
        #   - frozen group's FULL 3D positions      [includes active-dim coord of frozen group]
        N = int(target.n_solvent)
        group_mask_A = torch.arange(N) % 2 == 0   # (N,) True for even indices
        group_mask_B = torch.arange(N) % 2 == 1   # (N,) True for odd indices

        feature_extractor = CrossGroupFeatureExtractor(
            box_length_nm=L,
            solute_positions_nm=target.system.solute_positions_nm,
            n_rbf_aa=16,
            n_rbf_ab=16,
            n_rbf_su=16,
        )

        # 6-step cycle: (channel_mask, group_mask)
        cycle = [
            ([1, 0, 0], group_mask_A), ([1, 0, 0], group_mask_B),
            ([0, 1, 0], group_mask_A), ([0, 1, 0], group_mask_B),
            ([0, 0, 1], group_mask_A), ([0, 0, 1], group_mask_B),
        ]

        params_per_dim = 3 * num_bins - 1
        layers = []

        for i in range(n_layers):
            coord_mask, grp_mask = cycle[i % len(cycle)]
            conditioner = TorusEquivariantSplineConditioner(
                feature_dim=feature_extractor.feature_dim,
                hidden_dim=hidden_dim,
                out_dim=3 * params_per_dim,
                n_interaction_blocks=2,
            )
            layers.append(
                HybridGroupCoordSplineCoupling(
                    feature_extractor=feature_extractor,
                    conditioner=conditioner,
                    channel_mask=coord_mask,
                    group_mask=grp_mask,
                    num_bins=num_bins,
                    min_bin_width=1e-3,
                    min_bin_height=1e-3,
                    min_derivative=1e-3,
                )
            )

        return WrappedTorusSplineFlow(layers=layers, base=base)

    elif flow_type == "realnvp":
        flows = []
        for _ in range(n_layers):
            d = dim // 2
            param_map = nf.nets.MLP([d, hidden_dim, hidden_dim, 2 * (dim - d)], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
            flows.append(nf.flows.Permute(dim))
            if bool(cfg.flow.act_norm):
                flows.append(nf.flows.ActNorm(dim))

        flow = nf.NormalizingFlow(base, flows)
        return WrappedNormFlowModel(flow)

    else:
        raise NotImplementedError(f"Unknown flow type: {flow_type}")

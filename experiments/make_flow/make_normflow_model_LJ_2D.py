import math
import torch
from torch import nn
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

import torch
from torch import nn


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def mic_box(dx: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=dx.device, dtype=dx.dtype)
    return dx - L * torch.round(dx / L)


def box_to_unit_centered(x: torch.Tensor, box_length: float) -> torch.Tensor:
    """
    Map centered box coords in [-L/2, L/2) to unit coords in [0,1).
    """
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    return wrap_unit((x / L) + 0.5)

def mic_unit(u: torch.Tensor) -> torch.Tensor:
        return u - torch.round(u)


def unit_to_box_centered(u: torch.Tensor, box_length: float) -> torch.Tensor:
    """
    Map unit coords in [0,1) to centered box coords in [-L/2, L/2).
    """
    L = torch.as_tensor(box_length, device=u.device, dtype=u.dtype)
    return L * (wrap_unit(u) - 0.5)


class SoluteCenteredUnitTorusTransform2D(nn.Module):
    """
    2D solute-centered periodic transform.

    Internal coordinates:
        solvent positions relative to the solute, represented on the unit torus [0,1),
        flattened as (B, 2*n_solvent)

    Cartesian coordinates:
        full 2D coordinates [solute, solvent...] in nm, shape (B, 2*(n_solute+n_solvent))

    Assumes:
        - n_solute = 1 is the main use case
        - particle order is [solute first, then solvent particles]
        - periodic box in x,y with side length box_length_nm
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        n_solute: int = 1,
    ):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)

        self.internal_dim = 2 * self.n_solvent
        self.cartesian_dim = 2 * (self.n_solute + self.n_solvent)

        if self.n_solute != 1:
            raise ValueError("This simple 2D version currently assumes n_solute = 1.")

    def forward(self, u_flat: torch.Tensor):
        """
        u_flat: (B, 2*n_solvent), solvent relative coords on unit torus [0,1)

        returns:
            x_full: (B, 2*(1+n_solvent)) in nm
                    with solute fixed at (0,0)
            logdet: zeros
        """
        if u_flat.ndim != 2 or u_flat.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected u_flat shape (B, {self.internal_dim}), got {tuple(u_flat.shape)}"
            )

        B = u_flat.shape[0]
        u = wrap_unit(u_flat.view(B, self.n_solvent, 2))
        solvent_rel_xy = unit_to_box_centered(u, self.box_length_nm)  # (B, N, 2)

        solute_xy = torch.zeros((B, 1, 2), device=u.device, dtype=u.dtype)
        x_full = torch.cat([solute_xy, solvent_rel_xy], dim=1)

        logdet = torch.zeros(B, device=u.device, dtype=u.dtype)
        return x_full.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 2*(1+n_solvent)) in nm

        returns:
            u_flat: (B, 2*n_solvent) in [0,1)
            logdet: zeros
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 2)

        solute_xy = x[:, 0:1, :]         # (B,1,2)
        solvent_xy = x[:, 1:, :]         # (B,N,2)

        rel_xy = solvent_xy - solute_xy
        rel_xy = mic_box(rel_xy, self.box_length_nm)   # centered in [-L/2, L/2)

        u = box_to_unit_centered(rel_xy, self.box_length_nm)

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return u.reshape(B, -1), logdet


def _cartesian_partition_cycle_2d():
    return [
        [0],
        [1],
    ]


def torus_project_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., 2)
    returns: (..., 4)
    """
    angle = 2.0 * math.pi * x
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

class RBFFeatures(nn.Module):
    def __init__(self, n_rbf: int = 16, r_max: float = 1.0):
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


class PeriodicSolventFeatureExtractor2D(nn.Module):
    """
    Per-solvent features on the unit torus [0,1)^2.
    """

    def __init__(
        self,
        box_length_nm: float,
        n_rbf_ss: int = 16,
        rbf_ss_max_nm: float = None,
        crowding_alpha: float = 12.0,
    ):
        super().__init__()
        self.box_length_nm = float(box_length_nm)
        self.crowding_alpha = float(crowding_alpha)

        if rbf_ss_max_nm is None:
            rbf_ss_max_nm = 0.5 * self.box_length_nm

        self.rbf_ss = RBFFeatures(n_rbf=n_rbf_ss, r_max=float(rbf_ss_max_nm))

        # sin/cos embedding of 2D torus coords -> 4 dims
        # plus d_nn and crowding -> 2 dims
        # plus pooled ss rbf
        self.feature_dim = 4 + 2 + n_rbf_ss

    def forward(self, u: torch.Tensor):
        """
        u: (B, N, 2) in [0,1)

        returns:
            feats: (B, N, F)
            d_ss:  (B, N, N)
        """
        B, N, _ = u.shape
        dtype = u.dtype
        device = u.device

        L = torch.as_tensor(self.box_length_nm, device=device, dtype=dtype)

        angle = 2.0 * math.pi * u
        periodic_embed = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B,N,4)

        du_ss = mic_unit(u[:, :, None, :] - u[:, None, :, :])   # (B,N,N,2)
        dx_ss = L * du_ss
        d_ss = torch.linalg.norm(dx_ss, dim=-1)                 # (B,N,N)

        eye = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
        d_ss_masked = d_ss.masked_fill(eye, float("inf"))

        d_nn = d_ss_masked.min(dim=-1).values
        crowding = torch.exp(-self.crowding_alpha * d_ss).masked_fill(eye, 0.0).sum(dim=-1)
        ss_rbf = self.rbf_ss(d_ss).masked_fill(eye.unsqueeze(-1), 0.0).sum(dim=2)

        scalar = torch.stack([d_nn, crowding], dim=-1)
        feats = torch.cat([periodic_embed, scalar, ss_rbf], dim=-1)
        return feats, d_ss

class EquivariantInteraction2D(nn.Module):
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


class TorusEquivariantSplineConditioner2D(nn.Module):
    """
    Outputs spline parameters per particle for both coordinates.
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
            [EquivariantInteraction2D(feature_dim, hidden_dim) for _ in range(n_interaction_blocks)]
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

class PeriodicParticleSplineCoupling2D(nn.Module):
    """
    Periodic coupling on unit-torus coordinates in 2D.
    """

    def __init__(
        self,
        feature_extractor: PeriodicSolventFeatureExtractor2D,
        conditioner: TorusEquivariantSplineConditioner2D,
        channel_mask,
        num_bins: int = 8,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if len(channel_mask) != 2:
            raise ValueError("channel_mask must have length 2")

        self.feature_extractor = feature_extractor
        self.conditioner = conditioner
        self.num_bins = int(num_bins)
        self.tail_bound = 0.5

        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        mask = torch.as_tensor(channel_mask, dtype=torch.bool).view(1, 1, 2)
        self.register_buffer("channel_mask", mask)

        self.active_idx = [i for i, v in enumerate(channel_mask) if v == 1]
        self.n_active = len(self.active_idx)

        self.params_per_dim = 3 * self.num_bins - 1

    def center_unit(self, u: torch.Tensor) -> torch.Tensor:
        return wrap_unit(u) - 0.5

    def uncenter_unit(self, x: torch.Tensor) -> torch.Tensor:
        return wrap_unit(x + 0.5)

    def forward(self, u: torch.Tensor):
        param = next(self.conditioner.parameters())
        u = u.to(device=param.device, dtype=param.dtype)
        u = wrap_unit(u)

        mask = self.channel_mask.to(device=u.device)
        u_frozen = u.masked_fill(mask, 0.0)

        feats, d_ss = self.feature_extractor(u_frozen)
        raw_params = self.conditioner(feats, d_ss)  # (B,N, 2*params_per_dim)

        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 2, self.params_per_dim)

        active = torch.as_tensor(self.active_idx, device=u.device, dtype=torch.long)
        params = raw_params[:, :, active, :]

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

        feats, d_ss = self.feature_extractor(y_frozen)
        raw_params = self.conditioner(feats, d_ss)

        B, N, _ = raw_params.shape
        raw_params = raw_params.view(B, N, 2, self.params_per_dim)

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

class WrappedTorusSplineFlow2D(nn.Module):
    """
    Flow over flattened solvent unit-torus coordinates in 2D.
    """

    def __init__(self, layers, base):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.base = base
        self.event_shape = (base.shape[0],) if hasattr(base, "shape") else (base.dim,)

    def _flat_to_particle(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] % 2 != 0:
            raise ValueError(f"Expected (B, 2N), got {tuple(x.shape)}")
        return x.view(x.shape[0], -1, 2)

    def _particle_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected (B, N, 2), got {tuple(x.shape)}")
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

def make_lj_flow_2d(n_solvent, box_length_nm, base, n_layers=4, hidden_dim=128, num_bins=8):
    cycle = _cartesian_partition_cycle_2d()
    layers = []

    feature_extractor = PeriodicSolventFeatureExtractor2D(
        box_length_nm=box_length_nm,
        n_rbf_ss=16,
    )

    params_per_dim = 3 * num_bins - 1

    for i in range(n_layers):
        conditioner = TorusEquivariantSplineConditioner2D(
            feature_dim=feature_extractor.feature_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * params_per_dim,
            n_interaction_blocks=2,
        )

        mask = [1 if ax in cycle[i % len(cycle)] else 0 for ax in [0, 1]]

        layers.append(
            PeriodicParticleSplineCoupling2D(
                feature_extractor=feature_extractor,
                conditioner=conditioner,
                channel_mask=mask,
                num_bins=num_bins,
            )
        )

    return WrappedTorusSplineFlow2D(layers=layers, base=base)
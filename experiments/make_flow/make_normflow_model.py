import numpy as np
import normflows as nf
import larsflow as lf
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from fab.target_distributions.base import TargetDistribution

from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform, Transform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
from nflows.distributions.base import Distribution

import math
from typing import Optional,  Dict, Tuple

from dataclasses import dataclass


from fab.wrappers.normflows import WrappedNormFlowModel
from fab.trainable_distributions import TrainableDistribution


try:
    from normflows.distributions.base import BaseDistribution
except Exception:
    BaseDistribution = nn.Module


def make_perm_equi_joint_spline_flow_nf(cfg, target):
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters   # = 6 for your current transform

    flows = []
    for k in range(cfg.flow.layers):
        flow = HybridTorusSO3Flow(
            n_waters=n_waters,
            layers=cfg.flow.layers,
            hidden_dim=cfg.flow.hidden_units,
            n_hidden=cfg.flow.blocks_per_layer,
        )
        flows.append(flow)


    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-solute-gauss":
        base = make_structured_base_from_target(target, trainable=cfg.flow.base.learn_mean_var)
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-solute-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)

class StructuredSoluteWaterGaussian(BaseDistribution):
    """
    Structured exact-ish base with symmetric overlap-aware oxygen resampling.

    Layout
    ------
    z = [solute(6) | water_1(6) | ... | water_W(6)]
    water_k = [z_O(3), omega(3)]

    Notes
    -----
    - log_prob() is still the simple factorized structured density.
    - sample() applies an overlap-aware resampling heuristic for z_O.
    - therefore sample() and log_prob() are no longer exactly matched if
      overlap_resampling is enabled.
    - this is intended as a practical initialization aid.
    """

    def __init__(
        self,
        solute_mean: torch.Tensor,
        solute_log_std: torch.Tensor,
        oxygen_rho_mean: torch.Tensor,
        oxygen_rho_log_std: torch.Tensor,
        omega_mean: torch.Tensor,
        omega_log_std: torch.Tensor,
        n_waters: int,
        oxygen_decoder=None,
        trainable: bool = False,
        rho_eps: float = 1e-6,
        overlap_resampling: bool = True,
        min_oo_distance: float = 0.20,
        max_resample_rounds: int = 12,
    ):
        super().__init__()

        self.n_waters = int(n_waters)
        self.solute_dim = 6
        self.water_dim = 6
        self.dim = self.solute_dim + self.n_waters * self.water_dim
        self.rho_eps = float(rho_eps)

        self.overlap_resampling = bool(overlap_resampling)
        self.min_oo_distance = float(min_oo_distance)
        self.max_resample_rounds = int(max_resample_rounds)
        self.oxygen_decoder = oxygen_decoder

        oxygen_rho_mean = oxygen_rho_mean.reshape(1)
        oxygen_rho_log_std = oxygen_rho_log_std.reshape(1)

        if trainable:
            self.solute_mean = nn.Parameter(solute_mean.clone())
            self.solute_log_std = nn.Parameter(solute_log_std.clone())
            self.oxygen_rho_mean = nn.Parameter(oxygen_rho_mean.clone())
            self.oxygen_rho_log_std = nn.Parameter(oxygen_rho_log_std.clone())
            self.omega_mean = nn.Parameter(omega_mean.clone())
            self.omega_log_std = nn.Parameter(omega_log_std.clone())
        else:
            self.register_buffer("solute_mean", solute_mean.clone())
            self.register_buffer("solute_log_std", solute_log_std.clone())
            self.register_buffer("oxygen_rho_mean", oxygen_rho_mean.clone())
            self.register_buffer("oxygen_rho_log_std", oxygen_rho_log_std.clone())
            self.register_buffer("omega_mean", omega_mean.clone())
            self.register_buffer("omega_log_std", omega_log_std.clone())

    @property
    def event_shape(self):
        return (self.dim,)

    def _normal_log_prob(self, x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor):
        log_2pi = math.log(2.0 * math.pi)
        z = (x - mean) * torch.exp(-log_std)
        return -0.5 * (z**2 + 2.0 * log_std + log_2pi)

    def _sample_unit_sphere(self, n: int, device, dtype):
        v = torch.randn(n, 3, device=device, dtype=dtype)
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)

    def _sample_rho(self, n: int, device, dtype):
        eps = torch.randn(n, 1, device=device, dtype=dtype)
        mean = self.oxygen_rho_mean.to(device=device, dtype=dtype).view(1, 1)
        std = torch.exp(self.oxygen_rho_log_std.to(device=device, dtype=dtype)).view(1, 1)
        rho = mean + std * eps
        return rho.clamp_min(self.rho_eps)

    def _sample_zO_blocks(self, n: int, device, dtype):
        rho = self._sample_rho(n, device, dtype)        # (n,1)
        u = self._sample_unit_sphere(n, device, dtype)  # (n,3)
        return rho * u                                  # (n,3)

    def _decode_oxygen(self, z_O: torch.Tensor):
        """
        z_O: (B,W,3) -> O_rel: (B,W,3)
        """
        if self.oxygen_decoder is None:
            return z_O

        B, W, _ = z_O.shape
        O_rel, _ = self.oxygen_decoder(z_O.reshape(B * W, 3))
        return O_rel.view(B, W, 3)

    def _pairwise_oo_distances(self, O_rel: torch.Tensor):
        """
        O_rel: (B,W,3)
        returns dist: (B,W,W)
        """
        diff = O_rel[:, :, None, :] - O_rel[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        return dist

    def _clashing_water_mask(self, z_O: torch.Tensor):
        """
        z_O: (B,W,3)
        returns clash_mask: (B,W) boolean
        """
        O_rel = self._decode_oxygen(z_O)               # (B,W,3)
        dist = self._pairwise_oo_distances(O_rel)      # (B,W,W)

        B, W, _ = dist.shape
        eye = torch.eye(W, device=dist.device, dtype=torch.bool).view(1, W, W)
        clash_pairs = (dist < self.min_oo_distance) & (~eye)   # (B,W,W)

        # a water clashes if it clashes with any other water
        clash_mask = clash_pairs.any(dim=-1)                    # (B,W)

        print("[BASE CHECK] min_oo_distance threshold =", self.min_oo_distance)
        dist_masked = dist.masked_fill(eye, float("inf"))
        mins = dist_masked.amin(dim=(1, 2))
        print("[BASE CHECK] current min O-O median", mins.median().item(),
            "min", mins.min().item())
        return clash_mask

    def _resample_clashing_oxygens(self, z_O: torch.Tensor):
        """
        z_O: (B,W,3)
        returns updated z_O after several symmetric resampling rounds
        """
        if not self.overlap_resampling or self.n_waters <= 1:
            return z_O

        B, W, _ = z_O.shape
        device, dtype = z_O.device, z_O.dtype

        z = z_O.clone()

        for t in range(self.max_resample_rounds):
            clash_mask = self._clashing_water_mask(z)   # (B,W)

            n_bad_waters = clash_mask.sum().item()
            n_bad_samples = clash_mask.any(dim=1).sum().item()

            print(f"[BASE RESAMPLE] round={t} bad_waters={n_bad_waters} bad_samples={n_bad_samples}")

            if not clash_mask.any():
                print(f"[BASE RESAMPLE] converged at round {t}")
                break

            n_bad = int(clash_mask.sum().item())
            z_new = self._sample_zO_blocks(n_bad, device, dtype)

            z[clash_mask] = z_new
        
        clash_mask = self._clashing_water_mask(z)
        print("[BASE RESAMPLE] final bad_waters", clash_mask.sum().item(),
            "final bad_samples", clash_mask.any(dim=1).sum().item())

        return z

    def log_prob(self, z: torch.Tensor):
        B = z.shape[0]
        device, dtype = z.device, z.dtype

        z_sol = z[:, :self.solute_dim]
        logp_sol = self._normal_log_prob(
            z_sol,
            self.solute_mean.to(device=device, dtype=dtype).view(1, -1),
            self.solute_log_std.to(device=device, dtype=dtype).view(1, -1),
        ).sum(dim=-1)

        z_w = z[:, self.solute_dim:].view(B, self.n_waters, self.water_dim)
        z_O = z_w[..., 0:3]
        omega = z_w[..., 3:6]

        rho = torch.linalg.norm(z_O, dim=-1, keepdim=True).clamp_min(self.rho_eps)

        logp_rho = self._normal_log_prob(
            rho,
            self.oxygen_rho_mean.to(device=device, dtype=dtype).view(1, 1, 1),
            self.oxygen_rho_log_std.to(device=device, dtype=dtype).view(1, 1, 1),
        ).sum(dim=-1)

        logp_dir = -2.0 * torch.log(rho[..., 0].clamp_min(self.rho_eps)) - math.log(4.0 * math.pi)

        logp_omega = self._normal_log_prob(
            omega,
            self.omega_mean.to(device=device, dtype=dtype).view(1, 1, 3),
            self.omega_log_std.to(device=device, dtype=dtype).view(1, 1, 3),
        ).sum(dim=-1)

        return logp_sol + (logp_rho + logp_dir + logp_omega).sum(dim=-1)

    def sample(self, num_samples: int):
        device = self.solute_mean.device
        dtype = self.solute_mean.dtype

        # solute
        eps_sol = torch.randn(num_samples, self.solute_dim, device=device, dtype=dtype)
        z_sol = self.solute_mean.view(1, -1) + eps_sol * torch.exp(self.solute_log_std).view(1, -1)

        # oxygen latents
        z_O = self._sample_zO_blocks(num_samples * self.n_waters, device, dtype)
        z_O = z_O.view(num_samples, self.n_waters, 3)

        O_rel_raw = self._decode_oxygen(z_O)
        dist_raw = self._pairwise_oo_distances(O_rel_raw)

        W = dist_raw.shape[1]
        eye = torch.eye(W, device=dist_raw.device, dtype=torch.bool).unsqueeze(0)
        dist_raw_masked = dist_raw.masked_fill(eye, float("inf"))
        min_oo_raw = dist_raw_masked.amin(dim=(1, 2))

        print("[BASE RAW] min O-O median", min_oo_raw.median().item(),
            "p10", torch.quantile(min_oo_raw, 0.1).item(),
            "p90", torch.quantile(min_oo_raw, 0.9).item(),
            "min", min_oo_raw.min().item())

        # symmetric overlap-aware resampling
        z_O = self._resample_clashing_oxygens(z_O)

        O_rel_final = self._decode_oxygen(z_O)
        dist_final = self._pairwise_oo_distances(O_rel_final)
        dist_final_masked = dist_final.masked_fill(eye, float("inf"))
        min_oo_final = dist_final_masked.amin(dim=(1, 2))

        print("[BASE FINAL] min O-O median", min_oo_final.median().item(),
            "p10", torch.quantile(min_oo_final, 0.1).item(),
            "p90", torch.quantile(min_oo_final, 0.9).item(),
            "min", min_oo_final.min().item())

        # omega
        BW = num_samples * self.n_waters
        eps_omega = torch.randn(BW, 3, device=device, dtype=dtype)
        omega = self.omega_mean.view(1, 3) + eps_omega * torch.exp(self.omega_log_std).view(1, 3)
        omega = omega.view(num_samples, self.n_waters, 3)

        z_w = torch.cat([z_O, omega], dim=-1)
        z = torch.cat([z_sol, z_w.reshape(num_samples, -1)], dim=-1)

        log_q = self.log_prob(z)
        return z, log_q

    def forward(self, num_samples: int):
        return self.sample(num_samples)

def make_structured_solute_water_gaussian_from_target(
    target,
    trainable: bool = False,
    eps: float = 1e-3,
    overlap_resampling: bool = True,
    min_oo_distance: float = 0.20,
    max_resample_rounds: int = 20,
):
    if getattr(target, "train_data_i", None) is not None:
        data_i = target.train_data_i
    elif getattr(target, "val_data_i", None) is not None:
        data_i = target.val_data_i
    else:
        raise ValueError("Need transformed internal data to fit structured base.")

    data_i = data_i.detach()
    if data_i.ndim != 2:
        data_i = data_i.reshape(data_i.shape[0], -1)

    n_waters = target.num_solvent_molecules
    solute_dim = 6
    water_dim = 6
    expected_dim = solute_dim + n_waters * water_dim
    assert data_i.shape[1] == expected_dim, (data_i.shape[1], expected_dim)

    solute = data_i[:, :solute_dim]
    solute_mean = solute.mean(dim=0)
    solute_std = solute.std(dim=0, unbiased=False).clamp_min(eps)

    water = data_i[:, solute_dim:].view(data_i.shape[0], n_waters, water_dim)
    z_O = water[..., 0:3].reshape(-1, 3)
    omega = water[..., 3:6].reshape(-1, 3)

    rho = torch.linalg.norm(z_O, dim=-1, keepdim=True)
    oxygen_rho_mean = rho.mean(dim=0)
    oxygen_rho_std = rho.std(dim=0, unbiased=False).clamp_min(eps)

    omega_mean = omega.mean(dim=0)
    omega_std = omega.std(dim=0, unbiased=False).clamp_min(eps)

    print("[BASE FIT] solute std mean:", solute_std.mean().item())
    print("[BASE FIT] rho mean/std:", oxygen_rho_mean.item(), oxygen_rho_std.item())
    print("[BASE FIT] omega std mean:", omega_std.mean().item())

    return StructuredSoluteWaterGaussian(
        solute_mean=solute_mean,
        solute_log_std=torch.log(solute_std),
        oxygen_rho_mean=oxygen_rho_mean,
        oxygen_rho_log_std=torch.log(oxygen_rho_std),
        omega_mean=omega_mean,
        omega_log_std=torch.log(omega_std),
        n_waters=n_waters,
        oxygen_decoder=target.coordinate_transform.oxygen_latent_to_cartesian,
        trainable=trainable,
        overlap_resampling=overlap_resampling,
        min_oo_distance=min_oo_distance,
        max_resample_rounds=max_resample_rounds,
    )

# def make_structured_solute_water_gaussian_from_target(
#     target,
#     trainable: bool = False,
#     eps: float = 1e-3,
# ):
#     if getattr(target, "train_data_i", None) is not None:
#         data_i = target.train_data_i
#     elif getattr(target, "val_data_i", None) is not None:
#         data_i = target.val_data_i
#     else:
#         raise ValueError("Need transformed internal data to fit structured base.")

#     data_i = data_i.detach()
#     if data_i.ndim != 2:
#         data_i = data_i.reshape(data_i.shape[0], -1)

#     n_waters = target.num_solvent_molecules
#     solute_dim = 6
#     water_dim = 6
#     expected_dim = solute_dim + n_waters * water_dim
#     assert data_i.shape[1] == expected_dim, (data_i.shape[1], expected_dim)

#     # solute
#     solute = data_i[:, :solute_dim]
#     solute_mean = solute.mean(dim=0)
#     solute_std = solute.std(dim=0, unbiased=False).clamp_min(eps)

#     # pooled waters
#     water = data_i[:, solute_dim:].view(data_i.shape[0], n_waters, water_dim)
#     z_O = water[..., 0:3].reshape(-1, 3)
#     omega = water[..., 3:6].reshape(-1, 3)

#     rho = torch.linalg.norm(z_O, dim=-1, keepdim=True)

#     oxygen_rho_mean = rho.mean(dim=0)
#     oxygen_rho_std = rho.std(dim=0, unbiased=False).clamp_min(eps)

#     omega_mean = omega.mean(dim=0)
#     omega_std = omega.std(dim=0, unbiased=False).clamp_min(eps)

#     print("[BASE FIT] solute std mean:", solute_std.mean().item())
#     print("[BASE FIT] rho mean/std:", oxygen_rho_mean.item(), oxygen_rho_std.item())
#     print("[BASE FIT] omega std mean:", omega_std.mean().item())

#     return StructuredSoluteWaterGaussian(
#         solute_mean=solute_mean,
#         solute_log_std=torch.log(solute_std),
#         oxygen_rho_mean=oxygen_rho_mean,
#         oxygen_rho_log_std=torch.log(oxygen_rho_std),
#         omega_mean=omega_mean,
#         omega_log_std=torch.log(omega_std),
#         n_waters=n_waters,
#         trainable=trainable,
#     )


class SoluteSplineCoupling(nf.flows.Flow):
    """
    Invertible spline coupling on the 6D solute block [v1(3), v2(3)].
    Alternates transforming one 3D half conditioned on the other.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        transform_first: bool = True,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.transform_first = transform_first
        self.part_dim = 3

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        params_per_dim = 2 * num_bins + (num_bins + 1)
        out_dim = self.part_dim * params_per_dim

        layers = []
        d = self.part_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def _split(self, z_sol):
        a = z_sol[:, 0:3]
        b = z_sol[:, 3:6]
        if self.transform_first:
            conditioner, target = b, a
        else:
            conditioner, target = a, b
        return conditioner, target

    def _merge(self, conditioner, target):
        if self.transform_first:
            a, b = target, conditioner
        else:
            a, b = conditioner, target
        return torch.cat([a, b], dim=-1)

    def _reshape_params(self, params):
        B = params.shape[0]
        K = self.num_bins
        per_dim = 2 * K + (K + 1)
        params = params.view(B, self.part_dim, per_dim)
        uw = params[..., :K]
        uh = params[..., K:2*K]
        ud = params[..., 2*K:]
        return uw, uh, ud

    def _transform(self, x, params, inverse=False):
        B, D = x.shape
        uw, uh, ud = self._reshape_params(params)

        x_flat = x.reshape(B * D)
        uw_flat = uw.reshape(B * D, self.num_bins)
        uh_flat = uh.reshape(B * D, self.num_bins)
        ud_flat = ud.reshape(B * D, self.num_bins + 1)

        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=uw_flat,
            unnormalized_heights=uh_flat,
            unnormalized_derivatives=ud_flat,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = y_flat.view(B, D)
        log_det = logabsdet_flat.view(B, D).sum(dim=-1)
        return y, log_det

    def forward(self, z_sol):
        conditioner, target = self._split(z_sol)
        params = self.net(conditioner)
        target_out, log_det = self._transform(target, params, inverse=False)
        z_out = self._merge(conditioner, target_out)
        return z_out, log_det

    def inverse(self, z_sol):
        conditioner, target = self._split(z_sol)
        params = self.net(conditioner)
        target_out, log_det = self._transform(target, params, inverse=True)
        z_out = self._merge(conditioner, target_out)
        return z_out, log_det

class JointSoluteWaterFlowLayer(nf.flows.Flow):
    def __init__(self, solute_flow, water_flow, solute_dim):
        super().__init__()
        self.solute_flow = solute_flow
        self.water_flow = water_flow
        self.solute_dim = solute_dim

    def forward(self, z):
        z_sol = z[:, :self.solute_dim]
        z_w = z[:, self.solute_dim:]

        z_sol_out, log_det_sol = self.solute_flow(z_sol)
        z_joint = torch.cat([z_sol_out, z_w], dim=-1)

        z_out, log_det_w = self.water_flow(z_joint)
        return z_out, log_det_sol + log_det_w

    def inverse(self, z):
        z_mid, log_det_w = self.water_flow.inverse(z)

        z_sol = z_mid[:, :self.solute_dim]
        z_w = z_mid[:, self.solute_dim:]

        z_sol_out, log_det_sol = self.solute_flow.inverse(z_sol)
        z_out = torch.cat([z_sol_out, z_w], dim=-1)
        return z_out, log_det_sol + log_det_w


class PermEquiWaterSplineCoupling(nf.flows.Flow):
    """
    Permutation-equivariant spline coupling over water 6D blocks.

    Layout:
      z = [solute_block | water_1(6) | ... | water_W(6)]
    where each water block is:
      water_k = [O_body(3), omega_rel(3)]

    This layer is EXACTLY permutation equivariant because:
      - every water is treated with the same shared networks
      - global context is pooled symmetrically over all waters
      - no water-index-based masking is used

    This layer is EXACTLY invertible because it is a standard coupling,
    but the coupling split is WITHIN each water block:
      - either transform O_body using omega_rel as conditioner
      - or transform omega_rel using O_body as conditioner

    With PBCGlobal3PointSphericalTransform3, the full transform+flow
    combination is:
      - permutation equivariant over waters
      - translation equivariant
      - rotation equivariant
    """

    def __init__(
        self,
        solute_dim: int,
        n_waters: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        transform_oxygen: bool = True,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        assert block_size == 6

        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size

        # water_k = [O_body(3), omega_rel(3)]
        self.part_dim = 3
        self.transform_oxygen = transform_oxygen

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # invariant solute conditioning from Transform3 layout:
        # [t_global(3), v1(3), v2(3)]
        self.shape_dim = 3

        # Shared embedding over the conditioner part of each water (3 dims)
        context_layers = []
        d = self.part_dim
        for _ in range(n_hidden):
            context_layers.append(nn.Linear(d, hidden_dim))
            context_layers.append(nn.ReLU())
            if dropout > 0.0:
                context_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.context_embed = nn.Sequential(*context_layers)

        # Shared parameter net for transformed subblock
        # input = local conditioner(3) + pooled context(H) + solute_shape(3)
        param_in_dim = self.part_dim + hidden_dim + self.shape_dim

        # per scalar dim: widths(K) + heights(K) + derivs(K+1)
        params_per_dim = 2 * num_bins + (num_bins + 1)
        param_out_dim = self.part_dim * params_per_dim

        param_layers = []
        d = param_in_dim
        for _ in range(n_hidden):
            param_layers.append(nn.Linear(d, hidden_dim))
            param_layers.append(nn.ReLU())
            if dropout > 0.0:
                param_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        param_layers.append(nn.Linear(d, param_out_dim))
        self.param_net = nn.Sequential(*param_layers)

    def _split(self, z: torch.Tensor):
        z_sol = z[:, :self.solute_dim]  # (B, S)
        z_w = z[:, self.solute_dim:].reshape(z.shape[0], self.n_waters, self.block_size)  # (B,W,6)
        return z_sol, z_w

    def _merge(self, z_sol: torch.Tensor, z_w: torch.Tensor):
        return torch.cat([z_sol, z_w.reshape(z_sol.shape[0], -1)], dim=-1)

    def _split_water_parts(self, z_w: torch.Tensor):
        """
        z_w: (B, W, 6)
        returns conditioner, target, where each is (B, W, 3)
        """
        O = z_w[..., 0:3]
        omega = z_w[..., 3:6]

        if self.transform_oxygen:
            conditioner = omega
            target = O
        else:
            conditioner = O
            target = omega

        return conditioner, target

    def _merge_water_parts(self, conditioner: torch.Tensor, target: torch.Tensor):
        """
        inverse of _split_water_parts
        """
        if self.transform_oxygen:
            O = target
            omega = conditioner
        else:
            O = conditioner
            omega = target

        return torch.cat([O, omega], dim=-1)

    def _pooled_context(self, conditioner: torch.Tensor):
        """
        conditioner: (B, W, 3)
        returns: (B, H)
        """
        B = conditioner.shape[0]
        h = self.context_embed(conditioner)  # (B, W, H)
        return h.mean(dim=1)                 # symmetric pooling over waters

    def _solute_shape_features(self, z_sol: torch.Tensor) -> torch.Tensor:
        """
        z_sol layout for current PBC transform:
        [v1(3), v2(3)]
        Use rotation/translation-invariant solute shape features.
        """
        v1 = z_sol[:, 0:3]
        v2 = z_sol[:, 3:6]

        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        cos_theta = (v1 * v2).sum(dim=-1, keepdim=True) / (r1 * r2).clamp_min(1e-12)

        return torch.cat([r1, r2, cos_theta.clamp(-1.0, 1.0)], dim=-1)

    def _reshape_params(self, params: torch.Tensor):
        """
        params: (B, W, 3 * (2K + K+1))
        returns:
          uw: (B, W, 3, K)
          uh: (B, W, 3, K)
          ud: (B, W, 3, K+1)
        """
        B, W, _ = params.shape
        K = self.num_bins
        per_dim = 2 * K + (K + 1)

        params = params.view(B, W, self.part_dim, per_dim)
        uw = params[..., :K]
        uh = params[..., K:2 * K]
        ud = params[..., 2 * K:]
        return uw, uh, ud

    def _transform_subblock(self, x_sub: torch.Tensor, params: torch.Tensor, inverse: bool = False):
        """
        x_sub:  (B, W, 3)
        params: (B, W, 3 * (2K + K+1))
        """
        B, W, D = x_sub.shape
        assert D == self.part_dim

        uw, uh, ud = self._reshape_params(params)

        x_flat = x_sub.reshape(B * W * D)
        uw_flat = uw.reshape(B * W * D, self.num_bins)
        uh_flat = uh.reshape(B * W * D, self.num_bins)
        ud_flat = ud.reshape(B * W * D, self.num_bins + 1)

        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=uw_flat,
            unnormalized_heights=uh_flat,
            unnormalized_derivatives=ud_flat,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = y_flat.view(B, W, D)
        log_det = logabsdet_flat.view(B, W, D).sum(dim=(1, 2))
        return y, log_det

    def _compute_params(self, z_sol: torch.Tensor, conditioner: torch.Tensor):
        """
        conditioner: (B, W, 3)
        returns params: (B, W, param_out_dim)
        """
        pooled = self._pooled_context(conditioner)  # (B, H)

        B, W, _ = conditioner.shape
        pooled_rep = pooled[:, None, :].expand(B, W, pooled.shape[-1])

        shape_feat = self._solute_shape_features(z_sol)  # (B, 3)
        shape_rep = shape_feat[:, None, :].expand(B, W, shape_feat.shape[-1])

        inp = torch.cat([conditioner, pooled_rep, shape_rep], dim=-1)
        params = self.param_net(inp)
        return params

    def forward(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)

        conditioner, target = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner)

        target_out, log_det = self._transform_subblock(target, params, inverse=False)
        z_w_out = self._merge_water_parts(conditioner, target_out)

        z_out = self._merge(z_sol, z_w_out)
        return z_out, log_det

    def inverse(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)

        conditioner, target = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner)

        target_out, log_det = self._transform_subblock(target, params, inverse=True)
        z_w_out = self._merge_water_parts(conditioner, target_out)

        z_out = self._merge(z_sol, z_w_out)
        return z_out, log_det

def make_perm_equi_joint_spline_flow_nf(cfg, target):
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters   # = 6 for your current transform

    flows = []
    for k in range(cfg.flow.layers):
        solute_flow = SoluteSplineCoupling(
            hidden_dim=cfg.flow.hidden_units,
            n_hidden=cfg.flow.blocks_per_layer,
            transform_first=(k % 2 == 0),
            num_bins=cfg.flow.num_bins,
            tail_bound=cfg.flow.tail_bound,
        )

        water_flow = PermEquiWaterSplineCoupling(
            solute_dim=solute_dim,
            n_waters=n_waters,
            block_size=6,
            hidden_dim=cfg.flow.hidden_units,
            n_hidden=cfg.flow.blocks_per_layer,
            dropout=cfg.flow.dropout,
            transform_oxygen=(k % 2 == 0),
            num_bins=cfg.flow.num_bins,
            tail_bound=cfg.flow.tail_bound,
        )

        flows.append(
            JointSoluteWaterFlowLayer(
                solute_flow=solute_flow,
                water_flow=water_flow,
                solute_dim=solute_dim,
            )
        )

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-solute-gauss":
        base = make_structured_base_from_target(target, trainable=cfg.flow.base.learn_mean_var)
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-solute-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)

class PermEquiWaterSplineCouplingPairwiseO(nf.flows.Flow):
    """
    Permutation-equivariant spline coupling over water 6D blocks.

    Layout:
      z = [solute_block | water_1(6) | ... | water_W(6)]
    where each water block is:
      water_k = [O_rel(3), omega(3)]

    Exact properties:
      - exactly invertible (standard coupling)
      - permutation equivariant over waters
      - translation-compatible with your transform

    Important note:
      - pairwise O-O context is used only when O is the *conditioner* (i.e. when transforming omega),
        because using current O geometry while transforming O would break exact coupling invertibility.
    """

    def __init__(
        self,
        solute_dim: int,
        n_waters: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        transform_oxygen: bool = True,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        pair_rbf_dim: int = 16,
        pair_rbf_max_dist: float = 3.5,
        oxygen_decoder=None,
    ):
        super().__init__()
        assert block_size == 6

        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size
        self.part_dim = 3
        self.transform_oxygen = transform_oxygen

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.oxygen_decoder = oxygen_decoder

        # solute shape summary = [|v1|, |v2|, cos(theta)]
        self.shape_dim = 3

        # Shared embedding of local conditioner
        context_layers = []
        d = self.part_dim
        for _ in range(n_hidden):
            context_layers.append(nn.Linear(d, hidden_dim))
            context_layers.append(nn.ReLU())
            if dropout > 0.0:
                context_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.context_embed = nn.Sequential(*context_layers)

        # Pairwise O-O radial context
        self.pair_rbf_dim = pair_rbf_dim
        centers = torch.linspace(0.0, pair_rbf_max_dist, pair_rbf_dim)
        widths = torch.full((pair_rbf_dim,), (pair_rbf_max_dist / max(pair_rbf_dim - 1, 1)))
        self.register_buffer("rbf_centers", centers)
        self.register_buffer("rbf_widths", widths)

        pair_layers = []
        d = pair_rbf_dim
        for _ in range(max(1, n_hidden - 1)):
            pair_layers.append(nn.Linear(d, hidden_dim))
            pair_layers.append(nn.ReLU())
            if dropout > 0.0:
                pair_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.pair_embed = nn.Sequential(*pair_layers)

        # Extra pairwise context is only valid when conditioner = O
        extra_pair_dim = hidden_dim if not transform_oxygen else 0

        param_in_dim = self.part_dim + hidden_dim + self.shape_dim + extra_pair_dim

        params_per_dim = 2 * num_bins + (num_bins + 1)
        param_out_dim = self.part_dim * params_per_dim

        param_layers = []
        d = param_in_dim
        for _ in range(n_hidden):
            param_layers.append(nn.Linear(d, hidden_dim))
            param_layers.append(nn.ReLU())
            if dropout > 0.0:
                param_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        param_layers.append(nn.Linear(d, param_out_dim))
        self.param_net = nn.Sequential(*param_layers)

    def _split(self, z: torch.Tensor):
        z_sol = z[:, :self.solute_dim]
        z_w = z[:, self.solute_dim:].reshape(z.shape[0], self.n_waters, self.block_size)
        return z_sol, z_w

    def _merge(self, z_sol: torch.Tensor, z_w: torch.Tensor):
        return torch.cat([z_sol, z_w.reshape(z_sol.shape[0], -1)], dim=-1)

    def _split_water_parts(self, z_w):
        z_O = z_w[..., 0:3]
        omega = z_w[..., 3:6]

        if self.transform_oxygen:
            # transform O using omega as conditioner
            conditioner = omega
            target = z_O
        else:
            # transform omega using O as conditioner
            conditioner = z_O
            target = omega

        return conditioner, target, z_O, omega
    def _decode_oxygen(self, z_O: torch.Tensor):
        if self.oxygen_decoder is None:
            return z_O

        B, W, _ = z_O.shape
        z_flat = z_O.reshape(B * W, 3)

        O_flat, _ = self.oxygen_decoder(z_flat)  # ignore logdet
        return O_flat.view(B, W, 3)

    def _merge_water_parts(self, conditioner: torch.Tensor, target: torch.Tensor):
        if self.transform_oxygen:
            O = target
            omega = conditioner
        else:
            O = conditioner
            omega = target
        return torch.cat([O, omega], dim=-1)

    def _pooled_context(self, conditioner: torch.Tensor):
        h = self.context_embed(conditioner)    # (B,W,H)
        return h.mean(dim=1)                   # permutation-symmetric

    def _solute_shape_features(self, z_sol: torch.Tensor) -> torch.Tensor:
        """
        z_sol layout for your current transform:
          [v1(3), v2(3)]
        """
        v1 = z_sol[:, 0:3]
        v2 = z_sol[:, 3:6]

        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        cos_theta = (v1 * v2).sum(dim=-1, keepdim=True) / (r1 * r2).clamp_min(1e-12)

        return torch.cat([r1, r2, cos_theta.clamp(-1.0, 1.0)], dim=-1)

    def _rbf(self, d: torch.Tensor):
        """
        d: (...,)
        returns (..., K)
        """
        c = self.rbf_centers.view(*([1] * d.ndim), -1)
        w = self.rbf_widths.view(*([1] * d.ndim), -1).clamp_min(1e-6)
        return torch.exp(-0.5 * ((d.unsqueeze(-1) - c) / w) ** 2)

    def _pairwise_o_context(self, z_O):
        O = self._decode_oxygen(z_O)

        B, W, _ = O.shape
        diff = O[:, :, None, :] - O[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)

        rbf = self._rbf(dist)

        eye = torch.eye(W, device=O.device, dtype=O.dtype).view(1, W, W, 1)
        rbf = rbf * (1.0 - eye)

        pair_h = self.pair_embed(rbf)
        ctx = pair_h.sum(dim=2)
        return ctx

    def _reshape_params(self, params: torch.Tensor):
        B, W, _ = params.shape
        K = self.num_bins
        per_dim = 2 * K + (K + 1)

        params = params.view(B, W, self.part_dim, per_dim)
        uw = params[..., :K]
        uh = params[..., K:2 * K]
        ud = params[..., 2 * K:]
        return uw, uh, ud

    def _transform_subblock(self, x_sub: torch.Tensor, params: torch.Tensor, inverse: bool = False):
        B, W, D = x_sub.shape
        assert D == self.part_dim

        uw, uh, ud = self._reshape_params(params)

        x_flat = x_sub.reshape(B * W * D)
        uw_flat = uw.reshape(B * W * D, self.num_bins)
        uh_flat = uh.reshape(B * W * D, self.num_bins)
        ud_flat = ud.reshape(B * W * D, self.num_bins + 1)

        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=uw_flat,
            unnormalized_heights=uh_flat,
            unnormalized_derivatives=ud_flat,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = y_flat.view(B, W, D)
        log_det = logabsdet_flat.view(B, W, D).sum(dim=(1, 2))
        return y, log_det

    def _compute_params(self, z_sol: torch.Tensor, conditioner: torch.Tensor, z_O: torch.Tensor):
        pooled = self._pooled_context(conditioner)      # (B,H)

        B, W, _ = conditioner.shape
        pooled_rep = pooled[:, None, :].expand(B, W, pooled.shape[-1])

        shape_feat = self._solute_shape_features(z_sol) # (B,3)
        shape_rep = shape_feat[:, None, :].expand(B, W, shape_feat.shape[-1])

        pieces = [conditioner, pooled_rep, shape_rep]

        # Only when O is the unchanged conditioner (transforming omega)
        # do we inject pairwise O-O geometry exactly without breaking invertibility.
        if not self.transform_oxygen:
            pair_ctx = self._pairwise_o_context(z_O)      # (B,W,H)
            pieces.append(pair_ctx)

        inp = torch.cat(pieces, dim=-1)
        params = self.param_net(inp)
        return params

    def forward(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target, z_O, omega = self._split_water_parts(z_w)

        params = self._compute_params(z_sol, conditioner, z_O)
        target_out, log_det = self._transform_subblock(target, params, inverse=False)

        z_w_out = self._merge_water_parts(conditioner, target_out)
        z_out = self._merge(z_sol, z_w_out)
        return z_out, log_det

    def inverse(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target, z_O, omega = self._split_water_parts(z_w)

        params = self._compute_params(z_sol, conditioner, z_O)
        target_out, log_det = self._transform_subblock(target, params, inverse=True)

        z_w_out = self._merge_water_parts(conditioner, target_out)
        z_out = self._merge(z_sol, z_w_out)
        return z_out, log_det

def make_perm_equi_spline_flow_nf(cfg, target):
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters

    flows = []
    for k in range(cfg.flow.layers):
        flows.append(
            PermEquiWaterSplineCouplingPairwiseO(
                solute_dim=solute_dim,
                n_waters=n_waters,
                block_size=6,
                hidden_dim=cfg.flow.hidden_units,
                n_hidden=cfg.flow.blocks_per_layer,
                dropout=cfg.flow.dropout,
                transform_oxygen=(k % 2 == 0),   # alternate O <-> omega coupling
                num_bins=cfg.flow.num_bins,
                tail_bound=cfg.flow.tail_bound,
                oxygen_decoder=target.coordinate_transform.oxygen_latent_to_cartesian,
            )
        )

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target,
            learn_mean_var=cfg.flow.base.learn_mean_var,
        )
    elif cfg.flow.base.type == "structured-solute-gauss":
        # base = make_structured_base_from_target(target, trainable=cfg.flow.base.learn_mean_var)
        base = make_structured_solute_water_gaussian_from_target(
            target,
            trainable=cfg.flow.base.learn_mean_var,
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)

# def make_perm_equi_spline_flow_nf(cfg, target):
#     dim = target.internal_dim
#     n_waters = target.num_solvent_molecules
#     solute_dim = dim - 6 * n_waters

#     flows = []
#     for k in range(cfg.flow.layers):
#         flows.append(
#             PermEquiWaterSplineCoupling(
#                 solute_dim=solute_dim,
#                 n_waters=n_waters,
#                 block_size=6,
#                 hidden_dim=cfg.flow.hidden_units,
#                 n_hidden=cfg.flow.blocks_per_layer,
#                 dropout=cfg.flow.dropout,
#                 transform_oxygen=(k % 2 == 0),   # alternate O <-> omega coupling
#                 num_bins=cfg.flow.num_bins,
#                 tail_bound=cfg.flow.tail_bound,
#             )
#         )

#     if cfg.flow.base.type == "gauss":
#         base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
#     elif cfg.flow.base.type == "structured-gauss":
#         base = make_structured_diag_gaussian_from_target(
#             target,
#             learn_mean_var=cfg.flow.base.learn_mean_var,
#         )
#     elif cfg.flow.base.type == "structured-solute-gauss":
#         base = make_structured_base_from_target(target, trainable=cfg.flow.base.learn_mean_var)
#     else:
#         raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

#     flow = nf.NormalizingFlow(base, flows)
#     return WrappedNormFlowModel(flow)

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor, batch_size: int, device, dtype):
        """
        t: scalar tensor
        returns: (B, emb_dim)
        """
        half = self.emb_dim // 2
        if half == 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)

        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), half, device=device, dtype=dtype
            )
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=device, dtype=dtype)], dim=-1)
        return emb


class PermEquiWaterCNFVectorField(nn.Module):
    """
    Permutation-equivariant vector field over water 6D blocks.

    Input state:
      z = [solute_block | water_1(6) | ... | water_W(6)]

    solute_block layout is assumed to be:
      [t_global(3), v1(3), v2(3)]

    The vector field:
      - leaves solute block untouched
      - updates all waters simultaneously
      - uses shared per-water embedding
      - uses symmetric pooling over all waters
      - conditions only on invariant solute shape features
      - optional time embedding
    """

    def __init__(
        self,
        solute_dim: int,
        n_waters: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        time_emb_dim: int = 16,
    ):
        super().__init__()
        assert block_size == 6
        assert solute_dim >= 9, "Expected solute block [t_global(3), v1(3), v2(3)]"

        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.shape_dim = 3
        self.time_emb_dim = time_emb_dim

        self.time_embed = TimeEmbedding(time_emb_dim)

        # Shared embedding over all waters
        embed_layers = []
        d = block_size
        for _ in range(n_hidden):
            embed_layers.append(nn.Linear(d, hidden_dim))
            embed_layers.append(nn.ReLU())
            if dropout > 0.0:
                embed_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.water_embed = nn.Sequential(*embed_layers)

        # Shared per-water vector field network
        # input = local water(6) + pooled_all(H) + shape_feat(3) + time_emb(T)
        vf_in_dim = block_size + hidden_dim + self.shape_dim + self.time_emb_dim

        vf_layers = []
        d = vf_in_dim
        for _ in range(n_hidden):
            vf_layers.append(nn.Linear(d, hidden_dim))
            vf_layers.append(nn.Tanh())
            if dropout > 0.0:
                vf_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        vf_layers.append(nn.Linear(d, block_size))
        self.vf_net = nn.Sequential(*vf_layers)

    def _split(self, z: torch.Tensor):
        z_sol = z[:, :self.solute_dim]  # (B, S)
        z_w = z[:, self.solute_dim:].reshape(z.shape[0], self.n_waters, self.block_size)  # (B,W,6)
        return z_sol, z_w

    def _merge(self, z_sol: torch.Tensor, z_w: torch.Tensor):
        return torch.cat([z_sol, z_w.reshape(z_sol.shape[0], -1)], dim=-1)

    def _solute_shape_features(self, z_sol: torch.Tensor) -> torch.Tensor:
        """
        z_sol layout: [t_global(3), v1(3), v2(3), ...]
        Uses only rotation/translation-invariant solute shape.
        """
        v1 = z_sol[:, 3:6]
        v2 = z_sol[:, 6:9]

        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        cos_theta = (v1 * v2).sum(dim=-1, keepdim=True) / (r1 * r2).clamp_min(1e-12)
        return torch.cat([r1, r2, cos_theta.clamp(-1.0, 1.0)], dim=-1)

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        """
        t: scalar tensor shape (1,1) or (1,) or broadcastable
        z: (B, D)
        returns dz/dt: (B, D)
        """
        z_sol, z_w = self._split(z)

        B = z.shape[0]
        device = z.device
        dtype = z.dtype

        # Shared embedding over all waters
        h = self.water_embed(z_w)      # (B, W, H)
        pooled = h.mean(dim=1)         # (B, H)

        shape_feat = self._solute_shape_features(z_sol)  # (B, 3)
        t_emb = self.time_embed(
            t.reshape(1, 1).to(device=device, dtype=dtype), B, device, dtype
        )  # (B, T)

        pooled_rep = pooled[:, None, :].expand(B, self.n_waters, pooled.shape[-1])
        shape_rep = shape_feat[:, None, :].expand(B, self.n_waters, shape_feat.shape[-1])
        t_rep = t_emb[:, None, :].expand(B, self.n_waters, t_emb.shape[-1])

        inp = torch.cat([z_w, pooled_rep, shape_rep, t_rep], dim=-1)   # (B,W,*)
        dz_w = self.vf_net(inp)                                         # (B,W,6)

        # Keep pose block unchanged
        dz_sol = torch.zeros_like(z_sol)
        return self._merge(dz_sol, dz_w)


class PermEquiWaterCNF(nf.flows.Flow):
    """
    FFJORD-style permutation-equivariant CNF over water 6D blocks.

    Works with transformed coordinates from PBCGlobal3PointSphericalTransform3:
      z = [t_global(3), v1(3), v2(3), water_1(6), ..., water_W(6)]

    Properties:
      - exact permutation equivariance over waters
      - with Transform3, gives translation/rotation equivariance
      - leaves the 9D global pose block unchanged
      - updates all waters simultaneously (no index-based masking)

    Notes:
      - uses fixed-step RK4 integration
      - uses Hutchinson estimator for divergence
      - for debugging you can turn on exact_divergence=True, but it is much slower
    """

    def __init__(
        self,
        solute_dim: int,
        n_waters: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        time_emb_dim: int = 16,
        n_steps: int = 8,
        hutchinson_samples: int = 1,
        exact_divergence: bool = False,
    ):
        super().__init__()
        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size
        self.n_steps = n_steps
        self.hutchinson_samples = hutchinson_samples
        self.exact_divergence = exact_divergence

        self.vf = PermEquiWaterCNFVectorField(
            solute_dim=solute_dim,
            n_waters=n_waters,
            block_size=block_size,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )

    def _divergence_exact(self, t: torch.Tensor, z: torch.Tensor):
        """
        Exact divergence wrt the water block only.
        Very expensive. Mainly for debugging.
        """
        z = z.requires_grad_(True)
        dz = self.vf(t, z)

        z_w = z[:, self.solute_dim:]
        dz_w = dz[:, self.solute_dim:]

        B, D = z_w.shape
        div = torch.zeros(B, device=z.device, dtype=z.dtype)

        for j in range(D):
            grad_j = torch.autograd.grad(
                dz_w[:, j].sum(),
                z,
                retain_graph=True,
                create_graph=self.training,
            )[0][:, self.solute_dim + j]
            div = div + grad_j

        return dz, div

    def _divergence_hutchinson(self, t: torch.Tensor, z: torch.Tensor):
        """
        Hutchinson trace estimator for divergence wrt water block only.
        """
        z = z.requires_grad_(True)
        dz = self.vf(t, z)

        z_w = z[:, self.solute_dim:]
        dz_w = dz[:, self.solute_dim:]

        B, D = z_w.shape
        div = torch.zeros(B, device=z.device, dtype=z.dtype)

        for _ in range(self.hutchinson_samples):
            eps = torch.randn_like(z_w)
            # trace J ≈ eps^T J eps
            jvp = torch.autograd.grad(
                (dz_w * eps).sum(),
                z,
                retain_graph=True,
                create_graph=self.training,
            )[0][:, self.solute_dim:]
            div = div + (jvp * eps).sum(dim=1)

        div = div / float(self.hutchinson_samples)
        return dz, div

    def _aug_dynamics(self, t: torch.Tensor, z: torch.Tensor, logp: torch.Tensor):
        with torch.enable_grad():
            if self.exact_divergence:
                dz, div = self._divergence_exact(t, z)
            else:
                dz, div = self._divergence_hutchinson(t, z)

        dlogp = -div
        return dz, dlogp

    def _rk4_step(self, t, dt, z, logp):
        k1_z, k1_lp = self._aug_dynamics(t, z, logp)
        k2_z, k2_lp = self._aug_dynamics(t + 0.5 * dt, z + 0.5 * dt * k1_z, logp + 0.5 * dt * k1_lp)
        k3_z, k3_lp = self._aug_dynamics(t + 0.5 * dt, z + 0.5 * dt * k2_z, logp + 0.5 * dt * k2_lp)
        k4_z, k4_lp = self._aug_dynamics(t + dt, z + dt * k3_z, logp + dt * k3_lp)

        z_new = z + (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
        lp_new = logp + (dt / 6.0) * (k1_lp + 2.0 * k2_lp + 2.0 * k3_lp + k4_lp)
        return z_new, lp_new

    def _integrate(self, z0: torch.Tensor, t0: float, t1: float):
        B = z0.shape[0]
        logp = torch.zeros(B, device=z0.device, dtype=z0.dtype)

        ts = torch.linspace(t0, t1, self.n_steps + 1, device=z0.device, dtype=z0.dtype)
        z = z0
        for i in range(self.n_steps):
            t = ts[i].reshape(1, 1)
            dt = ts[i + 1] - ts[i]
            z, logp = self._rk4_step(t, dt, z, logp)
        return z, logp

    def forward(self, z: torch.Tensor):
        """
        Integrate from t=0 -> t=1.
        Returns transformed z and accumulated logdet.
        """
        z1, logdet = self._integrate(z, 0.0, 1.0)
        return z1, logdet

    def inverse(self, z: torch.Tensor):
        """
        Integrate from t=1 -> t=0.
        """
        z0, logdet = self._integrate(z, 1.0, 0.0)
        return z0, logdet


def make_perm_equi_cnf_flow_nf(cfg, target):
    """
    Factory for the permutation-equivariant CNF flow.

    Expected target.internal_dim layout:
      dim = 9 + 6 * n_waters
      solute block = [t_global(3), v1(3), v2(3)]
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters   # should be 9 for Transform3

    flows = []
    for _ in range(cfg.flow.layers):
        flows.append(
            PermEquiWaterCNF(
                solute_dim=solute_dim,
                n_waters=n_waters,
                block_size=6,
                hidden_dim=cfg.flow.hidden_units,
                n_hidden=cfg.flow.blocks_per_layer,
                dropout=cfg.flow.dropout,
                time_emb_dim=getattr(cfg.flow, "time_emb_dim", 16),
                n_steps=getattr(cfg.flow, "cnf_steps", 8),
                hutchinson_samples=getattr(cfg.flow, "hutchinson_samples", 1),
                exact_divergence=getattr(cfg.flow, "exact_divergence", False),
            )
        )

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target,
            learn_mean_var=cfg.flow.base.learn_mean_var,
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)


def make_normflow_flow(dim: int, n_flow_layers: int, layer_nodes_per_dim: int, act_norm: bool):
    # Define list of flows
    flows = []
    # layer_width = dim * layer_nodes_per_dim
    layer_width = 1024
    for i in range(n_flow_layers):
        # Neural network with two hidden layers having 32 units each
        # Last layer is initialized by zeros making training more stable
        d = int((dim / 2) + 0.5)
        param_map = nf.nets.MLP([d, layer_width, layer_width, 2 * (dim - d)], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
        # Swap dimensions
        flows.append(nf.flows.InvertibleAffine(dim))
        # ActNorm
        if act_norm:
            flows.append(nf.flows.ActNorm(dim))
    return flows


def make_normflow_snf(
    base: nf.distributions.BaseDistribution,
    target: nf.distributions.Target,
    dim: int,
    n_flow_layers: int,
    layer_nodes_per_dim: int,
    act_norm: bool,
    it_snf_layer: int = 2,
    mh_prop_scale: float = 0.1,
    mh_steps: int = 10,
    hmc_n_leapfrog_steps: int = 5,
    transition_operator_type="metropolis",
):
    """Setup stochastic normalising flow model."""
    assert transition_operator_type in ["metropolis", "hmc"]
    # Define list of flows
    flows = []
    layer_width = dim * layer_nodes_per_dim
    for i in range(n_flow_layers):
        # Neural network with two hidden layers having 32 units each
        # Last layer is initialized by zeros making training more stable
        d = int((dim / 2) + 0.5)
        param_map = nf.nets.MLP([d, layer_width, layer_width, 2 * (dim - d)], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
        # Swap dimensions
        flows.append(nf.flows.InvertibleAffine(dim))
        # ActNorm
        if act_norm:
            flows.append(nf.flows.ActNorm(dim))
        # Sampling layer of SNF
        if (i + 1) % it_snf_layer == 0:
            lam = (i + 1) / n_flow_layers
            dist = nf.distributions.LinearInterpolation(target, base, lam)
            if transition_operator_type == "metropolis":
                prop_scale = mh_prop_scale * np.ones(dim)
                proposal = nf.distributions.DiagGaussianProposal((dim,), prop_scale)
                flows.append(nf.flows.MetropolisHastings(dist, proposal, mh_steps))
            elif transition_operator_type == "hmc":
                flows.append(
                    nf.flows.HamiltonianMonteCarlo(
                        dist,
                        steps=hmc_n_leapfrog_steps,
                        log_step_size=torch.ones(dim) * torch.log(torch.tensor(mh_steps)),
                        log_mass=torch.zeros(dim),
                        max_abs_grad=1e4,
                    )
                )
            else:
                raise NotImplementedError
    return flows


def make_wrapped_normflow_realnvp(
    dim: int, n_flow_layers: int = 5, layer_nodes_per_dim: int = 10, act_norm: bool = True
) -> TrainableDistribution:
    """Created a wrapped normflows distribution using the example from the normflows page."""
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_flow(
        dim, n_flow_layers=n_flow_layers, layer_nodes_per_dim=layer_nodes_per_dim, act_norm=act_norm
    )
    model = nf.NormalizingFlow(base, flows)
    wrapped_dist = WrappedNormFlowModel(model)
    if act_norm:
        wrapped_dist.sample((500,))  # ensure we call sample to initialise the ActNorm layers
    return wrapped_dist


def make_wrapped_normflow_snf_model(
    dim: int,
    target: nf.distributions.Target,
    n_flow_layers: int = 5,
    layer_nodes_per_dim: int = 10,
    act_norm: bool = True,
    it_snf_layer: int = 2,
    mh_prop_scale: float = 0.1,
    mh_steps: int = 10,
    hmc_n_leapfrog_steps: int = 5,
    transition_operator_type="metropolis",
) -> TrainableDistribution:
    """Created normflows distribution with sampling layers."""
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_snf(
        base,
        target,
        dim,
        n_flow_layers=n_flow_layers,
        layer_nodes_per_dim=layer_nodes_per_dim,
        act_norm=act_norm,
        it_snf_layer=it_snf_layer,
        mh_prop_scale=mh_prop_scale,
        mh_steps=mh_steps,
        hmc_n_leapfrog_steps=hmc_n_leapfrog_steps,
        transition_operator_type=transition_operator_type,
    )
    model = nf.NormalizingFlow(base, flows, p=target)
    if act_norm:
        model.sample(500)  # ensure we call sample to initialise the ActNorm layers
    wrapped_dist = WrappedNormFlowModel(model)
    return wrapped_dist


def make_wrapped_normflow_resampled_flow(
    dim: int,
    n_flow_layers: int = 5,
    layer_nodes_per_dim: int = 10,
    act_norm: bool = True,
    a_hidden_layer: int = 2,
    a_hidden_units: int = 256,
    T: int = 100,
    eps: float = 0.05,
    resenet: bool = True,
) -> TrainableDistribution:
    """Created normflows distribution with resampled base."""
    if resenet:
        resnet = nf.nets.ResidualNet(dim, 1, a_hidden_units, num_blocks=a_hidden_layer)
        a = torch.nn.Sequential(resnet, torch.nn.Sigmoid())
    else:
        hu = [dim] + [a_hidden_units] * a_hidden_layer + [1]
        a = nf.nets.MLP(hu, output_fn="sigmoid")
    base = lf.distributions.ResampledGaussian(dim, a, T, eps, trainable=False)
    flows = make_normflow_flow(
        dim, n_flow_layers=n_flow_layers, layer_nodes_per_dim=layer_nodes_per_dim, act_norm=act_norm
    )
    model = nf.NormalizingFlow(base, flows)
    if act_norm:
        model.sample(500)  # ensure we call sample to initialise the ActNorm layers
    wrapped_dist = WrappedNormFlowModel(model)
    return wrapped_dist


def make_wrapped_normflow_solvent_flow(config, target):
    """
    Setup Flow model.
    """

    # Flow parameters
    flow_type = config["flow"]["type"]
    seed = config["training"]["seed"]
    dim = target.internal_dim  # 6 degrees of freedom are fixed in the target distribution

    # Periodic indices for solvent are phi and theta. In the representation with explicit dofs, these are all indices
    # except [0::3], which represent the radial distance instead. In the representation with the 6 dofs removed, these
    # are indices [2] (representing phi of the third atom), and then all indices except [3::3]. Indices 0 and 1 are the
    # radial distance of atom2 and atom3 (atom1 has all zeros for r, phi, theta).
    periodic_inds = np.array([2] + [i for i in range(3, dim) if i % 3 != 0])

    # Feature mixing happens through using different features as identity and transform features. This is controlled
    #  by the mask passed to the mask=True / permute_mask=True parameter. Tail bounds are shifting as well, so at
    #  every step the Spline still satisfies the conditions for periodicity. Note that the base distribution is not
    #  part of the Splines: it comes in when evaluating the log_prob of the flow, or when sampling from the flow, but
    #  not when flowing a given sample forward or backward.

    # Base distribution
    # Indices of periodic variables (e.g., phi, theta) are given by `periodic_inds`, these should have their owns scale
    # Original implementation uses 2 * pi / `std_of_angle` for base_scale of uniform distribution. I think it makes
    #  more sense to not use the std_of_angle here, as the want the flow to operate on unit scale (also if
    #  std_of_angle is not unit scale, then there will be a large difference between the scale of the Gaussian
    #  base dist N(0, 1), and the scale of the uniform base dist U(0, 2 * pi / std_of_angle).
    # Note that we have two different types of angles: phi and theta. But we can just use a pi range for both, and
    #  multiply the one for phi by 2 at Flow output (in principle the Flow can learn that phi angles have double
    #  range, but we might as well put that in manually, so that on initialisation the relative scale matches).
    bound_circ = np.pi
    # Bound of the Spline tails.
    tail_bound = 5.0 * torch.ones(dim)
    tail_bound[periodic_inds] = bound_circ

    circ_shift = None if not "circ_shift" in config["flow"] else config["flow"]["circ_shift"]

    # Base distribution
    if config["flow"]["base"]["type"] == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=config["flow"]["base"]["learn_mean_var"])
    elif config["flow"]["base"]["type"] == "gauss-uni":
        base_scale = torch.ones(dim)  # Stddev of Gaussian or width of uniform
        base_scale[periodic_inds] = bound_circ
        base = nf.distributions.UniformGaussian(dim, periodic_inds, scale=base_scale)
        base.shape = (dim,)
    else:
        raise NotImplementedError("The base distribution " + config["flow"]["base"]["type"] + " is not implemented")

    # Flow layers
    layers = []
    n_layers = config["flow"]["blocks"]

    for i in range(n_layers):
        if flow_type == "ar-nsf":  # Autoregressive Rational Spline Normalizing Flow
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            layers.append(
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    num_input_channels=dim,
                    num_blocks=bl,
                    num_hidden_channels=hu,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    permute_mask=True,
                    init_identity=ii,
                    dropout_probability=dropout,
                )
            )
        elif flow_type == "coup-nsf":  # Coupling Rational Spline Normalizing Flow
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            dropout = config["flow"]["dropout"]
            layers.append(
                nf.flows.CoupledRationalQuadraticSpline(
                    num_input_channels=dim,
                    num_blocks=bl,
                    num_hidden_channels=hu,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    dropout_probability=dropout,
                )
            )
        elif flow_type == "circ-ar-nsf":  # Circular AutoRegressive Rational Spline Normalizing Flow
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            layers.append(
                nf.flows.CircularAutoregressiveRationalQuadraticSpline(
                    num_input_channels=dim,
                    num_blocks=bl,
                    num_hidden_channels=hu,
                    ind_circ=periodic_inds,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    permute_mask=True,
                    init_identity=ii,
                    dropout_probability=dropout,
                )
            )
        elif flow_type == "circ-coup-nsf":  # Circular Coupled Rational Spline Normalizing Flow
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            if i % 2 == 0:
                mask = nf.utils.masks.create_random_binary_mask(dim, seed=seed + i)
            else:
                mask = 1 - mask
            layers.append(
                nf.flows.CircularCoupledRationalQuadraticSpline(
                    num_input_channels=dim,
                    num_blocks=bl,
                    num_hidden_channels=hu,
                    ind_circ=periodic_inds,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    init_identity=ii,
                    dropout_probability=dropout,
                    mask=mask,
                )
            )
        else:
            raise NotImplementedError("The flow type " + flow_type + " is not implemented for solvent systems.")

        if config["flow"]["mixing"] == "affine":
            layers.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        elif config["flow"]["mixing"] == "permute":
            layers.append(nf.flows.Permute(dim))

        if config["flow"]["actnorm"]:
            layers.append(nf.flows.ActNorm(dim))

        # Shift the periodic angles.
        if i % 2 == 1 and i != n_layers - 1:
            if circ_shift == "constant":
                layers.append(nf.flows.PeriodicShift(periodic_inds, bound=bound_circ, shift=bound_circ))
            elif circ_shift == "random":
                gen = torch.Generator().manual_seed(seed + i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                layers.append(nf.flows.PeriodicShift(periodic_inds, bound=bound_circ, shift=shift_scale * bound_circ))

        # SNF
        if "snf" in config["flow"]:
            if (i + 1) % config["flow"]["snf"]["every_n"] == 0:
                prop_scale = config["flow"]["snf"]["proposal_std"] * np.ones(dim)
                steps = config["flow"]["snf"]["steps"]
                proposal = nf.distributions.DiagGaussianProposal((dim,), prop_scale)
                lam = (i + 1) / n_layers
                dist = nf.distributions.LinearInterpolation(target, base, lam)
                layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # Map input to periodic interval
    # The purpose is that incoming samples from a dataset get periodically wrapped to the interval [-pi, pi],
    #  or the equivalent scaled version.
    layers.append(nf.flows.PeriodicWrap(periodic_inds, bound_circ))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)
    wrapped_flow = WrappedNormFlowModel(flow)

    return wrapped_flow

def make_wrapped_normflow_pbc_cartesian(config):
    flow_type = config["flow"]["type"]
    seed = config["training"]["seed"]

    L = config["target"]["box_length_nm"]
    dim = config["target"]["cartesian_dim"]

    periodic_inds = np.arange(dim)          # ALL coordinates periodic

    bound_circ = 0.5 * float(L)             # coords in [-L/2, L/2]

    tail_bound = bound_circ * torch.ones(dim)

    # base: uniform for periodic coords + (optional) gaussian noise
    base_scale = torch.ones(dim) * bound_circ
    base = nf.distributions.UniformGaussian(dim, periodic_inds, scale=base_scale)
    base.shape = (dim,)

    layers = []
    n_layers = config["flow"]["blocks"]

    mask = nf.utils.masks.create_random_binary_mask(dim, seed=seed)

    for i in range(n_layers):
        bl = config["flow"]["blocks_per_layer"]
        hu = config["flow"]["hidden_units"]
        nb = config["flow"]["num_bins"]
        ii = config["flow"]["init_identity"]
        dropout = config["flow"]["dropout"]
        
        if i % 2 == 1:
            mask = 1 - mask


        # use the *circular* spline versions
        layers.append(
            nf.flows.CircularCoupledRationalQuadraticSpline(
                num_input_channels=dim, 
                num_blocks=bl, 
                num_hidden_channels=hu, 
                ind_circ=periodic_inds,
                tail_bound=tail_bound,
                num_bins=nb,
                init_identity=ii,
                dropout_probability=dropout,
                mask=mask,
            )
        )

        if config["flow"]["mixing"] == "permute":
            layers.append(nf.flows.Permute(dim))

        if config["flow"]["actnorm"]:
            layers.append(nf.flows.ActNorm(dim))

    # ensure everything stays on the torus [-L/2, L/2]
    layers.append(nf.flows.PeriodicWrap(periodic_inds, bound_circ))

    flow = nf.NormalizingFlow(base, layers)
    return WrappedNormFlowModel(flow)

def water_block_permutation(dim: int, n_prefix: int, block_size: int, seed: int) -> torch.Tensor:
    """
    Keep the first n_prefix dims fixed, permute the remaining dims in whole blocks.
    """
    assert (dim - n_prefix) % block_size == 0
    n_blocks = (dim - n_prefix) // block_size

    rng = np.random.RandomState(seed)
    bperm = rng.permutation(n_blocks)

    perm = list(range(n_prefix))
    for b in bperm:
        start = n_prefix + b * block_size
        perm.extend(range(start, start + block_size))

    return torch.tensor(perm, dtype=torch.long)


class PermuteFixed(nf.flows.Flow):
    """A fixed permutation flow (so we can permute groups, not individual dims)."""
    def __init__(self, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("perm", perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        self.register_buffer("inv_perm", inv)

    def forward(self, z):
        z = z[:, self.perm]
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, log_det

    def inverse(self, z):
        z = z[:, self.inv_perm]
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, log_det


class TrainableDiagonalNormal(Distribution):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, trainable: bool = True):
        super().__init__()

        if trainable:
            self.loc = nn.Parameter(mean.clone())
            self.log_scale = nn.Parameter(log_std.clone())
        else:
            self.register_buffer("loc", mean.clone())
            self.register_buffer("log_scale", log_std.clone())

        self._shape = torch.Size([mean.numel()])

    def _log_prob(self, inputs, context):
        log_2pi = math.log(2.0 * math.pi)
        z = (inputs - self.loc) * torch.exp(-self.log_scale)
        log_prob = -0.5 * (z**2 + 2.0 * self.log_scale + log_2pi)
        return log_prob.sum(dim=-1)

    def _sample(self, num_samples, context):
        eps = torch.randn(
            num_samples,
            *self._shape,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return self.loc.unsqueeze(0) + eps * torch.exp(self.log_scale).unsqueeze(0)

def make_nflows_diag_gaussian_from_target(target, trainable: bool = False, eps: float = 1e-3):
    if getattr(target, "train_data_i", None) is not None:
        data_i = target.train_data_i
    elif getattr(target, "val_data_i", None) is not None:
        data_i = target.val_data_i
    else:
        with torch.no_grad():
            ref_x = target.transform_data.reshape(1, -1).to(target.device)
            data_i, _ = target.coordinate_transform.inverse(ref_x)

    data_i = data_i.detach()
    mean = data_i.mean(dim=0)
    std = data_i.std(dim=0, unbiased=False).clamp_min(eps)

    return TrainableDiagonalNormal(
        mean=mean,
        log_std=torch.log(std),
        trainable=trainable,
    )

def make_structured_diag_gaussian_from_target(target, learn_mean_var: bool = True, eps: float = 1e-3):
    """
    Create a diagonal Gaussian base initialized from target internal-coordinate statistics.
    """
    import normflows as nf
    import torch

    dim = target.internal_dim

    # Prefer train data, then val, then transform_data mapped to i-space
    if getattr(target, "train_data_i", None) is not None:
        data_i = target.train_data_i
    elif getattr(target, "val_data_i", None) is not None:
        data_i = target.val_data_i
    else:
        # fallback: use single reference transformed point
        with torch.no_grad():
            ref_x = target.transform_data.reshape(1, -1).to(target.device)
            data_i, _ = target.coordinate_transform.inverse(ref_x)

    data_i = data_i.detach()
    mean = data_i.mean(dim=0)
    std = data_i.std(dim=0, unbiased=False).clamp_min(eps)
    std = std.clamp(min=0.05)

    base = nf.distributions.DiagGaussian(dim, trainable=learn_mean_var)

    # normflows stores loc/log_scale as parameters in many versions
    with torch.no_grad():
        if hasattr(base, "loc"):
            base.loc.copy_(mean)
        if hasattr(base, "log_scale"):
            base.log_scale.copy_(torch.log(std))

    return base

def make_water_block_mask(
    dim: int,
    solute_dim: int,
    n_waters: int,
    block_size: int = 6,
    transform_even_waters: bool = True,
    solute_as_condition: bool = True,
) -> torch.Tensor:
    """
    nflows convention:
      mask[d] > 0  -> conditioning/identity dims
      mask[d] <= 0 -> transformed dims
    """
    assert dim == solute_dim + n_waters * block_size
    mask = torch.zeros(dim, dtype=torch.float32)

    if solute_as_condition:
        mask[:solute_dim] = 1.0

    for w in range(n_waters):
        start = solute_dim + w * block_size
        end = start + block_size
        is_even = (w % 2 == 0)
        transform_this = (is_even == transform_even_waters)

        if transform_this:
            mask[start:end] = 0.0
        else:
            mask[start:end] = 1.0

    return mask


def make_transform_net(hidden_features: int, num_blocks: int, dropout: float):
    def create_net(in_features: int, out_features: int):
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            activation=F.relu,
            dropout_probability=dropout,
            use_batch_norm=False,
        )
    return create_net

class BlockPermutation(Transform):
    def __init__(self, permutation: torch.Tensor):
        super().__init__()
        self.register_buffer("_permutation", permutation.long())
        inv = torch.empty_like(self._permutation)
        inv[self._permutation] = torch.arange(len(self._permutation), device=self._permutation.device)
        self.register_buffer("_inverse_permutation", inv)

    def forward(self, inputs, context=None):
        outputs = inputs[:, self._permutation]
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = inputs[:, self._inverse_permutation]
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

def water_block_permutation(dim: int, n_prefix: int, block_size: int, seed: int) -> torch.Tensor:
    assert (dim - n_prefix) % block_size == 0
    n_blocks = (dim - n_prefix) // block_size

    rng = np.random.RandomState(seed)
    bperm = rng.permutation(n_blocks)

    perm = list(range(n_prefix))
    for b in bperm:
        start = n_prefix + b * block_size
        perm.extend(range(start, start + block_size))

    return torch.tensor(perm, dtype=torch.long)

class WrappedNFlowsModel(torch.nn.Module):
    def __init__(self, flow, dim: int):
        super().__init__()
        self._flow = flow
        self.event_shape = torch.Size([dim])

    def log_prob(self, x):
        return self._flow.log_prob(x)

    def sample(self, shape):
        if isinstance(shape, tuple):
            n = shape[0]
        else:
            n = shape
        return self._flow.sample(n)

    def sample_and_log_prob(self, shape):
        if isinstance(shape, tuple):
            n = shape[0]
        else:
            n = shape
        x = self._flow.sample(n)
        log_q = self._flow.log_prob(x)
        return x, log_q

def make_coupled_spline_flow_nf(cfg, target):
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters

    transforms = []
    create_net = make_transform_net(cfg.flow.hidden_units, cfg.flow.blocks_per_layer, cfg.flow.dropout)

    for k in range(cfg.flow.layers):
        mask = make_water_block_mask(
            dim=dim,
            solute_dim=solute_dim,
            n_waters=n_waters,
            block_size=6,
            transform_even_waters=(k % 2 == 0),
            solute_as_condition=True,
        )

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net,
                num_bins=cfg.flow.num_bins,
                tails="linear",
                tail_bound=cfg.flow.tail_bound,
                apply_unconditional_transform=False,
            )
        )

        perm = water_block_permutation(
            dim=dim,
            n_prefix=solute_dim,
            block_size=6,
            seed=cfg.training.seed + k,
        )
        transforms.append(BlockPermutation(perm))

    transform = CompositeTransform(transforms)

    # Base distribution
    if cfg.flow.base.type == "gauss":
        base = StandardNormal(shape=[dim])
    elif cfg.flow.base.type == "structured-gauss":
        base = make_nflows_diag_gaussian_from_target(
            target,
            trainable=cfg.flow.base.learn_mean_var,
        )

        
    else:
        raise NotImplementedError(
            "Supported base types: 'gauss', 'structured-gauss'."
        )

    flow = Flow(transform, base)


    return WrappedNFlowsModel(flow, dim=dim)


# def make_coupled_spline_flow_nf(cfg: DictConfig, target: TargetDistribution) -> nf.NormalizingFlow:
#     """
#     Coupled RQS spline flow using normflows (nf.flows.*), with 6D-group masks.

#     Your i-space is Euclidean (solute 6 + each water 6), so no periodic wrapping.
#     """
#     dim = target.internal_dim
#     n_waters = target.num_solvent_molecules
#     solute_dim = dim - 6 * n_waters
    
#     # Tail bounds per-dimension (vector accepted by normflows spline flows)
#     tail_bound = cfg.flow.tail_bound * torch.ones(dim)

#     # Base distribution
#     if cfg.flow.base.type == "gauss":
#         base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
#     elif cfg.flow.base.type == "structured-gauss":
#         base = make_structured_diag_gaussian_from_target(
#             target,
#             learn_mean_var=cfg.flow.base.learn_mean_var,
#         )
#     elif cfg.flow.base.type == "structured-gauss2":
#         base = make_structured_gaussian_from_target(
#                 target,
#                 trainable=True,
#                 cov_eps=1e-4,
#                 cov_shrink=5e-2,
#             )
        
#     else:
#         raise NotImplementedError(
#             "Supported base types: 'gauss', 'structured-gauss'."
#         )


    # for k in range(cfg.flow.layers):
    #     flows.append(
    #         nf.flows.CoupledRationalQuadraticSpline(
    #             num_input_channels=dim,
    #             num_blocks=cfg.flow.blocks_per_layer,
    #             num_hidden_channels=cfg.flow.hidden_units,
    #             tail_bound=tail_bound,
    #             num_bins=cfg.flow.num_bins,
    #             dropout_probability=cfg.flow.dropout,
    #             reverse_mask=bool(k % 2),
    #         )
    #     )
        
        
    #     perm = water_block_permutation(
    #         dim=dim,
    #         n_prefix=solute_dim,
    #         block_size=6,
    #         seed=cfg.training.seed + k,
    #     )
    #     flows.append(PermuteFixed(perm))

        # # Mixing layer
        # if cfg.flow.mixing == "affine":
        #     flows.append(nf.flows.InvertibleAffine(dim, use_lu=True))

        # if cfg.flow.actnorm:
        #     flows.append(nf.flows.ActNorm(dim))
    # flows = []
    # for k in range(cfg.flow.layers):
    #     flows.append(
    #         WaterBlockAffineCoupling(
    #             solute_dim=solute_dim,
    #             n_waters=target.num_solvent_molecules,
    #             hidden=cfg.flow.hidden_units,
    #             transform_mask_even=bool(k % 2 == 0),
    #         )
    #     )
    #     flows.append(
    #         InvertibleWaterBlockMix(
    #             solute_dim=solute_dim,
    #             n_waters=target.num_solvent_molecules,
    #         )
    #     )
    #     if cfg.flow.actnorm:
    #         flows.append(nf.flows.ActNorm(dim))

    # flow = nf.NormalizingFlow(base, flows)
    # wrapped_flow = WrappedNormFlowModel(flow)

    # return wrapped_flow

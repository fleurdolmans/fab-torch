import math

import torch
from torch import nn
import normflows as nf
from nflows.distributions.base import Distribution

try:
    from normflows.distributions.base import BaseDistribution
except Exception:
    BaseDistribution = nn.Module


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

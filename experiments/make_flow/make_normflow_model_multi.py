import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf
from omegaconf import DictConfig

from fab.wrappers.normflows import WrappedNormFlowModel

# This import may need a tiny adjustment depending on your install.
# In many setups this works:
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class PermuteFixed(nf.flows.Flow):
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


class SharedWaterBlockSplineCoupling(nf.flows.Flow):
    """
    Shared-weight spline coupling over 6D water blocks.

    Layout:
        z = [solute_prefix | water_0(6) | ... | water_{W-1}(6)]

    Solute prefix is used as conditioner here, not transformed here.
    Water blocks are transformed in parity groups.
    """
    def __init__(
        self,
        dim: int,
        n_prefix: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        transform_odd_blocks: bool = False,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        assert (dim - n_prefix) % block_size == 0
        assert block_size == 6

        self.dim = dim
        self.n_prefix = n_prefix
        self.block_size = block_size
        self.n_blocks = (dim - n_prefix) // block_size
        self.transform_odd_blocks = transform_odd_blocks

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        block_ids = torch.arange(self.n_blocks)
        if transform_odd_blocks:
            transform_blocks = block_ids[block_ids % 2 == 1]
            context_blocks = block_ids[block_ids % 2 == 0]
        else:
            transform_blocks = block_ids[block_ids % 2 == 0]
            context_blocks = block_ids[block_ids % 2 == 1]

        self.register_buffer("transform_blocks", transform_blocks)
        self.register_buffer("context_blocks", context_blocks)

        # context = [solute_prefix, pooled_context_water(6)]
        in_dim = n_prefix + block_size

        # per transformed scalar dimension:
        #   widths: num_bins
        #   heights: num_bins
        #   derivatives: num_bins + 1
        params_per_dim = 2 * num_bins + (num_bins + 1)
        out_dim = block_size * params_per_dim

        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
        )

    def _split(self, z: torch.Tensor):
        B = z.shape[0]
        prefix = z[:, :self.n_prefix]
        waters = z[:, self.n_prefix:].view(B, self.n_blocks, self.block_size)
        return prefix, waters

    def _merge(self, prefix: torch.Tensor, waters: torch.Tensor):
        return torch.cat([prefix, waters.reshape(prefix.shape[0], -1)], dim=-1)

    def _pooled_context(self, context_waters: torch.Tensor, B: int, dtype, device):
        if context_waters.shape[1] == 0:
            return torch.zeros((B, self.block_size), dtype=dtype, device=device)
        return context_waters.mean(dim=1)

    def _shared_params(self, prefix: torch.Tensor, context_waters: torch.Tensor):
        B = prefix.shape[0]
        pooled = self._pooled_context(context_waters, B, prefix.dtype, prefix.device)
        context = torch.cat([prefix, pooled], dim=-1)  # (B, n_prefix+6)
        params = self.net(context)
        return params

    def _reshape_params(self, params: torch.Tensor):
        """
        params: (B, 6 * (2*num_bins + num_bins+1))
        returns:
            unnorm_widths  (B, 6, K)
            unnorm_heights (B, 6, K)
            unnorm_derivs  (B, 6, K+1)
        """
        B = params.shape[0]
        K = self.num_bins
        per_dim = 2 * K + (K + 1)

        params = params.view(B, self.block_size, per_dim)

        uw = params[:, :, :K]
        uh = params[:, :, K:2 * K]
        ud = params[:, :, 2 * K:]
        return uw, uh, ud

    def _transform_block(self, x_block: torch.Tensor, params: torch.Tensor, inverse: bool = False):
        """
        x_block: (B, Nt, 6)
        params:  (B, P) shared params from conditioner
        """
        B, Nt, D = x_block.shape
        assert D == self.block_size

        uw, uh, ud = self._reshape_params(params)  # each (B, 6, ...)
        # share across all transformed waters
        uw = uw[:, None, :, :].expand(B, Nt, D, self.num_bins)
        uh = uh[:, None, :, :].expand(B, Nt, D, self.num_bins)
        ud = ud[:, None, :, :].expand(B, Nt, D, self.num_bins + 1)

        x_flat = x_block.reshape(B * Nt * D)
        uw_flat = uw.reshape(B * Nt * D, self.num_bins)
        uh_flat = uh.reshape(B * Nt * D, self.num_bins)
        ud_flat = ud.reshape(B * Nt * D, self.num_bins + 1)

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

        y = y_flat.view(B, Nt, D)
        logabsdet = logabsdet_flat.view(B, Nt, D).sum(dim=(1, 2))
        return y, logabsdet

    def forward(self, z: torch.Tensor):
        prefix, waters = self._split(z)

        context_waters = waters[:, self.context_blocks, :]
        target_waters = waters[:, self.transform_blocks, :]

        params = self._shared_params(prefix, context_waters)
        transformed, log_det = self._transform_block(target_waters, params, inverse=False)

        waters_out = waters.clone()
        waters_out[:, self.transform_blocks, :] = transformed
        z_out = self._merge(prefix, waters_out)
        return z_out, log_det

    def inverse(self, z: torch.Tensor):
        prefix, waters = self._split(z)

        context_waters = waters[:, self.context_blocks, :]
        target_waters = waters[:, self.transform_blocks, :]

        params = self._shared_params(prefix, context_waters)
        inverted, log_det = self._transform_block(target_waters, params, inverse=True)

        waters_out = waters.clone()
        waters_out[:, self.transform_blocks, :] = inverted
        z_out = self._merge(prefix, waters_out)
        return z_out, log_det

class SoluteSplineCoupling(nf.flows.Flow):
    """
    Simple coupling acting only on the solute prefix.
    Conditioner = water mean-pooled context.
    Transform = spline on the solute dims.
    """
    def __init__(
        self,
        dim: int,
        n_prefix: int,
        block_size: int = 6,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        dropout: float = 0.0,
        reverse_mask: bool = False,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        assert n_prefix in (3, 6)
        assert (dim - n_prefix) % block_size == 0

        self.dim = dim
        self.n_prefix = n_prefix
        self.block_size = block_size
        self.n_blocks = (dim - n_prefix) // block_size

        self.reverse_mask = reverse_mask
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        idx = torch.arange(n_prefix)
        if reverse_mask:
            self.register_buffer("transform_idx", idx[idx % 2 == 1])
            self.register_buffer("context_idx", idx[idx % 2 == 0])
        else:
            self.register_buffer("transform_idx", idx[idx % 2 == 0])
            self.register_buffer("context_idx", idx[idx % 2 == 1])

        params_per_dim = 2 * num_bins + (num_bins + 1)
        in_dim = len(self.context_idx) + block_size
        out_dim = len(self.transform_idx) * params_per_dim

        self.net = MLP(in_dim, out_dim, hidden_dim, n_hidden, dropout)

    def _pooled_water(self, z: torch.Tensor):
        B = z.shape[0]
        waters = z[:, self.n_prefix:].view(B, self.n_blocks, self.block_size)
        return waters.mean(dim=1)

    def _reshape_params(self, params: torch.Tensor):
        B = params.shape[0]
        K = self.num_bins
        per_dim = 2 * K + (K + 1)
        D = params.shape[1] // per_dim
        params = params.view(B, D, per_dim)
        uw = params[:, :, :K]
        uh = params[:, :, K:2 * K]
        ud = params[:, :, 2 * K:]
        return uw, uh, ud

    def _transform(self, x: torch.Tensor, params: torch.Tensor, inverse: bool):
        B, D = x.shape
        uw, uh, ud = self._reshape_params(params)

        y, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x.reshape(B * D),
            unnormalized_widths=uw.reshape(B * D, self.num_bins),
            unnormalized_heights=uh.reshape(B * D, self.num_bins),
            unnormalized_derivatives=ud.reshape(B * D, self.num_bins + 1),
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        return y.view(B, D), logabsdet.view(B, D).sum(dim=1)

    def forward(self, z: torch.Tensor):
        z_out = z.clone()
        pooled_water = self._pooled_water(z)
        context = torch.cat([z[:, self.context_idx], pooled_water], dim=-1)
        params = self.net(context)

        x_t = z[:, self.transform_idx]
        y_t, log_det = self._transform(x_t, params, inverse=False)
        z_out[:, self.transform_idx] = y_t
        return z_out, log_det

    def inverse(self, z: torch.Tensor):
        z_out = z.clone()
        pooled_water = self._pooled_water(z)
        context = torch.cat([z[:, self.context_idx], pooled_water], dim=-1)
        params = self.net(context)

        y_t = z[:, self.transform_idx]
        x_t, log_det = self._transform(y_t, params, inverse=True)
        z_out[:, self.transform_idx] = x_t
        return z_out, log_det

def make_shared_water_spline_flow_nf(cfg: DictConfig, target):
    dim = target.internal_dim

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    else:
        raise NotImplementedError("Only gaussian base implemented.")

    solute_dim = target.internal_dim - 6 * target.num_solvent_molecules
    assert solute_dim in (3, 6)

    flows = []
    for k in range(cfg.flow.layers):
        # 1. solute subflow
        flows.append(
            SoluteSplineCoupling(
                dim=dim,
                n_prefix=solute_dim,
                block_size=6,
                hidden_dim=cfg.flow.hidden_units,
                n_hidden=cfg.flow.blocks_per_layer,
                dropout=cfg.flow.dropout,
                reverse_mask=bool(k % 2),
                num_bins=cfg.flow.num_bins,
                tail_bound=cfg.flow.tail_bound,
            )
        )

        # 2. shared water spline coupling
        flows.append(
            SharedWaterBlockSplineCoupling(
                dim=dim,
                n_prefix=solute_dim,
                block_size=6,
                hidden_dim=cfg.flow.hidden_units,
                n_hidden=cfg.flow.blocks_per_layer,
                dropout=cfg.flow.dropout,
                transform_odd_blocks=bool(k % 2),
                num_bins=cfg.flow.num_bins,
                tail_bound=cfg.flow.tail_bound,
            )
        )

        # 3. permute water blocks
        perm = water_block_permutation(
            dim=dim,
            n_prefix=solute_dim,
            block_size=6,
            seed=cfg.training.seed + k,
        )
        flows.append(PermuteFixed(perm))

        if getattr(cfg.flow, "mixing", None) == "affine":
            flows.append(nf.flows.InvertibleAffine(dim, use_lu=True))

        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)
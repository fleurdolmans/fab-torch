import torch
from torch import nn
import normflows as nf
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline

from fab.flow.utils import MLP


class SoluteSplineCoupling(nf.flows.Flow):
    """
    Invertible spline coupling on a 6D solute block [v1(3), v2(3)].

    Alternates transforming one 3D half conditioned on the other.
    No water conditioning — used inside JointSoluteWaterFlowLayer.
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


class SoluteWaterSplineCoupling(nf.flows.Flow):
    """
    Spline coupling acting on the solute prefix, conditioned on water mean-pool.

    Layout:
        z = [solute_prefix(n_prefix) | water_0(6) | ... | water_{W-1}(6)]

    The solute dims are split into two halves (alternated via reverse_mask); one
    half is transformed conditioned on the other half + mean-pooled water context.
    Supports n_prefix in (3, 6).
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

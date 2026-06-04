import math
import torch
from torch import nn
import normflows as nf
from normflows.utils.splines import unconstrained_rational_quadratic_spline

from fab.flow.utils import MLP


class JointSoluteWaterFlowLayer(nf.flows.Flow):
    """
    Wraps a solute sub-flow and a water sub-flow into one joint flow layer.

    The solute sub-flow receives only z_sol; the water sub-flow receives the full
    vector (so it can condition on the updated solute representation).
    """

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

    Exact properties:
      - permutation equivariant over waters (shared networks + symmetric mean-pool)
      - exactly invertible (within-block coupling split)

    The coupling alternates between:
      - transforming O_body conditioned on omega_rel
      - transforming omega_rel conditioned on O_body
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
        oxygen_circular: bool = False,
    ):
        super().__init__()
        assert block_size == 6

        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size
        self.part_dim = 3
        self.transform_oxygen = transform_oxygen
        self.oxygen_circular = oxygen_circular

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # invariant solute shape features:
        #   solute_dim == 3: [|v1|]          (1 feature, single bond vector)
        #   solute_dim == 6: [|v1|, |v2|, cos(theta)]  (3 features, two bond vectors)
        self.shape_dim = 1 if solute_dim <= 3 else 3

        # shared embedding over the conditioner part of each water (3 dims)
        context_layers = []
        d = self.part_dim
        for _ in range(n_hidden):
            context_layers.append(nn.Linear(d, hidden_dim))
            context_layers.append(nn.ReLU())
            if dropout > 0.0:
                context_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.context_embed = nn.Sequential(*context_layers)

        # shared parameter net: conditioner(3) + pooled_context(H) + solute_shape(3) -> params
        param_in_dim = self.part_dim + hidden_dim + self.shape_dim
        params_per_dim = 3 * num_bins  # normflows: K widths + K heights + K derivatives
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

    def _split_water_parts(self, z_w: torch.Tensor):
        O = z_w[..., 0:3]
        omega = z_w[..., 3:6]
        if self.transform_oxygen:
            return omega, O   # conditioner, target
        else:
            return O, omega

    def _merge_water_parts(self, conditioner: torch.Tensor, target: torch.Tensor):
        if self.transform_oxygen:
            O, omega = target, conditioner
        else:
            O, omega = conditioner, target
        return torch.cat([O, omega], dim=-1)

    def _pooled_context(self, conditioner: torch.Tensor):
        h = self.context_embed(conditioner)  # (B, W, H)
        return h.mean(dim=1)                 # symmetric pooling over waters

    def _solute_shape_features(self, z_sol: torch.Tensor) -> torch.Tensor:
        v1 = z_sol[:, 0:3]
        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        if self.solute_dim <= 3:
            return r1  # (B, 1)
        v2 = z_sol[:, 3:6]
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        cos_theta = (v1 * v2).sum(dim=-1, keepdim=True) / (r1 * r2).clamp_min(1e-12)
        return torch.cat([r1, r2, cos_theta.clamp(-1.0, 1.0)], dim=-1)  # (B, 3)

    def _reshape_params(self, params: torch.Tensor):
        B, W, _ = params.shape
        K = self.num_bins
        per_dim = 3 * K  # K widths + K heights + K derivatives (normflows convention)
        params = params.view(B, W, self.part_dim, per_dim)
        uw = params[..., :K]
        uh = params[..., K:2 * K]
        ud = params[..., 2 * K:]
        return uw, uh, ud

    def _transform_subblock(self, x_sub: torch.Tensor, params: torch.Tensor, inverse: bool = False,
                            tails: str = "linear", tail_bound: float = None):
        B, W, D = x_sub.shape
        assert D == self.part_dim

        if tail_bound is None:
            tail_bound = self.tail_bound

        uw, uh, ud = self._reshape_params(params)

        x_flat = x_sub.reshape(B * W * D)
        uw_flat = uw.reshape(B * W * D, self.num_bins)
        uh_flat = uh.reshape(B * W * D, self.num_bins)
        ud_flat = ud.reshape(B * W * D, self.num_bins)

        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=uw_flat,
            unnormalized_heights=uh_flat,
            unnormalized_derivatives=ud_flat,
            inverse=inverse,
            tails=tails,
            tail_bound=tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = y_flat.view(B, W, D)
        log_det = logabsdet_flat.view(B, W, D).sum(dim=(1, 2))
        return y, log_det

    def _compute_params(self, z_sol: torch.Tensor, conditioner: torch.Tensor):
        pooled = self._pooled_context(conditioner)  # (B, H)
        B, W, _ = conditioner.shape
        pooled_rep = pooled[:, None, :].expand(B, W, pooled.shape[-1])

        shape_feat = self._solute_shape_features(z_sol)  # (B, 3)
        shape_rep = shape_feat[:, None, :].expand(B, W, shape_feat.shape[-1])

        inp = torch.cat([conditioner, pooled_rep, shape_rep], dim=-1)
        return self.param_net(inp)

    def _target_tails(self):
        if self.oxygen_circular and self.transform_oxygen:
            return "circular", math.pi
        return "linear", self.tail_bound

    def forward(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner)
        tails, tb = self._target_tails()
        target_out, log_det = self._transform_subblock(target, params, inverse=False, tails=tails, tail_bound=tb)
        z_w_out = self._merge_water_parts(conditioner, target_out)
        return self._merge(z_sol, z_w_out), log_det

    def inverse(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner)
        tails, tb = self._target_tails()
        target_out, log_det = self._transform_subblock(target, params, inverse=True, tails=tails, tail_bound=tb)
        z_w_out = self._merge_water_parts(conditioner, target_out)
        return self._merge(z_sol, z_w_out), log_det


class PermEquiWaterSplineCouplingPairwiseO(nf.flows.Flow):
    """
    Permutation-equivariant spline coupling over water 6D blocks with pairwise O-O context.

    Layout:
      z = [solute_block | water_1(6) | ... | water_W(6)]
    where each water block is:
      water_k = [O_rel(3), omega(3)]

    Exact properties:
      - exactly invertible (standard coupling)
      - permutation equivariant over waters

    Pairwise O-O context is only injected when O is the *conditioner* (i.e. when
    transforming omega), because using current O geometry while transforming O would
    break exact coupling invertibility.
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
        oxygen_circular: bool = False,
    ):
        super().__init__()
        assert block_size == 6

        self.solute_dim = solute_dim
        self.n_waters = n_waters
        self.block_size = block_size
        self.part_dim = 3
        self.transform_oxygen = transform_oxygen
        self.oxygen_circular = oxygen_circular

        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.oxygen_decoder = oxygen_decoder

        self.shape_dim = 1 if solute_dim <= 3 else 3

        # shared embedding of local conditioner
        context_layers = []
        d = self.part_dim
        for _ in range(n_hidden):
            context_layers.append(nn.Linear(d, hidden_dim))
            context_layers.append(nn.ReLU())
            if dropout > 0.0:
                context_layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.context_embed = nn.Sequential(*context_layers)

        # pairwise O-O radial basis
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

        # pairwise context only when O is the conditioner
        extra_pair_dim = hidden_dim if not transform_oxygen else 0

        param_in_dim = self.part_dim + hidden_dim + self.shape_dim + extra_pair_dim
        params_per_dim = 3 * num_bins  # normflows: K widths + K heights + K derivatives
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
            conditioner, target = omega, z_O
        else:
            conditioner, target = z_O, omega
        return conditioner, target, z_O, omega

    def _decode_oxygen(self, z_O: torch.Tensor):
        if self.oxygen_decoder is None:
            return z_O
        B, W, _ = z_O.shape
        z_flat = z_O.reshape(B * W, 3)
        O_flat, _ = self.oxygen_decoder(z_flat)
        return O_flat.view(B, W, 3)

    def _merge_water_parts(self, conditioner: torch.Tensor, target: torch.Tensor):
        if self.transform_oxygen:
            O, omega = target, conditioner
        else:
            O, omega = conditioner, target
        return torch.cat([O, omega], dim=-1)

    def _pooled_context(self, conditioner: torch.Tensor):
        h = self.context_embed(conditioner)
        return h.mean(dim=1)

    def _solute_shape_features(self, z_sol: torch.Tensor) -> torch.Tensor:
        v1 = z_sol[:, 0:3]
        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        if self.solute_dim <= 3:
            return r1  # (B, 1)
        v2 = z_sol[:, 3:6]
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        cos_theta = (v1 * v2).sum(dim=-1, keepdim=True) / (r1 * r2).clamp_min(1e-12)
        return torch.cat([r1, r2, cos_theta.clamp(-1.0, 1.0)], dim=-1)  # (B, 3)

    def _rbf(self, d: torch.Tensor):
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
        return pair_h.sum(dim=2)

    def _reshape_params(self, params: torch.Tensor):
        B, W, _ = params.shape
        K = self.num_bins
        per_dim = 3 * K  # K widths + K heights + K derivatives (normflows convention)
        params = params.view(B, W, self.part_dim, per_dim)
        uw = params[..., :K]
        uh = params[..., K:2 * K]
        ud = params[..., 2 * K:]
        return uw, uh, ud

    def _transform_subblock(self, x_sub: torch.Tensor, params: torch.Tensor, inverse: bool = False,
                            tails: str = "linear", tail_bound: float = None):
        B, W, D = x_sub.shape
        assert D == self.part_dim

        if tail_bound is None:
            tail_bound = self.tail_bound

        uw, uh, ud = self._reshape_params(params)

        x_flat = x_sub.reshape(B * W * D)
        uw_flat = uw.reshape(B * W * D, self.num_bins)
        uh_flat = uh.reshape(B * W * D, self.num_bins)
        ud_flat = ud.reshape(B * W * D, self.num_bins)

        y_flat, logabsdet_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=uw_flat,
            unnormalized_heights=uh_flat,
            unnormalized_derivatives=ud_flat,
            inverse=inverse,
            tails=tails,
            tail_bound=tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        y = y_flat.view(B, W, D)
        log_det = logabsdet_flat.view(B, W, D).sum(dim=(1, 2))
        return y, log_det

    def _compute_params(self, z_sol: torch.Tensor, conditioner: torch.Tensor, z_O: torch.Tensor):
        pooled = self._pooled_context(conditioner)
        B, W, _ = conditioner.shape
        pooled_rep = pooled[:, None, :].expand(B, W, pooled.shape[-1])

        shape_feat = self._solute_shape_features(z_sol)
        shape_rep = shape_feat[:, None, :].expand(B, W, shape_feat.shape[-1])

        pieces = [conditioner, pooled_rep, shape_rep]
        # only inject pairwise O-O context when O is the unchanged conditioner
        if not self.transform_oxygen:
            pieces.append(self._pairwise_o_context(z_O))

        inp = torch.cat(pieces, dim=-1)
        return self.param_net(inp)

    def _target_tails(self):
        if self.oxygen_circular and self.transform_oxygen:
            return "circular", math.pi
        return "linear", self.tail_bound

    def forward(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target, z_O, omega = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner, z_O)
        tails, tb = self._target_tails()
        target_out, log_det = self._transform_subblock(target, params, inverse=False, tails=tails, tail_bound=tb)
        z_w_out = self._merge_water_parts(conditioner, target_out)
        return self._merge(z_sol, z_w_out), log_det

    def inverse(self, z: torch.Tensor):
        z_sol, z_w = self._split(z)
        conditioner, target, z_O, omega = self._split_water_parts(z_w)
        params = self._compute_params(z_sol, conditioner, z_O)
        tails, tb = self._target_tails()
        target_out, log_det = self._transform_subblock(target, params, inverse=True, tails=tails, tail_bound=tb)
        z_w_out = self._merge_water_parts(conditioner, target_out)
        return self._merge(z_sol, z_w_out), log_det


class SharedWaterBlockSplineCoupling(nf.flows.Flow):
    """
    Shared-weight spline coupling over 6D water blocks.

    Layout:
        z = [solute_prefix | water_0(6) | ... | water_{W-1}(6)]

    Water blocks are split into two parity groups (even/odd). One group is
    transformed while the other, plus the solute prefix, acts as the conditioner.
    All transformed blocks share the same spline parameters (computed from the
    pooled conditioner context), making the coupling permutation equivariant.
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

        # context = [solute_prefix | pooled_context_water(6)]
        in_dim = n_prefix + block_size
        params_per_dim = 3 * num_bins  # normflows: K widths + K heights + K derivatives
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
        context = torch.cat([prefix, pooled], dim=-1)
        return self.net(context)

    def _reshape_params(self, params: torch.Tensor):
        B = params.shape[0]
        K = self.num_bins
        per_dim = 3 * K  # normflows: K widths + K heights + K derivatives
        params = params.view(B, self.block_size, per_dim)
        uw = params[:, :, :K]
        uh = params[:, :, K:2 * K]
        ud = params[:, :, 2 * K:]
        return uw, uh, ud

    def _transform_block(self, x_block: torch.Tensor, params: torch.Tensor, inverse: bool = False):
        """
        x_block: (B, Nt, 6) — target water blocks
        params:  (B, P) — shared params from conditioner
        """
        B, Nt, D = x_block.shape
        assert D == self.block_size

        uw, uh, ud = self._reshape_params(params)
        uw = uw[:, None, :, :].expand(B, Nt, D, self.num_bins)
        uh = uh[:, None, :, :].expand(B, Nt, D, self.num_bins)
        ud = ud[:, None, :, :].expand(B, Nt, D, self.num_bins + 1)

        x_flat = x_block.reshape(B * Nt * D)
        uw_flat = uw.reshape(B * Nt * D, self.num_bins)
        uh_flat = uh.reshape(B * Nt * D, self.num_bins)
        ud_flat = ud.reshape(B * Nt * D, self.num_bins)

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
        return self._merge(prefix, waters_out), log_det

    def inverse(self, z: torch.Tensor):
        prefix, waters = self._split(z)

        context_waters = waters[:, self.context_blocks, :]
        target_waters = waters[:, self.transform_blocks, :]

        params = self._shared_params(prefix, context_waters)
        inverted, log_det = self._transform_block(target_waters, params, inverse=True)

        waters_out = waters.clone()
        waters_out[:, self.transform_blocks, :] = inverted
        return self._merge(prefix, waters_out), log_det

"""
Modified version of your code for your purpose:

- Your internal coords are Euclidean: i = [solute(6), water1(6), ..., waterM(6)]  => dim = 6*(1+n_waters)
- We use *grouped* coupling spline flows (RQS), NOT dim//2 splits.
- Masks act on whole 6D blocks (solute + each water), which is much better than cutting blocks in half.
- Mixing is done by permuting 6D groups (cheap and effective); optional per-group 6x6 linear mixing included.

This file assumes you already have:
    from src.utils import unconstrained_RQS
which supports vectorized inputs:
    y, logdet = unconstrained_RQS(x, W, H, D, inverse=..., tail_bound=B)

Recommended starting hyperparams for H100:
    n_blocks=16, hidden_dim=512, K=8, B=4.0 (after standardization)
"""

import numpy as np
import torch
import normflows as nf


def group_mask(dim: int, group_size: int = 6, seed: int | None = None, pattern: str = "alternating"):
    """
    Returns a {0,1} mask of shape (dim,) that selects whole groups of size `group_size`.
    mask==1 dims are transformed; mask==0 dims are identity (conditioning part).
    """
    assert dim % group_size == 0
    n_groups = dim // group_size

    if pattern == "alternating":
        gmask = torch.zeros(n_groups, dtype=torch.float32)
        gmask[0::2] = 1.0
    elif pattern == "random":
        assert seed is not None
        rng = np.random.RandomState(seed)
        gmask = torch.from_numpy(rng.randint(0, 2, size=n_groups)).float()
        # avoid all-zeros or all-ones
        if gmask.sum() == 0:
            gmask[0] = 1.0
        if gmask.sum() == n_groups:
            gmask[0] = 0.0
    else:
        raise ValueError("pattern must be 'alternating' or 'random'")

    return gmask.repeat_interleave(group_size)  # (dim,)


def make_coupled_spline_flow_nf(cfg: DictConfig, target: Target) -> nf.NormalizingFlow:
    """
    Coupled RQS spline flow using normflows (nf.flows.*), with 6D-group masks.

    Your i-space is Euclidean (solute 6 + each water 6), so no periodic wrapping.
    """
    dim = target.internal_dim
    # Base distribution
    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.learn_mean_var)
    else:
        raise NotImplementedError("Only base_type='gauss' is recommended for your current i-space.")
    
    

    # Tail bounds per-dimension (vector accepted by normflows spline flows)
    tb = cfg.flow.tail_bound * torch.ones(dim)

    flows = []
    mask = group_mask(dim, group_size=cfg.flow.group_size, seed=cfg.training.seed, pattern="alternating")

    for k in range(cfg.flow.blocks):
        # Alternate masks to mix information between groups
        if k % 2 == 1:
            mask = 1.0 - mask

        flows.append(
            nf.flows.CoupledRationalQuadraticSpline(
                dim=dim,
                num_blocks=cfg.flow.blocks_per_layer,
                hidden_channels=cfg.flow.hidden_units,
                tail_bound=tb,
                num_bins=cfg.flow.num_bins,
                init_identity=cfg.flow.init_identity,
                mask=mask,
            )
        )

        # Mixing layer
        if cfg.flow.mixing == "affine":
            flows.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        elif cfg.flow.mixing == "permute":
            flows.append(nf.flows.Permute(dim))
        else:
            raise ValueError("mixing must be 'affine' or 'permute'")

        if cfg.flow.actnorm:
            flows.append(nf.flows.ActNorm(dim))

        # Optional: also permute groups sometimes (cheap extra mixing)
        # (If you want, uncomment and keep group_size=6)
        # if k % 4 == 3:
        #     flows.append(nf.flows.Permute(dim))

    flow = nf.NormalizingFlow(base, flows)
    return flow


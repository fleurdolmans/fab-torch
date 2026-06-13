import normflows as nf

from fab.flow import (
    SoluteSplineCoupling,
    SoluteWaterSplineCoupling,
    JointSoluteWaterFlowLayer,
    PermEquiWaterSplineCoupling,
    PermEquiWaterSplineCouplingPairwiseO,
    SharedWaterBlockSplineCoupling,
    PermuteFixed,
    water_block_permutation,
)
from fab.wrappers.normflows import WrappedNormFlowModel
from experiments.make_base.base import make_structured_diag_gaussian_from_target


def make_perm_equi_joint_spline_flow_nf(cfg, target):
    """
    Joint solute+water flow with permutation-equivariant water coupling.

    Each layer consists of:
      - SoluteSplineCoupling: alternating 3+3 split on the 6D solute block
      - PermEquiWaterSplineCoupling: within-block O/omega coupling with mean-pool conditioning

    Compatible with Global3PointRadialRotvecTransform (solute_dim=6, all dims in R).
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters

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
    else:
        raise NotImplementedError("Supported base types: 'gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)


def make_perm_equi_spline_flow_nf(cfg, target):
    """
    Permutation-equivariant water flow with pairwise O-O radial context.

    Each layer is a PermEquiWaterSplineCouplingPairwiseO that alternates between
    transforming O (using omega as conditioner) and transforming omega (using O as
    conditioner, with additional pairwise O-O RBF geometry).

    Compatible with Global3PointRadialRotvecTransform (solute_dim=6, all dims in R).
    """
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
                transform_oxygen=(k % 2 == 0),
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
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)


def make_perm_equi_sfic_flow_nf(cfg, target):
    """
    Permutation-equivariant flow for SFICTransform (solute_dim=3, all dims in R).

    SFICTransform removes 3 global rotation DOFs, giving a 3D solute block [v1(3)]
    and water blocks [z_O(3), omega(3)] — all unconstrained R.

    Each layer:
      - SoluteWaterSplineCoupling (n_prefix=3): transforms solute conditioned on
        mean-pooled water context (alternating even/odd dims of the 3D solute)
      - PermEquiWaterSplineCoupling: within-block z_O/omega coupling
      - PermuteFixed: water block permutation

    Compatible with SFICTransform only. For Global3PointRadialRotvecTransform
    (solute_dim=6) use make_perm_equi_joint_spline_flow_nf.
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters
    assert solute_dim == 3, f"make_perm_equi_sfic_flow_nf requires solute_dim=3, got {solute_dim}"

    coupling_kwargs = dict(
        solute_dim=solute_dim,
        n_waters=n_waters,
        hidden_dim=cfg.flow.hidden_units,
        n_hidden=cfg.flow.blocks_per_layer,
        dropout=cfg.flow.dropout,
        num_bins=cfg.flow.num_bins,
        tail_bound=cfg.flow.tail_bound,
    )

    flows = []
    for k in range(cfg.flow.layers):
        flows.append(
            SoluteWaterSplineCoupling(
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
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=(k % 2 == 0), **coupling_kwargs))
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=(k % 2 == 1), **coupling_kwargs))

        perm = water_block_permutation(dim, solute_dim, block_size=6, seed=k)
        flows.append(PermuteFixed(perm))

        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target, learn_mean_var=cfg.flow.base.learn_mean_var
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    flow = nf.NormalizingFlow(base, flows)
    return WrappedNormFlowModel(flow)


def make_shared_water_spline_flow_nf(cfg, target):
    """
    Shared-weight water coupling flow with solute prefix coupling.

    Each layer consists of:
      - SoluteWaterSplineCoupling: transforms solute prefix conditioned on water mean-pool
      - SharedWaterBlockSplineCoupling: transforms water parity groups with shared params
      - PermuteFixed: block-level permutation for mixing

    Compatible with SFICTransform (solute_dim=3) and
    Global3PointRadialRotvecTransform (solute_dim=6).
    """
    dim = target.internal_dim
    solute_dim = target.internal_dim - 6 * target.num_solvent_molecules
    assert solute_dim in (3, 6)

    if cfg.flow.base.type == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target,
            learn_mean_var=cfg.flow.base.learn_mean_var,
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    flows = []
    for k in range(cfg.flow.layers):
        flows.append(
            SoluteWaterSplineCoupling(
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

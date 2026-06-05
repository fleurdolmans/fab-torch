import math
import torch
import normflows as nf

from fab.flow import (
    PermEquiWaterSplineCoupling,
    SoluteWaterSplineCoupling,
)
from fab.wrappers.normflows import WrappedNormFlowModel
from experiments.make_base.base import make_structured_diag_gaussian_from_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torus_ind_circ(solute_dim: int, n_waters: int) -> list:
    """
    Indices of the periodic tau dims in the torus internal coordinate vector.

    Layout: [solute(solute_dim) | tau_1(3) omega_1(3) | ... | tau_W(3) omega_W(3)]

    tau_k lives in [-pi, pi)^3 — the torus angles of the k-th oxygen.
    omega_k lives in R^3 — unconstrained rotation vector.

    Returns the flat index list of all tau dims.
    """
    ind_circ = []
    for k in range(n_waters):
        base = solute_dim + 6 * k
        ind_circ += [base, base + 1, base + 2]
    return ind_circ


def _torus_base_scale(dim: int, ind_circ: list) -> torch.Tensor:
    """
    Scale vector for UniformGaussian so that tau dims are Uniform[-pi, pi).

    normflows UniformGaussian samples eps_u = rand - 0.5  in [-0.5, 0.5), then
    multiplies by scale.  With scale=1 (default) the uniform dims are in [-0.5, 0.5),
    NOT [-pi, pi).  We need scale = 2*pi on the tau dims so that:
        tau_latent = 2*pi * (rand - 0.5)  in [-pi, pi)
    which matches the circular RQS tail_bound=pi domain.
    """
    scale = torch.ones(dim)
    for i in ind_circ:
        scale[i] = 2.0 * math.pi
    return scale


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_perm_equi_torus_flow_nf(cfg, target):
    """
    Permutation-equivariant flow for torus coordinate transforms
    (LabFrameTorusTransform, LabFrameCanonicalTorusTransform,
     LabFrameGeometricTorusTransform, SFICTorusTransform).

    Internal layout: [solute(6) | tau_1(3) omega_1(3) | ... | tau_W(3) omega_W(3)]
      tau  in [-pi, pi)^3  ->  circular RQS (tails='circular', tail_bound=pi)
      omega in R^3         ->  linear  RQS (tails='linear')

    Each flow layer:
      - SoluteWaterSplineCoupling(n_prefix=6)  (linear RQS on solute, conditioned on water mean-pool)
      - PermEquiWaterSplineCoupling(transform_oxygen=True,  oxygen_circular=True)
      - PermEquiWaterSplineCoupling(transform_oxygen=False, oxygen_circular=True)

    No inter-layer permutation mixing (PermuteFixed removed) — the mean-pool conditioning
    in PermEquiWaterSplineCoupling provides global information propagation while
    preserving exact permutation equivariance of the full flow.

    Base distribution:
      UniformGaussian with ind_circ = tau indices  (Uniform[-pi,pi) on tau, N(0,1) on omega)

    Config keys:
      cfg.flow.layers, hidden_units, blocks_per_layer (n_hidden), dropout,
      num_bins, tail_bound (used for solute and omega dims only)
      cfg.flow.base.type  ('gauss' | 'structured-gauss')
      cfg.flow.base.learn_mean_var
      cfg.flow.actnorm    (optional bool)
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters   # typically 6 for triatomic solute
    ind_circ = _torus_ind_circ(solute_dim, n_waters)

    coupling_kwargs = dict(
        solute_dim=solute_dim,
        n_waters=n_waters,
        hidden_dim=cfg.flow.hidden_units,
        n_hidden=cfg.flow.blocks_per_layer,
        dropout=cfg.flow.dropout,
        num_bins=cfg.flow.num_bins,
        tail_bound=cfg.flow.tail_bound,
        oxygen_circular=True,
    )

    flows = []
    for k in range(cfg.flow.layers):
        # solute coupling (linear RQS, conditioned on water mean-pool)
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
        # transform tau (circular) conditioned on omega
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=True, **coupling_kwargs))
        # transform omega (linear) conditioned on tau
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=False, **coupling_kwargs))

        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    if cfg.flow.base.type == "gauss":
        scale = _torus_base_scale(dim, ind_circ)
        base = nf.distributions.UniformGaussian(dim, ind_circ, scale=scale)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target, learn_mean_var=cfg.flow.base.learn_mean_var
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)
    return WrappedNormFlowModel(model)


def make_circ_rqs_torus_flow_nf(cfg, target):
    """
    Non-equivariant CircularCoupledRQS baseline for torus transforms.

    Uses the flat normflows CircularCoupledRationalQuadraticSpline with
    ind_circ = all tau indices.  Does NOT preserve permutation equivariance
    over water molecules; use make_perm_equi_torus_flow_nf for that.

    Base: UniformGaussian(dim, ind_circ)  — Uniform[-pi,pi) on tau, N(0,1) on omega.

    Config keys: same as make_perm_equi_torus_flow_nf.
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters
    ind_circ = _torus_ind_circ(solute_dim, n_waters)

    flows = []
    for k in range(cfg.flow.layers):
        flows.append(
            nf.flows.CircularCoupledRationalQuadraticSpline(
                num_input_channels=dim,
                num_blocks=cfg.flow.blocks_per_layer,
                num_hidden_channels=cfg.flow.hidden_units,
                ind_circ=ind_circ,
                num_bins=cfg.flow.num_bins,
                tail_bound=cfg.flow.tail_bound,
                dropout_probability=cfg.flow.dropout,
                reverse_mask=bool(k % 2),
            )
        )
        if getattr(cfg.flow, "mixing", None) == "affine":
            flows.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    scale = _torus_base_scale(dim, ind_circ)
    base = nf.distributions.UniformGaussian(dim, ind_circ, scale=scale)
    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)
    return WrappedNormFlowModel(model)


def make_perm_equi_sfic_torus_flow_nf(cfg, target):
    """
    Permutation-equivariant circular flow for SFICTorusTransform (solute_dim=3).

    SFICTorusTransform removes 3 global rotation DOFs, giving:
      [v1(3) | tau_1(3) omega_1(3) | ... | tau_W(3) omega_W(3)]
    where tau ∈ [-pi, pi)^3 is periodic and omega ∈ R^3 is unconstrained.

    Each layer:
      - SoluteWaterSplineCoupling (n_prefix=3): transforms solute conditioned on
        mean-pooled water context
      - PermEquiWaterSplineCoupling(oxygen_circular=True): circular RQS on tau,
        linear RQS on omega

    No inter-layer permutation mixing — mean-pool conditioning provides global
    information propagation while preserving exact permutation equivariance.

    Base: UniformGaussian with ind_circ = tau indices.

    Compatible with SFICTorusTransform only.
    For LabFrame torus transforms (solute_dim=6) use make_perm_equi_torus_flow_nf.
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 6 * n_waters
    assert solute_dim == 3, f"make_perm_equi_sfic_torus_flow_nf requires solute_dim=3, got {solute_dim}"
    ind_circ = _torus_ind_circ(solute_dim, n_waters)

    coupling_kwargs = dict(
        solute_dim=solute_dim,
        n_waters=n_waters,
        hidden_dim=cfg.flow.hidden_units,
        n_hidden=cfg.flow.blocks_per_layer,
        dropout=cfg.flow.dropout,
        num_bins=cfg.flow.num_bins,
        tail_bound=cfg.flow.tail_bound,
        oxygen_circular=True,
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
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=True, **coupling_kwargs))
        flows.append(PermEquiWaterSplineCoupling(transform_oxygen=False, **coupling_kwargs))

        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    scale = _torus_base_scale(dim, ind_circ)
    base = nf.distributions.UniformGaussian(dim, ind_circ, scale=scale)
    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)
    return WrappedNormFlowModel(model)

import normflows as nf

from fab.flow import (
    SoluteWaterSplineCoupling,
    PermEquiWater9DSplineCoupling,
)
from fab.wrappers.normflows import WrappedNormFlowModel
from experiments.make_base.base import make_structured_diag_gaussian_from_target


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_perm_equi_gps_flow_nf(cfg, target):
    """
    Permutation-equivariant flow for Global3PointSphericalTransform (GPS).

    Internal layout: [solute(3) | O_sph(3) H1_sph(3) H2_sph(3) | ... per water]
      - solute: [fr_H1 | fr_H2 fphi_H2]          (3 dims)
      - water k: [fr_O fphi_O ftheta_O |           (9 dims)
                  fr_H1 fphi_H1 ftheta_H1 |
                  fr_H2 fphi_H2 ftheta_H2]

    Each flow layer:
      - SoluteWaterSplineCoupling(n_prefix=3, block_size=9): transforms solute
        conditioned on mean-pooled 9D water context (alternating even/odd dims).
      - PermEquiWater9DSplineCoupling(transform_oxygen=True): transforms O_sph(3)
        conditioned on {H1,H2}_sph(6) + global mean-pool + solute shape.
      - PermEquiWater9DSplineCoupling(transform_oxygen=False): transforms
        {H1,H2}_sph(6) conditioned on O_sph(3) + global mean-pool + solute shape.

    The combination of GPS transform + this flow is permutation equivariant:
      - GPS maps water permutations in Cartesian space to 9D block permutations in z.
      - PermEquiWater9DSplineCoupling is equivariant over 9D water blocks via shared
        weights and symmetric mean-pool, so the full pipeline is equivariant.

    Compatible with Global3PointSphericalTransform only (solute_dim must equal 3).
    For non-PBC droplet systems use use_pbc=False (default).
    For PBC systems pass use_pbc=True, L=<box_nm> to the transform.

    Config keys:
      cfg.flow.layers, hidden_units, blocks_per_layer, dropout,
      num_bins, tail_bound
      cfg.flow.base.type  ('gauss' | 'structured-gauss')
      cfg.flow.base.learn_mean_var
      cfg.flow.actnorm    (optional bool)
    """
    dim = target.internal_dim
    n_waters = target.num_solvent_molecules
    solute_dim = dim - 9 * n_waters
    assert solute_dim == 3, (
        f"make_perm_equi_gps_flow_nf requires solute_dim=3, got {solute_dim}. "
        "Only compatible with Global3PointSphericalTransform."
    )

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
        # solute coupling: alternating even/odd dims, conditioned on water mean-pool
        flows.append(
            SoluteWaterSplineCoupling(
                dim=dim,
                n_prefix=solute_dim,
                block_size=9,
                hidden_dim=cfg.flow.hidden_units,
                n_hidden=cfg.flow.blocks_per_layer,
                dropout=cfg.flow.dropout,
                reverse_mask=bool(k % 2),
                num_bins=cfg.flow.num_bins,
                tail_bound=cfg.flow.tail_bound,
            )
        )
        # transform O_sph conditioned on {H1,H2}_sph
        flows.append(PermEquiWater9DSplineCoupling(transform_oxygen=True, **coupling_kwargs))
        # transform {H1,H2}_sph conditioned on O_sph
        flows.append(PermEquiWater9DSplineCoupling(transform_oxygen=False, **coupling_kwargs))

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

    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)
    return WrappedNormFlowModel(model)

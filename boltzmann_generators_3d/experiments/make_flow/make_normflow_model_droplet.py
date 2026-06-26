import normflows as nf

from boltzmann_generators_3d.fab.wrappers.normflows import WrappedNormFlowModel
from boltzmann_generators_3d.experiments.make_base.base import make_structured_diag_gaussian_from_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spherical_ind_circ(internal_dim: int) -> list:
    """
    Return the list of periodic dimension indices for Global3PointSphericalTransform.

    The transform produces a flat vector of length n_atoms*3 - 6 with this layout:

      index 0          fr   atom 1          (unconstrained ℝ)
      index 1          fr   atom 2          (unconstrained ℝ)
      index 2          fphi atom 2          (periodic [-π, π))
      index 3 + 3k     fr   atom k+3        (unconstrained ℝ)   k = 0 … n_atoms-4
      index 3 + 3k+1   fphi atom k+3        (periodic [-π, π))
      index 3 + 3k+2   ftheta atom k+3      (periodic [-π, π))

    phi and theta are already rescaled into [-π, π) by the transform
    (offset_phi = π, scale_theta = 0.5).
    """
    n_atoms = (internal_dim + 6) // 3
    ind_circ = [2]
    for k in range(n_atoms - 3):
        base = 3 + k * 3
        ind_circ += [base + 1, base + 2]
    return ind_circ


def _base_diag_gaussian(cfg, target):
    dim = target.internal_dim
    if cfg.flow.base.type == "gauss":
        return nf.distributions.DiagGaussian(dim, trainable=cfg.flow.base.learn_mean_var)
    elif cfg.flow.base.type == "structured-gauss":
        return make_structured_diag_gaussian_from_target(
            target, learn_mean_var=cfg.flow.base.learn_mean_var
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_spherical_circular_rqs_flow_nf(cfg, target):
    """
    Circular coupled RQS flow for Global3PointSphericalTransform.

    Uses CircularCoupledRationalQuadraticSpline so that phi and theta
    dimensions are treated as periodic ([-π, π)).

    Base distribution:
      'gauss'           -> UniformGaussian: Uniform([-π, π)) on circular dims,
                          N(0,1) on radial dims.  This is the principled choice.
      'structured-gauss' -> DiagGaussian fitted to training data (Gaussian on all
                           dims; only use this if the periodic dims do not wrap
                           in your training set).

    Config keys used:
      cfg.flow.layers, hidden_units, blocks_per_layer, dropout, num_bins, tail_bound
      cfg.flow.base.type  ('gauss' | 'structured-gauss')
      cfg.flow.base.learn_mean_var
      cfg.flow.mixing     (optional: 'affine' adds InvertibleAffine between layers)
      cfg.flow.actnorm    (optional bool)
    """
    dim = target.internal_dim
    ind_circ = _spherical_ind_circ(dim)

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

    if cfg.flow.base.type == "gauss":
        # Uniform on circular dims, N(0,1) on radial dims — the correct choice
        # for a flow whose circular outputs live on [-π, π).
        base = nf.distributions.UniformGaussian(dim, ind_circ)
    elif cfg.flow.base.type == "structured-gauss":
        base = make_structured_diag_gaussian_from_target(
            target, learn_mean_var=cfg.flow.base.learn_mean_var
        )
    else:
        raise NotImplementedError("Supported base types: 'gauss', 'structured-gauss'.")

    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)   # initialise ActNorm layers
    return WrappedNormFlowModel(model)


def make_coupled_rqs_flow_nf(cfg, target):
    """
    Standard (non-circular) coupled RQS flow.

    A simple baseline for any flat internal representation. Does NOT
    handle periodic dimensions — for Global3PointSphericalTransform use
    make_spherical_circular_rqs_flow_nf instead.

    Config keys: same as make_spherical_circular_rqs_flow_nf.
    """
    dim = target.internal_dim

    flows = []
    for k in range(cfg.flow.layers):
        flows.append(
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels=dim,
                num_blocks=cfg.flow.blocks_per_layer,
                num_hidden_channels=cfg.flow.hidden_units,
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

    base = _base_diag_gaussian(cfg, target)
    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)
    return WrappedNormFlowModel(model)


def make_realnvp_flow_nf(cfg, target):
    """
    RealNVP flow (affine coupling + invertible linear mixing).

    Each layer: AffineCouplingBlock → InvertibleAffine → (optional ActNorm).
    Simple and fast, but does not respect periodicity of phi/theta dims.

    Config keys used:
      cfg.flow.layers, hidden_units   (num_bins / tail_bound are not used)
      cfg.flow.base.type, base.learn_mean_var
      cfg.flow.actnorm    (optional bool)
    """
    dim = target.internal_dim
    d = int((dim / 2) + 0.5)   # first-half split point (ceil(dim/2))

    flows = []
    for k in range(cfg.flow.layers):
        param_map = nf.nets.MLP(
            [d, cfg.flow.hidden_units, cfg.flow.hidden_units, 2 * (dim - d)],
            init_zeros=True,
        )
        flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
        flows.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        if getattr(cfg.flow, "actnorm", False):
            flows.append(nf.flows.ActNorm(dim))

    base = _base_diag_gaussian(cfg, target)
    model = nf.NormalizingFlow(base, flows)
    if getattr(cfg.flow, "actnorm", False):
        model.sample(500)   # initialise ActNorm layers
    return WrappedNormFlowModel(model)

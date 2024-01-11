import numpy as np
import torch
import normflows as nf
import larsflow as lf
import torch

from fab.wrappers.normflows import WrappedNormFlowModel
from fab.trainable_distributions import TrainableDistribution


def make_normflow_flow(dim: int, n_flow_layers: int, layer_nodes_per_dim: int, act_norm: bool):
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

    # TODO: Check that base distribution is correct. Check scales of base distribution at initialisation. I think it
    #  might be initialised at [-pi, pi], instead of [0, pi]? Or even [-pi/2, pi/2], as a symmetric interval that has
    #  width pi = bound_circ. Compare the Circular Flow implementation here:
    #   https://github.com/francois-rozet/zuko/blob/master/zuko/flows.py#L493
    #  And the used CircularShiftTransform: https://github.com/francois-rozet/zuko/blob/master/zuko/transforms.py
    #  Note: MonotonicRQSTransform is simply the monotonic Rational Quadratic Spline transform.
    #  What is our PeriodicShift doing exactly?
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
    #  range, but we might as well put that in manually, so that on initialisation the relative scale matches.
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
                    dim,
                    bl,
                    hu,
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
                    dim,
                    bl,
                    hu,
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
                    dim,
                    bl,
                    hu,
                    periodic_inds,
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
                    dim,
                    bl,
                    hu,
                    periodic_inds,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    init_identity=ii,
                    dropout_probability=dropout,
                    mask=mask,
                )
            )
        else:
            raise NotImplementedError("The flow type " + flow_type + " is not implemented for solvent systems.")

        # if config["flow"]["mixing"] == "affine":
        #     layers.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        # elif config["flow"]["mixing"] == "permute":
        #     layers.append(nf.flows.Permute(dim))
        #
        # if config["flow"]["actnorm"]:
        #     layers.append(nf.flows.ActNorm(dim))
        #
        # # Shift the periodic angles.
        # if i % 2 == 1 and i != n_layers - 1:
        #     if circ_shift == "constant":
        #         layers.append(nf.flows.PeriodicShift(periodic_inds, bound=bound_circ, shift=bound_circ))
        #     elif circ_shift == "random":
        #         gen = torch.Generator().manual_seed(seed + i)
        #         shift_scale = torch.rand([], generator=gen) + 0.5
        #         layers.append(nf.flows.PeriodicShift(periodic_inds, bound=bound_circ, shift=shift_scale * bound_circ))
        #
        # # SNF
        # if "snf" in config["flow"]:
        #     if (i + 1) % config["flow"]["snf"]["every_n"] == 0:
        #         prop_scale = config["flow"]["snf"]["proposal_std"] * np.ones(dim)
        #         steps = config["flow"]["snf"]["steps"]
        #         proposal = nf.distributions.DiagGaussianProposal((dim,), prop_scale)
        #         lam = (i + 1) / n_layers
        #         dist = nf.distributions.LinearInterpolation(target, base, lam)
        #         layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # Map input to periodic interval
    # The purpose is that incoming samples from a dataset get periodically wrapped to the interval [-pi, pi],
    #  or the equivalent scaled version.
    layers.append(nf.flows.PeriodicWrap(periodic_inds, bound_circ))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)
    wrapped_flow = WrappedNormFlowModel(flow)

    return wrapped_flow

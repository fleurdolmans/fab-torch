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
    dim = target.dim

    # Base distribution
    # TODO: We only use the Gauss here, because we have no freely rotating dihedrals that we would need circular
    #  distributions for (circ_ind). Check that this is actually true!
    if config["flow"]["base"]["type"] == "gauss":
        base = nf.distributions.DiagGaussian(dim, trainable=config["flow"]["base"]["learn_mean_var"])
    else:
        raise NotImplementedError("The base distribution " + config["flow"]["base"]["type"] + " is not implemented")

    # Flow layers
    layers = []
    n_layers = config["flow"]["blocks"]
    tail_bound = 5.0 * torch.ones(dim)

    for i in range(n_layers):
        if flow_type == "ar-nsf":
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            # TODO: This is different from the Circular splines used in FAB paper. As far as I know, we
            #  don't have freely rotating angles that we need the Circular indices for. But if we do, we should
            #  change this.
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
        elif flow_type == "coup-nsf":
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            if i % 2 == 0:
                mask = nf.utils.masks.create_random_binary_mask(dim, seed=config["training"]["seed"] + i)
            else:
                mask = 1 - mask
            # TODO: This is different from the Circular splines used in FAB paper. As far as I know, we
            #  don't have freely rotating angles that we need the Circular indices for. But if we do, we should
            #  change this.
            layers.append(
                nf.flows.CoupledRationalQuadraticSpline(
                    dim,
                    bl,
                    hu,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    init_identity=ii,
                    dropout_probability=dropout,
                    mask=mask,
                )
            )
        else:
            raise NotImplementedError("The flow type " + flow_type + " is not implemented for solvent systems.")

        if config["flow"]["mixing"] == "affine":
            layers.append(nf.flows.InvertibleAffine(dim, use_lu=True))
        elif config["flow"]["mixing"] == "permute":
            layers.append(nf.flows.Permute(dim))

        if config["flow"]["actnorm"]:
            layers.append(nf.flows.ActNorm(dim))

        # SNF
        if "snf" in config["flow"]:
            if (i + 1) % config["flow"]["snf"]["every_n"] == 0:
                prop_scale = config["flow"]["snf"]["proposal_std"] * np.ones(dim)
                steps = config["flow"]["snf"]["steps"]
                proposal = nf.distributions.DiagGaussianProposal((dim,), prop_scale)
                lam = (i + 1) / n_layers
                dist = nf.distributions.LinearInterpolation(target, base, lam)
                layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)
    wrapped_flow = WrappedNormFlowModel(flow)

    return wrapped_flow

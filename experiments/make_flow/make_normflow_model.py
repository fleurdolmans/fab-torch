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


def make_wrapped_normflow_molecular_flow(config):
    """
    Setup Flow model.
    """

    # Flow parameters
    flow_type = config["flow"]["type"]
    ndim = 60

    ncarts = target.coordinate_transform.transform.len_cart_inds
    permute_inv = target.coordinate_transform.transform.permute_inv.cpu().numpy()
    dih_ind_ = target.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
    std_dih = target.coordinate_transform.transform.ic_transform.std_dih.cpu()

    ind = np.arange(ndim)
    ind = np.concatenate([ind[: 3 * ncarts - 6], -np.ones(6, dtype=np.int), ind[3 * ncarts - 6:]])
    ind = ind[permute_inv]
    dih_ind = ind[dih_ind_]

    ind_circ = dih_ind[ind_circ_dih]
    bound_circ = np.pi / std_dih[ind_circ_dih]

    tail_bound = 5.0 * torch.ones(ndim)
    tail_bound[ind_circ] = bound_circ

    circ_shift = None if not "circ_shift" in config["flow"] else config["flow"]["circ_shift"]

    # Base distribution
    if config["flow"]["base"]["type"] == "gauss":
        base = nf.distributions.DiagGaussian(ndim, trainable=config["flow"]["base"]["learn_mean_var"])
    elif config["flow"]["base"]["type"] == "gauss-uni":
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        base.shape = (ndim,)
    elif config["flow"]["base"]["type"] == "resampled-gauss-uni":
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base_ = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        pf = nf.utils.nn.PeriodicFeaturesCat(ndim, ind_circ, np.pi / bound_circ)
        resnet = nf.nets.ResidualNet(
            ndim + len(ind_circ),
            1,
            config["flow"]["base"]["params"]["a_hidden_units"],
            num_blocks=config["flow"]["base"]["params"]["a_n_blocks"],
            preprocessing=pf,
        )
        a = torch.nn.Sequential(resnet, torch.nn.Sigmoid())
        base = lf.distributions.ResampledDistribution(
            base_, a, config["flow"]["base"]["params"]["T"], config["flow"]["base"]["params"]["eps"]
        )
        base.shape = (ndim,)
    else:
        raise NotImplementedError("The base distribution " + config["flow"]["base"]["type"] + " is not implemented")

    # Flow layers
    layers = []
    n_layers = config["flow"]["blocks"]

    for i in range(n_layers):
        if flow_type == "rnvp":
            # Coupling layer
            hl = config["flow"]["hidden_layers"] * [config["flow"]["hidden_units"]]
            scale_map = config["flow"]["scale_map"]
            scale = scale_map is not None
            if scale_map == "tanh":
                output_fn = "tanh"
                scale_map = "exp"
            else:
                output_fn = None
            param_map = nf.nets.MLP(
                [(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)],
                init_zeros=config["flow"]["init_zeros"],
                output_fn=output_fn,
            )
            layers.append(nf.flows.AffineCouplingBlock(param_map, scale=scale, scale_map=scale_map))
        elif flow_type == "circular-ar-nsf":
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            layers.append(
                nf.flows.CircularAutoregressiveRationalQuadraticSpline(
                    ndim,
                    bl,
                    hu,
                    ind_circ,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    permute_mask=True,
                    init_identity=ii,
                    dropout_probability=dropout,
                )
            )
        elif flow_type == "circular-coup-nsf":
            bl = config["flow"]["blocks_per_layer"]
            hu = config["flow"]["hidden_units"]
            nb = config["flow"]["num_bins"]
            ii = config["flow"]["init_identity"]
            dropout = config["flow"]["dropout"]
            if i % 2 == 0:
                mask = nf.utils.masks.create_random_binary_mask(ndim, seed=seed + i)
            else:
                mask = 1 - mask
            layers.append(
                nf.flows.CircularCoupledRationalQuadraticSpline(
                    ndim,
                    bl,
                    hu,
                    ind_circ,
                    tail_bound=tail_bound,
                    num_bins=nb,
                    init_identity=ii,
                    dropout_probability=dropout,
                    mask=mask,
                )
            )
        else:
            raise NotImplementedError("The flow type " + flow_type + " is not implemented.")

        if config["flow"]["mixing"] == "affine":
            layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
        elif config["flow"]["mixing"] == "permute":
            layers.append(nf.flows.Permute(ndim))

        if config["flow"]["actnorm"]:
            layers.append(nf.flows.ActNorm(ndim))

        if i % 2 == 1 and i != n_layers - 1:
            if circ_shift == "constant":
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ, shift=bound_circ))
            elif circ_shift == "random":
                gen = torch.Generator().manual_seed(seed + i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ, shift=shift_scale * bound_circ))

        # SNF
        if "snf" in config["flow"]:
            if (i + 1) % config["flow"]["snf"]["every_n"] == 0:
                prop_scale = config["flow"]["snf"]["proposal_std"] * np.ones(ndim)
                steps = config["flow"]["snf"]["steps"]
                proposal = nf.distributions.DiagGaussianProposal((ndim,), prop_scale)
                lam = (i + 1) / n_layers
                dist = nf.distributions.LinearInterpolation(target, base, lam)
                layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # Map input to periodic interval
    layers.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)
    wrapped_flow = WrappedNormFlowModel(flow)

    return wrapped_flow

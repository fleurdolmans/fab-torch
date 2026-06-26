import re
import time
from typing import Union, Callable, Optional, List
import os
import pathlib
import wandb
import numpy as np
from omegaconf import DictConfig

from datetime import datetime

import matplotlib.pyplot as plt
import torch

from boltzmann_generators_3d.fab import Trainer, BufferTrainer, PrioritisedBufferTrainer
from boltzmann_generators_3d.fab.target_distributions.base import TargetDistribution
from boltzmann_generators_3d.fab.utils.replay_buffer import ReplayBuffer
from boltzmann_generators_3d.fab.utils.plotting import plot_history

from boltzmann_generators_3d.fab import FABModel, HamiltonianMonteCarlo, Metropolis
from boltzmann_generators_3d.fab.core import ALPHA_DIV_TARGET_LOSSES
from boltzmann_generators_3d.fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer

from boltzmann_generators_3d.experiments.logger_setup import setup_logger
from boltzmann_generators_3d.experiments.make_flow import (
    make_shared_water_spline_flow_nf,
    make_perm_equi_spline_flow_nf,
    make_perm_equi_joint_spline_flow_nf,
    make_spherical_circular_rqs_flow_nf,
    make_coupled_rqs_flow_nf,
    make_realnvp_flow_nf,
    make_perm_equi_torus_flow_nf,
    make_circ_rqs_torus_flow_nf,
    make_perm_equi_gps_flow_nf
)

Plotter = Callable[[FABModel], List[plt.Figure]]
SetupPlotterFn = Callable[
    [DictConfig, TargetDistribution, Optional[Union[ReplayBuffer, PrioritisedReplayBuffer]]], Plotter
]


def get_n_iterations(
    n_training_iter: Union[int, None],
    n_flow_forward_pass: Union[int, None],
    batch_size: int,
    loss_type: str,
    n_transition_operator_inner_steps: int,
    n_intermediate_ais_dist: int,
    transition_operator_type: str,
    use_buffer: bool,
    min_buffer_length: Optional[int] = None,
) -> int:
    """
    Calculate the number of training iterations, based on the run config.
    We define one "training iteration" as
        - for training by KLD: 1 forward pass of the flow to estimate KLD
        - for training by FAB: 1 forward pass of the flow and AIS
        - for training by FAB with buffer: 1 forward pass of the flow & AIS followed by
            n buffer sampling update steps.

    Note: We aim here to do the theoretical number of forward passes required for each method
    during training for fair comparison. Due to inefficiencies in implementation this will not match
    the actual number of flow forward passes.
    """
    # must specify either number of training iterations or flow forward passes.
    assert bool(n_training_iter) != bool(n_flow_forward_pass)

    if n_training_iter:
        return n_training_iter
    else:
        if loss_type[0:4] == "flow" or loss_type[:6] == "target":
            n_iter = n_flow_forward_pass // batch_size
        else:
            if transition_operator_type == "hmc":
                # Note this also requires differentiating the flow, which is fair as the
                # KLD forward pass also requires a differentiation of target and flow step.
                # +1 is for the initial sampling step.
                n_flow_eval_per_ais_forward = (n_transition_operator_inner_steps) * n_intermediate_ais_dist + 1
            else:
                assert transition_operator_type == "metropolis"
                # +1 for the initial sampling step
                n_flow_eval_per_ais_forward = n_transition_operator_inner_steps * n_intermediate_ais_dist + 1
            if use_buffer:
                buffer_init_flow_eval = n_flow_eval_per_ais_forward * min_buffer_length
                # we do another ais evaluation per iteration to calculate the log prob of the
                # samples from the buffer.
                n_flow_eval_per_iter = (n_flow_eval_per_ais_forward + 1) * batch_size
            else:
                buffer_init_flow_eval = 0
                n_flow_eval_per_iter = n_flow_eval_per_ais_forward * batch_size
            n_iter = int((n_flow_forward_pass - buffer_init_flow_eval) / n_flow_eval_per_iter)
    print(f"{n_iter} iter for {n_flow_forward_pass} flow forward passes")
    return n_iter


def setup_buffer(
    cfg: DictConfig, fab_model: FABModel, auto_fill_buffer: bool
) -> Union[ReplayBuffer, PrioritisedReplayBuffer]:
    if hasattr(fab_model.target_distribution, "internal_dim"):
        dim = fab_model.target_distribution.internal_dim  # Use internal dimension if provided
    else:
        dim = cfg.target.cartesian_dim  # applies to flow and target
    
    buffer_device = (
        "cuda"
        if torch.cuda.is_available() and cfg.training.use_gpu
        else "cpu"
    )
    print("Buffer device:", buffer_device)
    flow_device = next(fab_model.flow.parameters()).device 
    print("Flow device:", flow_device)
    if cfg.training.buffer.prioritised is False:
        def initial_sampler():
            # used to fill the replay buffer up to its minimum size
            x, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size, logging=False, purpose="init buffer fill"
            )
            return x, log_w

        buffer = ReplayBuffer(
            dim=dim,
            max_length=cfg.training.buffer.maximum_length,
            min_sample_length=cfg.training.buffer.min_length,
            initial_sampler=initial_sampler,
            temperature=cfg.training.buffer.temp,
        )
    else:
        # buffer
        def initial_sampler():
            # Calls AIS
            print("[buffer prefill] starting AIS...", flush=True)
            t0 = time.time()
            point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size, logging=False, purpose="init buffer fill"
            )
            dt = time.time() - t0
            print(f"[buffer prefill] AIS batch done in {dt:.2f}s", flush=True)
            return point.x.detach(), log_w, point.log_q.detach()

        buffer = PrioritisedReplayBuffer(
            dim=dim,
            max_length=cfg.training.buffer.maximum_length,
            min_sample_length=cfg.training.buffer.min_length,
            initial_sampler=initial_sampler,
            fill_buffer_during_init=auto_fill_buffer,
            device=buffer_device,
        )
        print("Replay buffer device:", buffer.buffer.x.device)
    return buffer


def get_load_checkpoint_dir(outer_checkpoint_dir, latest=False, continue_same_run=False):
    """Get directory of checkpoint of a specific run, or if latest is True,
    the most recent run inside outer_checkpoint_dir.
    
    run should be the full path to the folder of that run.
    """
    try:
        if not latest:
            chkpts_dir = os.path.join(outer_checkpoint_dir, "model_checkpoints")
        else:
            chkpts = [it.path for it in os.scandir(outer_checkpoint_dir) if it.is_dir()]
            folder_names = [it.name for it in os.scandir(outer_checkpoint_dir) if it.is_dir()]
            times = [datetime.fromisoformat(name).timestamp() for name in folder_names]
            chkpts_dir = os.path.join(chkpts[np.argmax(times)], "model_checkpoints")
        
        
        iter_dirs = [it.path for it in os.scandir(chkpts_dir) if it.is_dir()]
        matches = []
        for subdir in iter_dirs:
            m = re.search(r"iter_(\d+)$", os.path.basename(subdir))
            if m is not None:
                matches.append((subdir, int(m.group(1))))

        if not matches:
            raise FileNotFoundError(f"No iter_* checkpoint folders found in {chkpts_dir}")

        chkpt_dir, iter_number = max(matches, key=lambda x: x[1])
        
        if continue_same_run is False:
            iter_number = 0

    except Exception as e:
        print(f"Starting training from the beginning with no checkpoint. Reason: {e}")
        return None, 0

    return chkpt_dir, iter_number


def setup_model(cfg: DictConfig, target: TargetDistribution) -> FABModel:
    if hasattr(target, "internal_dim"):
        # Log probabilities are typically computed from representations in internal space (transforms are applied to
        #  compute Boltzmann probabilities). So we should be using the internal dimension for computing importance
        #  weights and the like in the AIS loss. Thus, we need the AIS samples to be in internal space.
        dim = target.internal_dim  # Use internal dimension if provided
    else:
        dim = cfg.target.cartesian_dim  # applies to flow and target
    p_target = cfg.fab.loss_type not in ALPHA_DIV_TARGET_LOSSES or not cfg.training.buffer.prioritised

    # Non-equivariant flows
    if cfg.flow.type == "shared-water-spline-nf":
        # Droplet. Shared-weight spline coupling over water blocks. No permutation equivariance.
        flow = make_shared_water_spline_flow_nf(cfg, target)
    elif cfg.flow.type == "spherical-circ-rqs-nf":
        # Droplet. Circular RQS coupling on mixed radial+spherical coordinates (Global3PointSphericalTransform).
        flow = make_spherical_circular_rqs_flow_nf(cfg, target)
    elif cfg.flow.type == "coupled-rqs-nf":
        # Droplet. Standard flat RQS coupling on unconstrained internal coordinates.
        flow = make_coupled_rqs_flow_nf(cfg, target)
    elif cfg.flow.type == "realnvp-nf":
        # Droplet. Affine coupling (RealNVP) on flat internal coordinates.
        flow = make_realnvp_flow_nf(cfg, target)
    elif cfg.flow.type == "circ-rqs-torus-nf":
        # PBC only. Circular RQS coupling on torus coordinates (LabFrameTorusTransform). No permutation equivariance.
        flow = make_circ_rqs_torus_flow_nf(cfg, target)

    # Permutation equivariant flows
    elif cfg.flow.type == "perm-equi-torus-nf":
        # PBC only. Permutation-equivariant spline flow on torus coordinates (LabFrameTorusTransform).
        # Water mean-pool conditioning. Handles periodic tau angles with circular RQS.
        flow = make_perm_equi_torus_flow_nf(cfg, target)
    elif cfg.flow.type == "perm-equi-joint-spline-nf":
        # Droplet. Permutation-equivariant flow that jointly updates solute (SoluteSplineCoupling)
        # and water (PermEquiWaterSplineCoupling) each layer. Water conditioned on mean-pool.
        flow = make_perm_equi_joint_spline_flow_nf(cfg, target)
    elif cfg.flow.type == "perm-equi-spline-nf":
        # Droplet. Permutation-equivariant water-only flow conditioned on pairwise O-O distances (RBF geometry).
        # Solute is not updated by the flow, only used as context.
        flow = make_perm_equi_spline_flow_nf(cfg, target)
    elif cfg.flow.type == "perm-equi-gps-nf":
        # Droplet or PBC (via use_pbc flag). Permutation-equivariant flow on Global3PointSphericalTransform
        # coordinates. Water blocks are 9D spherical (O+H1+H2). Mean-pool + solute shape conditioning.
        flow = make_perm_equi_gps_flow_nf(cfg, target)
    else:
        raise NotImplementedError(f"Flow type {cfg.flow.type} not implemented.")
    


    if cfg.fab.transition_operator.type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            dim=dim,
            base_log_prob=flow.log_prob,  # Flow
            target_log_prob=target.log_prob,  # Boltzmann: transforms I --> X, then gets unnormalised logprob.
            alpha=cfg.fab.alpha,
            p_target=p_target,
            target_p_accept=cfg.fab.transition_operator.target_p_accept,
            epsilon=cfg.fab.transition_operator.init_step_size,
            common_epsilon_init_weight=cfg.fab.transition_operator.init_step_size,
            L=cfg.fab.transition_operator.n_inner_steps,
        )
    elif cfg.fab.transition_operator.type == "metropolis":
        transition_operator = Metropolis(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            dim=dim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
            p_target=p_target,
            alpha=cfg.fab.alpha,
            n_updates=cfg.fab.transition_operator.n_inner_steps,
            adjust_step_size=cfg.fab.transition_operator.tune_step_size,
            target_p_accept=cfg.fab.transition_operator.target_p_accept,
            min_step_size=cfg.fab.transition_operator.init_step_size,
            max_step_size=cfg.fab.transition_operator.init_step_size,
        )
    else:
        transition_operator = None

    # use GPU if available
    if torch.cuda.is_available() and cfg.training.use_gpu:
        flow.cuda()
        transition_operator.cuda()
        print("\n*************  Utilising GPU  ****************** \n")
    else:
        print("\n*************  Utilising CPU  ****************** \n")

    fab_model = FABModel(
        flow=flow,
        target_distribution=target,
        n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
        transition_operator=transition_operator,
        alpha=cfg.fab.alpha,
        loss_type=cfg.fab.loss_type,
        use_ais=cfg.fab.use_ais,
    )
    return fab_model

def run_initial_flow_sanity_test(fab_model, target, batch_size: int = 8):
    """
    Flow-only sanity checks in internal space.

    Checks:
      1. base sampling/logprob works
      2. flow inverse(forward(x)) roundtrip in internal space
      3. permutation equivariance over water blocks
      4. logdet permutation invariance
      5. base permutation invariance
      6. sample stats

    Assumes internal layout:
      [solute | water_1 | ... | water_W]

    Water block size is inferred from target.transform_version:
      GPS              -> 9 dims/water (O + H1 + H2, each 3 spherical coords)
      GPR, LGT, other -> 6 dims/water (rigid-body / rotvec representation)
    """
    print("\n=== INITIAL FLOW SANITY TEST ===")

    flow = fab_model.flow
    n_waters = target.num_solvent_molecules
    # GPS uses 9 dims/water (3 atoms × 3 spherical); GPR/LGT use 6 dims/water (rigid-body)
    transform_version = getattr(target, 'transform_version', None)
    water_block_dim = 9 if transform_version == 'GPS' else 6
    solute_dim = target.internal_dim - n_waters * water_block_dim
    print(f"[SANITY] transform_version={transform_version!r}  water_block_dim={water_block_dim}  solute_dim={solute_dim}")


    def permute_water_blocks(i, perm):
        B = i.shape[0]
        w = i[:, solute_dim:].view(B, n_waters, water_block_dim)
        w_perm = w[:, perm, :]
        return torch.cat([i[:, :solute_dim], w_perm.reshape(B, -1)], dim=-1)

    def get_nf_model(flow):
        if hasattr(flow, "_nf_model"):
            return flow._nf_model
        return flow

    nf_model = get_nf_model(flow)

    def run_forward_all(x):
        z = x
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for fl in nf_model.flows:
            z, ld = fl(z)
            total_logdet = total_logdet + ld
        return z, total_logdet

    def run_inverse_all(z):
        x = z
        total_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for fl in reversed(nf_model.flows):
            x, ld = fl.inverse(x)
            total_logdet = total_logdet + ld
        return x, total_logdet
    
    def wrap(x: torch.Tensor, L: float) -> torch.Tensor:
        return torch.remainder(x, float(L))
    def dist_pbc(p, q, L):
        return torch.linalg.norm(mic(p - q, L), dim=-1)
    
    def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
        return dx - L_t * torch.round(dx / L_t)

    with torch.no_grad():
        # --------------------------------------------
        # 1) pick internal test batch
        # --------------------------------------------
        if getattr(target, "val_data_i", None) is not None:
            i = target.val_data_i[:batch_size].to(target.device).reshape(batch_size, -1)
        elif getattr(target, "train_data_i", None) is not None:
            i = target.train_data_i[:batch_size].to(target.device).reshape(batch_size, -1)
        else:
            raise ValueError("Need target.val_data_i or target.train_data_i for flow sanity test.")

        print("internal batch shape:", tuple(i.shape))

        # --------------------------------------------
        # 2) base sampling/logprob
        # --------------------------------------------
        try:
            z0, log_q0 = nf_model.q0(batch_size)
            print("[BASE] sample shape:", tuple(z0.shape))
            print("[BASE] log_prob shape:", tuple(log_q0.shape))
            print("[BASE] sample mean/std:", z0.mean().item(), z0.std(unbiased=False).item())
        except Exception as e:
            print("[BASE] sampling/logprob failed:", repr(e))

        # --------------------------------------------
        # 3) full flow roundtrip
        # --------------------------------------------
        try:
            z, ld_fwd = run_forward_all(i)
            i_rec, ld_inv = run_inverse_all(z)

            print("[FLOW] max |i - inv(fwd(i))|:", (i - i_rec).abs().max().item())
            print("[FLOW] mean |i - inv(fwd(i))|:", (i - i_rec).abs().mean().item())
            print("[FLOW] max |ld_fwd + ld_inv|:", (ld_fwd + ld_inv).abs().max().item())
        except Exception as e:
            print("[FLOW] roundtrip failed:", repr(e))

        # --------------------------------------------
        # 4) permutation equivariance of first layer
        # --------------------------------------------
        perm = torch.randperm(n_waters, device=i.device)
        i_perm = permute_water_blocks(i, perm)

        try:
            fl0 = nf_model.flows[0]
            z1, ld1 = fl0(i)
            z2, ld2 = fl0(i_perm)

            z1_expected = permute_water_blocks(z1, perm)

            print("[FLOW layer 0] perm equiv max err:",
                  (z2 - z1_expected).abs().max().item())
            print("[FLOW layer 0] perm equiv mean err:",
                  (z2 - z1_expected).abs().mean().item())
            print("[FLOW layer 0] logdet perm inv max err:",
                  (ld2 - ld1).abs().max().item())
        except Exception as e:
            print("[FLOW layer 0] permutation test failed:", repr(e))

        # --------------------------------------------
        # 5) permutation equivariance of whole flow
        # --------------------------------------------
        try:
            z_full_1, ld_full_1 = run_forward_all(i)
            z_full_2, ld_full_2 = run_forward_all(i_perm)

            z_full_expected = permute_water_blocks(z_full_1, perm)

            print("[FULL FLOW] perm equiv max err:",
                  (z_full_2 - z_full_expected).abs().max().item())
            print("[FULL FLOW] perm equiv mean err:",
                  (z_full_2 - z_full_expected).abs().mean().item())
            print("[FULL FLOW] logdet perm inv max err:",
                  (ld_full_2 - ld_full_1).abs().max().item())
        except Exception as e:
            print("[FULL FLOW] permutation test failed:", repr(e))

        # --------------------------------------------
        # 6) base permutation invariance
        # --------------------------------------------
        try:
            if hasattr(nf_model.q0, "log_prob"):
                logq1 = nf_model.q0.log_prob(i)
                logq2 = nf_model.q0.log_prob(i_perm)
                print("[BASE] permutation invariance max err:",
                      (logq1 - logq2).abs().max().item())
                print("[BASE] permutation invariance mean err:",
                      (logq1 - logq2).abs().mean().item())
            else:
                print("[BASE] no log_prob method; skipping permutation invariance test")
        except Exception as e:
            print("[BASE] permutation invariance test failed:", repr(e))

        # --------------------------------------------
        # 7) quick sample stats from current model
        # --------------------------------------------
        try:
            if hasattr(flow, "sample_and_log_prob"):
                samp, logq = flow.sample_and_log_prob((batch_size,))
            elif hasattr(nf_model, "sample"):
                samp, logq = nf_model.sample(batch_size)
            else:
                raise RuntimeError("No sample method found")

            print("[FLOW sample] shape:", tuple(samp.shape))
            print("[FLOW sample] mean/std:", samp.mean().item(), samp.std(unbiased=False).item())

            # oxygen norms in internal space (first 3 dims of each water block = O spherical coords)
            O = samp[:, solute_dim:].view(batch_size, n_waters, water_block_dim)[..., 0:3]
            O_norm = torch.linalg.norm(O, dim=-1)

            print("[FLOW sample] O norm median:",
                  O_norm.median().item(),
                  "p10:", torch.quantile(O_norm.reshape(-1), 0.10).item(),
                  "p90:", torch.quantile(O_norm.reshape(-1), 0.90).item(),
                  "max:", O_norm.max().item())
        except Exception as e:
            print("[FLOW sample] sampling stats failed:", repr(e))
        
        # [FLOW sample] already sampled in internal space as `samp`
        x_samp, _ = target.coordinate_transform.forward(samp)
        x_samp = x_samp.view(batch_size, -1, 3)
        L = float(target.box_length_nm)
        x_samp_w = wrap(x_samp, L)

        # oxygen positions
        O_idx = [3 + 3 * k for k in range(target.num_solvent_molecules)]
        O = x_samp_w[:, O_idx, :]  # (B, W, 3)

        # min O-O
        min_oo = []
        for i in range(target.num_solvent_molecules):
            for j in range(i + 1, target.num_solvent_molecules):
                dij = dist_pbc(O[:, i, :], O[:, j, :], L)
                min_oo.append(dij)
        min_oo = torch.stack(min_oo, dim=1).min(dim=1).values

        # solute atoms
        solute = x_samp_w[:, :3, :]  # (B,3,3)

        # min solute-O
        min_solO = []
        for i in range(target.num_solvent_molecules):
            for a in range(3):
                d = dist_pbc(O[:, i, :], solute[:, a, :], L)
                min_solO.append(d)
        min_solO = torch.stack(min_solO, dim=1).min(dim=1).values

        # hydrogens
        H_idx = []
        for k in range(target.num_solvent_molecules):
            base = 3 + 3 * k
            H_idx.extend([base + 1, base + 2])
        H = x_samp_w[:, H_idx, :]

        # min solute-H
        min_solH = []
        for h in range(H.shape[1]):
            for a in range(3):
                d = dist_pbc(H[:, h, :], solute[:, a, :], L)
                min_solH.append(d)
        min_solH = torch.stack(min_solH, dim=1).min(dim=1).values

        print("[FLOW sample distances] min O-O median", min_oo.median().item(),
            "p10", torch.quantile(min_oo, 0.1).item(),
            "p90", torch.quantile(min_oo, 0.9).item(),
            "min", min_oo.min().item())

        print("[FLOW sample distances] min solute-O median", min_solO.median().item(),
            "p10", torch.quantile(min_solO, 0.1).item(),
            "p90", torch.quantile(min_solO, 0.9).item(),
            "min", min_solO.min().item())

        print("[FLOW sample distances] min solute-H median", min_solH.median().item(),
            "p10", torch.quantile(min_solH, 0.1).item(),
            "p90", torch.quantile(min_solH, 0.9).item(),
            "min", min_solH.min().item())

    print("=== END INITIAL FLOW SANITY TEST ===\n")


def setup_trainer_and_run_flow(cfg: DictConfig, setup_plotter: SetupPlotterFn, target: TargetDistribution):
    """Setup model and train."""
    print("Starting setup...")
    start_time = time.time()

    if cfg.training.checkpoint_load_dir is not None:
        if not os.path.exists(cfg.training.checkpoint_load_dir):
            print("No checkpoint loaded, starting training from scratch.")
            chkpt_dir = None
            iter_number = 0
        else:
            chkpt_dir, iter_number = get_load_checkpoint_dir(cfg.training.checkpoint_load_dir, cfg.training.continue_same_run)
            print(f"Checkpoint directory: {chkpt_dir}, starting from iteration {iter_number}")
    else:
        chkpt_dir = None
        iter_number = 0

    # Take logger and save dir from Target if it has those, else create them for backwards compatibility.
    if hasattr(target, "logger"):
        logger = target.logger
        assert hasattr(target, "save_dir"), "Target must have a save_dir attribute when it has a logger."
        save_path = target.save_dir
    else:
        save_path = os.path.join(cfg.evaluation.save_path, str(datetime.now().isoformat()))
        logger = setup_logger(cfg, save_path)
        if hasattr(cfg.logger, "wandb"):
            # if using wandb then save to wandb path
            save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    n_iterations = cfg.training.n_iterations

    print(f"Running for {n_iterations} iterations.")
    cfg.training.n_iterations = n_iterations

    with open(os.path.join(save_path, "config.txt"), "w") as file:
        file.write(str(cfg))

    # 1) Build model
    print("Setting up model...")
    fab_model = setup_model(cfg, target)
    num_model_params = sum(p.numel() for p in fab_model.flow.parameters() if p.requires_grad)
    print(f" Model with {num_model_params} parameters")
    print(fab_model.flow)
    logger.write({"num_parameters": num_model_params})

    run_initial_flow_sanity_test(fab_model, target, batch_size=min(8, cfg.evaluation.n_eval if hasattr(cfg.evaluation, "n_eval") else 8))

    # 2) Initialize optimizer and its parameters
    #  Taken from ALDP's train.py.
    lr = cfg.training.lr
    weight_decay = cfg.training.wd
    optimizer_name = "adam" if not "optimizer" in cfg.training else cfg.training.optimizer
    optimizer_param = fab_model.flow.parameters()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(optimizer_param, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(optimizer_param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("The optimizer " + optimizer_name + " is not implemented.")
    
    # 3) Scheduler
    scheduler = None
    lr_step = 1

    has_sched = ("lr_scheduler" in cfg.training) and (cfg.training.lr_scheduler is not None)
    sched_type = None
    if has_sched:
        sched_type = cfg.training.lr_scheduler.type

    if (not has_sched) or (sched_type is None) or (str(sched_type).lower() in ["none", "null", "off", "disable", "disabled"]):
        scheduler = None
        lr_step = 1
    elif sched_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=cfg.training.lr_scheduler.rate_decay,
        )
        lr_step = cfg.training.lr_scheduler.decay_iter
    elif sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.training.n_iterations,
        )
        lr_step = 1
    elif sched_type == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg.training.lr_scheduler.decay_iter,
        )
        lr_step = 1
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.training.lr_scheduler.decay_iter,
            gamma=cfg.training.lr_scheduler.rate_decay,
        )
        lr_step = 1
    else:
        raise NotImplementedError(f"The scheduler {sched_type} is not implemented.")

    # 4) Scheduler warmup
    warmup_iters = cfg.training.warmup_iter if "warmup_iter" in cfg.training and cfg.training.warmup_iter is not None else 0
    warmup_scheduler = None
    if warmup_iters > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, s / warmup_iters)
        )
    
    # 5) Load checkpointed model
    if chkpt_dir is not None:
        map_location = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        fab_model.load(os.path.join(chkpt_dir, "model.pt"), map_location, partial_flow_load=cfg.training.transfer_learning)
        
        if cfg.training.load_optimizer_state:
            opt_state = torch.load(os.path.join(chkpt_dir, "optimizer.pt"), map_location)
            optimizer.load_state_dict(opt_state)

    buffer = None
    # 6) Create buffer if needed
    if cfg.training.buffer.use:
        print("Setting up buffer...")
        buffer_time = time.time()
        # Only use buffer when specified or when continuing the same run
        buffer = setup_buffer(cfg, fab_model, auto_fill_buffer=not cfg.training.buffer.load_buffer)

    
    # 7) If load_buffer, load saved buffer
    if cfg.training.buffer.load_buffer:
        buffer.load(path=os.path.join(chkpt_dir, "buffer.pt"))
        assert buffer.can_sample, (
            "If a buffer is loaded, it is expected to contain enough samples to sample from."
        )
    print(f"\n\n**************** Loaded checkpoint: {chkpt_dir}*******************\n\n")



    if buffer is not None:
        print(f" Initialised buffer with {buffer.get_buffer_size()} points.")
        print(f" Buffer setup time: {time.time() - buffer_time:.2f}s")

    print("Setting up plotter...")
    plot = setup_plotter(cfg, target, buffer)

    # Create trainer
    print("Create trainer...")     

    if buffer and cfg.training.buffer.prioritised:
        trainer = PrioritisedBufferTrainer(
            model=fab_model,
            optimizer=optimizer,
            logger=logger,
            plot=plot,
            optim_scheduler=scheduler,
            save_path=save_path,
            buffer=buffer,
            n_batches_buffer_sampling=cfg.training.buffer.n_batches_sampling,
            max_gradient_norm=cfg.training.max_grad_norm,
            w_adjust_max_clip=cfg.training.buffer.w_adjust_max_clip,
            alpha=cfg.fab.alpha,
            lr_step=lr_step,
            warmup_scheduler=warmup_scheduler,
            warmup_iters=warmup_iters
        )

    elif buffer:
        trainer = BufferTrainer(
            model=fab_model,
            optimizer=optimizer,
            logger=logger,
            plot=plot,
            optim_scheduler=scheduler,
            save_path=save_path,
            buffer=buffer,
            n_batches_buffer_sampling=cfg.training.buffer.n_batches_sampling,
            max_gradient_norm=cfg.training.max_grad_norm
        )

    else:
        trainer = Trainer(
            model=fab_model,
            optimizer=optimizer,
            logger=logger,
            plot=plot,
            optim_scheduler=scheduler,
            save_path=save_path,
            max_gradient_norm=cfg.training.max_grad_norm,
            lr_step=lr_step,
            print_eval=cfg.evaluation.print_eval,
            warmup_scheduler=warmup_scheduler,
            warmup_iters=warmup_iters,
            overlap_penalty=cfg.training.overlap_penalty,
            mixing=cfg.training.mixing,
            n_pretraining=cfg.training.n_pretraining,
        )

    print("Starting training...")
    trainer.run(
        n_iterations=n_iterations,
        batch_size=cfg.training.batch_size,
        n_plot=cfg.evaluation.n_plots,
        n_eval=cfg.evaluation.n_eval,
        eval_batch_size=cfg.evaluation.eval_batch_size,
        save=True,
        n_checkpoints=cfg.evaluation.n_checkpoints,
        tlimit=cfg.training.tlimit,
        start_time=start_time,
        start_iter=iter_number,
    )

    if hasattr(cfg.logger, "list_logger"):
        plot_history(trainer.logger.history)
        plt.show()
        print(trainer.logger.history["eval_ess_flow_p_target"][-10:])
        print(trainer.logger.history["eval_ess_ais_p_target"][-10:])
        print(trainer.logger.history["test_set_mean_log_prob_p_target"][-10:])

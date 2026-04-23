import os
import pathlib
import re
import time
from datetime import datetime
from typing import Union, Callable, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

from fab import TrainerLJ, BufferTrainer, PrioritisedBufferTrainer
from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from fab.core import ALPHA_DIV_TARGET_LOSSES
from fab.target_distributions.base import TargetDistribution
from fab.utils.plotting import plot_history
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer
from fab.utils.replay_buffer import ReplayBuffer

from experiments.logger_setup import setup_logger
from experiments.make_flow import make_lj_flow


Plotter = Callable[[FABModel], List[plt.Figure]]
SetupPlotterFn = Callable[
    [DictConfig, TargetDistribution, Optional[Union[ReplayBuffer, PrioritisedReplayBuffer]]], Plotter
]


def setup_model(cfg: DictConfig, target: TargetDistribution) -> FABModel:
    dim = target.internal_dim if hasattr(target, "internal_dim") else cfg.target.cartesian_dim
    p_target = cfg.fab.loss_type not in ALPHA_DIV_TARGET_LOSSES or not cfg.training.buffer.prioritised

    flow = make_lj_flow(cfg=cfg, target=target)
    if cfg.training.use_64_bit:
        flow = flow.double()

    if cfg.fab.transition_operator.type == "hmc":
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            dim=dim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
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

    if torch.cuda.is_available() and cfg.training.use_gpu:
        flow.cuda()
        if transition_operator is not None:
            transition_operator.cuda()
        print("\n*************  Utilising GPU  ******************\n")
    else:
        print("\n*************  Utilising CPU  ******************\n")

    return FABModel(
        flow=flow,
        target_distribution=target,
        n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
        transition_operator=transition_operator,
        alpha=cfg.fab.alpha,
        loss_type=cfg.fab.loss_type,
        use_ais=cfg.fab.use_ais,
    )


def run_initial_flow_sanity_test(fab_model: FABModel, target: TargetDistribution, batch_size: int = 8):
    print("\n=== INITIAL FLOW SANITY TEST ===")

    flow = fab_model.flow
    flow_param = next(flow.parameters())
    flow_device = flow_param.device
    flow_dtype = flow_param.dtype

    def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
        return dx - L_t * torch.round(dx / L_t)

    def pbc_abs_err(x: torch.Tensor, y: torch.Tensor, L: float) -> torch.Tensor:
        return torch.abs(mic(x - y, L))

    with torch.no_grad():
        if getattr(target, "val_data_i", None) is not None:
            x = target.val_data_i[:batch_size].reshape(batch_size, -1)
        elif getattr(target, "train_data_i", None) is not None:
            x = target.train_data_i[:batch_size].reshape(batch_size, -1)
        else:
            raise ValueError("Need target.val_data_i or target.train_data_i for sanity test.")

        x = x.to(device=flow_device, dtype=flow_dtype)

        print("batch shape:", tuple(x.shape))
        print("flow device/dtype:", flow_device, flow_dtype)

        try:
            if hasattr(flow, "base"):
                z0, log_q0 = flow.base(batch_size)
                base_obj = flow.base
            elif hasattr(flow, "_nf_model") and hasattr(flow._nf_model, "q0"):
                z0, log_q0 = flow._nf_model.q0(batch_size)
                base_obj = getattr(flow._nf_model, "q0", None)
            else:
                raise AttributeError("No accessible base distribution found")

            z0 = z0.to(device=flow_device, dtype=flow_dtype)
            log_q0 = log_q0.to(device=flow_device, dtype=flow_dtype)

            print("[BASE] sample shape:", tuple(z0.shape))
            print("[BASE] log_prob shape:", tuple(log_q0.shape))
            print("[BASE] mean/std:", z0.mean().item(), z0.std(unbiased=False).item())

            if hasattr(base_obj, "last_sampling_stats") and base_obj.last_sampling_stats is not None:
                print("[BASE stats]", base_obj.last_sampling_stats)
            elif hasattr(base_obj, "wrapped_base") and hasattr(base_obj.wrapped_base, "last_sampling_stats"):
                if base_obj.wrapped_base.last_sampling_stats is not None:
                    print("[BASE wrapped stats]", base_obj.wrapped_base.last_sampling_stats)

        except Exception as e:
            print("[BASE] failed:", repr(e))

        try:
            if hasattr(flow, "forward_map") and hasattr(flow, "inverse_map"):
                z, ld_fwd = flow.forward_map(x)
                x_rec, ld_inv = flow.inverse_map(z)

            elif hasattr(flow, "_nf_model") and hasattr(flow._nf_model, "flows"):
                z = x
                ld_fwd = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                for fl in flow._nf_model.flows:
                    z, ld = fl(z)
                    ld_fwd = ld_fwd + ld

                x_rec = z
                ld_inv = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                for fl in reversed(flow._nf_model.flows):
                    x_rec, ld = fl.inverse(x_rec)
                    ld_inv = ld_inv + ld
            else:
                raise AttributeError("No accessible flow mapping found")

            if hasattr(target, "box_length_nm"):
                err = pbc_abs_err(x, x_rec, float(target.box_length_nm))
            else:
                err = (x - x_rec).abs()

            print("[FLOW] max |x - inv(fwd(x))|:", err.max().item())
            print("[FLOW] mean |x - inv(fwd(x))|:", err.mean().item())
            print("[FLOW] max |ld_fwd + ld_inv|:", (ld_fwd + ld_inv).abs().max().item())

        except Exception as e:
            print("[FLOW] roundtrip failed:", repr(e))

        try:
            samp, logq = flow.sample_and_log_prob((batch_size,))
            samp = samp.to(device=flow_device, dtype=flow_dtype)
            logq = logq.to(device=flow_device, dtype=flow_dtype)

            print("[FLOW sample] shape:", tuple(samp.shape))
            print("[FLOW sample] mean/std:", samp.mean().item(), samp.std(unbiased=False).item())

            log_p = target.log_prob(samp)
            print(
                "[FLOW sample] target log_prob mean/std:",
                log_p.mean().item(),
                log_p.std(unbiased=False).item(),
            )

            if hasattr(target, "box_length_nm"):
                L = float(target.box_length_nm)
                X = samp.view(batch_size, -1, 3)
                dmin = torch.full((batch_size,), float("inf"), device=X.device, dtype=X.dtype)
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        dij = torch.linalg.norm(mic(X[:, i, :] - X[:, j, :], L), dim=-1)
                        dmin = torch.minimum(dmin, dij)
                print("[FLOW sample] min pair distance mean/min:", dmin.mean().item(), dmin.min().item())

        except Exception as e:
            print("[FLOW sample] failed:", repr(e))

    print("=== END INITIAL FLOW SANITY TEST ===\n")


def setup_buffer(
    cfg: DictConfig,
    fab_model: FABModel,
    target: TargetDistribution,
    auto_fill_buffer: bool,
) -> Union[ReplayBuffer, PrioritisedReplayBuffer]:
    dim = target.internal_dim if hasattr(target, "internal_dim") else cfg.target.cartesian_dim
    buffer_device = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"

    if not cfg.training.buffer.prioritised:
        def initial_sampler():
            print("[buffer prefill] starting AIS...", flush=True)
            t0 = time.time()
            x, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size,
                logging=False,
                purpose="init buffer fill",
            )
            dt = time.time() - t0
            print(f"[buffer prefill] AIS batch done in {dt:.2f}s", flush=True)
            return x, log_w

        return ReplayBuffer(
            dim=dim,
            max_length=cfg.training.buffer.maximum_length,
            min_sample_length=cfg.training.buffer.min_length,
            initial_sampler=initial_sampler,
            temperature=cfg.training.buffer.temp,
        )

    def initial_sampler():
        point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
            cfg.training.batch_size,
            logging=False,
            purpose="init buffer fill",
        )
        return point.x.detach(), log_w, point.log_q.detach()

    return PrioritisedReplayBuffer(
        dim=dim,
        max_length=cfg.training.buffer.maximum_length,
        min_sample_length=cfg.training.buffer.min_length,
        initial_sampler=initial_sampler,
        fill_buffer_during_init=auto_fill_buffer,
        device=buffer_device,
    )


def get_load_checkpoint_dir(
    outer_checkpoint_dir: str,
    latest: bool = False,
    continue_same_run: bool = False,
):
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

        if not continue_same_run:
            iter_number = 0

    except Exception as e:
        print(f"Starting from scratch. Reason: {e}")
        return None, 0

    return chkpt_dir, iter_number


def build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
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

    return scheduler, lr_step


def setup_trainer_and_run_flow(
    cfg: DictConfig,
    setup_plotter: SetupPlotterFn,
    target: TargetDistribution,
):
    start_time = time.time()

    if cfg.training.checkpoint_load_dir is not None and os.path.exists(cfg.training.checkpoint_load_dir):
        chkpt_dir, iter_number = get_load_checkpoint_dir(
            cfg.training.checkpoint_load_dir,
            latest=cfg.training.latest_run,
            continue_same_run=cfg.training.continue_same_run,
        )
        print("Loaded checkpoints from:", chkpt_dir)
    else:
        chkpt_dir, iter_number = None, 0

    if hasattr(target, "logger"):
        logger = target.logger
        save_path = target.save_dir
    else:
        save_path = os.path.join(cfg.evaluation.save_path, str(datetime.now().isoformat()))
        logger = setup_logger(cfg, save_path)
        if hasattr(cfg.logger, "wandb"):
            save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    n_iterations = cfg.training.n_iterations
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(save_path, "config.txt"), "w") as file:
        file.write(str(cfg))

    fab_model = setup_model(cfg, target)
    num_model_params = sum(p.numel() for p in fab_model.flow.parameters() if p.requires_grad)
    print(f"Model with {num_model_params} parameters")
    print(fab_model.flow)
    logger.write({"num_parameters": num_model_params})

    run_initial_flow_sanity_test(
        fab_model,
        target,
        batch_size=min(8, cfg.evaluation.n_eval if hasattr(cfg.evaluation, "n_eval") else 8),
    )
    print("Set optimizer...")

    optimizer_name = cfg.training.optimizer if "optimizer" in cfg.training else "adam"
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            fab_model.flow.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.wd,
        )
    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(
            fab_model.flow.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.wd,
        )
    else:
        raise NotImplementedError(optimizer_name)
    
    print("Set scheduler...")

    scheduler, lr_step = build_scheduler(cfg, optimizer)

    warmup_iters = cfg.training.warmup_iter if "warmup_iter" in cfg.training and cfg.training.warmup_iter is not None else 0
    warmup_scheduler = None
    if warmup_iters > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, s / warmup_iters),
        )

    if chkpt_dir is not None:
        print("Get model (and optimizer) state from checkpoint...")
        map_location = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        fab_model.load(
            os.path.join(chkpt_dir, "model.pt"),
            map_location,
            partial_flow_load=cfg.training.transfer_learning,
        )
        if cfg.training.load_optimizer_state:
            opt_state = torch.load(os.path.join(chkpt_dir, "optimizer.pt"), map_location)
            optimizer.load_state_dict(opt_state)
    buffer = None
    if cfg.training.buffer.use:
        print("Set up buffer...")
        buffer = setup_buffer(
            cfg,
            fab_model,
            target,
            auto_fill_buffer=not cfg.training.buffer.load_buffer,
        )

    if cfg.training.buffer.load_buffer and chkpt_dir is not None:
        print("Load buffer checkpoints...")
        buffer.load(path=os.path.join(chkpt_dir, "buffer.pt"))
        assert buffer.can_sample
    
    print("Set up plotter...")

    plot = setup_plotter(cfg, target, buffer)

    print("Initialize trainer...")
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
            warmup_iters=warmup_iters,
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
            max_gradient_norm=cfg.training.max_grad_norm,
        )
    else:
        trainer = TrainerLJ(
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
            overlap_penalty=cfg.training.overlap.penalty,
            dist_ssolv=cfg.training.overlap.dist_ssolv,
            dist_solute=cfg.training.overlap.dist_solute,
            mixing=cfg.training.mixing,
            n_pretraining=cfg.training.n_pretraining,
        )

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
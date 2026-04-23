import os
import pathlib
import re
import time
import warnings
from datetime import datetime
from time import time as walltime
from typing import Union, Callable, Optional, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler as lr_scheduler

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


# ============================================================
# 2D sanity test
# ============================================================

def run_initial_flow_sanity_test_2d(
    fab_model: FABModel,
    target: TargetDistribution,
    batch_size: int = 8,
):
    print("\n=== INITIAL FLOW SANITY TEST (2D) ===")

    flow = fab_model.flow
    flow_param = next(flow.parameters())
    flow_device = flow_param.device
    flow_dtype = flow_param.dtype

    def wrap_unit(x: torch.Tensor) -> torch.Tensor:
        return x - torch.floor(x)

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

        # -------------------------
        # base
        # -------------------------
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

        # -------------------------
        # flow roundtrip in internal coords
        # -------------------------
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

            # internal coords live on unit torus
            err = torch.abs(wrap_unit(x) - wrap_unit(x_rec))
            err = torch.minimum(err, 1.0 - err)

            print("[FLOW] max |x - inv(fwd(x))| on unit torus:", err.max().item())
            print("[FLOW] mean |x - inv(fwd(x))| on unit torus:", err.mean().item())
            print("[FLOW] max |ld_fwd + ld_inv|:", (ld_fwd + ld_inv).abs().max().item())

        except Exception as e:
            print("[FLOW] roundtrip failed:", repr(e))

        # -------------------------
        # transform roundtrip internal -> cartesian -> internal
        # -------------------------
        try:
            x_cart, ld1 = target.coordinate_transform.forward(x)
            x_back, ld2 = target.coordinate_transform.inverse(x_cart)

            err_int = torch.abs(wrap_unit(x) - wrap_unit(x_back))
            err_int = torch.minimum(err_int, 1.0 - err_int)

            print("[TRANSFORM] max internal roundtrip error:", err_int.max().item())
            print("[TRANSFORM] mean internal roundtrip error:", err_int.mean().item())
            print("[TRANSFORM] max |ld1 + ld2|:", (ld1 + ld2).abs().max().item())

        except Exception as e:
            print("[TRANSFORM] roundtrip failed:", repr(e))

        # -------------------------
        # sample from flow and evaluate target
        # -------------------------
        try:
            samp, logq = flow.sample_and_log_prob((batch_size,))
            samp = samp.to(device=flow_device, dtype=flow_dtype)
            logq = logq.to(device=flow_device, dtype=flow_dtype)

            print("[FLOW sample] internal shape:", tuple(samp.shape))
            print("[FLOW sample] internal mean/std:", samp.mean().item(), samp.std(unbiased=False).item())

            x_cart, _ = target.coordinate_transform.forward(samp)
            log_p = target.log_prob(samp)

            print(
                "[FLOW sample] target log_prob mean/std:",
                log_p.mean().item(),
                log_p.std(unbiased=False).item(),
            )

            if hasattr(target, "box_length_nm"):
                L = float(target.box_length_nm)
                X = x_cart.view(batch_size, -1, 2)
                dmin = torch.full((batch_size,), float("inf"), device=X.device, dtype=X.dtype)
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        dij = torch.linalg.norm(mic(X[:, i, :] - X[:, j, :], L), dim=-1)
                        dmin = torch.minimum(dmin, dij)
                print("[FLOW sample] min pair distance mean/min:", dmin.mean().item(), dmin.min().item())

        except Exception as e:
            print("[FLOW sample] failed:", repr(e))

    print("=== END INITIAL FLOW SANITY TEST (2D) ===\n")


# ============================================================
# model setup
# ============================================================

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


# ============================================================
# buffer
# ============================================================

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


# ============================================================
# checkpoints
# ============================================================

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


# ============================================================
# scheduler
# ============================================================

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


# ============================================================
# 2D trainer
# ============================================================

class TrainerLJ2D:
    def __init__(
        self,
        model: FABModel,
        save_path: str,
        optimizer: torch.optim.Optimizer,
        optim_scheduler: Optional[lr_scheduler] = None,
        logger=None,
        plot: Optional[Plotter] = None,
        max_gradient_norm: Optional[float] = 5.0,
        lr_step=1,
        warmup_scheduler: Optional[lr_scheduler] = None,
        warmup_iters: int = 0,
        print_eval: bool = False,
        overlap_penalty: Optional[float] = 0.0,
        dist_ssolv: Optional[float] = 0.30,
        dist_solute: Optional[float] = 0.32,
        mixing: Optional[float] = 0.0,
        n_pretraining: Optional[int] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.optim_scheduler = optim_scheduler
        self.lr_step = lr_step
        self.logger = logger
        self.plot = plot
        self.flow_device = next(model.flow.parameters()).device
        self.max_gradient_norm = max_gradient_norm if max_gradient_norm else float("inf")
        self.save_dir = save_path
        self.print_eval = print_eval
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.checkpoints_dir = os.path.join(self.save_dir, "model_checkpoints")
        self.warmup_scheduler = warmup_scheduler
        self.warmup_iters = warmup_iters
        self.overlap_penalty = overlap_penalty
        self.dist_ssolv = dist_ssolv
        self.dist_solute = dist_solute
        self.mixing = mixing
        self.n_pretraining = n_pretraining

    def save_checkpoint(self, i):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
        pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
        self.model.save(os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        if self.optim_scheduler:
            torch.save(self.optim_scheduler.state_dict(), os.path.join(self.checkpoints_dir, "scheduler.pt"))
        if self.warmup_scheduler:
            torch.save(self.warmup_scheduler.state_dict(), os.path.join(self.checkpoints_dir, "warmup_scheduler.pt"))

    def make_and_save_plots(self, i, save):
        figures = self.plot(self.model, {})
        for j, figure in enumerate(figures):
            if save:
                figure.savefig(os.path.join(self.plots_dir, f"{j}_iter_{i}.png"))
            else:
                plt.show()
            plt.close(figure)

    def perform_eval(self, i, eval_batch_size, batch_size):
        eval_info = self.model.get_eval_info(
            outer_batch_size=eval_batch_size,
            inner_batch_size=batch_size,
            iteration=i,
        )
        eval_info.update(iteration=i)
        self.logger.write(eval_info)
        if self.print_eval:
            print(
                "   Eval metrics: " +
                str({key: "{:.4f}".format(value) for key, value in eval_info.items() if key != "step"})
            )

    @staticmethod
    def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
        return dx - L * torch.round(dx / L)

    def pair_clash_penalty(
        self,
        A: torch.Tensor,   # (B, NA, 2)
        B: torch.Tensor,   # (B, NB, 2)
        L: float,
        r0: float,
        k: float = 200.0,
    ) -> torch.Tensor:
        dx = self.mic(A[:, :, None, :] - B[:, None, :, :], L)
        d = torch.linalg.norm(dx, dim=-1)
        pen = torch.nn.functional.softplus(k * (r0 - d)) / k
        return pen.sum(dim=(1, 2)).mean()

    def lj_overlap_penalty(
        self,
        x_flat: torch.Tensor,         # (B, 2N)
        L: float,
        n_solute: int,
        n_solvent: int,
        r0_ssolv: float = 0.24,
        r0_solv_solute: float = 0.24,
        k: float = 200.0,
        chunk: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, D = x_flat.shape
        N = D // 2
        X = x_flat.view(B, N, 2)

        solute = X[:, :n_solute, :]
        solvent = X[:, n_solute:n_solute + n_solvent, :]

        solv_solute_pen = self.pair_clash_penalty(
            solvent, solute, L=L, r0=r0_solv_solute, k=k
        )

        L_t = torch.as_tensor(L, device=x_flat.device, dtype=x_flat.dtype)
        r0_t = torch.as_tensor(r0_ssolv, device=x_flat.device, dtype=x_flat.dtype)
        solv_solv_pen_per_batch = torch.zeros((B,), device=x_flat.device, dtype=x_flat.dtype)

        for i0 in range(0, n_solvent, chunk):
            i1 = min(n_solvent, i0 + chunk)
            Xi = solvent[:, i0:i1, :]
            ci = i1 - i0

            dx = self.mic(Xi[:, :, None, :] - solvent[:, None, :, :], L_t)
            d = torch.linalg.norm(dx, dim=-1)

            rows = torch.arange(ci, device=x_flat.device)
            cols = rows + i0
            mask_self = torch.zeros((ci, n_solvent), device=x_flat.device, dtype=torch.bool)
            mask_self[rows, cols] = True
            d = d.masked_fill(mask_self.unsqueeze(0), 1e9)

            pen = torch.nn.functional.softplus(k * (r0_t - d)) / k
            solv_solv_pen_per_batch = solv_solv_pen_per_batch + pen.sum(dim=(1, 2))

        solv_solv_pen = solv_solv_pen_per_batch.mean()
        return solv_solv_pen, solv_solute_pen

    def save_flow_samples_h5(
        self,
        n_samples: int = 1000,
        filename: str = "flow_samples_2d.h5",
        batch_size: int = 256,
    ):
        target_dist = self.model.target_distribution
        xs = []

        with torch.no_grad():
            n_done = 0
            while n_done < n_samples:
                n_now = min(batch_size, n_samples - n_done)
                z, _ = self.model.flow.sample_and_log_prob((n_now,))
                x_full, _ = target_dist.coordinate_transform.forward(z)
                xs.append(x_full.detach().cpu())
                n_done += n_now

        x = torch.cat(xs, dim=0).view(n_samples, target_dist.n_particles, 2).numpy()

        save_path = os.path.join(self.save_dir, filename)
        with h5py.File(save_path, "w") as f:
            f.create_dataset("coordinates", data=x)

        print(f"Saved flow samples H5 to: {save_path}")

    def run(
        self,
        n_iterations: int,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        n_eval: Optional[int] = None,
        n_plot: Optional[int] = None,
        n_checkpoints: Optional[int] = None,
        save: bool = True,
        tlimit: Optional[float] = None,
        start_time: Optional[float] = None,
        start_iter: Optional[int] = 0,
    ) -> None:
        if save:
            pathlib.Path(self.plots_dir).mkdir(exist_ok=True)
            pathlib.Path(self.checkpoints_dir).mkdir(exist_ok=True)

        if n_checkpoints:
            checkpoint_iter = list(np.linspace(0, n_iterations, n_checkpoints + 1, dtype="int")[1:])
        else:
            checkpoint_iter = []

        if n_eval is not None:
            eval_iter = list(np.linspace(0, n_iterations, n_eval + 1, dtype="int")[1:])
            assert eval_batch_size is not None
        else:
            eval_iter = []

        if n_plot is not None:
            plot_iter = list(np.linspace(0, n_iterations, n_plot + 1, dtype="int")[1:])
        else:
            plot_iter = []

        if start_time is None:
            start_time = walltime()

        target_dist = self.model.target_distribution

        if n_eval is not None:
            self.perform_eval(0, eval_batch_size, batch_size)
        if n_plot is not None:
            self.make_and_save_plots(0, save)

        if getattr(target_dist, "train_data_i", None) is None:
            raise ValueError("Target distribution must provide train_data_i for training.")

        target_dist.train_data_i = target_dist.train_data_i.reshape(-1, target_dist.internal_dim).contiguous()
        target_dist.train_logdet_xi = target_dist.train_logdet_xi.reshape(-1).contiguous()

        if self.n_pretraining is not None:
            train_data = target_dist.train_data_i[:self.n_pretraining]
            train_logdet_xi = target_dist.train_logdet_xi[:self.n_pretraining]
        else:
            train_data = target_dist.train_data_i
            train_logdet_xi = target_dist.train_logdet_xi

        overlap_w = self.overlap_penalty
        global_step = 0
        k, epoch = 0, 0
        max_it_time = 0.0

        for t in range(start_iter, n_iterations):
            i = t + 1
            if i % 10 == 1:
                print(f"Iteration {i}/{n_iterations}")

            it_start_time = walltime()
            self.optimizer.zero_grad()

            if self.model.loss_type == "forward_kl":
                if k == 0:
                    perm = torch.randperm(train_data.shape[0], device=train_data.device)

                idx = perm[k * batch_size:(k + 1) * batch_size]
                i_batch = train_data[idx].to(self.flow_device, non_blocking=True)
                logdet_batch = train_logdet_xi[idx].to(self.flow_device, non_blocking=True)

                flow_loss = self.model.loss(i_batch)
                transform_loss = -logdet_batch.mean()
                loss = flow_loss + transform_loss

                if (k + 1) * batch_size >= len(train_data):
                    k = 0
                    epoch += 1
                else:
                    k += 1

            else:
                loss = self.model.loss(batch_size)

            if overlap_w > 0.0:
                z_pen, _ = self.model.flow.sample_and_log_prob((batch_size,))
                x_pen, _ = target_dist.coordinate_transform.forward(z_pen)

                oo_pen, sw_pen = self.lj_overlap_penalty(
                    x_pen,
                    L=float(target_dist.box_length_nm),
                    n_solute=int(target_dist.n_solute),
                    n_solvent=int(target_dist.n_solvent),
                    r0_ssolv=self.dist_ssolv,
                    r0_solv_solute=self.dist_solute,
                    k=100.0,
                )
                loss = loss + overlap_w * (oo_pen + sw_pen)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                    global_step += 1

                    if self.warmup_scheduler is not None and global_step <= self.warmup_iters:
                        self.warmup_scheduler.step()
                    elif self.optim_scheduler is not None and (global_step % self.lr_step == 0):
                        self.optim_scheduler.step()
                else:
                    warnings.warn("Encountered inf grad norm!")
            else:
                warnings.warn("NaN loss encountered! No update performed.")

            self.optimizer.zero_grad()

            info = self.model.get_iter_info()
            info.update(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "iteration": i,
                    "epoch": epoch,
                }
            )
            self.logger.write(info)

            if i % 10 == 0:
                print(f"   Iter {i}, Train loss: {loss.detach().cpu().item():.4f}")

            if n_eval is not None and i in eval_iter:
                self.perform_eval(i, eval_batch_size, batch_size)

            if n_plot is not None and i in plot_iter:
                self.make_and_save_plots(i, save)

            if n_checkpoints is not None and i in checkpoint_iter:
                self.save_checkpoint(i)

            max_it_time = max(max_it_time, walltime() - it_start_time)

            if tlimit is not None:
                time_past = (walltime() - start_time) / 3600
                if (time_past + max_it_time / 3600) > tlimit:
                    if i not in checkpoint_iter:
                        self.save_checkpoint(i)
                    self.logger.close()
                    print(
                        f"\nEnding training at iteration {i}, after training for {time_past:.2f} "
                        f"hours as timelimit {tlimit:.2f} hours has been reached.\n"
                    )
                    return

        self.save_flow_samples_h5(n_samples=1000, filename="flow_samples_2d.h5", batch_size=256)
        print(f"\nRun completed in {(walltime() - start_time) / 3600:.2f} hours\n")
        self.logger.close()


# ============================================================
# main setup
# ============================================================

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

    run_initial_flow_sanity_test_2d(
        fab_model,
        target,
        batch_size=min(8, cfg.evaluation.n_eval if hasattr(cfg.evaluation, "n_eval") else 8),
    )

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

    plot = setup_plotter(cfg, target, buffer)

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
        trainer = TrainerLJ2D(
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
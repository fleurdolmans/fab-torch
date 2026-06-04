from typing import Callable, Any, Optional, List

import torch
import torch.nn.functional as F
import torch.optim.optimizer
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from time import time
import os

from fab.utils.logging import Logger, ListLogger, WandbLogger
from fab.core import FABModel
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer


lr_scheduler = Any  # a learning rate scheduler from torch.optim.lr_scheduler
Plotter = Callable[[FABModel], List[plt.Figure]]


def _mic(dx: torch.Tensor, L: float) -> torch.Tensor:
    L_t = torch.as_tensor(L, device=dx.device, dtype=dx.dtype)
    return dx - L_t * torch.round(dx / L_t)


def _pair_clash_penalty(
    A: torch.Tensor,   # (B, NA, 2)
    B: torch.Tensor,   # (B, NB, 2)
    L: float,
    r0: float,
    k: float = 200.0,
) -> torch.Tensor:
    dx = _mic(A[:, :, None, :] - B[:, None, :, :], L)
    d = torch.linalg.norm(dx, dim=-1)
    pen = F.softplus(k * (r0 - d)) / k
    return pen.sum(dim=(1, 2)).mean()


def _lj_overlap_penalty(
    x_flat: torch.Tensor,   # (B, 2*(n_solute+n_solvent))
    L: float,
    n_solute: int,
    n_solvent: int,
    r0_ssolv: float = 0.24,
    r0_solv_solute: float = 0.24,
    k: float = 200.0,
    chunk: int = 64,
) -> torch.Tensor:
    B, D = x_flat.shape
    X = x_flat.view(B, D // 2, 2)
    solute  = X[:, :n_solute, :]
    solvent = X[:, n_solute:n_solute + n_solvent, :]

    sw_pen = _pair_clash_penalty(solvent, solute, L=L, r0=r0_solv_solute, k=k)

    L_t = torch.as_tensor(L, device=x_flat.device, dtype=x_flat.dtype)
    r0_t = torch.as_tensor(r0_ssolv, device=x_flat.device, dtype=x_flat.dtype)
    ss_pen_per_batch = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)
    for i0 in range(0, n_solvent, chunk):
        i1  = min(n_solvent, i0 + chunk)
        Xi  = solvent[:, i0:i1, :]
        ci  = i1 - i0
        dx  = _mic(Xi[:, :, None, :] - solvent[:, None, :, :], L_t)
        d   = torch.linalg.norm(dx, dim=-1)
        rows = torch.arange(ci, device=x_flat.device)
        mask = torch.zeros(ci, n_solvent, device=x_flat.device, dtype=torch.bool)
        mask[rows, rows + i0] = True
        d = d.masked_fill(mask.unsqueeze(0), 1e9)
        ss_pen_per_batch = ss_pen_per_batch + (F.softplus(k * (r0_t - d)) / k).sum(dim=(1, 2))

    return ss_pen_per_batch.mean() + sw_pen


class PrioritisedBufferTrainer:
    """A trainer for the FABModel for use with a prioritised replay buffer, and a different form
    of loss. In this training loop we target p^\alpha / q^(\alpha - 1) instead of p."""
    def __init__(
        self,
        model: FABModel,
        optimizer: torch.optim.Optimizer,
        buffer: PrioritisedReplayBuffer,
        alpha: float,
        n_batches_buffer_sampling: int = 2,
        optim_scheduler: Optional[lr_scheduler] = None,
        logger: Logger = ListLogger(),
        plot: Optional[Plotter] = None,
        max_gradient_norm: Optional[float] = 5.0,
        w_adjust_max_clip: Optional[float] = 10.0,
        w_adjust_in_buffer_after_update: bool = False,
        save_path: str = "",
        lr_step=1,
        warmup_scheduler: Optional[lr_scheduler] = None,
        warmup_iters: int = 0,
        print_eval: bool = False,
        overlap_w: float = 0.0,
        overlap_L: float = 1.0,
        overlap_n_solute: int = 1,
        overlap_n_solvent: int = 1,
        overlap_r0: float = 0.24,
    ):
        self.model = model
        self.alpha = alpha

        # Ensure we have p^\alpha q^{1-\alpha} as the AIS target distribution.
        self.model.p_target = False
        self.model.annealed_importance_sampler.p_target = False

        self.optimizer = optimizer
        self.optim_scheduler = optim_scheduler
        self.lr_step = lr_step
        self.logger = logger
        self.plot = plot
        # if no gradient clipping set max_gradient_norm to inf
        self.max_gradient_norm = max_gradient_norm if max_gradient_norm else float("inf")
        self.save_dir = save_path
        self.print_eval = print_eval
        self.plots_dir = os.path.join(self.save_dir, f"plots")
        self.checkpoints_dir = os.path.join(self.save_dir, f"model_checkpoints")
        self.buffer = buffer
        self.n_batches_buffer_sampling = n_batches_buffer_sampling
        self.flow_device = next(model.flow.parameters()).device
        self.max_adjust_w_clip = w_adjust_max_clip
        self.w_adjust_in_buffer_after_update = w_adjust_in_buffer_after_update
        self.warmup_scheduler = warmup_scheduler
        self.warmup_iters = warmup_iters
        self.overlap_w = overlap_w
        self.overlap_L = overlap_L
        self.overlap_n_solute = overlap_n_solute
        self.overlap_n_solvent = overlap_n_solvent
        self.overlap_r0 = overlap_r0

    def save_checkpoint(self, i):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
        pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
        self.model.save(os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        self.buffer.save(os.path.join(checkpoint_path, "buffer.pt"))
        if self.optim_scheduler:
            torch.save(self.optim_scheduler.state_dict(), os.path.join(self.checkpoints_dir, "scheduler.pt"))
        if self.warmup_scheduler:
            torch.save(self.warmup_scheduler.state_dict(), os.path.join(self.checkpoints_dir, "warmup_scheduler.pt"))

    def make_and_save_plots(self, i, save):
        if hasattr(self.model.target_distribution, "plot_marginal_hists"):
            plot_dict = {
                "plot_md_energies": (i == 0 and self.model.target_distribution.plot_MD_energies),
                "plot_marginal_hists": self.model.target_distribution.plot_marginal_hists,
            }
            figures = self.plot(self.model, plot_dict)
        else:
            figures = self.plot(self.model)

        for j, figure in enumerate(figures):
            if save:
                if isinstance(self.logger, WandbLogger):
                    self.logger.write({f"it{i}_fig{j}": wandb.Image(figure), "iteration": i})
                else:
                    figure.savefig(os.path.join(self.plots_dir, f"{j}_iter_{i}.png"))
            else:
                plt.show()
            plt.close(figure)

    def perform_eval(self, i, eval_batch_size, batch_size):
        # Set ais distribution to target for evaluation of ess, freeze transition operator params.
        self.model.annealed_importance_sampler.transition_operator.set_eval_mode(True)
        eval_info_true_target = self.model.get_eval_info(
            outer_batch_size=eval_batch_size,
            inner_batch_size=batch_size,
            set_p_target=True,
            iteration=i,
        )
        # Double check the ais distribution has been set back to p^\alpha q^{1-\alpha}.
        assert self.model.annealed_importance_sampler.p_target is False
        assert self.model.annealed_importance_sampler.transition_operator.p_target is False
        # Evaluation with the AIS ESS with target set as p^\alpha q^{1-\alpha}.
        eval_info_practical_target = self.model.get_eval_info(
            outer_batch_size=eval_batch_size,
            inner_batch_size=batch_size,
            set_p_target=False,
            ais_only=True,
            iteration=i,
        )
        self.model.annealed_importance_sampler.transition_operator.set_eval_mode(False)

        eval_info = {}
        eval_info.update({key + "_p_target": val for key, val in eval_info_true_target.items()})
        eval_info.update({key + "_min_var_target": val for key, val in eval_info_practical_target.items()})
        eval_info.update(iteration=i)
        self.logger.write(eval_info)
        if self.print_eval:
            print(
                "   Eval metrics: " +
                str({key: "{:.4f}".format(value) for key, value in eval_info.items() if key != "step"})
            )

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
        # Linspace (0, 100, 11) = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        #  We can remove the first entry, since we want special behaviour before training anyway.
        #  Our primary loop uses a 0-indexed `t` variable, but the `iteration` or `i` count used for evaluation is
        #   1-indexed, so n_iteration=100 will give `t`=99 for the final iteration, but `i`=100 so it will be evaluated.
        if n_checkpoints:
            checkpoint_iter = list(np.linspace(0, n_iterations, n_checkpoints + 1, dtype="int")[1:])
        if n_eval is not None:
            eval_iter = list(np.linspace(0, n_iterations, n_eval + 1, dtype="int")[1:])
            assert eval_batch_size is not None
        if n_plot is not None:
            plot_iter = list(np.linspace(0, n_iterations, n_plot + 1, dtype="int")[1:])
        if tlimit is not None:
            assert n_checkpoints is not None, "Time limited specified but no checkpoints are " "being saved."
        if start_time is None:
            start_time = time()
        if start_iter >= n_iterations:
            raise Exception("Not running training as start_iter >= total training iterations")

        max_it_time = 0.0
        global_step = 0
        # pbar = tqdm(range(n_iterations - start_iter))
        # for pbar_iter in pbar:
        #     i = pbar_iter + start_iter + 1
        for t in range(start_iter, n_iterations, 1):
            i = t + 1
            print(f"Iteration: {i}/{n_iterations}")
            iter_start = time()
            it_start_time = time()
            self.optimizer.zero_grad()
            # collect samples and log weights with AIS and add to the buffer
            ais_time = time()
            point_ais, log_w_ais = self.model.annealed_importance_sampler.sample_and_log_weights(
                batch_size, purpose="fill buffer"
            )
            if i % 10 == 0:
                print(f" AIS time: {time() - ais_time:.2f}s.")
            x_ais = point_ais.x.detach()
            log_w_ais = log_w_ais.detach()
            log_q_x_ais = point_ais.log_q.detach()
            self.buffer.add(x_ais.detach(), log_w_ais.detach(), log_q_x_ais.detach())
            if i % 10 == 0:
                print(f" Buffer contains {self.buffer.get_buffer_size()} points.")

            # we log info from the step of the recently generated ais points.
            info = self.model.get_iter_info()

            # We now take self.n_batches_buffer_sampling gradient steps using
            # data from the replay buffer.
            mini_dataset = self.buffer.sample_n_batches(batch_size=batch_size, n_batches=self.n_batches_buffer_sampling)
            if i % 10 == 0:
                print(f" Sampled {self.n_batches_buffer_sampling} batches of size {batch_size}.")
            for (x, log_w, log_q_old, indices) in mini_dataset:
                x, log_w, log_q_old, indices = (
                    x.to(self.flow_device),
                    log_w.to(self.flow_device),
                    log_q_old.to(self.flow_device),
                    indices.to(self.flow_device),
                )
                self.optimizer.zero_grad()
                log_q_x = self.model.flow.log_prob(x)
                # adjustment to account for change to theta since sample was last added/adjusted
                log_w_adjust = (1 - self.alpha) * (log_q_x.detach() - log_q_old)
                w_adjust_pre_clip = torch.exp(log_w_adjust)  # no grad
                if self.max_adjust_w_clip is not None:
                    w_adjust = torch.clip(w_adjust_pre_clip, max=self.max_adjust_w_clip)
                else:
                    w_adjust = w_adjust_pre_clip
                # manually calculate the new form of the loss
                loss = -torch.mean(w_adjust * log_q_x)
                # add overlap penalty on fresh flow samples
                if self.overlap_w > 0.0:
                    x_pen, _ = self.model.flow.sample_and_log_prob((batch_size,))
                    # prepend pinned solute at origin to get full (B, 2*(n_solute+n_solvent)) layout
                    solute_pen = torch.zeros(
                        batch_size, self.overlap_n_solute, 2,
                        device=x_pen.device, dtype=x_pen.dtype,
                    )
                    x_full_pen = torch.cat(
                        [solute_pen.view(batch_size, -1), x_pen], dim=1
                    )
                    pen = _lj_overlap_penalty(
                        x_full_pen,
                        L=self.overlap_L,
                        n_solute=self.overlap_n_solute,
                        n_solvent=self.overlap_n_solvent,
                        r0_ssolv=self.overlap_r0,
                        r0_solv_solute=self.overlap_r0,
                    )
                    loss = loss + self.overlap_w * pen

       
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                        global_step += 1
                        # Step schedulers after optimizer update
                        if self.warmup_scheduler is not None and global_step <= self.warmup_iters:
                            # warmup steps every optimizer step
                            self.warmup_scheduler.step()

                        else:
                            # Normal scheduler uses your existing lr_step cadence
                            if self.optim_scheduler is not None and (global_step % self.lr_step == 0):
                                self.optim_scheduler.step()
                    else:
                        print(f"nan grad norm in replay step (batch size: {batch_size}")
                else:
                    print(f"nan loss in replay step (batch size: {batch_size}")

                # Adjust log weights in the buffer on the fly.
                if not self.w_adjust_in_buffer_after_update:
                    with torch.no_grad():
                        self.buffer.adjust(log_w_adjust, log_q_x, indices)

            info.update(
                loss=loss.cpu().detach().item(),
                step=i,
                grad_norm=grad_norm.cpu().detach().item(),
                sampled_log_w_std=torch.std(log_w).detach().cpu().item(),
                sampled_log_w_mean=torch.mean(log_w).detach().cpu().item(),
                w_adjust_mean=torch.mean(w_adjust_pre_clip).detach().cpu().item(),
                w_adjust_min=torch.min(w_adjust_pre_clip).detach().cpu().item(),
                w_adjust_max=torch.max(w_adjust_pre_clip).detach().cpu().item(),
                log_q_x_mean=torch.mean(log_q_x).cpu().item(),
            )

            if self.w_adjust_in_buffer_after_update:
                with torch.no_grad():
                    for (x, log_w, log_q_old, indices) in mini_dataset:
                        """Adjust importance weights in the buffer for the points in the
                        `mini_dataset` to account for the updated theta."""
                        x, log_w, log_q_old, indices = (
                            x.to(self.flow_device),
                            log_w.to(self.flow_device),
                            log_q_old.to(self.flow_device),
                            indices.to(self.flow_device),
                        )
                        log_q_new = self.model.flow.log_prob(x)
                        log_w_adjust_insert = (1 - self.alpha) * (log_q_new - log_q_old)
                        self.buffer.adjust(log_w_adjust_insert, log_q_new, indices)
                    info.update(
                        log_w_adjust_insert_mean=torch.mean(log_w_adjust_insert).detach().cpu().item(),
                        log_q_mean=torch.mean(log_q_new).detach().cpu().item(),
                    )

            self.logger.write(info)
            loss_str = (
                f" Train loss: {loss.cpu().detach().item():.4f}, "
                f"ess base: {info['ess_base']:.4f}, "
                f"ess ais: {info['ess_ais']:.4f}"
            )
            # pbar.set_description(loss_str)
            if i % 10 == 0:
                print(loss_str)

            if n_eval is not None:
                if i in eval_iter:
                    self.perform_eval(i, eval_batch_size, batch_size)

            if n_plot is not None:
                if i in plot_iter:
                    self.make_and_save_plots(i, save)

            if n_checkpoints is not None:
                if i in checkpoint_iter:
                    self.save_checkpoint(i)

            if i % 10 == 0:
                print(f" Iteration time: {time() - iter_start:.2f}s.")
            max_it_time = max(max_it_time, time() - it_start_time)

            # End job if necessary
            if tlimit is not None:
                time_past = (time() - start_time) / 3600
                if (time_past + max_it_time / 3600) > tlimit:
                    # self.perform_eval(i, eval_batch_size, batch_size)
                    # self.make_and_save_plots(i, save)
                    if i not in checkpoint_iter:
                        self.save_checkpoint(i)
                    self.logger.close()
                    print(
                        f"\nEnding training at iteration {i}, after training for {time_past:.2f} "
                        f"hours as timelimit {tlimit:.2f} hours has been reached.\n"
                    )
                    return

        print(f"\n Run completed in {(time() - start_time) / 3600:.2f} hours \n")
        if tlimit is not None:
            print(f"Run finished before timelimit of {tlimit:.2f} hours was reached. \n")

        self.logger.close()

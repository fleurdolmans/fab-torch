import pathlib
import os
import wandb
import warnings
from time import time
from typing import Callable, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch.optim.optimizer

from fab.utils.logging import Logger, ListLogger, WandbLogger
from fab.types_ import Model
from fab.core import FABModel

import torch
import torch.nn.functional as F

lr_scheduler = Any  # a learning rate scheduler from torch.optim.lr_scheduler
Plotter = Callable[[Model], List[plt.Figure]]


class Trainer:
    def __init__(
        self,
        model: FABModel,
        save_path: str,
        optimizer: torch.optim.Optimizer,
        optim_scheduler: Optional[lr_scheduler] = None,
        logger: Logger = ListLogger(),
        plot: Optional[Plotter] = None,
        max_gradient_norm: Optional[float] = 5.0,
        lr_step=1,
        warmup_scheduler: Optional[lr_scheduler] = None,
        warmup_iters: int = 0,
        print_eval: bool = False,
        overlap_penalty: Optional[float] = 0.2,
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
        # if no gradient clipping set max_gradient_norm to inf
        self.max_gradient_norm = max_gradient_norm if max_gradient_norm else float("inf")
        self.save_dir = save_path
        self.print_eval = print_eval
        self.plots_dir = os.path.join(self.save_dir, f"plots")
        self.checkpoints_dir = os.path.join(self.save_dir, f"model_checkpoints")
        self.warmup_scheduler = warmup_scheduler
        self.warmup_iters = warmup_iters
        self.overlap_penalty = overlap_penalty
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
    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        return dx - L * torch.round(dx / L)

    def solute_water_clash_penalty(
        self,
        x_flat: torch.Tensor,   # (B, 3N)
        L: float,
        n_solute: int,
        n_waters: int,
        r0_SO: float = 0.26,
        r0_OO: float = 0.25,
        r0_SH: float = 0.18,
        r0_OH: float = 0.18,
        k: float = 200.0,
        include_H: bool = False,
    ) -> torch.Tensor:
        """
        Clash penalty between solute atoms and water atoms.
        Assumes solute atoms are [S, O, O] and waters are [O,H,H].
        """
        B, D = x_flat.shape
        X = x_flat.view(B, -1, 3)

        # solute atoms
        S  = X[:, 0:1, :]   # (B,1,3)
        Os = X[:, 1:3, :]   # (B,2,3)

        # water atoms
        O_idx  = [n_solute + 3*w for w in range(n_waters)]
        H1_idx = [n_solute + 3*w + 1 for w in range(n_waters)]
        H2_idx = [n_solute + 3*w + 2 for w in range(n_waters)]

        Owat = X[:, O_idx, :]    # (B,n_waters,3)

        pen = 0.0
        pen = pen + self.pair_clash_penalty(S,  Owat, L=L, r0=r0_SO, k=k)
        pen = pen + self.pair_clash_penalty(Os, Owat, L=L, r0=r0_OO, k=k)

        if include_H:
            H1 = X[:, H1_idx, :]
            H2 = X[:, H2_idx, :]
            pen = pen + self.pair_clash_penalty(S,  H1, L=L, r0=r0_SH, k=k)
            pen = pen + self.pair_clash_penalty(S,  H2, L=L, r0=r0_SH, k=k)
            pen = pen + self.pair_clash_penalty(Os, H1, L=L, r0=r0_OH, k=k)
            pen = pen + self.pair_clash_penalty(Os, H2, L=L, r0=r0_OH, k=k)

        return pen

    def pair_clash_penalty(
        self,
        A: torch.Tensor,   # (B, NA, 3)
        B: torch.Tensor,   # (B, NB, 3)
        L: float,
        r0: float,
        k: float = 200.0,
    ) -> torch.Tensor:
        """
        Soft clash penalty between two atom sets A and B under PBC.
        Returns mean penalty over batch.
        """
        L_t = torch.as_tensor(L, device=A.device, dtype=A.dtype)

        dx = self.mic(A[:, :, None, :] - B[:, None, :, :], L_t)
        d = torch.linalg.norm(dx, dim=-1)               # (B, NA, NB)

        r0_t = torch.as_tensor(r0, device=A.device, dtype=A.dtype)
        pen = F.softplus(k * (r0_t - d)) / k            # (B, NA, NB)
        return pen.sum(dim=(1, 2)).mean()

    def oo_clash_penalty(self, 
        x_flat: torch.Tensor,      # (B,3N) nm
        L: float,
        n_solute: int,
        n_waters: int,
        r0: float = 0.22,          # nm
        k: float = 200.0,          # softness
        chunk: int = 64,
    ) -> torch.Tensor:
        """
        Differentiable soft penalty: sum softplus(k*(r0 - d))/k over O_O pairs.
        Returns scalar.
        """
        B, D = x_flat.shape
        N = D // 3
        X = x_flat.view(B, N, 3)

        # O indices: n_solute + 3*w
        O = torch.stack([X[:, n_solute + 3*w, :] for w in range(n_waters)], dim=1)  # (B,W,3)

        L_t = torch.as_tensor(L, device=x_flat.device, dtype=x_flat.dtype)
        r0_t = torch.as_tensor(r0, device=x_flat.device, dtype=x_flat.dtype)

        pen = torch.zeros((B,), device=x_flat.device, dtype=x_flat.dtype)

        for i0 in range(0, n_waters, chunk):
            i1 = min(n_waters, i0 + chunk)
            Oi = O[:, i0:i1, :]  # (B,ci,3)
            ci = i1 - i0

            dx = self.mic(Oi[:, :, None, :] - O[:, None, :, :], L_t)  # (B,ci,W,3)
            d = torch.linalg.norm(dx, dim=-1)                    # (B,ci,W)

            # Mask self-distances without in-place ops:
            # For each local row kk in [0,ci), the "self" column is (i0+kk)
            rows = torch.arange(ci, device=x_flat.device)
            cols = rows + i0
            mask_self = torch.zeros((ci, n_waters), device=x_flat.device, dtype=torch.bool)
            mask_self[rows, cols] = True
            d = d.masked_fill(mask_self.unsqueeze(0), 1e9)

            pen = pen + (F.softplus(k * (r0_t - d)) / k).sum(dim=(1, 2))

        return pen.mean()

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
            assert n_checkpoints is not None, "Time limited specified but not checkpoints are " "being saved."
        if start_time is not None:
            start_time = time()

        if start_iter >= n_iterations:
            raise Exception("Not running training as start_iter >= total training iterations")

        max_it_time = 0.0
        k, epoch, next_epoch = 0, 0, True  # Used for Maximum Likelihood (forward KL) training.
        target_dist = self.model.target_distribution

        if n_eval is not None:  # Save any pre-training eval metrics
            self.perform_eval(0, eval_batch_size, batch_size)

        if n_plot is not None:  # Save any pre-training plots
            self.make_and_save_plots(0, save)
        
        target_dist.train_data_i = target_dist.train_data_i.reshape(-1, target_dist.internal_dim).contiguous()
        target_dist.train_logdet_xi = target_dist.train_logdet_xi.reshape(-1).contiguous()

        if self.n_pretraining is not None:
            print(f" Pretraining set: training with {self.n_pretraining} MD samples.")
            train_data = target_dist.train_data_i[:self.n_pretraining]
            train_logdet_xi = target_dist.train_logdet_xi[:self.n_pretraining]
        else:
            train_data = target_dist.train_data_i
            train_logdet_xi = target_dist.train_logdet_xi

        overlap_w = self.overlap_penalty

        global_step = 0
        for t in range(start_iter, n_iterations, 1):
            print("Iteration {}/{}".format(t + 1, n_iterations))
            i = t + 1
            if i % 100 == 1:
                print(f"  Iteration: {i}/{n_iterations}")
            it_start_time = time()
            self.optimizer.zero_grad()

            if self.model.loss_type == "forward_kl":
                # MD training: get the next batch of data and compute the likelihood (loss) under the Flow.
                # 'i' here represents that the data has already been transformed to internal coordinates, rather than
                #  Cartesian. This is what we feed into the flow.
                
                # shuffle indices once per epoch
                if k == 0:
                    perm = torch.randperm(train_data.shape[0], device=train_data.device)
                # slice
                idx = perm[k*batch_size:(k+1)*batch_size]
                i_batch = train_data[idx].to(self.flow_device, non_blocking=True)
                logdet_batch = train_logdet_xi[idx].to(self.flow_device, non_blocking=True)

                flow_loss = self.model.loss(i_batch)
                transform_loss = -logdet_batch.mean()
                loss = flow_loss + transform_loss
                
   
                # train_data = target_dist.train_data_i.clone().reshape(-1, target_dist.internal_dim)
                # Log determinant Jacobian for the transformation from Cartesian to internal coordinates.
                # train_logdet_xi = target_dist.train_logdet_xi.clone()
                # Shuffle train data if first iteration
                # if k == 0:
                #     permutation = torch.randperm(len(train_data))
                #     train_data = train_data[permutation]
                #     train_logdet_xi = train_logdet_xi[permutation]
                # i_batch = train_data[k * batch_size: (k + 1) * batch_size, ...].to(self.flow_device)
                # # Loss (log likelihood slash forward KL divergence) on this batch
                # flow_loss = self.model.loss(i_batch)
                # transform_loss = -train_logdet_xi.mean()  # negative because loss is neg of p log q.
                # # TODO: Maybe add a term for OH bond lengths and angles? Bit strange, because we will no longer be
                # #  doing likelihood minimisation exactly, but the optimum does not change, so it might be okay.
                # loss = flow_loss + transform_loss
                if (k + 1) * batch_size >= len(train_data):
                    k = 0  # Restart epoch if current batch exceeds number of training data points
                    epoch += 1
                    next_epoch = True
                else:
                    k += 1
                    next_epoch = False
            else:
                # Not doing MD training here.
                # Loss function generates samples from the Flow and computes the loss value internally.
                loss = self.model.loss(batch_size)
            
            # -------------------------------------------------
            # Optional MD mixing term
            # -------------------------------------------------
            if self.mixing > 0.0:
                train_data = target_dist.train_data_i
                train_logdet_xi = target_dist.train_logdet_xi

                n_mix = int(round(self.mixing * batch_size))
                n_mix = max(1, min(n_mix, batch_size))

                perm = torch.randperm(train_data.shape[0], device=train_data.device)
                idx = perm[:n_mix]

                i_batch_mix = train_data[idx].to(self.flow_device, non_blocking=True)
                logdet_batch_mix = train_logdet_xi[idx].to(self.flow_device, non_blocking=True)

                # MD likelihood term under the flow
                flow_loss_mix = -self.model.flow.log_prob(i_batch_mix).mean()
                transform_loss_mix = -logdet_batch_mix.mean()
                data_loss_mix = flow_loss_mix + transform_loss_mix

                loss = self.mixing * data_loss_mix + (1 - self.mixing) * loss


            # -------------------------------------------------
            # Overlap penalty on flow samples
            # -------------------------------------------------
            if overlap_w > 0.0:
                B_rev = batch_size
                z, log_q = self.model.flow.sample_and_log_prob((B_rev,))

                x_pen, _ = target_dist.coordinate_transform.forward(z)  # (B_rev, 3N) in nm

                oo_pen = self.oo_clash_penalty(
                    x_pen,
                    L=float(target_dist.box_length_nm),
                    n_solute=3,
                    n_waters=int(target_dist.num_solvent_molecules),
                    r0=0.24,
                    k=200.0,
                    chunk=64,
                )

                sw_pen = self.solute_water_clash_penalty(
                    x_pen,
                    L=float(target_dist.box_length_nm),
                    n_solute=3,
                    n_waters=int(target_dist.num_solvent_molecules),
                    r0_SO=0.25,
                    r0_OO=0.20,
                    include_H=False,
                    k=100.0,
                )

                pen = oo_pen + sw_pen
                loss = loss + overlap_w * pen



            # Update parameters
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
                old_grad_norm = torch.cat(grads).norm().clone()
                # Clip grad norm
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
                    warnings.warn("Encountered inf grad norm!")
          
            else:
                warnings.warn("NaN loss encountered! No update performed.")
                old_grad_norm = torch.zeros_like(loss)
                grad_norm = torch.zeros_like(loss)

            self.optimizer.zero_grad()
            info = self.model.get_iter_info()
            info.update(
                {
                    "loss": loss.cpu().detach().item(),
                    "old_grad_norm": old_grad_norm.cpu().detach().item(),
                    "grad_norm": grad_norm.cpu().detach().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "iteration": i,
                }
            )

            if overlap_w > 0.0:
                info["oo_pen"] = oo_pen.detach().cpu().item()
                info["sw_pen"] = sw_pen.detach().cpu().item()

            ct = getattr(target_dist, "coordinate_transform", None)
            if ct is not None and hasattr(ct, "get_stats"):

                s = ct.get_stats()
                B = max(int(s.get("B", 0)), 1)
                # Rates
                info.update({
                    "geom/seam_x_rate": s.get("seam_x", 0) / B,
                    "geom/seam_z_rate": s.get("seam_z", 0) / B,
                    "geom/degenerate_rate": s.get("degenerate", 0) / B,
                })

                # Raw counts (optional but nice for debugging)
                info.update({
                    "geom/seam_x_count": s.get("seam_x", 0),
                    "geom/seam_z_count": s.get("seam_z", 0),
                    "geom/degenerate_count": s.get("degenerate", 0),
                    "geom/calls": s.get("calls", 0),
                    "geom/B": s.get("B", 0),
                })

            self.logger.write(info)

            loss_str = f"   Iter {i}, Train loss: {loss.cpu().detach().item():.4f}"
            if "ess_ais" in info.keys():
                loss_str += f" ess base: {info['ess_base']:.4f}, ess ais: {info['ess_ais']:.4f}"
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

            max_it_time = max(max_it_time, time() - it_start_time)
            # End job if necessary
            if tlimit is not None:
                time_past = (time() - start_time) / 3600
                if (time_past + max_it_time / 3600) > tlimit:
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

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
        print_eval: bool = False,
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

    def save_checkpoint(self, i):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
        pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
        self.model.save(os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        if self.optim_scheduler:
            torch.save(self.optim_scheduler.state_dict(), os.path.join(self.checkpoints_dir, "scheduler.pt"))

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

        for t in range(start_iter, n_iterations, 1):
            i = t + 1
            if self.model.loss_type == "forward_kl" and next_epoch:
                print(f" The following iterations correspond to epoch {epoch} of Forward KL training.")
            if i % 100 == 1:
                print(f"  Iteration: {i}/{n_iterations}")
            it_start_time = time()
            self.optimizer.zero_grad()

            if self.model.loss_type == "forward_kl":
                # MD training: get the next batch of data and compute the likelihood (loss) under the Flow.
                # 'i' here represents that the data has already been transformed to internal coordinates, rather than
                #  Cartesian. This is what we feed into the flow.
                train_data = target_dist.train_data_i.clone().reshape(-1, target_dist.internal_dim)
                # Log determinant Jacobian for the transformation from Cartesian to internal coordinates.
                train_logdet_xi = target_dist.train_logdet_xi.clone()
                # Shuffle train data if first iteration
                if k == 0:
                    permutation = torch.randperm(len(train_data))
                    train_data = train_data[permutation]
                    train_logdet_xi = train_logdet_xi[permutation]
                i_batch = train_data[k * batch_size: (k + 1) * batch_size, ...].to(self.flow_device)
                # Loss (log likelihood slash forward KL divergence) on this batch
                flow_loss = self.model.loss(i_batch)
                transform_loss = -train_logdet_xi.mean()  # negative because loss is neg of p log q.
                # TODO: Maybe add a term for OH bond lengths and angles? Bit strange, because we will no longer be
                #  doing likelihood minimisation exactly, but the optimum does not change, so it might be okay.
                loss = flow_loss + transform_loss
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

            # Update parameters
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
                old_grad_norm = torch.cat(grads).norm()
                # Clip grad norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                else:
                    warnings.warn("Encountered inf grad norm!")
                if self.optim_scheduler and (i + 1) % self.lr_step == 0:
                    self.optim_scheduler.step()
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

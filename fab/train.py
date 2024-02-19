from typing import Callable, Any, Optional, List

import torch.optim.optimizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from fab.utils.logging import Logger, ListLogger
from fab.types_ import Model
from fab.core import FABModel
import pathlib
import os
from time import time

lr_scheduler = Any  # a learning rate schedular from torch.optim.lr_scheduler
Plotter = Callable[[Model], List[plt.Figure]]


class Trainer:
    def __init__(
        self,
        model: FABModel,
        optimizer: torch.optim.Optimizer,
        optim_schedular: Optional[lr_scheduler] = None,
        logger: Logger = ListLogger(),
        plot: Optional[Plotter] = None,
        max_gradient_norm: Optional[float] = 5.0,
        save_path: str = "",
        lr_step=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.optim_schedular = optim_schedular
        self.lr_step = lr_step
        self.logger = logger
        self.plot = plot
        self.flow_device = next(model.flow.parameters()).device
        # if no gradient clipping set max_gradient_norm to inf
        self.max_gradient_norm = max_gradient_norm if max_gradient_norm else float("inf")
        self.save_dir = save_path
        self.plots_dir = os.path.join(self.save_dir, f"plots")
        self.checkpoints_dir = os.path.join(self.save_dir, f"model_checkpoints")

    def save_checkpoint(self, i):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
        pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
        self.model.save(os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        if self.optim_schedular:
            torch.save(self.optim_schedular.state_dict(), os.path.join(self.checkpoints_dir, "scheduler.pt"))

    def make_and_save_plots(self, i, save):
        figures = self.plot(self.model)
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
        eval_info.update(step=i)
        self.logger.write(eval_info)
        print("   Eval metrics: " + str({key: "{:.4f}".format(value) for key, value in eval_info.items() if key != "step"}))

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
            checkpoint_iter = list(np.linspace(1, n_iterations, n_checkpoints, dtype="int"))
        if n_eval is not None:
            eval_iter = list(np.linspace(1, n_iterations, n_eval, dtype="int"))
            assert eval_batch_size is not None
        if n_plot is not None:
            plot_iter = list(np.linspace(1, n_iterations, n_plot, dtype="int"))
        if tlimit is not None:
            assert n_checkpoints is not None, "Time limited specified but not checkpoints are " "being saved."
        if start_time is not None:
            start_time = time()

        if start_iter >= n_iterations:
            raise Exception("Not running training as start_iter >= total training iterations")

        max_it_time = 0.0
        # pbar = tqdm(range(n_iterations - start_iter))
        # for pbar_iter in pbar:
        #     i = pbar_iter + start_iter + 1
        k, epoch, next_epoch = 0, 0, True  # Used for Maximum Likelihood (forward KL) training.
        target_dist = self.model.target_distribution
        for t in range(start_iter, n_iterations, 1):
            i = t + 1
            if self.model.loss_type == "forward_kl" and next_epoch:
                print(f" The following iterations correspond to epoch {epoch} of Forward KL training.")

            if i % 100 == 1:
                print(f"  Iteration: {i}/{n_iterations}")
            iter_start = time()
            it_start_time = time()
            self.optimizer.zero_grad()

            # with torch.no_grad():
            #     s = self.model.flow.sample((10000,))
            #     print("r range:", s[:, 3::3].min(), s[:, 3::3].max())
            #     print("phi range:", s[:, 4::3].min(), s[:, 4::3].max())
            #     print("theta range:", s[:, 5::3].min(), s[:, 5::3].max())

            if self.model.loss_type == "forward_kl":
                # 'z' here represents that the data has already been transformed to internal coordinates, rather than
                #  Cartesian. This is what we feed into the flow.
                train_data = target_dist.train_data_i.clone().reshape(-1, target_dist.internal_dim)
                # Log determinant Jacobian for the transformation from Cartesian to internal coordinates.
                train_logdet_xi = target_dist.train_logdet_xi.clone()
                # shuffle train data if first iteration
                if k == 0:
                    permutation = torch.randperm(len(train_data))
                    train_data = train_data[permutation]
                    train_logdet_xi = train_logdet_xi[permutation]
                i_batch = train_data[k * batch_size: (k + 1) * batch_size, ...].to(self.flow_device)
                flow_loss = self.model.loss(i_batch)
                transform_loss = -train_logdet_xi.mean()  # negative because loss is neg of p log q.
                # print("J_XI", transform_loss)
                loss = flow_loss + transform_loss
                if (k + 1) * batch_size >= len(train_data):
                    k = 0  # Restart epoch if current batch exceeds number of training data points
                    epoch += 1
                    next_epoch = True
                else:
                    k += 1
                    next_epoch = False
            else:
                loss = self.model.loss(batch_size)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                else:
                    print("encountered inf grad norm")
                if self.optim_schedular and (i + 1) % self.lr_step == 0:
                    self.optim_schedular.step()
            else:
                print("nan loss encountered")

            self.optimizer.zero_grad()
            info = self.model.get_iter_info()
            info.update(loss=loss.cpu().detach().item(), step=i)
            info.update(grad_norm=grad_norm.cpu().detach().item())
            self.logger.write(info)

            loss_str = f"   Iter {i}, Train loss: {loss.cpu().detach().item():.4f}"
            if "ess_ais" in info.keys():
                loss_str += f" ess base: {info['ess_base']:.4f}, ess ais: {info['ess_ais']:.4f}"
            # pbar.set_description(loss_str)
            if i % 10 == 1:
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

            # print(f"  Iteration time: {time() - iter_start:.2f}s.")
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

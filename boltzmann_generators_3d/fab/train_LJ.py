import pathlib
import os
import wandb
import warnings
from time import time
from typing import Callable, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim.optimizer
import h5py

from boltzmann_generators_3d.fab.utils.logging import Logger, ListLogger, WandbLogger
from boltzmann_generators_3d.fab.types_ import Model
from boltzmann_generators_3d.fab.core import FABModel

lr_scheduler = Any
Plotter = Callable[[Model], List[plt.Figure]]


class TrainerLJ:
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

    @staticmethod
    def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
        return dx - L * torch.round(dx / L)

    def pair_clash_penalty(
        self,
        A: torch.Tensor,   # (B, NA, 3)
        B: torch.Tensor,   # (B, NB, 3)
        L: float,
        r0: float,
        k: float = 200.0,
    ) -> torch.Tensor:
        dx = self.mic(A[:, :, None, :] - B[:, None, :, :], L)
        d = torch.linalg.norm(dx, dim=-1)
        pen = F.softplus(k * (r0 - d)) / k
        return pen.sum(dim=(1, 2)).mean()

    def lj_overlap_penalty(
        self,
        x_flat: torch.Tensor,         # (B, 3N)
        L: float,
        n_solute: int,
        n_solvent: int,
        r0_ssolv: float = 0.24,       # solvent-solvent minimum soft radius
        r0_solv_solute: float = 0.24, # solvent-solute minimum soft radius
        k: float = 200.0,
        chunk: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generic LJ overlap penalty for:
        - solvent-solvent pairs
        - solvent-solute pairs

        Assumes atom order:
        [solute particles..., solvent particles...]
        """
        B, D = x_flat.shape
        N = D // 3
        X = x_flat.view(B, N, 3)

        solute = X[:, :n_solute, :]                  # (B, n_solute, 3)
        solvent = X[:, n_solute:n_solute + n_solvent, :]  # (B, n_solvent, 3)

        # solvent-solute penalty
        solv_solute_pen = self.pair_clash_penalty(
            solvent, solute, L=L, r0=r0_solv_solute, k=k
        )

        # solvent-solvent penalty without double counting / self terms
        L_t = torch.as_tensor(L, device=x_flat.device, dtype=x_flat.dtype)
        r0_t = torch.as_tensor(r0_ssolv, device=x_flat.device, dtype=x_flat.dtype)
        solv_solv_pen_per_batch = torch.zeros((B,), device=x_flat.device, dtype=x_flat.dtype)

        for i0 in range(0, n_solvent, chunk):
            i1 = min(n_solvent, i0 + chunk)
            Xi = solvent[:, i0:i1, :]  # (B, ci, 3)
            ci = i1 - i0

            dx = self.mic(Xi[:, :, None, :] - solvent[:, None, :, :], L_t)
            d = torch.linalg.norm(dx, dim=-1)  # (B, ci, n_solvent)

            # mask lower triangle (j <= i) to count each pair {i,j} exactly once
            global_row = torch.arange(i0, i1, device=x_flat.device).view(ci, 1)
            col = torch.arange(n_solvent, device=x_flat.device).view(1, n_solvent)
            mask_lower_tri = col <= global_row  # (ci, n_solvent), True where j <= i (incl. diagonal)
            d = d.masked_fill(mask_lower_tri.unsqueeze(0), 1e9)

            pen = F.softplus(k * (r0_t - d)) / k
            solv_solv_pen_per_batch = solv_solv_pen_per_batch + pen.sum(dim=(1, 2))

        solv_solv_pen = solv_solv_pen_per_batch.mean()

        return solv_solv_pen, solv_solute_pen
    
    def _sample_flow_cartesian(self, n_samples: int, batch_size: int = 256) -> torch.Tensor:
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

        x = torch.cat(xs, dim=0)
        n_atoms = target_dist.cartesian_dim // 3
        return x.view(n_samples, n_atoms, 3)

    def save_flow_samples_h5(
        self,
        n_samples: int = 1000,
        filename: str = "flow_samples.h5",
        batch_size: int = 256,
    ):
        save_path = os.path.join(self.save_dir, filename)
        x = self._sample_flow_cartesian(n_samples=n_samples, batch_size=batch_size).numpy()

        with h5py.File(save_path, "w") as f:
            f.create_dataset("coordinates", data=x)

        print(f"Saved flow samples H5 to: {save_path}")

    def _clean_topology_for_pdb(self, topology):
        from openmm import app, unit, Vec3

        new_top = app.Topology()
        atom_map = {}

        for chain in topology.chains():
            new_chain = new_top.addChain(chain.id)
            for residue in chain.residues():
                new_res = new_top.addResidue(residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    new_atom = new_top.addAtom(atom.name, atom.element, new_res, atom.id)
                    atom_map[atom] = new_atom

        for bond in topology.bonds():
            new_top.addBond(atom_map[bond[0]], atom_map[bond[1]])

        box = topology.getPeriodicBoxVectors()
        if box is not None:
            a, b, c = box
            a_nm = a.value_in_unit(unit.nanometer)
            b_nm = b.value_in_unit(unit.nanometer)
            c_nm = c.value_in_unit(unit.nanometer)

            new_top.setPeriodicBoxVectors((
                Vec3(a_nm[0], a_nm[1], a_nm[2]),
                Vec3(b_nm[0], b_nm[1], b_nm[2]),
                Vec3(c_nm[0], c_nm[1], c_nm[2]),
            ))

        return new_top

    def save_flow_samples_pdb(
        self,
        n_samples: int = 1000,
        filename: str = "flow_samples.pdb",
        batch_size: int = 256,
    ):
        from openmm import app, unit, Vec3

        target_dist = self.model.target_distribution
        save_path = os.path.join(self.save_dir, filename)

        x = self._sample_flow_cartesian(n_samples=n_samples, batch_size=batch_size).numpy()
        topology = self._clean_topology_for_pdb(target_dist.system.topology)

        with open(save_path, "w") as f:
            for i in range(n_samples):
                positions = [Vec3(*xyz) for xyz in x[i]] * unit.nanometer
                if i == 0:
                    app.PDBFile.writeFile(topology, positions, f, keepIds=True)
                else:
                    app.PDBFile.writeModel(topology, positions, f, modelIndex=i + 1)

        print(f"Saved flow samples PDB to: {save_path}")

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

        if tlimit is not None:
            assert n_checkpoints is not None

        if start_time is None:
            start_time = time()

        if start_iter >= n_iterations:
            raise Exception("Not running training as start_iter >= total training iterations")

        max_it_time = 0.0
        k, epoch, next_epoch = 0, 0, True
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
            print(f"Pretraining set: training with {self.n_pretraining} MD samples.")
            train_data = target_dist.train_data_i[:self.n_pretraining]
            train_logdet_xi = target_dist.train_logdet_xi[:self.n_pretraining]
        else:
            train_data = target_dist.train_data_i
            train_logdet_xi = target_dist.train_logdet_xi

        overlap_w = self.overlap_penalty
        global_step = 0

        for t in range(start_iter, n_iterations, 1):
            i = t + 1
            if i % 10 == 1:
                print(f"Iteration {i}/{n_iterations}")

            it_start_time = time()
            self.optimizer.zero_grad()

            # metrics placeholders so logging always works
            flow_loss = None
            transform_loss = None
            md_loss = None
            reverse_loss = None

            mean_log_q_flow = None
            mean_logdet_transform = None
            mean_log_q_total = None
            neg_mean_log_q_total = None
            mean_log_p = None
            neg_mean_log_p = None
            bad_frac_batch = None

            mean_log_q_flow_md = None
            mean_logdet_transform_md = None
            mean_log_q_total_md = None
            mean_log_p_md = None
            bad_frac_md = None

            mean_log_q_flow_model = None
            mean_logdet_transform_model = None
            mean_log_q_total_model = None
            mean_log_p_model = None
            reverse_kl_model_est = None
            bad_frac_model = None

            # -------------------------
            # forward KL
            # -------------------------
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
                    next_epoch = True
                else:
                    k += 1
                    next_epoch = False

                if self.mixing > 0.0:
                    n_mix = int(round(self.mixing * batch_size))
                    n_mix = max(1, min(n_mix, batch_size))

                    perm_mix = torch.randperm(train_data.shape[0], device=train_data.device)
                    idx_mix = perm_mix[:n_mix]

                    i_batch_mix = train_data[idx_mix].to(self.flow_device, non_blocking=True)
                    logdet_batch_mix = train_logdet_xi[idx_mix].to(self.flow_device, non_blocking=True)

                    flow_loss_mix = -self.model.flow.log_prob(i_batch_mix).mean()
                    transform_loss_mix = -logdet_batch_mix.mean()
                    data_loss_mix = flow_loss_mix + transform_loss_mix

                    md_loss = data_loss_mix
                    loss = self.mixing * data_loss_mix + (1.0 - self.mixing) * loss

                with torch.no_grad():
                    mean_log_q_flow = self.model.flow.log_prob(i_batch).mean()
                    mean_logdet_transform = logdet_batch.mean()
                    mean_log_q_total = mean_log_q_flow + mean_logdet_transform

                    log_p_batch = target_dist.log_prob(i_batch)
                    mean_log_p = log_p_batch.mean()

                    neg_mean_log_q_total = -mean_log_q_total
                    neg_mean_log_p = -mean_log_p
                    bad_frac_batch = (log_p_batch <= -1e7).double().mean()

            # -------------------------
            # base transport on data
            # -------------------------
            elif self.model.loss_type == "base_transport":
                if k == 0:
                    perm = torch.randperm(train_data.shape[0], device=train_data.device)

                idx = perm[k * batch_size:(k + 1) * batch_size]
                i_batch = train_data[idx].to(self.flow_device, non_blocking=True)
                logdet_batch = train_logdet_xi[idx].to(self.flow_device, non_blocking=True)

                loss = self.model.loss(i_batch)

                if (k + 1) * batch_size >= len(train_data):
                    k = 0
                    epoch += 1
                    next_epoch = True
                else:
                    k += 1
                    next_epoch = False

                with torch.no_grad():
                    mean_log_q_flow = self.model.flow.log_prob(i_batch).mean()
                    mean_logdet_transform = logdet_batch.mean()
                    mean_log_q_total = mean_log_q_flow + mean_logdet_transform

                    log_p_batch = target_dist.log_prob(i_batch)
                    mean_log_p = log_p_batch.mean()

                    neg_mean_log_q_total = -mean_log_q_total
                    neg_mean_log_p = -mean_log_p
                    bad_frac_batch = (log_p_batch <= -1e7).double().mean()

            # -------------------------
            # reverse KL / flow-sampled branch
            # -------------------------
            else:
                reverse_loss = self.model.loss(batch_size)
                loss = reverse_loss

                if self.mixing > 0.0:
                    n_mix = int(round(self.mixing * batch_size))
                    n_mix = max(1, min(n_mix, batch_size))

                    perm_mix = torch.randperm(train_data.shape[0], device=train_data.device)
                    idx_mix = perm_mix[:n_mix]

                    x_md = train_data[idx_mix].to(self.flow_device, non_blocking=True)
                    logdet_md = train_logdet_xi[idx_mix].to(self.flow_device, non_blocking=True)

                    flow_loss_md = -self.model.flow.log_prob(x_md).mean()
                    transform_loss_md = -logdet_md.mean()
                    md_loss = flow_loss_md + transform_loss_md

                    loss = (1.0 - self.mixing) * reverse_loss + self.mixing * md_loss

                with torch.no_grad():
                    # data-side diagnostics on an MD minibatch
                    perm_md_eval = torch.randperm(train_data.shape[0], device=train_data.device)
                    idx_md_eval = perm_md_eval[:batch_size]
                    i_batch = train_data[idx_md_eval].to(self.flow_device, non_blocking=True)
                    logdet_batch = train_logdet_xi[idx_md_eval].to(self.flow_device, non_blocking=True)

                    mean_log_q_flow_md = self.model.flow.log_prob(i_batch).mean()
                    mean_logdet_transform_md = logdet_batch.mean()
                    mean_log_q_total_md = mean_log_q_flow_md + mean_logdet_transform_md

                    log_p_md = target_dist.log_prob(i_batch)
                    mean_log_p_md = log_p_md.mean()
                    bad_frac_md = (log_p_md <= -1e7).double().mean()

                    # model-side diagnostics on flow samples
                    z_model, log_q_flow_model = self.model.flow.sample_and_log_prob((batch_size,))
                    x_model, logdet_model = target_dist.coordinate_transform.forward(z_model)

                    mean_log_q_flow_model = log_q_flow_model.mean()
                    mean_logdet_transform_model = logdet_model.mean()
                    mean_log_q_total_model = mean_log_q_flow_model + mean_logdet_transform_model

                    # assumes target_dist.log_prob expects internal coordinates
                    log_p_model = target_dist.log_prob(z_model)
                    mean_log_p_model = log_p_model.mean()
                    reverse_kl_model_est = (log_q_flow_model - log_p_model).mean()
                    bad_frac_model = (log_p_model <= -1e7).double().mean()

            # overlap penalty on flow samples
            oo_pen = None
            sw_pen = None
            if overlap_w > 0.0:
                z_pen, _ = self.model.flow.sample_and_log_prob((batch_size,))
                x_pen, _ = target_dist.coordinate_transform.forward(z_pen)

                n_atoms = target_dist.cartesian_dim // 3
                n_solvent = int(getattr(target_dist, "n_solvent"))
                n_solute = n_atoms - n_solvent

                oo_pen, sw_pen = self.lj_overlap_penalty(
                    x_pen,
                    L=float(target_dist.box_length_nm),
                    n_solute=n_solute,
                    n_solvent=n_solvent,
                    r0_ssolv=self.dist_ssolv,
                    r0_solv_solute=self.dist_solute,
                    k=100.0,
                )
                loss = loss + overlap_w * (oo_pen + sw_pen)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]

                if len(grads) > 0:
                    old_grad_norm = torch.cat(grads).norm().clone()
                else:
                    old_grad_norm = torch.tensor(0.0, device=self.flow_device)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                    global_step += 1

                    if self.warmup_scheduler is not None and global_step <= self.warmup_iters:
                        self.warmup_scheduler.step()
                    else:
                        if self.optim_scheduler is not None and (global_step % self.lr_step == 0):
                            self.optim_scheduler.step()
                else:
                    warnings.warn("Encountered inf grad norm!")
            else:
                warnings.warn("NaN loss encountered! No update performed.")
                old_grad_norm = torch.tensor(0.0, device=self.flow_device)
                grad_norm = torch.tensor(0.0, device=self.flow_device)

            self.optimizer.zero_grad()

            info = self.model.get_iter_info()
            info.update(
                {
                    "loss": loss.detach().cpu().item(),
                    "old_grad_norm": old_grad_norm.detach().cpu().item(),
                    "grad_norm": grad_norm.detach().cpu().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "iteration": i,
                    "epoch": epoch,
                }
            )

            if flow_loss is not None:
                info["flow_loss_only"] = flow_loss.detach().cpu().item()
            if transform_loss is not None:
                info["transform_loss_only"] = transform_loss.detach().cpu().item()
            if reverse_loss is not None:
                info["reverse_loss_only"] = reverse_loss.detach().cpu().item()
            if md_loss is not None:
                info["md_loss_only"] = md_loss.detach().cpu().item()

            # generic batch metrics for forward/data branches
            if mean_log_q_flow is not None:
                info["mean_log_q_flow"] = mean_log_q_flow.detach().cpu().item()
            if mean_logdet_transform is not None:
                info["mean_logdet_transform"] = mean_logdet_transform.detach().cpu().item()
            if mean_log_q_total is not None:
                info["mean_log_q_total"] = mean_log_q_total.detach().cpu().item()
            if neg_mean_log_q_total is not None:
                info["neg_mean_log_q_total"] = neg_mean_log_q_total.detach().cpu().item()
            if mean_log_p is not None:
                info["mean_log_p"] = mean_log_p.detach().cpu().item()
            if neg_mean_log_p is not None:
                info["neg_mean_log_p"] = neg_mean_log_p.detach().cpu().item()
            if bad_frac_batch is not None:
                info["bad_sample_fraction_batch"] = bad_frac_batch.detach().cpu().item()

            # reverse branch: MD-side diagnostics
            if mean_log_q_flow_md is not None:
                info["mean_log_q_flow_md"] = mean_log_q_flow_md.detach().cpu().item()
            if mean_logdet_transform_md is not None:
                info["mean_logdet_transform_md"] = mean_logdet_transform_md.detach().cpu().item()
            if mean_log_q_total_md is not None:
                info["mean_log_q_total_md"] = mean_log_q_total_md.detach().cpu().item()
            if mean_log_p_md is not None:
                info["mean_log_p_md"] = mean_log_p_md.detach().cpu().item()
            if bad_frac_md is not None:
                info["bad_sample_fraction_md"] = bad_frac_md.detach().cpu().item()

            # reverse branch: flow/model-side diagnostics
            if mean_log_q_flow_model is not None:
                info["mean_log_q_flow_model"] = mean_log_q_flow_model.detach().cpu().item()
            if mean_logdet_transform_model is not None:
                info["mean_logdet_transform_model"] = mean_logdet_transform_model.detach().cpu().item()
            if mean_log_q_total_model is not None:
                info["mean_log_q_total_model"] = mean_log_q_total_model.detach().cpu().item()
            if mean_log_p_model is not None:
                info["mean_log_p_model"] = mean_log_p_model.detach().cpu().item()
            if reverse_kl_model_est is not None:
                info["reverse_kl_model_est"] = reverse_kl_model_est.detach().cpu().item()
            if bad_frac_model is not None:
                info["bad_sample_fraction_model"] = bad_frac_model.detach().cpu().item()

            if overlap_w > 0.0 and oo_pen is not None and sw_pen is not None:
                info["solv_solv_pen"] = oo_pen.detach().cpu().item()
                info["solv_solute_pen"] = sw_pen.detach().cpu().item()

            self.logger.write(info)

            if i % 10 == 0:
                print(f"   Iter {i}, Train loss: {loss.detach().cpu().item():.4f}")

            if n_eval is not None and i in eval_iter:
                self.perform_eval(i, eval_batch_size, batch_size)

            if n_plot is not None and i in plot_iter:
                self.make_and_save_plots(i, save)

            if n_checkpoints is not None and i in checkpoint_iter:
                self.save_checkpoint(i)

            max_it_time = max(max_it_time, time() - it_start_time)

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

        self.save_flow_samples_h5(n_samples=1000, filename="flow_samples.h5", batch_size=256)
        self.save_flow_samples_pdb(n_samples=1000, filename="flow_samples.pdb", batch_size=256)

        print(f"\nRun completed in {(time() - start_time) / 3600:.2f} hours\n")
        if tlimit is not None:
            print(f"Run finished before timelimit of {tlimit:.2f} hours was reached.\n")

        self.logger.close()

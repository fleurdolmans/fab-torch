import os
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

from experiments.logger_setup import setup_logger
from experiments.setup_run import setup_trainer_and_run_flow, Plotter
from fab import FABModel
from fab.target_distributions.h2o_in_h2o import H2OinH2O

# Removed params:
# dim = cfg.target.dim,
# n_mixes = cfg.target.n_mixes,
# loc_scaling = cfg.target.loc_scaling,
# log_var_scaling = cfg.target.log_var_scaling,
# use_gpu = cfg.training.use_gpu,


def setup_h2o_plotter(cfg: DictConfig, target: H2OinH2O, buffer=None) -> Plotter:
    def plot(fab_model: FABModel, plot_only_md_energies: bool) -> List[plt.Figure]:
        figs = []
        # Plot energies of the MD data as a sanity check if desired.
        if plot_only_md_energies:
            if target.eval_mode == "val":
                target_data_i = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
            elif target.eval_mode == "test":
                target_data_i = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)

            prob, jac = target.log_prob_and_jac(target_data_i)
            energy = -1 * (prob - jac).cpu()
            energy_in_kJ_per_mol = energy * 0.008314 * target.temperature  # R = 8.314 J/(mol K)
            fig = plt.figure(figsize=(8, 5))
            plt.plot(list(range(len(target_data_i))), energy_in_kJ_per_mol)
            plt.xlabel("MD sample index")
            plt.ylabel(f"Boltzmann energy (kJ/mol)")
            plt.ylim(min(energy_in_kJ_per_mol) * 1.05, 0)
            figs.append(fig)

        else:
            # Check some integrals (sums over grids) of certain feature dimensions.
            dim_labels = ["H11", "H12", "H12"]
            for dim in range(3, target.internal_dim, 9):
                dim_labels.extend(3 * [f"O{dim // 9 + 2}"] + 3 * [f"H{dim // 9 + 2}1"] + 3 * [f"H{dim // 9 + 2}2"])

            num_molecules_to_plot = min(2, target.cartesian_dim // 9)
            num_first_molecule_dims = 3
            remaining_dims = (num_molecules_to_plot - 1) * 3 * 3
            total_dims = num_first_molecule_dims + remaining_dims
            assert len(dim_labels) >= total_dims, "Not enough atom labels given for number of dimensions."
            r_dims = [0, 1] + [i for i in range(3, target.cartesian_dim, 3)]
            phi_dims = [2] + [i for i in range(4, target.cartesian_dim, 3)]
            theta_dims = [i for i in range(5, target.cartesian_dim, 3)]

            grid_resolution = 100
            # Flow output ranges to plot for
            # fr is in R in principle (positive after softplus, but generally r values are smaller than 1nm,
            #  so fr < 1 is sufficient scale.
            # fphi and ftheta are in [-pi, pi] (see PeriodicWrap in make_wrapped_normflow_solvent_flow() with
            #  bound_circ = np.pi). Note that we want phi in [0, 2pi] and theta in [-pi/2, pi/2].
            min_fr, max_fr = -3, -0.5
            min_fphi, max_fphi = -np.pi, np.pi
            min_ftheta, max_ftheta = -np.pi, np.pi

            # Figure setup
            ncols = 3
            nrows = total_dims // ncols if total_dims % ncols == 0 else total_dims // ncols + 1

            if target.eval_mode == "val":  # TODO: batch this?
                target_data_i = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
            elif target.eval_mode == "test":
                target_data_i = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)

            # Use first MD data point as anchor for the grid
            data_point = target_data_i[0]
            fig = plt.figure(figsize=(13, 4 * nrows))
            # Plot each dimension
            for dim in range(total_dims):
                plt.subplot(nrows, ncols, dim + 1)
                md_f_val = data_point[dim].cpu()  # MD value of feature in flow-output space
                if dim in r_dims:
                    f_vals = torch.linspace(min_fr, max_fr, grid_resolution)
                    i_vals = F.softplus(f_vals)
                    md_val = F.softplus(md_f_val)  # Transformation to r coordinate from flow output
                    label, unit = "r", "nm"
                    plt.title(f"r({dim_labels[dim]}), r_MD = {md_val:.4f}nm")
                elif dim in phi_dims:
                    # 1e-6 to prevent getting exactly 0, which tends to give nans.
                    f_vals = torch.linspace(min_fphi + 1e-6, max_fphi, grid_resolution)
                    i_vals = f_vals + np.pi
                    md_val = md_f_val + np.pi  # Transformation to phi coordinate from flow output
                    label, unit = "phi", "rad"
                    plt.title(f"phi({dim_labels[dim]}), phi_MD = {md_val:.4f}rad")
                elif dim in theta_dims:
                    # 1e-6 to prevent getting exactly 0, which tends to give nans.
                    f_vals = torch.linspace(min_ftheta + 1e-6, max_ftheta, grid_resolution)
                    theta_scale = 1.0 / 2.0
                    i_vals = f_vals * theta_scale
                    md_val = md_f_val * theta_scale  # Transformation to theta coordinate from flow output
                    label, unit = "theta", "rad"
                    plt.title(f"theta({dim_labels[dim]}), theta_MD = {md_val:.4f}rad")
                else:
                    raise ValueError(f"Unexpected dim index {dim}.")
                # Axes labels
                if dim % ncols == 0:
                    plt.ylabel("probability")
                if dim >= total_dims - ncols:
                    plt.xlabel(f"{label} ({unit})")
                # Repeat data point and replace the dimension we want to plot with the value grid.
                data = data_point.clone().repeat(grid_resolution, 1)
                data[:, dim] = f_vals

                with torch.no_grad():
                    # Get prob of grid under flow
                    # Add logdetjac to compensate for the fact that we are plotting as a function of r, phi
                    #  and theta, but computing logprobs starting from the flow output space. Essentially, we
                    #  are plotting prob densities, rather than prob volumes of the grid-patches, and this
                    #  means we need to compensate for the fact that the grid patches are not of equal size
                    #  under our transforms from flow output to (true) internal coordinates.
                    flow_sliceprobs = torch.exp(fab_model.flow.log_prob(data))  # TODO: See below for potential issue.
                    flow_probs = flow_sliceprobs.cpu() / flow_sliceprobs.cpu().sum()
                    # Get prob of grid under Boltzmann
                    # TODO: We are plotting a grid, but we are evaluating a continuous logprob. This means that
                    #  naively plotting would lead to us plotting densities evaluated at a point within a grid
                    #  patch, rather than the probability volume of the grid patch (we would need to integrate
                    #  the logprob over the whole grid patch for that). To compensate, we need to subtract the
                    #  logdetjac from the energies again to visualise correctly. This does mean that the
                    #  energy values we visualise here and the actual energies seen by the flow are offset,
                    #  since the flow deals with densities, whereas we are visualising probability volumes here.
                    #  Not sure how to deal with this properly for the flow logprobs though... Do we need to
                    #  also remove the whole logdetjac to the base probability?
                    boltzmann_logprobs, logdetjac = target.log_prob_and_jac(data)
                    boltzmann_sliceprobs = torch.exp(target.log_prob(data) - logdetjac)
                    boltzmann_probs = boltzmann_sliceprobs.cpu() / boltzmann_sliceprobs.cpu().sum()
                plt.plot(i_vals, flow_probs, label="flow")
                plt.plot(i_vals, boltzmann_probs, label="boltzmann")
                ymax = max(flow_probs.max().item(), boltzmann_probs.max().item())
                plt.vlines(md_val, 0, ymax, label=f"MD {label}", color="k", linestyle="--")
                plt.legend()
            plt.tight_layout()
            figs.append(fig)

            # Plot marginals hists of actual MD data and flow generated data in internal coordinate space.
            num_flow_samples = 1000
            with torch.no_grad():
                flow_samples, flow_logprob = fab_model.flow.sample_and_log_prob((num_flow_samples,))
            flow_samples_kl = flow_samples.cpu().clone().numpy()
            num_md_samples_to_compute = 1000
            target_data_kl = target_data_i.cpu().clone().numpy()
            if len(target_data_kl) < num_md_samples_to_compute:
                sampled_target_data = target_data_kl
            else:
                perm = np.random.permutation(num_md_samples_to_compute)
                perm_target_data_kl = target_data_kl[perm]
                sampled_target_data = perm_target_data_kl[:num_md_samples_to_compute, :]

            # Figure setup
            ncols = 6
            nrows = (
                target.internal_dim // ncols if target.internal_dim % ncols == 0 else target.internal_dim // ncols + 1
            )
            nbins = 50
            hist_range = [-5, 5]
            fig = plt.figure(figsize=(3 * ncols, 4 * nrows))
            for dim in range(target.internal_dim):
                plt.subplot(nrows, ncols, dim + 1)
                plt.hist(
                    sampled_target_data[:, dim], bins=nbins, range=hist_range, density=True, label="MD data", alpha=0.4
                )
                plt.hist(
                    flow_samples_kl[:, dim], bins=nbins, range=hist_range, density=True, label="Flow samples", alpha=0.4
                         )
                if dim in r_dims:
                    label, unit = "r", "in flow units"
                    plt.title(f"r({dim_labels[dim]})")
                elif dim in phi_dims:
                    label, unit = "phi", "in flow units"
                    plt.title(f"phi({dim_labels[dim]})")
                elif dim in theta_dims:
                    label, unit = "theta", "in flow units"
                    plt.title(f"theta({dim_labels[dim]})")
                else:
                    raise ValueError(f"Unexpected dim index {dim}.")
                # Axes labels
                if dim % ncols == 0:
                    plt.ylabel("relative frequency")
                if dim >= total_dims - ncols:
                    plt.xlabel(f"{label} ({unit})")
                plt.legend()

            plt.tight_layout()
            figs.append(fig)
        return figs
    return plot


def _run(cfg: DictConfig) -> None:
    # Seeds
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)  # seed of 0 for setup.

    # Gets output dir that Hydra created.
    base_save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_dir = os.path.join(base_save_dir, str(datetime.now().isoformat()))
    # Setup logger
    logger = setup_logger(cfg, save_dir)
    # If using Wandb, use its save path. Typically of form: HYDRA_RUN_DIR/wandb/run-YYYMMDD_HHMMSS-RUN_ID/files/
    # This is a result of Wandb working on top of the Hydra output directory.
    if hasattr(cfg.logger, "wandb"):
        import wandb
        save_dir = wandb.run.dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Example of structure of wandb/run-YYYMMDD_HHMMSS-RUN_ID/files/ dir:
    # - config.yaml: Contains Hydra config.
    # - config.txt: Contains Hydra config in plaintext.
    # - output.log: Contains stdout of run.
    # - wandb-summary.json: JSON file containing logged metrics.
    # - wandb-metadata.json: JSON file containing metadata about the run.
    # - requirements.txt: Plaintext file of pip packages used.
    # - media: Directory containing any media files logged to Wandb, such as images.
    # - plots: Directory containing any plots saved on disk (typically not used when already sending images to Wandb).
    # - metrics: Directory containing any metrics saved on disk.
    # - model_checkpoints: Directory containing any model checkpoints saved on disk.

    # Target distribution setup
    if cfg.target.solute == "water" and cfg.target.solvent == "water":
        assert cfg.target.dim % 3 == 0, "Dimension must be divisible by 3 for water in water."
        target = H2OinH2O(
            solvent_pdb_path=cfg.target.solvent_pdb_path,
            dim=cfg.target.dim,
            temperature=cfg.target.temperature,
            energy_cut=cfg.target.energy_cut,
            energy_max=cfg.target.energy_max,
            n_threads=cfg.target.n_threads,
            device="cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu",
            train_samples_path=cfg.target.train_samples_path,
            val_samples_path=cfg.target.val_samples_path,
            test_samples_path=cfg.target.test_samples_path,
            eval_mode=cfg.evaluation.eval_mode,
            logger=logger,
            save_dir=save_dir,
            plot_MD_energies=cfg.evaluation.plot_MD_energies,
            external_constraints=cfg.target.external_constraints,
            internal_constraints=cfg.target.internal_constraints,
            rigidwater=cfg.target.rigidwater,
        )
    else:
        raise NotImplementedError("Solute/solvent combination not implemented.")
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    # Setup model and start training
    # Will grab logger and save_dir from target, if target has those defined.
    setup_trainer_and_run_flow(cfg, setup_h2o_plotter, target)


@hydra.main(config_path="./config/", config_name="h2oinh2o_fab_pbuff.yaml", version_base="1.1")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _run(cfg)


if __name__ == "__main__":
    run()

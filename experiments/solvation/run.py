import os
import json
import pathlib
import hydra
import warnings
from omegaconf import DictConfig, OmegaConf
from typing import List

import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fab import FABModel
from fab.target_distributions.solute_in_water import SoluteInWater
from experiments.logger_setup import setup_logger
from experiments.setup_run import setup_trainer_and_run_flow, Plotter

SAVE_DIR = None

def setup_triatomic_in_h2o_plotter(cfg: DictConfig, target: SoluteInWater, buffer=None) -> Plotter:
    def plot(fab_model: FABModel, plot_dict: dict) -> List[plt.Figure]:
        figs = []
        R, T = 8.314e-3, target.temperature

        if target.eval_mode == "val":
            target_data_i = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
        elif target.eval_mode == "test":
            target_data_i = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)

        # Plot energies of the MD data as a sanity check if desired.
        if plot_dict["plot_md_energies"]:
            prob, jac = target.log_prob_and_jac(target_data_i)
            energy = -1 * (prob - jac).cpu()
            energy_in_kJ_per_mol = energy * R * target.temperature  # R = 8.314 J/(mol K)
            fig = plt.figure(figsize=(8, 5))
            plt.plot(list(range(len(target_data_i))), energy_in_kJ_per_mol)
            plt.xlabel("MD sample index")
            plt.ylabel(f"Boltzmann energy (kJ/mol)")
            plt.ylim(min(energy_in_kJ_per_mol) * 1.05, 0)
            figs.append(fig)

        if plot_dict["plot_marginal_hists"]:
            # Plot marginals hists of actual MD data and flow generated data in internal coordinate space.
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

        # RDF of flow samples vs MD samples
        num_flow_samples = 10000
        # num_flow_samples = 1000
        # num_flow_samples = 100
        with torch.no_grad():
            flow_samples = fab_model.flow.sample((num_flow_samples,))
        # Distance between primary solute atom and solvent water oxygens.
        # Assumes triatomic solute and water solvent.
        flow_samples_r_oxygen = F.softplus(flow_samples[:, 3::9]).flatten().cpu().numpy()
        md_samples_r_oxygen = F.softplus(target_data_i[:, 3::9]).flatten().cpu().numpy()
        # Potential energy evaluation of flow samples vs MD samples.
        # To obtain energy of Cartesian system: subtract log det jacobian from logprob.
        flow_samples_boltz_logprob, flow_jac = target.p.log_prob_and_jac(flow_samples)
        flow_samples_energy = -1 * (flow_samples_boltz_logprob - flow_jac).detach().cpu().numpy() * R * T
        md_samples_boltz_logprob, md_jac = target.p.log_prob_and_jac(target_data_i)
        md_samples_energy = -1 * (md_samples_boltz_logprob - md_jac).detach().cpu().numpy() * R * T

        # TODO: There seems to be one sample in every flow generation that has 2.5e8 kJ/mol energy.
        #  Both for 8mol water and 3mol so2.
        #  This is likely a numerical issue? Fix this.
        #  Actually, for 3mol so2 this crazy peak moves towards 1e8 and then 3-8e6 after about 800-1000 iterations.
        # md_samples_boltz_logprob[:3], md_jac[:3], md_samples_energy[:3], R*T, flow_samples_boltz_logprob[:3], flow_jac[:3], flow_samples_energy[:3]
        fig = plt.figure(figsize=(17, 5))
        plt.subplot(1, 3, 1)
        hist_range = (
            min(min(flow_samples_r_oxygen), min(md_samples_r_oxygen)),
            max(max(flow_samples_r_oxygen), max(md_samples_r_oxygen))
        )  # nm
        nbins = 101
        plt.hist(flow_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="Flow RDF", alpha=0.4)
        plt.hist(md_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="MD RDF", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("r (nm)")
        plt.title("RDF of flow samples vs MD samples")
        plt.legend()

        plt.subplot(1, 3, 2)
        hist_range = (min(md_samples_energy) - 1, max(md_samples_energy) + 1)  # kJ / mol
        plt.hist(flow_samples_energy, bins=nbins, range=hist_range, density=True, label="Flow energy", alpha=0.4)
        plt.hist(md_samples_energy, bins=nbins, range=hist_range, density=True, label="MD energy", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("energy (kJ/mol)")
        plt.title("Potential energy of flow samples vs MD samples (truncated)")
        plt.legend()

        plt.subplot(1, 3, 3)
        hist_range = (
            min(min(flow_samples_energy), min(md_samples_energy)),
            max(max(flow_samples_energy), max(md_samples_energy))
        )  # kJ / mol
        nbins = 101
        plt.hist(flow_samples_energy, bins=nbins, range=hist_range, density=True, label="Flow energy", alpha=0.4)
        plt.hist(md_samples_energy, bins=nbins, range=hist_range, density=True, label="MD energy", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("energy (kJ/mol)")
        plt.title("Potential energy of flow samples vs MD samples (full)")
        plt.legend()

        plt.tight_layout()
        figs.append(fig)

        # Potential alternatives: plot with MDtraj+NGLView or RDKit.
        # from simtk.openmm import app
        # from rdkit import Chem
        # from rdkit.Chem import Draw
        # import nglview
        # import mdtraj
        # Setup tmp folder for saving temporary pdb files.
        # tmp_folder = pathlib.Path(SAVE_DIR) / "tmp"
        # tmp_folder.mkdir(parents=True, exist_ok=True)
        # tmp_file = tmp_folder / f"tmp{i}.pdb"
        # app.PDBFile.writeFile(target.system.topology, pos.reshape(-1, 3), open(tmp_file, 'w'))
        # Using MDtraj + NGLView to visualise the high energy states.
        # traj = mdtraj.load(str(tmp_file))
        # view = nglview.show_mdtraj(traj)
        # view.add_representation("ball+stick")
        # figs.append(view)
        # Using RDKit
        # molecule = Chem.MolFromPDBFile(tmp_file, removeHs=False)
        # img = Draw.MolToImage(molecule)
        # figs.append(img)
        sorted_energy = flow_samples_energy.argsort()
        high_energy_inds = sorted_energy[-2:]
        low_energy_inds = sorted_energy[:2]
        with torch.no_grad():
            flow_samples_cartesian = target.coordinate_transform.forward(flow_samples)[0].cpu().numpy()
        high_positions = flow_samples_cartesian[high_energy_inds, ...]
        low_positions = flow_samples_cartesian[low_energy_inds, ...]
        high_energies = flow_samples_energy[high_energy_inds]
        low_energies = flow_samples_energy[low_energy_inds]
        md_inds = [0, -1]
        with torch.no_grad():
            md_cartesian = target.coordinate_transform.forward(target_data_i)[0].cpu().numpy()
        md_positions = md_cartesian[md_inds]
        md_energies = md_samples_energy[md_inds]

        fig = plt.figure(figsize=(12, 10))
        # Setting up the color map based on atom type
        colours = {'S': 'blue', 'O': 'red', 'H': 'black'}  # Define more colors if you have more atom types
        atom_types = [a[:1] for a in target.system.atoms]  # Take indices off the atom names
        color_list = [colours[atype] for atype in atom_types]

        dim = md_cartesian.shape[-1]
        md_reshaped = md_cartesian.reshape(-1, dim // 3, 3)

        x_lim = (md_reshaped[:, :, 0].min(), md_reshaped[:, :, 0].max())
        y_lim = (md_reshaped[:, :, 1].min(), md_reshaped[:, :, 1].max())
        z_lim = (md_reshaped[:, :, 2].min(), md_reshaped[:, :, 2].max())
        lim = (
            np.floor(min(x_lim[0], y_lim[0], z_lim[0]) * 10) / 10.0,
            np.ceil(max(x_lim[1], y_lim[1], z_lim[1]) * 10) / 10.0
        )

        def subplot_molecular_system(ax, pos, energy, title_str):
            # print("H2 coords:", pos[2, :])
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color_list, label=atom_types)
            for i in range(0, len(pos) - 2, 3):  # Draw bond lines
                if i + 2 < len(pos):  # Ensure we don't go out of bounds
                    # Draw line from atom i to i+1
                    ax.plot([pos[i][0], pos[i + 1][0]],
                            [pos[i][1], pos[i + 1][1]],
                            [pos[i][2], pos[i + 1][2]], color='grey')
                    # Draw line from atom i to i+2
                    ax.plot([pos[i][0], pos[i + 2][0]],
                            [pos[i][1], pos[i + 2][1]],
                            [pos[i][2], pos[i + 2][2]], color='grey')

            # Draw the coordinate planes for debugging
            # plane_edges = [-0.4, 0.4]  # Just needs to be bigger than the xyz plot limits.
            # xx, yy = np.meshgrid(plane_edges, plane_edges)
            # ax.plot_surface(xx * 0, yy, yy, alpha=0.2, color="blue")
            # ax.plot_surface(xx, yy * 0, yy, alpha=0.2, color="blue")
            # ax.plot_surface(xx, yy, yy * 0, alpha=0.2, color="blue")
            # Adding labels
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_zlabel('z (nm)')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)
            ax.set_title(f"{title_str}: {energy:.3g} kJ/mol")
            ax.view_init(elev=30, azim=45)  # Rotate 90 degrees around the z-axis
            legend_elements = [
                Patch(facecolor=colours[atype], edgecolor=colours[atype], label=atype)
                for atype in colours if atype in atom_types
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax = fig.add_subplot(2, 3, 1, projection='3d')
        subplot_molecular_system(ax, md_positions[0].reshape(-1, 3), md_energies[0], "First MD frame")
        ax = fig.add_subplot(2, 3, 2, projection='3d')
        subplot_molecular_system(ax, low_positions[0].reshape(-1, 3), low_energies[0], "Lowest energy")
        ax = fig.add_subplot(2, 3, 3, projection='3d')
        subplot_molecular_system(ax, high_positions[0].reshape(-1, 3), high_energies[-1], "Highest energy")
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        subplot_molecular_system(ax, md_positions[1].reshape(-1, 3), md_energies[1], "Last MD frame")
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        subplot_molecular_system(ax, low_positions[1].reshape(-1, 3), low_energies[1], "Second lowest energy")
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        subplot_molecular_system(ax, high_positions[1].reshape(-1, 3), high_energies[0], "Second highest energy")
        plt.tight_layout()
        figs.append(fig)

        return figs
    return plot


def _run(cfg: DictConfig) -> None:
    # Seeds
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)  # seed of 0 for setup.

    # Gets output dir that Hydra created: defined in hydra.run.dir in the config.
    global SAVE_DIR  # Necessary for plotting functions to save to the correct directory.
    SAVE_DIR = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    # Save config dict to output dir as yaml and json.
    with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    # Example structure of hydra.run.dir:
    # - plots: Directory containing any plots saved on disk (typically not used when already sending images to Wandb).
    # - metrics: Directory containing any metrics saved on disk.
    # - model_checkpoints: Directory containing any model checkpoints saved on disk.
    # - wandb: Wandb logging files, see below.

    # Setup logger: do that here, so the logger can be passed to the target distribution in case it needs to log stuff.
    logger = setup_logger(cfg, SAVE_DIR)
    # Example of structure of wandb/run-YYYMMDD_HHMMSS-RUN_ID/files/ dir inside the hydra.run.dir.
    # - config.yaml: Contains Hydra config.
    # - config.txt: Contains Hydra config in plaintext.
    # - output.log: Contains stdout of run.
    # - wandb-summary.json: JSON file containing logged metrics.
    # - wandb-metadata.json: JSON file containing metadata about the run.
    # - requirements.txt: Plaintext file of pip packages used.
    # - media: Directory containing any media files logged to Wandb, such as images.

    # Target distribution setup
    if cfg.target.solvent == "water":
        target = SoluteInWater(
            solute_pdb_path=cfg.target.solute_pdb_path,
            solute_xml_path=cfg.target.solute_xml_path,
            solute_inpcrd_path=cfg.target.solute_inpcrd_path,
            solute_prmtop_path=cfg.target.solute_prmtop_path,
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
            save_dir=SAVE_DIR,
            plot_MD_energies=cfg.evaluation.plot_MD_energies,
            plot_marginal_hists=cfg.evaluation.plot_marginal_hists,
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
    # TODO: Plotter assumes a triatomic solute in water solvent. Also that the transformation from flow output coords to
    #  internal coordinates is the same as in the target distribution (e.g., for r it's just a softplus).
    setup_trainer_and_run_flow(cfg, setup_triatomic_in_h2o_plotter, target)

# Run with hydra configuration.
@hydra.main(config_path="./config/", config_name="h2oinh2o_forwardkl.yaml", version_base="1.1")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _run(cfg)


if __name__ == "__main__":
    run()

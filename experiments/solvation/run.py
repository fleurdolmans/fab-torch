import os
import json
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List
from openmm import unit

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
from experiments.solvation.test_run import run_transform_test_droplet, run_test_pbc
from fab.transforms.transform_pbc import PBCPreprocessTransform
from matplotlib.patches import Patch

SAVE_DIR = None


def setup_triatomic_in_h2o_plotter(cfg: DictConfig, target: SoluteInWater, buffer=None) -> Plotter:
    
    def plot_droplet(fab_model: FABModel, plot_dict: dict) -> List[plt.Figure]:
        figs = []
        R, T = 8.314e-3, target.temperature

        print("Loading target data for plotting...")

        if target.eval_mode == "val":
            target_data_i = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
        elif target.eval_mode == "test":
            target_data_i = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)
        
        print("Loaded")

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

        # RDF and energies of flow samples vs MD samples
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

        fig = plt.figure(figsize=(17, 10))
        plt.subplot(2, 2, 1)
        hist_range = (min(md_samples_r_oxygen), max(md_samples_r_oxygen))  # nm
        nbins = 101
        plt.hist(flow_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="Flow RDF", alpha=0.4)
        plt.hist(md_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="MD RDF", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("r (nm)")
        plt.title("RDF of flow samples vs MD samples (truncated)")
        plt.legend()

        plt.subplot(2, 2, 2)
        hist_range = (
            min(min(flow_samples_r_oxygen), min(md_samples_r_oxygen)),
            max(max(flow_samples_r_oxygen), max(md_samples_r_oxygen))
        )  # nm
        plt.hist(flow_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="Flow RDF", alpha=0.4)
        plt.hist(md_samples_r_oxygen, bins=nbins, range=hist_range, density=True, label="MD RDF", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("r (nm)")
        plt.title("RDF of flow samples vs MD samples (full)")
        plt.legend()

        plt.subplot(2, 2, 3)
        hist_range = (min(md_samples_energy), max(md_samples_energy))  # kJ / mol
        plt.hist(flow_samples_energy, bins=nbins, range=hist_range, density=True, label="Flow energy", alpha=0.4)
        plt.hist(md_samples_energy, bins=nbins, range=hist_range, density=True, label="MD energy", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("energy (kJ/mol)")
        plt.title("Potential energy of flow samples vs MD samples (truncated)")
        plt.legend()

        plt.subplot(2, 2, 4)
        hist_range = (
            min(min(flow_samples_energy), min(md_samples_energy)),
            max(max(flow_samples_energy), max(md_samples_energy))
        )  # kJ / mol
        plt.hist(flow_samples_energy, bins=nbins, range=hist_range, density=True, label="Flow energy", alpha=0.4)
        plt.hist(md_samples_energy, bins=nbins, range=hist_range, density=True, label="MD energy", alpha=0.4)
        plt.ylabel("density")
        plt.xlabel("energy (kJ/mol)")
        plt.title("Potential energy of flow samples vs MD samples (full)")
        plt.legend()

        plt.tight_layout()
        figs.append(fig)

        # Plot some of the molecular states in Cartesian space.
        # Plots MD samples, and lowest + highest energy states from the flow samples.
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

        # Get plot limits from MD data
        dim = md_cartesian.shape[-1]
        md_reshaped = md_cartesian.reshape(-1, dim // 3, 3)
        x_lim = (np.floor(md_reshaped[:, :, 0].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 0].max() * 10) / 10.0)
        y_lim = (np.floor(md_reshaped[:, :, 1].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 1].max() * 10) / 10.0)
        z_lim = (np.floor(md_reshaped[:, :, 2].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 2].max() * 10) / 10.0)

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

            # Adding labels
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.set_zlabel('z (nm)')
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
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

    def plot_pbc(fab_model: FABModel, plot_dict: dict) -> List[plt.Figure]:
        figs = []
        R, T = 8.314e-3, target.temperature
        L = float(target.box_length_nm)

        # triatomic solute + waters (O,H,H)
        n_solute = 3
        n_waters = int(target.num_solvent_molecules)

        kBT = R * T  # kJ/mol

        def wrap(x: torch.Tensor, L: float) -> torch.Tensor:
            return torch.remainder(x, float(L))

        # local MIC helper
        def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
            L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
            return dx - L_t * torch.round(dx / L_t)
        
        def dist_pbc(p, q, L):
            return torch.linalg.norm(mic(p - q, L), dim=-1)

        def angle(a, b, c, L):
            # angle ABC at B using MIC vectors BA and BC
            ba = mic(a - b, L)
            bc = mic(c - b, L)
            ba_n = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            bc_n = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosang = (ba_n * bc_n).sum(dim=-1).clamp(-1.0, 1.0)
            return torch.acos(cosang)

        # visualization helper: locally "unwrap" bonds for plotting and center solute atom0
        def make_whole_for_viz(Xp_flat: torch.Tensor, L: float, n_solute: int = 3, n_waters: int = 0) -> torch.Tensor:
            if Xp_flat.ndim != 2 or (Xp_flat.shape[1] % 3 != 0):
                raise ValueError(f"Expected Xp_flat (B,3N), got {tuple(Xp_flat.shape)}")

            device = Xp_flat.device
            dtype = Xp_flat.dtype
            L_t = torch.as_tensor(float(L), device=device, dtype=dtype)

            B, D = Xp_flat.shape
            N = D // 3
            X = Xp_flat.view(B, N, 3).clone()  # (B,N,3)

            def mic_local(dx: torch.Tensor) -> torch.Tensor:
                return dx - L_t * torch.round(dx / L_t)

            # unwrap solute relative to atom0
            a0 = X[:, 0, :]
            for a in range(1, min(n_solute, N)):
                X[:, a, :] = a0 + mic_local(X[:, a, :] - a0)

            # unwrap each water: H relative to its O
            start = n_solute
            for w in range(n_waters):
                i = start + 3 * w
                if i + 2 >= N:
                    break
                O = X[:, i + 0, :]
                H1 = X[:, i + 1, :]
                H2 = X[:, i + 2, :]
                X[:, i + 1, :] = O + mic_local(H1 - O)
                X[:, i + 2, :] = O + mic_local(H2 - O)

            # center solute atom0 at origin
            X = X - X[:, 0:1, :]

            return X  # (B,N,3)
        
        def min_dist_stats(Xp_flat: torch.Tensor, L: float, n_solute: int, n_waters: int):
            X = Xp_flat.view(Xp_flat.shape[0], -1, 3)

            solute = X[:, :n_solute, :]  # (B,3,3)

            O_idx  = [n_solute + 3*w for w in range(n_waters)]
            H1_idx = [n_solute + 3*w + 1 for w in range(n_waters)]
            H2_idx = [n_solute + 3*w + 2 for w in range(n_waters)]

            Owat = X[:, O_idx, :]
            Hwat = torch.cat([X[:, H1_idx, :], X[:, H2_idx, :]], dim=1)

            def pairwise_min(A, B):
                d = mic(A[:, :, None, :] - B[:, None, :, :], L)
                d = torch.linalg.norm(d, dim=-1)
                return d.amin(dim=(1, 2)).detach().cpu().numpy()

            # O-O among waters
            min_oo = min_mic_OO_distance(Xp_flat, L, n_solute, n_waters, chunk=64).detach().cpu().numpy()

            min_solO = pairwise_min(solute, Owat)
            min_solH = pairwise_min(solute, Hwat)

            return min_oo, min_solO, min_solH
        
        def energy_by_force(context, system, x_flat, L_nm=None):
            """
            Returns a list of (force_index, force_name, energy_kJmol).
            """
            x = x_flat.view(-1, 3).detach().cpu().numpy()
            context.setPositions(x * unit.nanometer)

            # Make sure box vectors are set (optional if already correct)
            if L_nm is not None:
                import openmm as mm
                context.setPeriodicBoxVectors(
                    mm.Vec3(L_nm,0,0)*unit.nanometer,
                    mm.Vec3(0,L_nm,0)*unit.nanometer,
                    mm.Vec3(0,0,L_nm)*unit.nanometer,
                )

            out = []
            for j, frc in enumerate(system.getForces()):
                st = context.getState(getEnergy=True, groups={j})
                Ej = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                out.append((j, frc.getName(), Ej))
            return out
        
        def water_orientation_angles(Xp_flat: torch.Tensor, L: float, n_solute: int, n_waters: int):
            X = Xp_flat.view(Xp_flat.shape[0], -1, 3)

            sol0 = X[:, 0, :]  # use first solute atom as reference

            O_idx  = [n_solute + 3*w for w in range(n_waters)]
            H1_idx = [n_solute + 3*w + 1 for w in range(n_waters)]
            H2_idx = [n_solute + 3*w + 2 for w in range(n_waters)]

            O  = X[:, O_idx, :]
            H1 = X[:, H1_idx, :]
            H2 = X[:, H2_idx, :]

            r = mic(O - sol0[:, None, :], L)                 # solute -> O
            b = mic((H1 + H2) / 2.0 - O, L)                  # O -> H-bisector

            r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            cosang = (r * b).sum(dim=-1).clamp(-1.0, 1.0)
            ang = torch.acos(cosang) * 180.0 / np.pi         # degrees
            return ang.detach().cpu().numpy().reshape(-1)
        
        def split_water_blocks(i: torch.Tensor, n_waters: int):
            """
            i: (B, 6 + 6*n_waters)
            Returns:
            O:     (B, n_waters, 3)
            omega: (B, n_waters, 3)
            """
            B = i.shape[0]
            water = i[:, 6:]                       # (B, 6*n_waters)
            water = water.view(B, n_waters, 6)     # (B, n_waters, 6)

            O = water[:, :, 0:3]
            omega = water[:, :, 3:6]
            return O, omega
        
        def norm_stats(name: str, x: torch.Tensor):
            """
            x: (B, n_waters, 3)
            """
            norms = torch.linalg.norm(x, dim=-1).reshape(-1).detach().cpu().numpy()
            print(
                f"[{name}] norm median={np.median(norms):.4f} "
                f"p10={np.percentile(norms, 10):.4f} "
                f"p90={np.percentile(norms, 90):.4f} "
                f"p99={np.percentile(norms, 99):.4f} "
                f"max={np.max(norms):.4f}"
            )
            return norms
        
        def min_pairwise_O_stats(name: str, O: torch.Tensor):
            """
            O: (B, n_waters, 3)
            Computes per-frame minimum pairwise O-O distance in internal O-space.
            """
            d = O[:, :, None, :] - O[:, None, :, :]             # (B,W,W,3)
            d = torch.linalg.norm(d, dim=-1)                    # (B,W,W)

            W = O.shape[1]
            eye = torch.eye(W, device=O.device, dtype=torch.bool)[None]
            d = d.masked_fill(eye, 1e9)

            min_per_frame = d.amin(dim=(1, 2)).detach().cpu().numpy()

            print(
                f"[{name}] internal O-O min median={np.median(min_per_frame):.4f} "
                f"p10={np.percentile(min_per_frame, 10):.4f} "
                f"p90={np.percentile(min_per_frame, 90):.4f} "
                f"min={np.min(min_per_frame):.4f}"
            )
            return min_per_frame

        # Load internal MD data
        if target.eval_mode == "val":
            target_data_i = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
        elif target.eval_mode == "test":
            target_data_i = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)
        else:
            raise ValueError(f"Unknown eval_mode: {target.eval_mode}")

        # Sample from flow in internal space
        num_flow_samples = 10000
        with torch.no_grad():
            flow_i = fab_model.flow.sample((1000,))  # (B, internal_dim)

            md_i = target_data_i[:1000].detach().cpu().numpy()

        print("MD internal mean/std:", md_i.mean().item(), md_i.std().item())
        print("FLOW internal mean/std:", flow_i.mean().item(), flow_i.std().item())


        with torch.no_grad():
            z = fab_model.flow.sample((64,))
            x, _ = target.coordinate_transform.forward(z)
            z_rec, _ = target.coordinate_transform.inverse(x)
            err = (z - z_rec).abs()
        print("max z roundtrip err", err.max().item())
        print("mean z roundtrip err", err.mean().item())

        # Map both MD-i and flow-i into Cartesian (wrapped to [0,L) by your transform)
        with torch.no_grad():
            flow_i = fab_model.flow.sample((num_flow_samples,))
            flowXp, _ = target.coordinate_transform.forward(flow_i)         # (B, 3N)
            print("flowXp std", flowXp.std().item(), "min", flowXp.min().item(), "max", flowXp.max().item())
            print("max pairwise diff in batch",
                (flowXp[:8] - flowXp[0:1]).abs().max().item())
            mdXp, _ = target.coordinate_transform.forward(target_data_i)    # (B, 3N)

            flowXp = wrap(flowXp, L)
            mdXp   = wrap(mdXp, L)

        n_diag_i = min(64, target_data_i.shape[0], flow_i.shape[0])
        md_i_diag = target_data_i[:n_diag_i]
        flow_i_diag = flow_i[:n_diag_i]

        md_O, md_omega = split_water_blocks(md_i_diag, n_waters)
        fl_O, fl_omega = split_water_blocks(flow_i_diag, n_waters)

        md_O_norms = norm_stats("MD O", md_O)
        fl_O_norms = norm_stats("FLOW O", fl_O)

        md_omega_norms = norm_stats("MD omega", md_omega)
        fl_omega_norms = norm_stats("FLOW omega", fl_omega)

        md_OO_internal = min_pairwise_O_stats("MD", md_O)
        fl_OO_internal = min_pairwise_O_stats("FLOW", fl_O)

        fig = plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(md_O_norms, bins=60, density=True, alpha=0.4, label="MD")
        plt.hist(fl_O_norms, bins=60, density=True, alpha=0.4, label="Flow")
        plt.xlabel("||O||")
        plt.ylabel("density")
        plt.title("Water O-position norm")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.hist(md_omega_norms, bins=60, density=True, alpha=0.4, label="MD")
        plt.hist(fl_omega_norms, bins=60, density=True, alpha=0.4, label="Flow")
        plt.xlabel("||omega||")
        plt.title("Water rotation-vector norm")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.hist(md_OO_internal, bins=60, density=True, alpha=0.4, label="MD")
        plt.hist(fl_OO_internal, bins=60, density=True, alpha=0.4, label="Flow")
        plt.xlabel("min pairwise O-O in internal space")
        plt.title("Closest O-O in O-space")
        plt.legend()

        plt.tight_layout()
        figs.append(fig)

        n_diag = min(64, mdXp.shape[0], flowXp.shape[0])
        mdXp_diag = mdXp[:n_diag]
        flowXp_diag = flowXp[:n_diag]
        
        md_min_oo, md_min_solO, md_min_solH = min_dist_stats(mdXp_diag, L, n_solute, n_waters)
        fl_min_oo, fl_min_solO, fl_min_solH = min_dist_stats(flowXp_diag, L, n_solute, n_waters)

        md_ang = water_orientation_angles(mdXp_diag, L, n_solute, n_waters)
        fl_ang = water_orientation_angles(flowXp_diag, L, n_solute, n_waters)

        print("[MD distances]  min O-O median", np.median(md_min_oo),
            "min solute-O median", np.median(md_min_solO),
            "min solute-H median", np.median(md_min_solH))

        print("[FLOW distances] min O-O median", np.median(fl_min_oo),
            "min solute-O median", np.median(fl_min_solO),
            "min solute-H median", np.median(fl_min_solH))

        print("[MD orientation] median", np.median(md_ang), "p10", np.percentile(md_ang, 10), "p90", np.percentile(md_ang, 90))
        print("[FLOW orientation] median", np.median(fl_ang), "p10", np.percentile(fl_ang, 10), "p90", np.percentile(fl_ang, 90))

        fig = plt.figure(figsize=(8, 5))
        plt.hist(md_ang, bins=60, density=True, alpha=0.4, label="MD")
        plt.hist(fl_ang, bins=60, density=True, alpha=0.4, label="Flow")
        plt.xlabel("Angle(solute0→O, O→H-bisector) [deg]")
        plt.ylabel("density")
        plt.title("Water orientation relative to solute")
        plt.legend()
        plt.tight_layout()
        figs.append(fig)

        # ----------------------------
        # RDF: solute atom0 -> solvent oxygens (PBC MIC)
        # ----------------------------
        def rdf_solute0_to_solventO_pbc(Xp_flat: torch.Tensor, L: float, n_solute: int, n_waters: int, dr: float = 0.005):
            B, D = Xp_flat.shape
            N = D // 3
            X = Xp_flat.view(B, N, 3)
            sol0 = X[:, 0, :]  # (B,3)

            # oxygen indices: n_solute + 3*w, for w in [0, n_waters)
            Oxyz = X[:, n_solute : n_solute + 3 * n_waters : 3, :]  # (B, n_waters, 3)

            d = mic(Oxyz - sol0[:, None, :], L)
            r = torch.linalg.norm(d, dim=-1).reshape(-1).detach().cpu().numpy()

            r_max = 0.5 * L
            nbins = int(np.floor(r_max / dr))
            edges = np.linspace(0.0, nbins * dr, nbins + 1)
            counts, _ = np.histogram(r, bins=edges)

            r_centers = 0.5 * (edges[:-1] + edges[1:])
            shell_vol = 4.0 * np.pi * (r_centers**2) * dr
            V = L**3
            rho = n_waters / V
            expected = (B * n_waters) * rho * shell_vol

            g_r = counts / np.maximum(expected, 1e-12)
            g_r[0] = 0.0
            return r_centers, g_r

        r_md, g_md = rdf_solute0_to_solventO_pbc(mdXp, L, n_solute=n_solute, n_waters=n_waters, dr=0.005)
        r_fl, g_fl = rdf_solute0_to_solventO_pbc(flowXp, L, n_solute=n_solute, n_waters=n_waters, dr=0.005)

        # ----------------------------
        # Energies: compute from CARTESIAN density only
        # ----------------------------
        with torch.no_grad():
            md_U_kT = (-target.p.log_prob_x(mdXp)).detach().cpu().numpy()
            fl_U_kT = (-target.p.log_prob_x(flowXp)).detach().cpu().numpy()

        md_U_kJ = md_U_kT * kBT
        fl_U_kJ = fl_U_kT * kBT

        def summarize_energy(name, U_kT, U_kJ, energy_cut):
            print(
                f"[{name}] median(kT)={np.median(U_kT):.3f} "
                f"p90(kT)={np.percentile(U_kT, 90):.3f} "
                f"p99(kT)={np.percentile(U_kT, 99):.3f} "
                f"max(kT)={np.max(U_kT):.3f}"
            )
            print(
                f"[{name}] median(kJ/mol)={np.median(U_kJ):.3f} "
                f"p90(kJ/mol)={np.percentile(U_kJ, 90):.3f} "
                f"p99(kJ/mol)={np.percentile(U_kJ, 99):.3f} "
                f"max(kJ/mol)={np.max(U_kJ):.3f}"
            )
            print(f"[{name}] clip_frac={np.mean(U_kT >= energy_cut - 1e-6):.4f}")

        summarize_energy("MD", md_U_kT, md_U_kJ, target.energy_cut)
        summarize_energy("FLOW", fl_U_kT, fl_U_kJ, target.energy_cut)


        # Optional: see if you're hitting energy_cut a lot (very informative)
        if hasattr(target, "energy_cut"):
            cut = float(target.energy_cut)
            print("[clip frac] md", np.mean(md_U_kT >= cut - 1e-6), "flow", np.mean(fl_U_kT >= cut - 1e-6))

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

        # ----------------------------
        # 4-panel: RDF truncated/full + Energy truncated/full
        # ----------------------------
        fig = plt.figure(figsize=(17, 10))

        # RDF truncated (truncate y-axis)
        plt.subplot(2, 2, 1)
        plt.plot(r_md, g_md, label="MD", alpha=0.9)
        plt.plot(r_fl, g_fl, label="Flow", alpha=0.9)
        plt.xlim(0, L / 2)
        y_hi = max(np.percentile(g_md, 99.5), 1.0) * 1.2
        plt.ylim(0, y_hi)
        plt.xlabel("r (nm)")
        plt.ylabel("g(r)")
        plt.title("RDF (truncated y)")
        plt.legend()

        # RDF full
        plt.subplot(2, 2, 2)
        plt.plot(r_md, g_md, label="MD", alpha=0.9)
        plt.plot(r_fl, g_fl, label="Flow", alpha=0.9)
        plt.xlim(0, L / 2)
        plt.xlabel("r (nm)")
        plt.ylabel("g(r)")
        plt.title("RDF (full)")
        plt.legend()

        # Energy truncated to MD percentiles
        plt.subplot(2, 2, 3)
        nbins = 120
        e_lo, e_hi = np.percentile(md_U_kJ, [0.5, 99.5])
        plt.hist(md_U_kJ, bins=nbins, range=(e_lo, e_hi), density=True, alpha=0.4, label="MD")
        plt.hist(fl_U_kJ, bins=nbins, range=(e_lo, e_hi), density=True, alpha=0.4, label="Flow")
        plt.xlabel("Potential energy (kJ/mol)")
        plt.ylabel("density")
        plt.title("Energy (truncated to MD percentiles)")
        plt.legend()

        # Energy “full-ish” (robust): combined 0.1–99.9 percentiles
        plt.subplot(2, 2, 4)
        e2_lo, e2_hi = np.percentile(np.concatenate([md_U_kJ, fl_U_kJ]), [0.1, 99.9])
        plt.hist(md_U_kJ, bins=nbins, range=(e2_lo, e2_hi), density=True, alpha=0.4, label="MD")
        plt.hist(fl_U_kJ, bins=nbins, range=(e2_lo, e2_hi), density=True, alpha=0.4, label="Flow")
        plt.xlabel("Potential energy (kJ/mol)")
        plt.ylabel("density")
        plt.title("Energy (full, robust range)")
        plt.legend()

        plt.tight_layout()
        figs.append(fig)


        # ----------------------------
        # Visualize: MD + lowest/highest energy FLOW samples (by Cartesian energy)
        # ----------------------------
        # choose representative flow samples
        sorted_inds = np.argsort(fl_U_kJ)
        low_inds = sorted_inds[:3]
        mid_inds = sorted_inds[len(sorted_inds)//2 : len(sorted_inds)//2 + 3]
        high_inds = sorted_inds[-3:]

        for idx in low_inds:
            print("LOW FLOW", idx)
            print(energy_by_force(target.p.sim_context, target.system.system, flowXp[idx], L_nm=target.box_length_nm))

        for idx in high_inds:
            print("HIGH FLOW", idx)
            print(energy_by_force(target.p.sim_context, target.system.system, flowXp[idx], L_nm=target.box_length_nm))
        
        for idx in mid_inds:
            print("MID FLOW", idx)
            print(energy_by_force(target.p.sim_context, target.system.system, flowXp[idx], L_nm=target.box_length_nm))

        
        # reshape flattened Cartesian to (B, N, 3)
        mdX = mdXp_diag.view(mdXp_diag.shape[0], -1, 3)
        flowX = flowXp_diag.view(flowXp_diag.shape[0], -1, 3)

        # --- 4) Rigid water bond lengths + angle checks ---
        oh_md = []
        oh_flow = []
        hoh_md = []
        hoh_flow = []

        for k in range(target.num_solvent_molecules):
            O_idx = 3 + 3 * k
            H1_idx = O_idx + 1
            H2_idx = O_idx + 2

            # distances
            oh1_md = dist_pbc(mdX[:, H1_idx, :], mdX[:, O_idx, :], L)
            oh2_md = dist_pbc(mdX[:, H2_idx, :], mdX[:, O_idx, :], L)
            oh1_flow = dist_pbc(flowX[:, H1_idx, :], flowX[:, O_idx, :], L)
            oh2_flow = dist_pbc(flowX[:, H2_idx, :], flowX[:, O_idx, :], L)

            # angle H-O-H
            ang_md = angle(mdX[:, H1_idx, :], mdX[:, O_idx, :], mdX[:, H2_idx, :], L)
            ang_flow = angle(flowX[:, H1_idx, :], flowX[:, O_idx, :], flowX[:, H2_idx, :], L)

            oh_md.append(torch.stack([oh1_md, oh2_md], dim=1))
            oh_flow.append(torch.stack([oh1_flow, oh2_flow], dim=1))
            hoh_md.append(ang_md)
            hoh_flow.append(ang_flow)

        oh_md = torch.cat(oh_md, dim=0)
        oh_flow = torch.cat(oh_flow, dim=0)
        hoh_md = torch.cat(hoh_md, dim=0)
        hoh_flow = torch.cat(hoh_flow, dim=0)

        print("[BONDS] MD OH mean (nm):", oh_md.mean().item(), "std:", oh_md.std(unbiased=False).item(), "max:", oh_md.max().item())
        print("[BONDS] FLOW OH mean (nm):", oh_flow.mean().item(), "std:", oh_flow.std(unbiased=False).item(), "max:", oh_flow.max().item())
        print("[ANGLE] MD HOH mean (deg):", (hoh_md.mean() * 180 / np.pi).item())
        print("[ANGLE] FLOW HOH mean (deg):", (hoh_flow.mean() * 180 / np.pi).item())
        print("[ANGLE] max |ΔHOH| flow-vs-rigid? (deg):", ((hoh_flow - hoh_md.mean()).abs().max() * 180 / np.pi).item())

        # --- 5) Solute bond lengths + angle checks ---
        # first 3 atoms are solute: [S, O, O]
        S_idx = 0
        O1_idx = 1
        O2_idx = 2

        r1_md = dist_pbc(mdX[:, O1_idx, :], mdX[:, S_idx, :], L)
        r2_md = dist_pbc(mdX[:, O2_idx, :], mdX[:, S_idx, :], L)
        r1_flow = dist_pbc(flowX[:, O1_idx, :], flowX[:, S_idx, :], L)
        r2_flow = dist_pbc(flowX[:, O2_idx, :], flowX[:, S_idx, :], L)

        ang_sol_md = angle(mdX[:, O1_idx, :], mdX[:, S_idx, :], mdX[:, O2_idx, :], L)
        ang_sol_flow = angle(flowX[:, O1_idx, :], flowX[:, S_idx, :], flowX[:, O2_idx, :], L)

        print("[SOLUTE BONDS] MD S-O mean (nm):", torch.cat([r1_md, r2_md]).mean().item(),
            "std:", torch.cat([r1_md, r2_md]).std(unbiased=False).item(),
            "max:", torch.cat([r1_md, r2_md]).max().item())

        print("[SOLUTE BONDS] FLOW S-O mean (nm):", torch.cat([r1_flow, r2_flow]).mean().item(),
            "std:", torch.cat([r1_flow, r2_flow]).std(unbiased=False).item(),
            "max:", torch.cat([r1_flow, r2_flow]).max().item())

        print("[SOLUTE ANGLE] MD O-S-O mean (deg):", (ang_sol_md.mean() * 180 / np.pi).item())
        print("[SOLUTE ANGLE] FLOW O-S-O mean (deg):", (ang_sol_flow.mean() * 180 / np.pi).item())

        sorted_inds = np.argsort(fl_U_kJ)
        low_energy_inds = sorted_inds[:2]
        high_energy_inds = sorted_inds[-2:]

        high_positions = flowXp[high_energy_inds, ...]
        low_positions = flowXp[low_energy_inds, ...]
        high_energies = fl_U_kJ[high_energy_inds]
        low_energies = fl_U_kJ[low_energy_inds]

        md_inds = [0, -1]
        md_positions = mdXp[md_inds]
        md_energies = md_U_kJ[md_inds]

        with torch.no_grad():
            md_viz = make_whole_for_viz(md_positions, L, n_solute=n_solute, n_waters=n_waters).cpu().numpy()
            low_viz = make_whole_for_viz(low_positions, L, n_solute=n_solute, n_waters=n_waters).cpu().numpy()
            high_viz = make_whole_for_viz(high_positions, L, n_solute=n_solute, n_waters=n_waters).cpu().numpy()

        fig = plt.figure(figsize=(12, 10))
        colours = {'S': 'blue', 'O': 'red', 'H': 'black'}
        atom_types = [a[:1] for a in target.system.atoms]
        color_list = [colours.get(atype, 'grey') for atype in atom_types]

        md_reshaped = md_viz.reshape(-1, md_viz.shape[1], 3)
        x_lim = (np.floor(md_reshaped[:, :, 0].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 0].max() * 10) / 10.0)
        y_lim = (np.floor(md_reshaped[:, :, 1].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 1].max() * 10) / 10.0)
        z_lim = (np.floor(md_reshaped[:, :, 2].min() * 10) / 10.0, np.ceil(md_reshaped[:, :, 2].max() * 10) / 10.0)

        def subplot_molecular_system(ax, pos, energy, title_str):
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color_list)
            for i in range(0, len(pos) - 2, 3):
                if i + 2 < len(pos):
                    ax.plot([pos[i][0], pos[i + 1][0]], [pos[i][1], pos[i + 1][1]], [pos[i][2], pos[i + 1][2]], color='grey')
                    ax.plot([pos[i][0], pos[i + 2][0]], [pos[i][1], pos[i + 2][1]], [pos[i][2], pos[i + 2][2]], color='grey')

            ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)'); ax.set_zlabel('z (nm)')
            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
            ax.set_title(f"{title_str}: {energy:.3g} kJ/mol")
            ax.view_init(elev=30, azim=45)

            legend_elements = [
                Patch(facecolor=colours[atype], edgecolor=colours[atype], label=atype)
                for atype in colours if atype in atom_types
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax = fig.add_subplot(2, 3, 1, projection='3d')
        subplot_molecular_system(ax, md_viz[0], md_energies[0], "First MD frame")
        ax = fig.add_subplot(2, 3, 2, projection='3d')
        subplot_molecular_system(ax, low_viz[0], low_energies[0], "Lowest energy (flow)")
        ax = fig.add_subplot(2, 3, 3, projection='3d')
        subplot_molecular_system(ax, high_viz[0], high_energies[-1], "Highest energy (flow)")
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        subplot_molecular_system(ax, md_viz[1], md_energies[1], "Last MD frame")
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        subplot_molecular_system(ax, low_viz[1], low_energies[1], "2nd lowest energy (flow)")
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        subplot_molecular_system(ax, high_viz[1], high_energies[0], "2nd highest energy (flow)")
        plt.tight_layout()
        figs.append(fig)

        return figs

    def plot(fab_model: FABModel, plot_dict: dict) -> List[plt.Figure]:
        # return plot_droplet(fab_model, plot_dict)
        if cfg.target.boundary_condition == "droplet":
            return plot_droplet(fab_model, plot_dict)
        else:
            return plot_pbc(fab_model, plot_dict)

    return plot

def pick_system_keys(d: dict, keys: list) -> dict:
    """Keep only keys that define the physical system / target distribution."""
    if d is None:
        return None
    # adjust these to match your JSON structure
    # if your JSON is a full resolved hydra config, it'll likely have a "target" section
    tgt = d.get("target", d)  # support either nested or flat
    return {k: tgt.get(k) for k in keys if k in tgt}


def load_json(path):
    """Load the JSON file if it exists; otherwise return None."""
    # If no path provided, return None
    if not path:
        return None

    p = pathlib.Path(path).with_suffix(".json")
    # Check if the file exists before trying to load it
    if not p.exists():
        return None

    with p.open("r") as f:
        return json.load(f)

def overwrite_cfg(cfg, system_cfgs):
    """Overwrite hydra cfg.target with the system config from the MD data JSON, 
    if it exists, and check consistency if multiple JSONs exist. Only overwrite MD_KEYS."""
    if not system_cfgs:
        return cfg
    else:
        # Choose train as canonical if present, otherwise first available
        base_name, base_system = next(((n, s) for (n, s) in system_cfgs if n == "train"), system_cfgs[0])

        # Check all system configs match base
        mismatches = [(n, s) for (n, s) in system_cfgs if s != base_system]
        if mismatches:
            raise ValueError(f"Train/val/test system configs differ. Base={base_name}, mismatches={[n for n,_ in mismatches]}")

        # Overwrite hydra cfg.target with dataset physics config
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"target": base_system}))
        return cfg


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
    
    # Set platform properties based on the selected platform
    platform_list = ["Reference", "CPU", "OpenCL", "CUDA", "None"]
    if cfg.target.platform_name == "CUDA":
        platform_properties = {
        "Precision": "mixed",   # Best speed/accuracy tradeoff
        "DeviceIndex": "0",         # Pick GPU 0
    }
    elif cfg.target.platform_name in platform_list:
        platform_properties = None
    else:
        raise NotImplementedError(f"Platform {cfg.target.platform_name} not implemented. Either use 'Reference', 'CPU', 'CUDA', 'OpenCL' or 'None'.")
    
 
    # Target distribution setup
    if cfg.target.solvent_name == "water":
        target = SoluteInWater(
            solute_pdb_path=cfg.target.solute_pdb_path,
            solute_xml_path=cfg.target.solute_xml_path,
            solute_inpcrd_path=cfg.target.solute_inpcrd_path,
            solute_prmtop_path=cfg.target.solute_prmtop_path,
            dim=cfg.target.cartesian_dim,
            num_solvent_molecules=cfg.target.num_solvent_molecules,
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
            boundary_condition = cfg.target.boundary_condition,
            box_length_nm = cfg.target.box_length_nm,
            nonbonded_cutoff_nm = cfg.target.nonbonded_cutoff_nm,
            rigid_water=cfg.target.rigid_water,
            internal_constraints=cfg.target.internal_constraints,
            external_constraints=cfg.target.external_constraints,
            constraint_radius=cfg.target.constraint_radius,
            constraint_force=cfg.target.constraint_force,
            platform_name=cfg.target.platform_name,
            platform_properties=platform_properties 
            
        )
    else:
        raise NotImplementedError("Solute/solvent combination not implemented.")
    
    if cfg.target.boundary_condition == "pbc":
        L = float(target.box_length_nm)

        def mic(dx, L):
            return dx - L * torch.round(dx / L)

        def wrap(x, L):
            return torch.remainder(x, L)

        def dist_pbc(p, q, L):
            return torch.linalg.norm(mic(p - q, L), dim=-1)

        def angle(a, b, c, L):
            # angle ABC at B using MIC vectors BA and BC
            ba = mic(a - b, L)
            bc = mic(c - b, L)
            ba_n = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            bc_n = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosang = (ba_n * bc_n).sum(dim=-1).clamp(-1.0, 1.0)
            return torch.acos(cosang)

        @torch.no_grad()
        def min_mic_OO_distance(x_flat, L, n_solute, n_waters):
            """
            Returns min MIC O-O distance between waters per frame.
            x_flat: (B, 3N)
            """
            B = x_flat.shape[0]
            x = x_flat.view(B, -1, 3)
            O = []
            for k in range(n_waters):
                O_idx = n_solute + 3 * k
                O.append(x[:, O_idx, :])
            O = torch.stack(O, dim=1)  # (B, n_waters, 3)

            # pairwise distances
            dmin = torch.full((B,), float("inf"), device=x.device, dtype=x.dtype)
            for i in range(n_waters):
                for j in range(i + 1, n_waters):
                    dij = torch.linalg.norm(mic(O[:, i, :] - O[:, j, :], L), dim=-1)
                    dmin = torch.minimum(dmin, dij)
            return dmin
        
        def energy_by_force(context, system, x_flat, L_nm=None):
            """
            Returns a list of (force_index, force_name, energy_kJmol).
            """
            x = x_flat.view(-1, 3).detach().cpu().numpy()
            context.setPositions(x * unit.nanometer)

            # Make sure box vectors are set (optional if already correct)
            if L_nm is not None:
                import openmm as mm
                context.setPeriodicBoxVectors(
                    mm.Vec3(L_nm,0,0)*unit.nanometer,
                    mm.Vec3(0,L_nm,0)*unit.nanometer,
                    mm.Vec3(0,0,L_nm)*unit.nanometer,
                )

            out = []
            for j, frc in enumerate(system.getForces()):
                st = context.getState(getEnergy=True, groups={j})
                Ej = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                out.append((j, frc.getName(), Ej))
            return out

        with torch.no_grad():
            print("INITIAL TEST (NEW TRANSFORM)!")

            bs = 8
            md_x = target.val_data_x[:bs].to(target.device).reshape(bs, -1)  # (B, 3N)

            # # --- 1) Transform inverse ---

            md_i, _ = target.coordinate_transform.inverse(md_x)  # (B, internal_dim), (B,)
            print("md_x shape:", tuple(md_x.shape), "md_i shape:", tuple(md_i.shape))

            # --- 2) Energy consistency ---
            # U(x) from direct Cartesian energy
            md_x0 = md_x
            md_x = torch.remainder(md_x, L)
            U_from_x = -target.log_prob_x(md_x)

            # U(i): compute log_prob(i) and jac from the TransformedBoltzmann wrapper
            lp_i, jac_i = target.log_prob_and_jac(md_i)
            # In your earlier code you used: U = -(lp - jac)
            U_from_i = -(lp_i - jac_i)

            print("U_from_x (kT units):", U_from_x.detach().cpu().numpy())
            print("U_from_i (kT units):", U_from_i.detach().cpu().numpy())
            print("max |diff|:", (U_from_x - U_from_i).abs().max().item())

            # --- 3) Round-trip x reconstruction (gauge-invariant) ---
            x_from_i, _ = target.coordinate_transform.forward(md_i)  # (B, 3N)


            N = md_x.shape[1] // 3

            mdX = md_x.view(bs, N, 3)
            recX = x_from_i.view(bs, N, 3)

            # Compare everything RELATIVE to solute atom 0 using MIC (translation-invariant)
            d_md  = mic(mdX - mdX[:, 0:1, :], L)
            d_rec = mic(recX - recX[:, 0:1, :], L)

            rel_err = torch.linalg.norm(d_md - d_rec, dim=-1)  # (B,N)
            print("[DEBUG] max rel-to-sol0 MIC err (nm):", rel_err.max().item())
            print("[DEBUG] mean rel-to-sol0 MIC err (nm):", rel_err.mean().item())

            worst = rel_err.view(-1).argmax().item()
            b = worst // rel_err.shape[1]
            a = worst %  rel_err.shape[1]
            print("[DEBUG] worst frame", b, "atom", a, "err", rel_err[b, a].item())

            # Example usage inside your test:
            b = 0
            x_from_i, _ = target.coordinate_transform.forward(md_i)

            E_md  = energy_by_force(target.p.sim_context, target.system.system, md_x[b], L_nm=target.box_length_nm)
            E_rec = energy_by_force(target.p.sim_context, target.system.system, x_from_i[b],  L_nm=target.box_length_nm)

            print("Per-force energy differences (kJ/mol):")
            for (j, name, e0), (_, _, e1) in zip(E_md, E_rec):
                diff = e1 - e0
                if abs(diff) > 1e-3:  # threshold
                    print(f"{j:2d} {name:30s}  md={e0: .6f}  rec={e1: .6f}  diff={diff: .6f}")
            
        
            # --- 4) Rigid water bond lengths + angle checks ---
            mdXw  = wrap(mdX, L)
            recXw = wrap(recX, L)

            oh_md = []
            oh_rec = []
            hoh_md = []
            hoh_rec = []

            for k in range(target.num_solvent_molecules):
                O_idx = 3 + 3 * k
                H1_idx = O_idx + 1
                H2_idx = O_idx + 2

                # distances
                oh1_md = dist_pbc(mdXw[:, H1_idx, :], mdXw[:, O_idx, :], L)
                oh2_md = dist_pbc(mdXw[:, H2_idx, :], mdXw[:, O_idx, :], L)
                oh1_rc = dist_pbc(recXw[:, H1_idx, :], recXw[:, O_idx, :], L)
                oh2_rc = dist_pbc(recXw[:, H2_idx, :], recXw[:, O_idx, :], L)

                # angle H-O-H
                ang_md = angle(mdXw[:, H1_idx, :], mdXw[:, O_idx, :], mdXw[:, H2_idx, :], L)
                ang_rc = angle(recXw[:, H1_idx, :], recXw[:, O_idx, :], recXw[:, H2_idx, :], L)

                oh_md.append(torch.stack([oh1_md, oh2_md], dim=1))   # (B,2)
                oh_rec.append(torch.stack([oh1_rc, oh2_rc], dim=1))
                hoh_md.append(ang_md)   # (B,)
                hoh_rec.append(ang_rc)

            oh_md = torch.cat(oh_md, dim=0)       # (B*n_waters, 2)
            oh_rec = torch.cat(oh_rec, dim=0)
            hoh_md = torch.cat(hoh_md, dim=0)     # (B*n_waters,)
            hoh_rec = torch.cat(hoh_rec, dim=0)

            print("[BONDS] MD OH mean (nm):", oh_md.mean().item(), "std:", oh_md.std(unbiased=False).item(), "max:", oh_md.max().item())
            print("[BONDS] REC OH mean (nm):", oh_rec.mean().item(), "std:", oh_rec.std(unbiased=False).item(), "max", oh_rec.max().item())
            print("[ANGLE] MD HOH mean (deg):", (hoh_md.mean() * 180 / np.pi).item())
            print("[ANGLE] REC HOH mean (deg):", (hoh_rec.mean() * 180 / np.pi).item())
            print("[ANGLE] max |ΔHOH| (deg):", ((hoh_rec - hoh_md).abs().max() * 180 / np.pi).item())

            # --- 5) Overlap / clash check: min MIC O-O distance ---
            x_from_i_wrapped = wrap(x_from_i, L)
            min_oo = min_mic_OO_distance(x_from_i_wrapped, L, n_solute=3, n_waters=target.num_solvent_molecules)
            print("[CLASH] X(from i) min MIC O-O distance (nm): mean", min_oo.mean().item(), "min", min_oo.min().item())
               
    else:
        with torch.no_grad():
            bs = 8
            md_i = target.val_data_i[:bs].to(target.device)
            lp, jac = target.log_prob_and_jac(md_i)
            U = -(lp - jac)
            print("[DEBUG] MD U(kBT):", U.cpu().numpy(), flush=True)

    
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()

    # Setup model and start training
    # Will grab logger and save_dir from target, if target has those defined.
    # TODO: Plotter assumes a triatomic solute in water solvent. Also that the transformation from flow output coords to
    #  internal coordinates is the same as in the target distribution (e.g., for r it's just a softplus).
    setup_trainer_and_run_flow(cfg, setup_triatomic_in_h2o_plotter, target)

def min_mic_OO_distance(
    Xp_flat: torch.Tensor,
    L: float,
    n_solute: int,
    n_waters: int,
    chunk: int = 64,
) -> torch.Tensor:
    """
    Per-frame minimum MIC O-O distance among solvent waters.
    Memory-safe: O(B*chunk*W) not O(B*W^2).
    Xp_flat: (B, 3N) wrapped
    Returns: (B,)
    """
    # Accept (B,3N) or (B,N,3)
    if Xp_flat.ndim == 3:
        # (B,N,3) -> (B,3N)
        Xp_flat = Xp_flat.reshape(Xp_flat.shape[0], -1)
    elif Xp_flat.ndim != 2:
        raise ValueError

    B, D = Xp_flat.shape
    N = D // 3
    X = Xp_flat.view(B, N, 3)

    start = n_solute
    O = torch.stack([X[:, start + 3*w, :] for w in range(n_waters)], dim=1)  # (B,W,3)

    L_t = torch.as_tensor(L, device=Xp_flat.device, dtype=Xp_flat.dtype)
    
    def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
        return dx - L * torch.round(dx / L)


    min_d2 = torch.full((B,), float("inf"), device=Xp_flat.device, dtype=Xp_flat.dtype)

    for i0 in range(0, n_waters, chunk):
        i1 = min(n_waters, i0 + chunk)
        Oi = O[:, i0:i1, :]  # (B,ci,3)

        dx = mic(Oi[:, :, None, :] - O[:, None, :, :], L_t)  # (B,ci,W,3)
        d2 = (dx * dx).sum(dim=-1)                      # (B,ci,W)

        # mask self-distances where they occur
        for k in range(i1 - i0):
            d2[:, k, i0 + k] = float("inf")

        block_min = d2.amin(dim=(1, 2))  # (B,)
        min_d2 = torch.minimum(min_d2, block_min)

    return torch.sqrt(min_d2)

# Run with hydra configuration.
@hydra.main(config_path="./config/", config_name="SoluteInSolvent", version_base="1.1")
def run(cfg: DictConfig) -> None:
    # "solute_pdb_path", "solute_xml_path", "solute_inpcrd_path", "solute_prmtop_path",
    MD_KEYS = [
        "cartesian_dim", "temperature", "boundary_condition", "nonbonded_cutoff_nm", 
        "internal_constraints", "rigid_water", "box_length_nm", "num_solvent_molecules",
         "femtoseconds_per_timestep"
        #  "external_constraints", "constraint_radius", "constraint_force",
    ]
    # Load MD data specifics from MD data JSON files if they exist.
    # Use the specific listed in the MD_KEYS
    train_data_config = load_json(cfg.target.train_samples_path)
    val_data_config   = load_json(cfg.target.val_samples_path)
    test_data_config  = load_json(cfg.target.test_samples_path)

    system_cfgs = []
    for name, dc in [("train", train_data_config), ("val", val_data_config), ("test", test_data_config)]:
        if dc is None:
            continue
        sys_part = pick_system_keys(dc, MD_KEYS)
        if not sys_part:
            raise ValueError(f"{name} config JSON exists but has none of the expected keys: {MD_KEYS}")
        system_cfgs.append((name, sys_part))

    # Overwrite hydra cfg.target with the system config from the MD data JSON, 
    cfg = overwrite_cfg(cfg, system_cfgs)
    print(OmegaConf.to_yaml(cfg))

    if cfg.training.only_test_run:
        print("When test=True, only run a test run and then return")
        if cfg.target.boundary_condition == "droplet":
            run_transform_test_droplet(cfg)
        else:
            run_test_pbc(cfg)
    
        return

    _run(cfg)


if __name__ == "__main__":
    run()
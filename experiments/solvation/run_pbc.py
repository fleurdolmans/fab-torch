import os
import json
import pathlib
import hydra
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
from experiments.solvation.test_run import run_transform_test_droplet, run_test_pbc
from fab.transforms.transform_pbc import PBCPreprocessTransform

SAVE_DIR = None

def plot_pbc(fab_model: FABModel, plot_dict: dict) -> List[plt.Figure]:
    figs = []
    R, T = 8.314e-3, target.temperature
    device = target.device
    L = float(target.box_length_nm)

    atom_types = target.system.atoms  # if this is list of names like ["O","H","H"...]
    colours = {"O":"red","H":"white","S":"yellow"} 

    # triatomic solute + waters (O,H,H)
    n_solute = 3
    n_waters = int(target.num_solvent_molecules)

    # per-frame minimum MIC O-O distance (memory-safe)
    def min_mic_distance(Xp_flat: torch.Tensor, L: float, chunk: int = 64) -> torch.Tensor:
        # Xp_flat: (B, 3N) already wrapped into [0, L)
        B, D = Xp_flat.shape
        N = D // 3
        X = Xp_flat.view(B, N, 3)

        # solvent O positions
        O = X[:, n_solute::3, :]         # (B, nO, 3)
        nO = O.shape[1]

        min_d2 = torch.full((B,), float("inf"), device=Xp_flat.device, dtype=Xp_flat.dtype)

        for i0 in range(0, nO, chunk):
            i1 = min(nO, i0 + chunk)
            Oi = O[:, i0:i1, :]  # (B, ci, 3)

            dx = target.coordinate_transform.mic(Oi[:, :, None, :] - O[:, None, :, :], L)  # (B, ci, nO, 3)
            d2 = (dx * dx).sum(dim=-1)                         # (B, ci, nO)

            # mask self distances for rows where Oi overlaps O
            for k in range(i1 - i0):
                d2[:, k, i0 + k] = float("inf")

            block_min = d2.amin(dim=(1, 2))  # (B,)
            min_d2 = torch.minimum(min_d2, block_min)

        return torch.sqrt(min_d2)

    print("Loading target data for plotting...")

    if target.eval_mode == "val":
        mdI = target.val_data_i.reshape(-1, target.internal_dim).to(target.device)
    elif target.eval_mode == "test":
        mdI = target.test_data_i.reshape(-1, target.internal_dim).to(target.device)
    
    print("Loaded")


    n_md = min(5000, mdI.shape[0])
    mdI = mdI[:n_md]


    # Internal -> wrapped Cartesian
    with torch.no_grad():
        mdXp, _ = target.coordinate_transform.forward(mdI)

        n_flow = min(5000, n_md)
        flowI = fab_model.flow.sample((n_flow,)).to(device)
        flowXp, _ = target.coordinate_transform.forward(flowI)
    
    # Energies in kJ/mol via log_prob_and_jac
    with torch.no_grad():
        lp_f, jac_f = target.p.log_prob_and_jac(flowI)
        U_f = (-(lp_f - jac_f) * (R * T)).detach().cpu().numpy()

        lp_m, jac_m = target.p.log_prob_and_jac(mdI)
        U_m = (-(lp_m - jac_m) * (R * T)).detach().cpu().numpy()

    # RDF: solute atom0 -> solvent oxygens
    def rdf_sol0_to_O(Xp_flat: torch.Tensor, dr: float = 0.005):
        B, D = Xp_flat.shape
        N = D // 3
        X = Xp_flat.view(B, N, 3)
        sol0 = X[:, 0, :]  # (B,3)
        Oxyz = X[:, n_solute::3, :]  # (B, n_waters, 3)

        d = target.coordinate_transform.mic(Oxyz - sol0[:, None, :], L)
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
        g = counts / np.maximum(expected, 1e-12)
        g[0] = 0.0
        return r_centers, g

    r_m, g_m = rdf_sol0_to_O(mdXp)
    r_f, g_f = rdf_sol0_to_O(flowXp)

    # Plot: RDF and Energy histogram
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(r_m, g_m, label="MD")
    ax1.plot(r_f, g_f, label="Flow")
    ax1.set_xlim(0, L / 2)
    ax1.set_xlabel("r (nm)")
    ax1.set_ylabel("g(r)")
    ax1.set_title("RDF: solute atom0 → solvent O (MIC)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    nbins = 100
    hist_range = (np.nanmin(U_m), np.nanmax(U_m))
    ax2.hist(U_f, bins=nbins, range=hist_range, density=True, alpha=0.4, label="Flow")
    ax2.hist(U_m, bins=nbins, range=hist_range, density=True, alpha=0.4, label="MD")
    ax2.set_xlabel("Potential energy (kJ/mol)")
    ax2.set_ylabel("density")
    ax2.set_title("Energy distribution")
    ax2.legend()

    figs.append(fig)
    
    def subplot_molecular_system_pbc(ax, pos, energy, title_str, L, atom_types, colours):
        # pos: (N,3) in [0,L)
        # mic_fn: function(dx)->dx_mic using same L

        def mic_fn(dx_np, L):
            dx = torch.tensor(dx_np, device=device, dtype=mdXp.dtype)
            out = target.coordinate_transform.mic(dx, L).detach().cpu().numpy()
            return out

        # colors per atom
        color_list = [colours.get(t, "k") for t in atom_types]

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color_list, s=10)

        # draw O-H bonds within each water/solute triad using MIC vectors
        # assumes ordering O H H repeated, and solute is also triatomic
        for i in range(0, len(pos) - 2, 3):
            o = pos[i]
            h1 = pos[i + 1]
            h2 = pos[i + 2]

            d1 = mic_fn(h1 - o, L)
            d2 = mic_fn(h2 - o, L)

            p1 = o + d1
            p2 = o + d2

            # optionally wrap endpoints for nicer display
            p1 = np.mod(p1, L)
            p2 = np.mod(p2, L)

            ax.plot([o[0], p1[0]], [o[1], p1[1]], [o[2], p1[2]], color="grey", lw=1)
            ax.plot([o[0], p2[0]], [o[1], p2[1]], [o[2], p2[2]], color="grey", lw=1)

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel("z (nm)")
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
        ax.set_title(f"{title_str}: {energy:.3g} kJ/mol")
        ax.view_init(elev=30, azim=45)

        legend_elements = [
            Patch(facecolor=colours[t], edgecolor=colours[t], label=t)
            for t in sorted(set(atom_types)) if t in colours
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    md_positions = mdXp[:2].detach().cpu().numpy()
        
    md_energies = U_m[:2]

    idx_sorted = np.argsort(U_f)
    low_positions  = flowXp[idx_sorted[:2]].detach().cpu().numpy()
    low_energies   = U_f[idx_sorted[:2]]
    high_positions = flowXp[idx_sorted[-2:]].detach().cpu().numpy()
    high_energies  = U_f[idx_sorted[-2:]]

    fig2 = plt.figure(figsize=(14, 8))
    axes = [
        fig2.add_subplot(2, 3, 1, projection="3d"),
        fig2.add_subplot(2, 3, 2, projection="3d"),
        fig2.add_subplot(2, 3, 3, projection="3d"),
        fig2.add_subplot(2, 3, 4, projection="3d"),
        fig2.add_subplot(2, 3, 5, projection="3d"),
        fig2.add_subplot(2, 3, 6, projection="3d"),
    ]
    subplot_molecular_system_pbc(axes[0], md_positions[0].reshape(-1, 3), md_energies[0], "MD frame 0")
    subplot_molecular_system_pbc(axes[1], low_positions[0].reshape(-1, 3), low_energies[0], "Flow lowest U")
    subplot_molecular_system_pbc(axes[2], high_positions[0].reshape(-1, 3), high_energies[0], "Flow highest U")
    subplot_molecular_system_pbc(axes[3], md_positions[1].reshape(-1, 3), md_energies[1], "MD frame 1")
    subplot_molecular_system_pbc(axes[4], low_positions[1].reshape(-1, 3), low_energies[1], "Flow 2nd lowest U")
    subplot_molecular_system_pbc(axes[5], high_positions[1].reshape(-1, 3), high_energies[1], "Flow 2nd highest U")
    plt.tight_layout()
    figs.append(fig2)

    return figs

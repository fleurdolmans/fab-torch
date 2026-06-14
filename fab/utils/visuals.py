# Visualization Library
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from matplotlib import animation
from torch import distributions
from scipy.stats import multivariate_normal
from matplotlib.patches import Circle

import matplotlib as mpl
mpl.rcParams["animation.html"] = "jshtml"

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 12
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)


def make_2D_traj_circles(
    x_traj,
    box=None,
    sigma=1.1,
    fps=30,
    solute_color="orangered",
    solvent_color="steelblue",
    figsize=(6, 6),
    xlim=None,
    ylim=None,
    interval=None,
    fontsize=14,
    animate=True,
    ax=None,
    title=None,
):
    """
    Animate a 2D trajectory with particles drawn as circles whose size is in data units.
    If ax is provided, draws a static snapshot (frame 0) onto that axes instead.

    Parameters
    ----------
    x_traj : np.ndarray
        Shape (n_frames, n_particles, 2)
    box : tuple or None
        Simulation box lengths (Lx, Ly). Used if xlim/ylim are not given.
    sigma : float
        Particle diameter in simulation units.
    fps : int
        Frames per second for the animation.
    solute_color : str
        Color for the first particle.
    solvent_color : str
        Color for all remaining particles.
    figsize : tuple
        Figure size (ignored when ax is provided).
    xlim, ylim : tuple or None
        Plot limits. If None, inferred from box.
    interval : float or None
        Milliseconds between frames. If None, computed from fps.
    fontsize : int
        Font size for axis labels.
    animate : bool
        If True, returns a FuncAnimation. Ignored when ax is provided.
    ax : matplotlib.axes.Axes or None
        If provided, draw a static snapshot onto this axes (for subplots).
    title : str or None
        Optional axes title.

    Returns
    -------
    (fig, ani)  when animate=True and ax is None
    (fig, ax)   when animate=False or ax is provided
    """
    x_traj = np.asarray(x_traj)
    n_frames, n_particles, dim = x_traj.shape
    if dim != 2:
        raise ValueError(f"x_traj must have shape (n_frames, n_particles, 2), got {x_traj.shape}")

    radius = sigma / 2.0
    if interval is None:
        interval = 1000 / fps

    box_rect = None
    if box is not None:
        x_half = box[0] / 2
        y_half = box[1] / 2
        box_rect = (-x_half, -y_half, box[0], box[1])  # (x0, y0, width, height)
        if xlim is None:
            xlim = (-x_half - 0.5, x_half + 0.5)
        if ylim is None:
            ylim = (-y_half - 0.5, y_half + 0.5)
    elif xlim is None or ylim is None:
        raise ValueError("Provide either box or both xlim and ylim.")

    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X axis", fontsize=fontsize)
    ax.set_ylabel("Y axis", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=16)

    # Draw dashed box outline
    if box_rect is not None:
        ax.add_patch(plt.Rectangle(
            (box_rect[0], box_rect[1]), box_rect[2], box_rect[3],
            fill=False, edgecolor="gray", linestyle="--", linewidth=1.2,
        ))

    # Create particle circles (frame 0)
    circles = []
    for p in range(n_particles):
        color = solute_color if p < 1 else solvent_color
        alpha = 1.0 if p < 1 else 0.5
        x, y = x_traj[0, p]
        circ = Circle((x, y), radius=radius, facecolor=color, edgecolor="black", alpha=alpha)
        ax.add_patch(circ)
        circles.append(circ)

    if animate and not external_ax:
        def update(frame):
            for p, circ in enumerate(circles):
                circ.center = tuple(x_traj[frame, p])
            return circles

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=True
        )
        return fig, ani

    return fig, ax

def visualize_rdf(r=None, gr=None, label=None, sigma=1.1, title='Solute-Solvent RDF', gr_std=None, colors=None):
    _, ax = plt.subplots(figsize=(8, 5))

    # Normalize inputs to lists
    if r is None:
        r = []
    elif not isinstance(r, list):
        r = [r]

    if gr is None:
        gr = []
    elif not isinstance(gr, list):
        gr = [gr]

    if gr_std is None:
        gr_std = [None] * len(r)
    elif not isinstance(gr_std, list):
        gr_std = [gr_std]

    if len(r) != len(gr):
        raise ValueError(
            f"r and gr must have the same length, "
            f"got {len(r)} and {len(gr)}"
        )

    if label is None:
        raise ValueError(
            f"You must provide a label for the RDF plots, got None. "
        )
    elif isinstance(label, str):
        label = [label]

    if len(label) != len(r):
        raise ValueError(
            f"label must have the same length as r/gr, "
            f"got {len(label)} and {len(r)}"
        )
    if colors is None:
        colors = ['black', 'steelblue', 'orangered', 'purple', 'darkorange', 'crimson']

    # Plot curves with optional ± std bands
    for i, (ri, gri) in enumerate(zip(r, gr)):
        if ri is not None and gri is not None:
            color = colors[i % len(colors)]
            ax.plot(ri, gri, lw=2, color=color, label=label[i])
            if gr_std[i] is not None:
                ax.fill_between(
                    ri,
                    gri - gr_std[i],
                    gri + gr_std[i],
                    color=color, alpha=0.15,
                )

    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='Ideal gas')
    ax.axvline(sigma, color='k', ls=':', lw=0.8, label=f'σ = {sigma}')

    ax.set_xlabel('r  (nm)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.show()



def visualize_distances(ss, vv, label,
    sigma=None, measure="min", sub_titles=[
                                'Solute – Solvent  (particle 0 vs rest)',
                                'Solvent – Solvent  (all solvent pairs)'
                            ], colors=None
    ):
    """ Visualize distributions of minimum pairwise distances for solute-solvent and 
    solvent-solvent pairs. Supports multiple BG datasets with custom labels."""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Normalize BG inputs to lists
    if ss is None:
        ss = []
    elif not isinstance(ss, list):
        ss = [ss]

    if vv is None:
        vv = []
    elif not isinstance(vv, list):
        vv = [vv]

    if len(ss) != len(vv):
        raise ValueError(
            f"ss and vv must have the same length, "
            f"got {len(ss)} and {len(vv)}"
        )

    all_arrays = [a for sublist in [ss, vv] for a in sublist if a is not None]
    max_val = np.max(np.concatenate(all_arrays))
    bins = np.linspace(0, max_val, 80)

    if colors is None:
        colors = ['black','steelblue', 'orangered', 'purple', 'darkorange', 'crimson']
    
    for ax, dist_list, title in zip(axes, [ss, vv], sub_titles):
        for i, d in enumerate(dist_list):
            if d is not None:
                ax.hist(d,
                    bins=bins,
                    density=True,
                    alpha=0.3,
                    color=colors[i % len(colors)],
                    label=label[i],
                    histtype='stepfilled'
                )
        if sigma is not None:
            ax.axvline(sigma, color='k', ls='--', lw=1.2, label=f'σ = {sigma}')
        ax.set_xlabel(f'{measure.capitalize()} distance  (nm)', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(title, fontsize=18)
        if ax == axes[0]:  # Only add legend to the last subplot
            ax.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.show()

def position_density(datasets, l_box, title='Position density  (red = solute, blue = solvent)'):
    # Side-by-side snapshot density: MC vs BG

    fig, axes = plt.subplots(1, len(datasets), figsize=(len(datasets)*3, 4))

    # Make sure axes is iterable even if there is only one dataset
    if len(datasets) == 1:
        axes = [axes]

    for ax, (traj, subtitle) in zip(axes, datasets):
        print(subtitle)

        ax.scatter(
            traj[:, 1:, 0].ravel(), traj[:, 1:, 1].ravel(),
            s=1, c='steelblue', alpha=0.04, label='Solvent'
        )
        ax.scatter(
            traj[:, 0, 0], traj[:, 0, 1],
            s=2, c='orangered', alpha=0.3, label='Solute'
        )

        ax.add_patch(
            plt.Rectangle(
                (-l_box, -l_box), 2*l_box, 2*l_box,
                fill=False, ec='gray', ls='--'
            )
        )

        ax.set_xlim(-l_box-0.5, l_box+0.5)
        ax.set_ylim(-l_box-0.5, l_box+0.5)
        ax.set_aspect('equal')

        ax.set_title(subtitle, fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_visible(False)  # Hide X-axis
        ax.yaxis.set_visible(False)  # Hide Y-axis
        if ax == axes[-1]:  # Only add legend to the last subplot
            ax.legend(
                    markerscale=6,
                    loc='upper right',
                    fontsize=12
                )

    # fig.suptitle(title, fontsize=18)

    plt.tight_layout()
    plt.show()

def plot_energies(energies, label, title='Energy distribution: MC vs BG', x_max=1000, colors=None):
    
    fig, ax = plt.subplots(figsize=(7, 4))
    if colors is None:
        colors = ['black', 'steelblue', 'orangered', 'purple', 'darkorange', 'crimson']
    for e in range(len(energies)):
        e_plot = energies[e][energies[e] < x_max]
        ax.hist(e_plot, bins=80, density=True, alpha=0.3, color=colors[e], label=f'{label[e]}({len(e_plot)/len(energies[e])*100:.0f}% shown)')
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_loss(loss, title='ML training loss'):
    window   = max(1, len(loss) // 200)
    smoothed = np.convolve(loss, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(8, 3))
    plt.plot(smoothed)
    plt.xlabel('Batch iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_position_density_mc_vs_bg(xtraj, x_gen, l_box, mc_stride=5):
    """
    Side-by-side scatter plot comparing MC and BG position density.

    Parameters
    ----------
    xtraj : np.ndarray, shape (n_frames, N, 2)
    x_gen : np.ndarray, shape (n_gen, N, 2)
    l_box : float
        Box half-width.
    mc_stride : int
        Use every mc_stride-th MC frame for the scatter plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    datasets = [
        (xtraj[::mc_stride], "MC samples"),
        (x_gen, "BG generated"),
    ]
    for ax, (traj, panel_title) in zip(axes, datasets):
        ax.scatter(
            traj[:, 1:, 0].ravel(), traj[:, 1:, 1].ravel(),
            s=1, c="steelblue", alpha=0.04, label="Solvent",
        )
        ax.scatter(
            traj[:, 0, 0], traj[:, 0, 1],
            s=8, c="orangered", alpha=0.30, label="Solute",
        )
        ax.add_patch(plt.Rectangle(
            (-l_box, -l_box), 2 * l_box, 2 * l_box,
            fill=False, ec="gray", ls="--",
        ))
        ax.set_xlim(-l_box - 1, l_box + 1)
        ax.set_ylim(-l_box - 1, l_box + 1)
        ax.set_aspect("equal")
        ax.set_title(panel_title, fontsize=16)
        ax.legend(markerscale=6, loc="upper right", fontsize=12)
    plt.suptitle("Position density  (red = solute, blue = solvent)", fontsize=18)
    plt.tight_layout()
    plt.show()


# ─── 3D Molecular System Visualization ──────────────────────────────────────


def plot_mol_position_density(datasets, L, O_IDX, N_SOLUTE_ATOMS, IS_PBC=True):
    """
    Scatter plot of water-O position density in XY, XZ, YZ projections.

    Parameters
    ----------
    datasets : list of (traj, subtitle)
        traj : (N, N_ATOMS, 3) array
    L : float
        Box length (nm).
    O_IDX : list[int]
        Water oxygen atom indices.
    N_SOLUTE_ATOMS : int
        Number of solute atoms.
    IS_PBC : bool
        Draw periodic box outline if True.
    """
    _proj_pairs = [
        (0, 1, "X (nm)", "Y (nm)"),
        (0, 2, "X (nm)", "Z (nm)"),
        (1, 2, "Y (nm)", "Z (nm)"),
    ]

    def _scatter_row(axes_row, traj, subtitle):
        o_pos   = traj[:, O_IDX, :]
        sol_pos = traj[:, :N_SOLUTE_ATOMS, :]
        for ax, (i, j, xl, yl) in zip(axes_row, _proj_pairs):
            ax.scatter(o_pos[:, :, i].ravel(), o_pos[:, :, j].ravel(),
                       s=1, c="steelblue", alpha=0.03, label="Water O")
            ax.scatter(sol_pos[:, :, i].ravel(), sol_pos[:, :, j].ravel(),
                       s=4, c="orangered", alpha=0.5, label="Solute")
            if IS_PBC:
                ax.add_patch(plt.Rectangle((0, 0), L, L,
                                           fill=False, ec="gray", ls="--", lw=1.2))
                ax.set_xlim(-0.05 * L, 1.05 * L)
                ax.set_ylim(-0.05 * L, 1.05 * L)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
        axes_row[1].set_title(subtitle, fontsize=14)

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 3, figsize=(12, 4.2 * n_ds), squeeze=False)
    for row_idx, (traj, subtitle) in enumerate(datasets):
        _scatter_row(axes[row_idx], np.asarray(traj), subtitle)
    axes[-1, -1].legend(markerscale=6, loc="upper right", fontsize=11)
    fig.suptitle("Position density  (blue = water-O, red = solute)", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_water_oxygen_heatmaps(datasets, L, O_IDX, N_SOLUTE_ATOMS,
                                IS_PBC=True, n_bins=60):
    """
    Log-scale 2D marginal histograms of water-O positions (XY, XZ, YZ).

    Parameters
    ----------
    datasets : list of (traj, label)
        traj : (N, N_ATOMS, 3)
    L : float
    O_IDX : list[int]
    N_SOLUTE_ATOMS : int
    IS_PBC : bool
    n_bins : int
    """
    from matplotlib.colors import LogNorm

    _proj_labels = [
        (2, "XY", "X (nm)", "Y (nm)"),
        (1, "XZ", "X (nm)", "Z (nm)"),
        (0, "YZ", "Y (nm)", "Z (nm)"),
    ]
    _Lplot = L if IS_PBC else None

    def _oxy_hist2d(traj, axis_drop):
        keep = [a for a in range(3) if a != axis_drop]
        oxy  = traj[:, O_IDX, :].reshape(-1, 3)
        H, _, _ = np.histogram2d(
            oxy[:, keep[0]], oxy[:, keep[1]],
            bins=n_bins,
            range=[[0, L], [0, L]] if _Lplot else None,
        )
        p = H / H.sum()
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        return H, entropy, keep

    n_ds   = len(datasets)
    n_proj = len(_proj_labels)
    fig, axes = plt.subplots(n_ds, n_proj,
                              figsize=(5.5 * n_proj, 4.8 * n_ds),
                              constrained_layout=True, squeeze=False)
    for row, (traj, ds_lbl) in enumerate(datasets):
        traj = np.asarray(traj)
        for col, (axis_drop, proj_name, xl, yl) in enumerate(_proj_labels):
            H, S, keep = _oxy_hist2d(traj, axis_drop)
            _pos  = H[H > 0]
            _norm = LogNorm(vmin=max(1, np.percentile(_pos, 1)),
                            vmax=max(2, np.percentile(_pos, 99.5)))
            ax  = axes[row, col]
            ext = (0, L, 0, L) if IS_PBC else None
            im  = ax.imshow(H.T, origin="lower", extent=ext,
                            aspect="equal", cmap="magma", norm=_norm)
            for si in range(N_SOLUTE_ATOMS):
                ax.scatter(traj[:, si, keep[0]].mean(),
                           traj[:, si, keep[1]].mean(),
                           s=40, c="cyan", edgecolors="black",
                           linewidths=0.5, zorder=3)
            if IS_PBC:
                ax.add_patch(plt.Rectangle((0, 0), L, L,
                                           fill=False, ec="white", ls="--",
                                           lw=1.0, alpha=0.7))
            ax.set_title(f"{ds_lbl} — {proj_name}  (H = {S:.3f})", fontsize=13)
            ax.set_xlabel(xl, fontsize=11)
            ax.set_ylabel(yl, fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.85).set_label("count (log)", fontsize=10)
    fig.suptitle("Water-oxygen density (marginal projections)", fontsize=16)
    plt.show()


def plot_rdf_3d(sw_curves, ww_curves, L=None, IS_PBC=True, sigma_oo=0.31507):
    """
    Plot solute–water-O and water-O–water-O RDF panels.

    Each curve is a precomputed (r, g_r, label) tuple produced by
    ``mol_analysis.compute_rdf_3d``.  Pass as many curves as you like per
    panel to overlay multiple models or temperatures in one plot.

    Parameters
    ----------
    sw_curves : list of (r, g_r, label)
        Solute(S) – water-O RDF curves.
    ww_curves : list of (r, g_r, label)
        Water-O – water-O RDF curves.
    L : float or None
        Box length (nm); used to set x-axis limit to L/2 when IS_PBC=True.
    IS_PBC : bool
    sigma_oo : float or None
        O-O LJ sigma (nm) to draw as a reference line on the water-water panel.
        TIP3P value: 0.31507 nm.
    """
    _colors = ["black", "steelblue", "orangered", "purple", "darkorange", "crimson"]
    panels = [
        (sw_curves, "Solute(S) – Water-O RDF", False),
        (ww_curves, "Water-O – Water-O RDF",   True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (curves, title, is_ww) in zip(axes, panels):
        for i, (r, g, lbl) in enumerate(curves):
            ax.plot(r, g, lw=2, color=_colors[i % len(_colors)], label=lbl)
        ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal gas")
        if is_ww and sigma_oo is not None:
            ax.axvline(sigma_oo, color="gray", ls=":", lw=1.2,
                       label=f"σ_OO = {sigma_oo:.4f} nm")
        ax.set_xlabel("r  (nm)", fontsize=12)
        ax.set_ylabel("g(r)", fontsize=12)
        ax.set_title(title, fontsize=16)
        if IS_PBC and L is not None:
            ax.set_xlim(0, 0.5 * L)
        ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mol_oo_distances(ww_dists, sw_dists, colors=None, sigma_oo=None):
    """
    Plot minimum O-O distance distributions (water–water and solute–water).

    Parameters
    ----------
    ww_dists : list of (distances, label)
        Per-frame minimum water-O – water-O distances from
        ``mol_analysis.compute_min_distances``.
    sw_dists : list of (distances, label)
        Per-frame minimum solute-atom-0 – water-O distances.
    colors : list[str] or None
    sigma_oo : float or None
        O-O LJ sigma (nm) to draw as a reference line on the water-water panel.
        TIP3P value: 0.31507 nm.
    """
    if colors is None:
        colors = ["black", "steelblue", "orangered", "purple", "darkorange", "crimson"]

    _all  = np.concatenate([v for v, _ in ww_dists + sw_dists])
    _bins = np.linspace(0, np.nanpercentile(_all[np.isfinite(_all)], 99), 80)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for ax, dists, stitle, show_sigma in zip(
        axes,
        [ww_dists, sw_dists],
        ["Water-O – Water-O ", "Solute(S) – Water-O "],
        [True, False],
    ):
        for i, (d, lbl) in enumerate(dists):
            ax.hist(d, bins=_bins, density=True, alpha=0.35,
                    color=colors[i % len(colors)], label=lbl, histtype="stepfilled")
        if show_sigma and sigma_oo is not None:
            ax.axvline(sigma_oo, color="gray", ls=":", lw=1.2,
                       label=f"σ_OO = {sigma_oo:.4f} nm")
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(stitle, fontsize=14)
        if ax == axes[0]:
            ax.legend(fontsize=12, loc="upper left")
            ax.set_xlabel("Min distance  (nm)", fontsize=14)
            ax.xaxis.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_water_geometry(oh_datasets, hoh_datasets):
    """
    Plot water O-H bond lengths and H-O-H angles.

    Parameters
    ----------
    oh_datasets : list of (oh_lengths, label, color)
        O-H bond length arrays from ``mol_analysis.compute_water_geometry``.
    hoh_datasets : list of (hoh_angles, label, color)
        H-O-H angle arrays (degrees).
    """
    _RIGID_STD_THRESH = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for oh, lbl, c in oh_datasets:
        oh = np.asarray(oh)
        if oh.std() < _RIGID_STD_THRESH:
            axes[0].axvline(oh.mean(), color=c, lw=2, alpha=0.8,
                            label=f"{lbl} (rigid, {oh.mean():.5f} nm)")
        else:
            axes[0].hist(oh, bins=80, density=True, alpha=0.4, color=c,
                         histtype="stepfilled", label=lbl)

    for hoh, lbl, c in hoh_datasets:
        hoh = np.asarray(hoh)
        if hoh.std() < _RIGID_STD_THRESH:
            axes[1].axvline(hoh.mean(), color=c, lw=2, alpha=0.8,
                            label=f"{lbl} (rigid, {hoh.mean():.4f}°)")
        else:
            axes[1].hist(hoh, bins=80, density=True, alpha=0.4, color=c,
                         histtype="stepfilled", label=lbl)

    axes[0].axvline(0.096, color="gray", ls=":", lw=1.0, label="TIP3P O-H = 0.096 nm")
    axes[1].axvline(104.52, color="gray", ls=":", lw=1.0, label="TIP3P H-O-H = 104.52°")
    axes[0].set_xlabel("O–H distance (nm)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Water O–H bond length", fontsize=16)
    axes[0].legend(fontsize=11)
    axes[1].set_xlabel("H–O–H angle (°)", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Water H–O–H angle", fontsize=16)
    axes[1].legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_solute_geometry(bond_datasets, angle_datasets, ref_bonds=None, ref_angles=None):
    """
    Plot solute bond length and angle distributions.

    Parameters
    ----------
    bond_datasets : list of (bonds, label, color)
        bonds : list of (bond_label_str, lengths_nm_array)
            Output of ``compute_solute_geometry`` (bond_data field), pooled
            across repeats.
    angle_datasets : list of (angles, label, color)
        angles : list of (angle_label_str, angles_deg_array)
    ref_bonds : list of (bond_label_str, ref_nm) or None
        Reference values shown as vertical dashed lines (e.g. equilibrium
        bond lengths from the force field).
    ref_angles : list of (angle_label_str, ref_deg) or None
    """
    bond_labels  = [lbl for lbl, _ in (bond_datasets[0][0]  if bond_datasets  else [])]
    angle_labels = [lbl for lbl, _ in (angle_datasets[0][0] if angle_datasets else [])]
    n_panels = len(bond_labels) + len(angle_labels)
    if n_panels == 0:
        return

    ref_bonds_dict  = dict(ref_bonds)  if ref_bonds  else {}
    ref_angles_dict = dict(ref_angles) if ref_angles else {}

    fig, axes = plt.subplots(n_panels, 1, figsize=(3 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    for b_idx, b_label in enumerate(bond_labels):
        ax = axes[panel]
        for bonds, lbl, c in bond_datasets:
            arr = np.asarray(bonds[b_idx][1])
            ax.hist(arr, bins=80, density=True, alpha=0.4, color=c,
                    histtype="stepfilled", label=lbl)
        if b_label in ref_bonds_dict:
            v = ref_bonds_dict[b_label]
            ax.axvline(v, color="gray", ls=":", lw=1.5,
                       label=f"ref = {v:.4f} nm")
        ax.set_xlabel(f"Bond length  {b_label}  (nm)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Solute bond  {b_label}", fontsize=14)
        panel += 1

    for a_idx, a_label in enumerate(angle_labels):
        ax = axes[panel]
        for angles, lbl, c in angle_datasets:
            arr = np.asarray(angles[a_idx][1])
            ax.hist(arr, bins=80, density=True, alpha=0.4, color=c,
                    histtype="stepfilled", label=lbl)
        if a_label in ref_angles_dict:
            v = ref_angles_dict[a_label]
            ax.axvline(v, color="gray", ls=":", lw=1.5,
                       label=f"ref = {v:.1f}°")
        ax.set_xlabel(f"Angle  {a_label}  (°)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Solute angle  {a_label}", fontsize=14)
        if ax == axes[-1]:  # Only add legend to the last subplot
            ax.legend(fontsize=12, loc="upper left")
        panel += 1

    plt.tight_layout()
    plt.show()


def plot_mol_energies(u_flow, u_md=None, title='Energy distribution', x_max=None):
    """
    Plot energy (reduced, = -log p) distributions for flow and optional MD reference.

    Parameters
    ----------
    u_flow : np.ndarray
    u_md : np.ndarray or None
    title : str
    x_max : float or None
        Upper x-axis limit; defaults to 99th percentile of finite values.
    """
    u_flow = np.asarray(u_flow)
    _finite = u_flow[np.isfinite(u_flow)]
    if x_max is None:
        x_max = float(np.percentile(_finite, 99))
        if u_md is not None:
            u_md = np.asarray(u_md)
            x_max = max(x_max, float(np.percentile(u_md[np.isfinite(u_md)], 99)))

    _sets = [(u_flow, "Flow", "steelblue")]
    if u_md is not None:
        _sets.insert(0, (u_md, "MD reference", "black"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for e, lbl, c in _sets:
        ep   = e[e < x_max]
        frac = len(ep) / len(e) * 100
        ax.hist(ep, bins=80, density=True, alpha=0.35, color=c,
                label=f"{lbl} ({frac:.0f}% shown)")
    ax.set_xlabel("Reduced energy  (−log p)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_mol_snapshots(datasets, L, N_SOLUTE_ATOMS, O_IDX, H1_IDX, H2_IDX,
                       IS_PBC=True, n_snap=4):
    """
    Gallery of 3D scatter snapshots for one or more trajectory sources.

    Parameters
    ----------
    datasets : list of (traj, label)
        traj : (N, N_ATOMS, 3)
    L : float
    N_SOLUTE_ATOMS : int
    O_IDX, H1_IDX, H2_IDX : list[int]
    IS_PBC : bool
    n_snap : int
        Number of snapshots per source.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _draw(ax3d, pos, title=None):
        o_pos = pos[O_IDX]
        h_pos = pos[H1_IDX + H2_IDX]
        sol   = pos[:N_SOLUTE_ATOMS]
        ax3d.scatter(*h_pos.T, s=10, c="lightblue",  alpha=0.5, edgecolors="none")
        ax3d.scatter(*o_pos.T, s=25, c="steelblue",  alpha=0.8, edgecolors="none",
                     label="Water O")
        ax3d.scatter(*sol.T,   s=80, c="orangered",  alpha=1.0, edgecolors="black",
                     linewidths=0.5, label="Solute")
        if IS_PBC and L is not None:
            for _i in range(2):
                for _j in range(2):
                    ax3d.plot([0, L], [_i*L, _i*L], [_j*L, _j*L],
                              c="gray", ls="--", lw=0.5, alpha=0.5)
                    ax3d.plot([_i*L, _i*L], [0, L], [_j*L, _j*L],
                              c="gray", ls="--", lw=0.5, alpha=0.5)
                    ax3d.plot([_i*L, _i*L], [_j*L, _j*L], [0, L],
                              c="gray", ls="--", lw=0.5, alpha=0.5)
            ax3d.set_xlim(0, L); ax3d.set_ylim(0, L); ax3d.set_zlim(0, L)
        ax3d.set_xlabel("X", fontsize=7)
        ax3d.set_ylabel("Y", fontsize=7)
        ax3d.set_zlabel("Z", fontsize=7)
        ax3d.tick_params(labelsize=6)
        if title:
            ax3d.set_title(title, fontsize=10)

    _ncols = n_snap * len(datasets)
    fig = plt.figure(figsize=(4.5 * _ncols, 4.5))
    k = 0
    for traj, lbl in datasets:
        traj = np.asarray(traj)
        for n in range(n_snap):
            ax3d = fig.add_subplot(1, _ncols, k + 1, projection="3d")
            _draw(ax3d, traj[n], title=f"{lbl} #{n+1}")
            k += 1
    plt.suptitle("Configuration snapshots", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_pretty_density_histograms(
    hist_mc, hist_bg_mean, H_mc, H_bg_mean, H_bg_std, l_box
):
    """
    Side-by-side log-scale 2D solvent density histograms for MC and BG.

    Parameters
    ----------
    hist_mc : np.ndarray, shape (bins, bins)
    hist_bg_mean : np.ndarray, shape (bins, bins)
    H_mc, H_bg_mean, H_bg_std : float
        Entropy values to display in panel titles.
    l_box : float
        Box half-width (used for the image extent and box outline).
    """
    from matplotlib.colors import LogNorm

    hist_mc = np.asarray(hist_mc)
    hist_bg_mean = np.asarray(hist_bg_mean)

    positive_values = np.concatenate([
        hist_mc[hist_mc > 0].ravel(),
        hist_bg_mean[hist_bg_mean > 0].ravel(),
    ])

    if len(positive_values) == 0:
        norm = None
    else:
        vmin = max(1, np.percentile(positive_values, 1))
        vmax = np.percentile(positive_values, 99.5)
        if vmax <= vmin:
            vmax = positive_values.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = [
        (hist_mc, f"MC solvent density\nH = {H_mc:.4f}"),
        (hist_bg_mean, f"BG mean solvent density\nH = {H_bg_mean:.4f} ± {H_bg_std:.4f}"),
    ]

    last_im = None
    for ax, (hist, title) in zip(axes, panels):
        last_im = ax.imshow(
            hist.T, origin="lower",
            extent=(-l_box, l_box, -l_box, l_box),
            aspect="equal", cmap="magma", norm=norm,
        )
        ax.scatter([0], [0], s=35, c="cyan", edgecolors="black",
                   linewidths=0.5, label="solute", zorder=3)
        ax.add_patch(plt.Rectangle(
            (-l_box, -l_box), 2 * l_box, 2 * l_box,
            fill=False, ec="white", ls="--", lw=1.0, alpha=0.7,
        ))
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(loc="upper right", frameon=True, fontsize=12)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.9)
    cbar.set_label("solvent count per bin (log scale)", fontsize=12)
    plt.show()




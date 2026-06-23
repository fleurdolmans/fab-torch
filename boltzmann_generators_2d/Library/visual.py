import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib import animation
from matplotlib.patches import Circle

mpl.rcParams["animation.html"] = "jshtml"

rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'size': 12})
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')


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
        ax.set_title(title, fontsize=22)

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

    ax.axvline(sigma, color='k', ls=':', lw=0.8, label=f'σ = {sigma}')

    ax.set_xlabel('r  (nm)', fontsize=16)
    ax.set_ylabel('g(r)', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.show()



def visualize_distances(ss, vv, label,
    sigma=None, measure="min", sub_titles=[
                                'Solute – Solvent',
                                'Solvent – Solvent'
                            ], colors=None
    ):
    """ Visualize distributions of minimum pairwise distances for solute-solvent and 
    solvent-solvent pairs. Supports multiple BG datasets with custom labels."""
    _, axes = plt.subplots(1, 2, figsize=(13, 5))

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
        ax.set_xlabel(f'{measure.capitalize()} distance  (nm)', fontsize=22)
        if ax == axes[0]:
            ax.set_ylabel('Density', fontsize=22)
        ax.set_title(title, fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        if ax == axes[0]:  # Only add legend to the last subplot
            ax.legend(fontsize=14, loc='upper left')

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
        if ax == axes[0]:  # Only add legend to the last subplot
            ax.legend(
                    markerscale=6,
                    loc='upper left',
                    fontsize=12
                )

    plt.tight_layout()
    plt.show()

def plot_energies(energies, label, title='Energy distribution: MC vs BG', x_max=100, colors=None):
    
    fig, ax = plt.subplots(figsize=(7, 4))
    if colors is None:
        colors = ['black', 'steelblue', 'orangered', 'purple', 'darkorange', 'crimson']
    for e in range(len(energies)):
        e_plot = energies[e][energies[e] < x_max]
        ax.hist(e_plot, bins=80, density=True, alpha=0.3, color=colors[e], label=f'{label[e]}({len(e_plot)/len(energies[e])*100:.0f}% shown)')
    ax.set_xlabel('Energy (kJ mol⁻¹)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
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




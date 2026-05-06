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
from density_estimator import density_estimator
from matplotlib.patches import Circle

import matplotlib as mpl
mpl.rcParams["animation.html"] = "jshtml"

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
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
    solute_color="red",
    solvent_color="blue",
    figsize=(6, 6),
    xlim=None,
    ylim=None,
    interval=None,
):
    """
    Animate a 2D trajectory with particles drawn as circles whose size is in data units.

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
        Color for the first two particles.
    solvent_color : str
        Color for all remaining particles.
    figsize : tuple
        Figure size.
    xlim, ylim : tuple or None
        Plot limits. If None, inferred from box.
    interval : float or None
        Milliseconds between frames. If None, computed from fps.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    x_traj = np.asarray(x_traj)
    n_frames, n_particles, dim = x_traj.shape
    if dim != 2:
        raise ValueError(f"x_traj must have shape (n_frames, n_particles, 2), got {x_traj.shape}")

    radius = sigma / 2.0
    if interval is None:
        interval = 1000 / fps

    if xlim is None or ylim is None:
        if box is None:
            raise ValueError("Provide either box or both xlim and ylim.")
        x_half = box[0] / 2
        y_half = box[1] / 2
        xlim = (-x_half, x_half)
        ylim = (-y_half, y_half)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    # Create particle circles
    circles = []
    for p in range(n_particles):
        color = solute_color if p < 1 else solvent_color
        alpha = 1.0 if p < 1 else 0.5   # solute vs solvent
        x, y = x_traj[0, p]
        circ = Circle((x, y), radius=radius, facecolor=color, edgecolor="black", alpha=alpha)
        ax.add_patch(circ)
        circles.append(circ)


    def update(frame):
        # Move particles
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

    return ani

def compute_gr_solute(traj, N, l_box, n_bins=60, r_max=4.0):
    """
    Solute-solvent g(r).  traj: (n_frames, N, 2).
    """
    r_edges   = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr        = r_centers[1] - r_centers[0]
    counts    = np.zeros(n_bins)
    for frame in traj:
        dists = np.linalg.norm(frame[1:] - frame[0], axis=1)
        hist, _ = np.histogram(dists, bins=r_edges)
        counts += hist
    rho  = (N - 1) / (2 * l_box)**2   # number density of solvent
    norm = len(traj) * rho * 2 * np.pi * r_centers * dr
    return r_centers, counts / norm


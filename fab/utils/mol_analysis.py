"""
Compute functions for 3D molecular trajectory analysis.

These functions are pure NumPy — they accept (N, N_ATOMS, 3) arrays and
return statistics ready for plotting.  Keep them separate from visuals.py
so you can compute once and build any combination of plots you need.
"""
import numpy as np


def compute_rdf_3d(traj, ref_idx, target_idx, L, dr=0.01):
    """
    3-D radial distribution function g(r) with minimum-image convention (PBC).

    Parameters
    ----------
    traj : array_like, shape (N, N_ATOMS, 3)
    ref_idx : list[int]
        Atom indices for the reference group.
    target_idx : list[int]
        Atom indices for the target group.
    L : float or None
        Box length (nm). None → no MIC (droplet).
    dr : float
        Bin width (nm).

    Returns
    -------
    r_c : np.ndarray   Bin centres.
    g_r : np.ndarray   g(r) values.
    """
    traj = np.asarray(traj)
    N    = traj.shape[0]
    ref  = traj[:, ref_idx, :]
    tgt  = traj[:, target_idx, :]

    diff = tgt[:, None, :, :] - ref[:, :, None, :]   # (N, n_ref, n_tgt, 3)
    if L is not None:
        diff -= L * np.round(diff / L)
    r = np.linalg.norm(diff, axis=-1).ravel()

    r_max  = 0.5 * L if L is not None else r.max()
    n_bins = int(np.floor(r_max / dr))
    edges  = np.linspace(0.0, n_bins * dr, n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)

    r_c      = 0.5 * (edges[:-1] + edges[1:])
    shell    = 4.0 * np.pi * r_c ** 2 * dr
    vol      = L ** 3 if L is not None else (4 / 3 * np.pi * r_max ** 3)
    rho      = len(target_idx) / vol
    expected = N * len(ref_idx) * rho * shell

    g_r    = counts / np.maximum(expected, 1e-12)
    g_r[0] = 0.0
    return r_c, g_r


def compute_min_distances(traj, O_IDX, L, IS_PBC=True):
    """
    Per-frame minimum pairwise distances.

    Parameters
    ----------
    traj : array_like, shape (N, N_ATOMS, 3)
    O_IDX : list[int]
        Water oxygen atom indices.
    L : float
        Box length (nm).
    IS_PBC : bool

    Returns
    -------
    d_ww : np.ndarray, shape (N,)
        Minimum water-O – water-O distance per frame.
    d_sw : np.ndarray, shape (N,)
        Minimum solute-atom-0 – water-O distance per frame.
    """
    traj = np.asarray(traj)

    # Water–water
    oxy  = traj[:, O_IDX, :]
    diff = oxy[:, :, None, :] - oxy[:, None, :, :]   # (N, W, W, 3)
    if IS_PBC:
        diff -= L * np.round(diff / L)
    d = np.linalg.norm(diff, axis=-1)
    eye = np.eye(d.shape[1], dtype=bool)[None, :, :]
    d_ww = np.where(eye, np.inf, d).min(axis=(1, 2))

    # Solute(atom 0)–water
    sol  = traj[:, 0:1, :]
    diff = oxy - sol
    if IS_PBC:
        diff -= L * np.round(diff / L)
    d_sw = np.linalg.norm(diff, axis=-1).min(axis=1)

    return d_ww, d_sw


def compute_water_geometry(traj, N_WATERS, O_IDX, H1_IDX, H2_IDX, L, IS_PBC=True):
    """
    Water O-H bond lengths and H-O-H angles for all frames.

    Parameters
    ----------
    traj : array_like, shape (N, N_ATOMS, 3)
    N_WATERS : int
    O_IDX, H1_IDX, H2_IDX : list[int]
    L : float
    IS_PBC : bool

    Returns
    -------
    oh_lengths : np.ndarray   All O-H bond lengths (nm).
    hoh_angles : np.ndarray   All H-O-H angles (degrees).
    """
    traj = np.asarray(traj)

    def _mic(a, b):
        d = b - a
        if IS_PBC:
            d -= L * np.round(d / L)
        return d

    oh_lengths, hoh_angles = [], []
    for k in range(N_WATERS):
        o  = traj[:, O_IDX[k],  :]
        h1 = traj[:, H1_IDX[k], :]
        h2 = traj[:, H2_IDX[k], :]

        v1 = _mic(o, h1)
        v2 = _mic(o, h2)
        oh_lengths.append(np.linalg.norm(v1, axis=-1))
        oh_lengths.append(np.linalg.norm(v2, axis=-1))

        cos = (v1 * v2).sum(-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12
        )
        hoh_angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    return np.concatenate(oh_lengths), np.concatenate(hoh_angles)

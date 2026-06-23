import copy
import itertools
import yaml

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

def overlap_penalty(x_flat, sys_dim, sigma, cutoff_factor=1.0, k=200.0):
    """
    Differentiable soft overlap penalty using softplus.

    Parameters
    ----------
    x_flat : (B, N*D) tensor
    sys_dim : tuple (N, D)
    sigma : float — particle diameter
    cutoff_factor : float — fraction of sigma used as the clash distance threshold
    k : float — sharpness of the softplus; larger k → closer to a hard wall

    Returns
    -------
    scalar — mean over batch of sum_{i<j} softplus(k*(r0 - r_ij)) / k
    """
    N, D = sys_dim
    B = x_flat.shape[0]
    coords = x_flat.reshape(B, N, D)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)           # (B, N, N, D)
    d = torch.linalg.norm(diff, dim=-1)                        # (B, N, N)
    idx = torch.triu_indices(N, N, offset=1, device=x_flat.device)
    d_pairs = d[:, idx[0], idx[1]]                             # (B, N*(N-1)/2)
    r0 = cutoff_factor * sigma
    return (F.softplus(k * (r0 - d_pairs)) / k).sum(-1).mean()


def apply_hungarian_alg(ref_coord, coords, n_particles=None, dim=None):
    """
    Reorder solvent particles (indices 1:) in each frame.

    Particle 0 is the fixed solute — left untouched.
    Particles 1..N-1 are reordered to minimise total Euclidean distance to
    a fixed reference configuration.

    Parameters
    ----------
    ref_coord : np.ndarray, shape (N, dim) or (N*dim,)
        Reference configuration, e.g. the last frame of the MC trajectory.
    coords : np.ndarray, shape (n_frames, N, dim) or (n_frames, N*dim)
        Trajectory to reorder.
    n_particles : int or None
        Total particle count (including solute). Inferred from coords if None.
    dim : int or None
        Spatial dimension (2 or 3). Inferred from coords if None.

    Returns
    -------
    new_coords : np.ndarray, same shape as ``coords``
    """
    if dim is None or n_particles is None:
        if coords.ndim == 3:
            n_particles, dim = coords.shape[1], coords.shape[2]
        else:
            raise ValueError(
                "Provide n_particles and dim when coords is a flattened (n_frames, N*dim) array."
            )

    flat_input = (coords.ndim == 2 and coords.shape[-1] == n_particles * dim)
    n_frames = len(coords)

    coords_p = coords.reshape(n_frames, n_particles, dim).copy()
    ref_p = ref_coord.reshape(n_particles, dim)
    ref_sol = ref_p[1:]   # skip fixed solute at index 0

    for i in tqdm(range(n_frames), desc='Hungarian reorder', leave=False):
        sol = coords_p[i, 1:]                            # (N-1, dim)
        diff = ref_sol[:, None, :] - sol[None, :, :]     # (N-1, N-1, dim)
        cost = np.linalg.norm(diff, axis=-1)              # (N-1, N-1)
        _, ci = linear_sum_assignment(cost)
        coords_p[i, 1:] = sol[ci]

    if flat_input:
        return coords_p.reshape(n_frames, n_particles * dim)
    return coords_p

def load_model(model_path, BoltzmannGenerator2D, solute_sys, flow_type, w_overlap, device="cpu"):
    with open(model_path + '.yml', 'r') as f:
        saved_params = yaml.unsafe_load(f)

    # Infer architecture from checkpoint weights (YAML can have stale values)
    state_dict = torch.load(model_path, map_location='cpu')
    cond_keys = sorted(
        [k for k in state_dict
            if 'coupling_layers.0.conditioner.net.' in k and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[-2])
    )
    if cond_keys:
        n_hidden_inferred = len(cond_keys) - 1
        n_nodes_inferred  = int(state_dict[cond_keys[-1]].shape[1])
        n_frozen_coords   = int(state_dict[cond_keys[0]].shape[1])
        out_size          = int(state_dict[cond_keys[-1]].shape[0])
        # Use layer 1's conditioner input size as n_active for layer 0
        # (masks alternate, so layer 1 freezes what layer 0 transforms)
        cond_keys_l1 = sorted(
            [k for k in state_dict
                if 'coupling_layers.1.conditioner.net.' in k and k.endswith('.weight')],
            key=lambda k: int(k.split('.')[-2])
        )
        if cond_keys_l1:
            n_active_coords = int(state_dict[cond_keys_l1[0]].shape[1])
        else:
            n_active_coords = n_frozen_coords  # fallback: assume equal split
        params_per_coord  = out_size // n_active_coords
        num_bins_inferred = (params_per_coord + 1) // 3
        saved_params['n_hidden']  = n_hidden_inferred
        saved_params['n_nodes']   = n_nodes_inferred
        saved_params['num_bins']  = num_bins_inferred

    saved_params['flow_type'] = flow_type
    saved_params['w_overlap'] = w_overlap

    BG = BoltzmannGenerator2D(saved_params)
    model_template = BG.build(solute_sys)
    if hasattr(model_template, "to"):
        model_template = model_template.to(device)
    model_loaded, loss = BG.load(model_template, model_path)
    model_loaded.eval()
    return model_loaded, loss


def distances(traj, measure='min'):
    """
    For each frame compute:
    - solute-solvent distance  (particle 0 vs 1..N-1)
    - solvent-solvent distance (all pairs within 1..N-1)

    Parameters
    ----------
    traj : np.ndarray, shape (n_frames, N, dim), where dim can be 2 or 3
    measure : str
        One of 'min', 'mean', 'median', 'max'.

    Returns
    -------
    d_sol_solv : np.ndarray, shape (n_frames,)
    d_solv_solv : np.ndarray, shape (n_frames,)
    """
    d_sol_solv = np.zeros(len(traj))
    d_solv_solv = np.zeros(len(traj))

    for i, frame in enumerate(traj):
        solute = frame[0]      # (dim,)
        solvent = frame[1:]    # (N-1, dim)

        # solute-solvent
        dists_ss = np.linalg.norm(solvent - solute, axis=-1)
        if measure == 'min':
            d_sol_solv[i] = dists_ss.min()
        elif measure == 'mean':
            d_sol_solv[i] = dists_ss.mean()
        elif measure == 'median':
            d_sol_solv[i] = np.median(dists_ss)
        elif measure == 'max':
            d_sol_solv[i] = dists_ss.max()
        else:
            raise ValueError(f'Unknown measure: {measure}')

        # solvent-solvent
        diff = solvent[:, None, :] - solvent[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(r, np.inf)
        if measure == 'min':
            d_solv_solv[i] = r[r < np.inf].min()
        elif measure == 'mean':
            d_solv_solv[i] = r[r < np.inf].mean()
        elif measure == 'median':
            idx = np.triu_indices(len(solvent), k=1)
            d_solv_solv[i] = np.median(r[idx])
        elif measure == 'max':
            idx = np.triu_indices(len(solvent), k=1)
            d_solv_solv[i] = r[idx].max()
        else:
            raise ValueError(f'Unknown measure: {measure}')

    return d_sol_solv, d_solv_solv


def compute_gr_solute(traj, n_particles, l_box, n_bins=60, r_max=None, dim=None):
    """
    Solute-solvent radial distribution function g(r).

    Parameters
    ----------
    traj : np.ndarray, shape (n_frames, N, dim)
    n_particles : int
        Total number of particles (including solute).
    l_box : float
        Box half-width.
    n_bins : int
        Number of histogram bins.
    r_max : float or None
        Maximum radius. Defaults to l_box - 0.3.
    dim : int
        Spatial dimension (2 or 3). Determines shell normalisation:
        - dim=2: 2D ring area  2*pi*r*dr
        - dim=3: 3D shell volume  4*pi*r^2*dr

    Returns
    -------
    r_centers : np.ndarray, shape (n_bins,)
    gr : np.ndarray, shape (n_bins,)
    """
    if dim is None:
        raise ValueError(
            "dim must be specified: pass dim=2 for 2D systems or dim=3 for 3D systems. "
            "Using the wrong dimension inflates/deflates g(r) by a large factor."
        )

    if r_max is None:
        r_max = l_box - 0.3

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_centers[1] - r_centers[0]
    counts = np.zeros(n_bins)

    for frame in traj:
        dists = np.linalg.norm(frame[1:] - frame[0], axis=1)
        hist, _ = np.histogram(dists, bins=r_edges)
        counts += hist

    # Number density of solvent in the box
    n_solvent = n_particles - 1
    box_vol = (2 * l_box) ** dim
    rho = n_solvent / box_vol

    if dim == 3:
        norm = len(traj) * rho * 4 * np.pi * r_centers ** 2 * dr
    elif dim == 2:
        norm = len(traj) * rho * 2 * np.pi * r_centers * dr
    else:
        raise ValueError(f"Unsupported dim={dim}. Use 2 or 3.")

    return r_centers, counts / norm


# ---------------------------------------------------------------------------
# Naming / config helpers
# ---------------------------------------------------------------------------

def format_float_for_name(x):
    """Turn floats into filename-safe strings, e.g. 0.5 -> '0p5', 1e-4 -> '1em4'."""
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if x.is_integer():
        return str(int(x))
    s = f"{x:g}"
    s = s.replace(".", "p").replace("-", "m").replace("+", "")
    return s


def make_run_name(version, flow, stage, wloss, N, L, energy_cap=None):
    """
    Build a model filename stem matching the convention used in the notebook.

    Examples
    --------
    make_run_name("v1", flow="spline", stage=1, wloss=[1.0, 0.0, 0.0], N=33, L=4.5)
        -> "v1_spline_s1_w_loss-1p0-0p0-0p0_N33_L4p5"
    make_run_name("v1", flow="spline", stage=2, wloss=[0.5, 0.5, 10.0], N=33, L=4.5,)
        -> "v1_spline_s2_w_loss-0p5-0p5-10p0_N33_L4p5"
    """
    name = f"{version}_{flow}_s{stage}_w_loss"
    
    for w in wloss:
        name += f"-{format_float_for_name(w)}"
    name += f"_N{N}_L{format_float_for_name(L)}"
    if energy_cap is not None:
        name += f"_E{format_float_for_name(energy_cap)}"
    return name


def make_bg_config(flow_type, N_SOLVENT, DIM, w_overlap, n_blocks=8, n_nodes=256, n_layers=3, n_epochs=25, batch_size=512,
                    lr=1e-4, prior_sigma=1.0, extra_config=None, min_delta=1e-4, patience=20, n_hidden=3, num_bins=8, 
                    tail_bound=5, hidden=128
                    ):
    """Create the config dictionary for BoltzmannGenerator2D."""

    config = {
        "flow_type": flow_type,
        "lr": lr, 
        "batch_size": batch_size, 
        "reshape": (N_SOLVENT, DIM),
        "dimension": N_SOLVENT * DIM,
        "min_delta": min_delta,
        "n_blocks": n_blocks,
        "n_epochs": n_epochs,
        "n_layers": n_layers,
        "n_nodes": n_nodes,
        "patience": patience,
        "prior_sigma": prior_sigma,
        "n_hidden": n_hidden,
        "num_bins": num_bins,
        'w_overlap'  : w_overlap,
        "tail_bound": tail_bound,
        "hidden": hidden
    }
    if extra_config is not None:
        config.update(extra_config)
    return config


def clone_model(model):
    """Return an independent deep copy of a model."""
    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Coordinate initialisation
# ---------------------------------------------------------------------------

def build_solute_coords(N=37, l_box=3.0, noise=0.1, seed=42):
    """
    Build an initial configuration on a grid with the solute fixed at the origin.

    Parameters
    ----------
    N : int
        Total number of particles (solute + solvent).
    l_box : float
        Box half-width.
    noise : float
        Gaussian noise amplitude added to grid positions.
    seed : int
        Random seed.

    Returns
    -------
    coords : np.ndarray, shape (N, 2)
    """
    np.random.seed(seed)
    d = (2 * l_box) / N ** 0.5
    n_side = int(np.ceil(np.sqrt(N)))
    pos = np.linspace(-l_box + 0.5 * d, l_box - 0.5 * d, n_side)
    coords = np.array(list(itertools.product(pos, repeat=2))[:N], dtype=np.float32)
    if noise:
        coords += (noise * np.random.randn(*coords.shape)).astype(np.float32)
    nearest = np.argmin(np.linalg.norm(coords, axis=1))
    coords[[0, nearest]] = coords[[nearest, 0]]
    coords[0] = [0.0, 0.0]
    return coords


# ---------------------------------------------------------------------------
# Per-frame distance statistics
# ---------------------------------------------------------------------------

def mean_distances(traj):
    """
    For each frame compute the mean solute-solvent and mean solvent-solvent distance.

    Parameters
    ----------
    traj : np.ndarray, shape (n_frames, N, 2)
        Particle 0 is the solute; particles 1..N-1 are solvent.

    Returns
    -------
    d_sol_solv : np.ndarray, shape (n_frames,)
    d_solv_solv : np.ndarray, shape (n_frames,)
    """
    d_sol_solv = np.zeros(len(traj))
    d_solv_solv = np.zeros(len(traj))
    for i, frame in enumerate(traj):
        solute = frame[0]
        solvent = frame[1:]
        d_sol_solv[i] = np.linalg.norm(solvent - solute, axis=1).mean()
        diff = solvent[:, None, :] - solvent[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        upper = r[np.triu_indices(len(solvent), k=1)]
        d_solv_solv[i] = upper.mean()
    return d_sol_solv, d_solv_solv


# ---------------------------------------------------------------------------
# Spatial density entropy
# ---------------------------------------------------------------------------

def density_entropy_2d(x, bins=50, box=None, normalize=True, eps=1e-12):
    """
    Shannon entropy of the 2D spatial particle density.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_frames, n_particles, 2) or (n_samples, 2).
    bins : int
        Number of bins per dimension.
    box : tuple or None
        ((xmin, xmax), (ymin, ymax)). Inferred from data if None.
    normalize : bool
        If True, divide by log(n_bins^2) so values are in ~[0, 1].
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    H : float
    hist : np.ndarray, shape (bins, bins)
    """
    x = np.asarray(x)
    if x.ndim == 3:
        positions = x.reshape(-1, 2)
    elif x.ndim == 2 and x.shape[1] == 2:
        positions = x
    else:
        raise ValueError("x must have shape (n_frames, n_particles, 2) or (n_samples, 2).")

    if box is None:
        hist_range = [
            (positions[:, 0].min(), positions[:, 0].max()),
            (positions[:, 1].min(), positions[:, 1].max()),
        ]
    else:
        hist_range = [box[0], box[1]]

    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins, range=hist_range
    )
    p = counts.ravel()
    p = p / p.sum()
    p_nonzero = p[p > eps]
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    if normalize:
        H = H / np.log(len(p))
    return H, counts


# ---------------------------------------------------------------------------
# Effective Sample Size
# ---------------------------------------------------------------------------

def effective_sample_size(log_weights):
    """
    Effective Sample Size (ESS) from log importance weights.
    For a Boltzmann generator the importance weights are:

        w_i = p(x_i) / q(x_i)  =  exp(-U(x_i)) / q(x_i)

    so the log weights are:

        log w_i = -U(x_i) - log q(x_i)

    where log q(x_i) is the log-density of the flow at x_i, obtained from
    the change-of-variables formula:

        log q(x_i) = log p_prior(z_i) + log |det J_i|

    The ESS is computed in log-space for numerical stability:

        ESS = (Σ wᵢ)² / Σ wᵢ²

        log ESS = 2 · logsumexp(log_w) − logsumexp(2 · log_w)

    Parameters
    ----------
    log_weights : array-like, shape (N,)
        Per-sample log importance weights. Need not be normalised.

    Returns
    -------
    ess : float
        Effective sample size in (0, N].
    ess_fraction : float
        ESS / N, in (0, 1]. A value close to 1 means the flow closely
        matches the Boltzmann distribution; a value near 0 indicates
        severe weight collapse.

    Examples
    --------
    Typical usage with a trained flow::

        with torch.no_grad():
            z = model.prior.sample(N)
            x, log_det = model.generator(z)          # log_det = log|det J|
            log_q = model.prior.log_prob(z) + log_det
            energies = system.get_energy_batch(x)     # U(x)
            log_w = -energies - log_q.numpy()

        ess, ess_frac = utils.effective_sample_size(log_w)
        print(f"ESS = {ess:.1f} / {N}  ({ess_frac*100:.1f}%)")
    """
    log_weights = np.asarray(log_weights, dtype=float)
    n = len(log_weights)
    log_ess = 2.0 * logsumexp(log_weights) - logsumexp(2.0 * log_weights)
    ess = float(np.exp(log_ess))
    return ess, ess / n


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def eval_energies(solute_sys, x_gen, etraj, verbose=True, energy_threshold=1000):
    """
    Compute energy statistics for generated samples and the MC reference.

    Parameters
    ----------
    solute_sys : object
        Full system object; must implement ``get_energy(x)`` where x has
        shape (n_particles, 2).
    x_gen : np.ndarray, shape (N, n_particles, 2)
        Full system coordinates (solute + solvent) as returned by
        ``generate_bg_samples``.
    etraj : array-like
        MC reference energies (computed with the same ``solute_sys.get_energy``).
    verbose : bool
        Print a summary table.
    energy_threshold : float
        Samples with energy < threshold are considered physical (default 1000).

    Returns
    -------
    dict with keys bg_energies, mc_energies, bg_stats, mc_stats.
    """
    with torch.no_grad():
        energies = torch.tensor([solute_sys.get_energy(s) for s in x_gen])
    energies = energies.cpu().numpy()
    etraj_np = np.asarray(etraj)

    bg_stats = {
        "mean": float(energies.mean()),
        "std": float(energies.std()),
        "min": float(energies.min()),
        "max": float(energies.max()),
        "frac_clipped_gt_1e7": float((energies > 1e7).mean()),
        "n_above_energy_cap": int((energies >= 1e8).sum()),
        "n_overlap_gt_1000": int((energies > energy_threshold).sum()),
        "frac_overlap_gt_1000": float((energies > energy_threshold).mean()),
        "n_physical_lt_1000": int((energies < energy_threshold).sum()),
        "frac_physical_lt_1000": float((energies < energy_threshold).mean()),
    }
    mc_stats = {
        "mean": float(etraj_np.mean()),
        "std": float(etraj_np.std()),
        "min": float(etraj_np.min()),
        "max": float(etraj_np.max()),
        "frac_clipped_gt_1e7": float((etraj_np > 1e7).mean()),
    }

    if verbose:
        print("Generated sample energies:")
        print(f"  mean : {bg_stats['mean']:.1f}")
        print(f"  std  : {bg_stats['std']:.1f}")
        print(f"  min  : {bg_stats['min']:.1f}")
        print(f"  max  : {bg_stats['max']:.1f}")
        print(f"  frac clipped (>1e7): {bg_stats['frac_clipped_gt_1e7']:.3f}")
        print("\nMC reference energies:")
        print(f"  mean : {mc_stats['mean']:.1f}")
        print(f"  std  : {mc_stats['std']:.1f}")
        print(f"  min  : {mc_stats['min']:.1f}")
        print(f"  max  : {mc_stats['max']:.1f}")
        print(f"  frac clipped (>1e7): {mc_stats['frac_clipped_gt_1e7']:.3f}")
        print(
            f"Samples above energy cap (max overlap): "
            f"{bg_stats['n_above_energy_cap']} / {len(energies)}"
        )
        print(
            f"Samples with overlap (E > {energy_threshold}):        "
            f"{bg_stats['n_overlap_gt_1000']} / {len(energies)}  "
            f"({bg_stats['frac_overlap_gt_1000'] * 100:.1f}%)"
        )
        print(
            f"Samples physical (E < {energy_threshold}):            "
            f"{bg_stats['n_physical_lt_1000']} / {len(energies)}  "
            f"({bg_stats['frac_physical_lt_1000'] * 100:.1f}%)"
        )

    return {
        "bg_energies": energies,
        "mc_energies": etraj_np,
        "bg_stats": bg_stats,
        "mc_stats": mc_stats,
    }


# ---------------------------------------------------------------------------
# Physical/unphysical sample splitting
# ---------------------------------------------------------------------------

def split_physical_samples(solute_sys, x_gen, energy_threshold=27.0):
    """
    Split generated samples into physical and unphysical subsets.

    Parameters
    ----------
    solute_sys : object
        Full system object; must implement ``get_energy(x)``.
    x_gen : np.ndarray, shape (N, n_particles, 2)
        Full system coordinates as returned by ``generate_bg_samples``.
    energy_threshold : float
        Samples with energy <= threshold are considered physical (default 27.0).

    Returns
    -------
    dict with keys:
        energies    : torch.Tensor (N,), energy of every sample in original order
        mask_phys   : bool tensor (N,), True for physical samples
        idx_phys    : LongTensor, indices of physical samples in original order
        idx_unphys  : LongTensor, indices of unphysical samples in original order
        e_phys      : energies[idx_phys]
        e_unphys    : energies[idx_unphys]
        n_phys      : int
        n_unphys    : int
        frac_phys   : float
    """
    with torch.no_grad():
        energies = torch.tensor([solute_sys.get_energy(s) for s in x_gen])

    mask_phys  = energies <= energy_threshold
    idx_phys   = mask_phys.nonzero(as_tuple=True)[0]
    idx_unphys = (~mask_phys).nonzero(as_tuple=True)[0]

    return {
        "energies":   energies,
        "mask_phys":  mask_phys,
        "idx_phys":   idx_phys,
        "idx_unphys": idx_unphys,
        "e_phys":     energies[idx_phys],
        "e_unphys":   energies[idx_unphys],
        "n_phys":     int(mask_phys.sum()),
        "n_unphys":   int((~mask_phys).sum()),
        "frac_phys":  float(mask_phys.float().mean()),
    }

# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def summarize_array(values):
    """Return mean/std/min/max dict for an array of scalar values."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def mean_and_std(values):
    """Return (mean, std) for an array of scalar values."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


# ---------------------------------------------------------------------------
# Training trajectory visualisation
# ---------------------------------------------------------------------------
def plot_training_trajectory(model_paths, stage_names, flow_labels=None, smooth_window=None, figsize=(12, 4), title='Training loss trajectory',
    show_raw=False, ylim=None, show_gradient=False, stage_boundaries=None, boundary_unit='iteration', iters_per_epoch=None,
    stage_label_y=0.97, show_stage_shading=True, show_stage_lines=True, colors=['steelblue','orangered','seagreen', 'purple', 'darkorange', 'crimson'],
    ):
    """
    Plot the concatenated training loss trajectory across multiple training stages,
    supporting one or more flows on the same axes.

    Loss values are read directly from the .yml files stored alongside each
    model checkpoint, then stitched together in the order supplied.

    You can either let the function infer stage boundaries from the individual
    .yml files, or manually pass stage boundaries. Manual boundaries are useful
    when your losses are already concatenated but you still want to draw boxes
    and vertical lines showing the training stages.

    Parameters
    ----------
    model_paths : list of str or list of lists of str
        Single flow:
            ['path/s1', 'path/s2', ...]
        Multiple flows:
            [['path/s1_flowA', ...], ['path/s1_flowB', ...]]

    stage_names : list of str or list of lists of str
        Stage labels matching model_paths.
        For a single flow:
            ['ML', 'KL', ...]
        For multiple flows:
            [['ML', 'KL'], ['ML', 'KL']]

    flow_labels : list of str or None
        Legend labels for each flow, e.g. ['T=4.5', 'T=5.5'].

    smooth_window : int or None
        Width of the moving-average window.
        Defaults to max(1, total_iters // 200), computed per flow.

    figsize : tuple
        Figure size passed to plt.subplots.

    title : str
        Plot title.

    show_raw : bool
        If True, also draw the unsmoothed loss as a faint background line.

    ylim : tuple or None
        If provided, sets the y-axis limits explicitly, e.g. (0, 5).

    show_gradient : bool
        If True, also plot d(loss) / d(iteration) on a secondary y-axis.

    stage_boundaries : None, list, or list of lists
        Manual stage boundaries.

        If None:
            Boundaries are inferred from the lengths of losses in each .yml file.

        If boundary_unit='iteration':
            Boundaries are interpreted as batch-iteration indices.

        If boundary_unit='epoch':
            Boundaries are interpreted as epoch numbers and converted to
            iterations using iters_per_epoch.

        You may pass either full boundaries including 0 and total length:
            stage_boundaries=[0, 5000, 12000, 20000]
        or only internal boundaries:
            stage_boundaries=[5000, 12000]
        In the latter case, the function automatically adds 0 and total length.
        For multiple flows, you can pass one shared list:

            stage_boundaries=[0, 5000, 12000, 20000]

        or one list per flow:

            stage_boundaries=[
                [0, 5000, 12000, 20000],
                [0, 4000, 11000, 18000],
            ]

    boundary_unit : {'iteration', 'epoch'}
        Unit of manually supplied stage_boundaries.

    iters_per_epoch : int, float, list, or None
        Required when boundary_unit='epoch'.

        If a scalar is passed, the same conversion is used for all flows.

        If a list is passed for multiple flows, each flow gets its own value:
            iters_per_epoch=[1000, 950]

    stage_label_y : float
        Vertical position of stage labels in axes coordinates.
        0.97 places labels near the top of the plotting area.

    show_stage_shading : bool
        If True, draw shaded boxes for stages.

    show_stage_lines : bool
        If True, draw vertical dashed lines at stage boundaries.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    def _is_nested_list(x):
        return (
            isinstance(x, (list, tuple))
            and len(x) > 0
            and isinstance(x[0], (list, tuple))
        )

    def _read_loss_from_yml(path_without_extension):
        yml_path = path_without_extension + '.yml'

        with open(yml_path, 'r') as f:
            lines = f.readlines()

        loss = None

        for line in lines:
            if 'loss: [' in line:
                loss_str = line.split('[', 1)[1].rsplit(']', 1)[0].split(',')
                loss = [float(v) for v in loss_str if v.strip()]
                break

        if loss is None:
            raise ValueError(f"No loss data found in {yml_path}.")

        return loss

    def _normalise_flows(model_paths, stage_names):
        multi_flow = _is_nested_list(model_paths)

        if not multi_flow:
            model_paths = [model_paths]
            stage_names = [stage_names]

        return model_paths, stage_names, multi_flow

    def _normalise_iters_per_epoch(iters_per_epoch, n_flows):
        if iters_per_epoch is None:
            return None

        if isinstance(iters_per_epoch, (list, tuple, np.ndarray)):
            if len(iters_per_epoch) != n_flows:
                raise ValueError(
                    "When iters_per_epoch is a list, it must have the same "
                    f"length as the number of flows. Got {len(iters_per_epoch)} "
                    f"for {n_flows} flows."
                )
            return list(iters_per_epoch)

        return [iters_per_epoch] * n_flows

    def _manual_boundaries_for_flow(
        stage_boundaries,
        flow_idx,
        n_flows,
        total,
        n_stage_names,
        boundary_unit,
        iters_per_epoch_for_flow,
    ):
        if stage_boundaries is None:
            return None

        if boundary_unit not in {'iteration', 'epoch'}:
            raise ValueError(
                "boundary_unit must be either 'iteration' or 'epoch'."
            )

        # For multiple flows, accept either:
        #   [30, 40, 50, 600] shared by all flows
        # or:
        #   [[30, 40, 50, 600], [30, 40, 50, 600]] one list per flow
        if _is_nested_list(stage_boundaries):
            if len(stage_boundaries) != n_flows:
                raise ValueError(
                    "Nested stage_boundaries must have one boundary list per flow. "
                    f"Got {len(stage_boundaries)} boundary lists for {n_flows} flows."
                )
            boundaries = list(stage_boundaries[flow_idx])
        else:
            boundaries = list(stage_boundaries)

        if len(boundaries) == 0:
            raise ValueError("stage_boundaries cannot be an empty list.")

        if boundary_unit == 'epoch':
            if iters_per_epoch_for_flow is None:
                raise ValueError(
                    "iters_per_epoch is required when boundary_unit='epoch'."
                )

            boundaries = [
                int(round(epoch * iters_per_epoch_for_flow))
                for epoch in boundaries
            ]
        else:
            boundaries = [int(round(boundary)) for boundary in boundaries]

        # Three accepted styles:
        #
        # 1. Full boundaries:
        #       [0, 30, 40, 50, 600]
        #    for 4 stages.
        #
        # 2. Stage end points:
        #       [30, 40, 50, 600]
        #    for 4 stages. We prepend 0.
        #
        # 3. Internal boundaries:
        #       [30, 40, 50]
        #    for 4 stages. We prepend 0 and append total.
        #
        if len(boundaries) == n_stage_names + 1:
            # Full boundaries, possibly already including 0.
            if boundaries[0] != 0:
                raise ValueError(
                    "When stage_boundaries has len(stage_names) + 1 entries, "
                    "it should include the initial 0 boundary."
                )

        elif len(boundaries) == n_stage_names:
            # Stage end points.
            boundaries = [0] + boundaries

        elif len(boundaries) == n_stage_names - 1:
            # Internal boundaries.
            boundaries = [0] + boundaries + [total]

        else:
            raise ValueError(
                "stage_boundaries must be one of:\n"
                "  - full boundaries: len(stage_names) + 1 entries, e.g. [0, 30, 40, 50, 600]\n"
                "  - stage end points: len(stage_names) entries, e.g. [30, 40, 50, 600]\n"
                "  - internal boundaries: len(stage_names) - 1 entries, e.g. [30, 40, 50]\n"
                f"Got {len(boundaries)} entries for {n_stage_names} stage names."
            )

        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(
                "stage_boundaries must be strictly increasing after conversion "
                "to iterations."
            )

        if boundaries[0] < 0:
            raise ValueError(
                f"stage_boundaries cannot start below 0. Got {boundaries}."
            )

        if boundaries[-1] > total:
            raise ValueError(
                f"The final stage boundary is larger than the number of losses. "
                f"Got final boundary {boundaries[-1]} but total={total}. "
                "If your boundaries are epochs, set boundary_unit='epoch' and "
                "iters_per_epoch=..."
            )

        return boundaries

    # Normalise to list-of-flows format.
    model_paths, stage_names, multi_flow = _normalise_flows(model_paths, stage_names)
    n_flows = len(model_paths)

    if len(stage_names) != n_flows:
        raise ValueError(
            "stage_names must have the same number of flows as model_paths. "
            f"Got {len(stage_names)} and {n_flows}."
        )

    if flow_labels is not None and len(flow_labels) != n_flows:
        raise ValueError(
            "flow_labels must have the same length as model_paths. "
            f"Got {len(flow_labels)} and {n_flows}."
        )

    if multi_flow and flow_labels is None:
        flow_labels = [f'Flow {i + 1}' for i in range(n_flows)]

    iters_per_epoch_by_flow = _normalise_iters_per_epoch(iters_per_epoch, n_flows)

    flow_colors = colors

    stage_colors = [
        '#d0e8f5',
        '#fde8d0',
        '#d0f5d8',
        '#f5d0e8',
        '#ede8f5',
        '#f5f5d0',
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax2 = None
    if show_gradient:
        ax2 = ax.twinx()
        ax2.set_ylabel('d(Loss) / d(iteration)', fontsize=11, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.axhline(0, color='gray', lw=0.6, ls=':', alpha=0.6)

    flow0_last_boundary = 0

    for flow_idx, (paths, names) in enumerate(zip(model_paths, stage_names)):
        if len(paths) == 0:
            raise ValueError(f"Flow {flow_idx}: model_paths cannot be empty.")

        if len(names) == 0:
            raise ValueError(f"Flow {flow_idx}: stage_names cannot be empty.")

        if stage_boundaries is None and len(paths) != len(names):
            raise ValueError(
                f"Flow {flow_idx}: when stage_boundaries is None, model_paths "
                "and stage_names must have the same length so stage boundaries "
                "can be inferred. "
                f"Got {len(paths)} paths and {len(names)} stage names."
            )

        # Read all losses for this flow.
        stage_losses = []
        for path in paths:
            stage_losses.append(_read_loss_from_yml(path))

        all_losses = np.array(
            [value for loss in stage_losses for value in loss],
            dtype=float,
        )

        total = len(all_losses)

        if total == 0:
            raise ValueError(f"Flow {flow_idx}: no loss values found.")

        # Determine stage boundaries.
        if stage_boundaries is None:
            inferred_boundaries = [0]
            running_total = 0

            for loss in stage_losses:
                running_total += len(loss)
                inferred_boundaries.append(running_total)

            boundaries = inferred_boundaries

        else:
            boundaries = _manual_boundaries_for_flow(
                stage_boundaries=stage_boundaries,
                flow_idx=flow_idx,
                n_flows=n_flows,
                total=total,
                n_stage_names=len(names),
                boundary_unit=boundary_unit,
                iters_per_epoch_for_flow=(
                    None
                    if iters_per_epoch_by_flow is None
                    else iters_per_epoch_by_flow[flow_idx]
                ),
            )

        # Smooth loss.
        win = smooth_window if smooth_window is not None else max(1, total // 200)
        win = int(win)

        if win < 1:
            raise ValueError("smooth_window must be >= 1.")

        if win > total:
            raise ValueError(
                f"smooth_window={win} is larger than the number of loss values "
                f"for flow {flow_idx}, which is {total}."
            )

        smoothed = np.convolve(all_losses, np.ones(win) / win, mode='valid')

        # smoothed[i] is centered around original iteration i + win // 2.
        x_smooth = np.arange(win // 2, win // 2 + len(smoothed))

        color = flow_colors[flow_idx % len(flow_colors)]
        label = flow_labels[flow_idx] if flow_labels is not None else None

        # Stage boxes.
        if show_stage_shading:
            for i, (name, start, end) in enumerate(
                zip(names, boundaries[:-1], boundaries[1:])
            ):
                ax.axvspan(
                    start,
                    end,
                    alpha=0.12,
                    color=stage_colors[i % len(stage_colors)],
                )

        # Stage separator lines.
        if show_stage_lines:
            for boundary in boundaries[1:-1]:
                ax.axvline(
                    boundary,
                    color='gray',
                    ls='--',
                    lw=1.0,
                    alpha=0.7,
                )

        # Stage labels.
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        for name, start, end in zip(names, boundaries[:-1], boundaries[1:]):
            if flow_idx == 0 or start >= flow0_last_boundary:
                ax.text(
                    (start + end) / 2,
                    stage_label_y,
                    name,
                    transform=trans,
                    ha='center',
                    va='top',
                    fontsize=14,
                    alpha=0.85,
                )

        if flow_idx == 0:
            flow0_last_boundary = boundaries[-1]

        # Raw and smoothed trajectories.
        if show_raw:
            ax.plot(
                np.arange(total),
                all_losses,
                color=color,
                lw=0.4,
                alpha=0.3,
            )

        ax.plot(
            x_smooth,
            smoothed,
            color=color,
            lw=1.5,
            label=label,
        )

        # Optional gradient.
        if show_gradient:
            grad = np.gradient(smoothed)

            grad_smoothed = np.convolve(
                grad,
                np.ones(win) / win,
                mode='same',
            )

            ax2.plot(
                x_smooth,
                grad_smoothed,
                color=color,
                lw=1.0,
                ls='--',
                alpha=1.0,
                label=(
                    f'{label} gradient'
                    if label is not None
                    else 'gradient'
                ),
            )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(left=0)
    ax.set_xlabel('Batch iteration', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(title, fontsize=16)

    if flow_labels is not None:
        ax.legend(fontsize=14, loc='lower right')

    plt.tight_layout()
    plt.show()

    return fig, ax
"""
Evaluation helpers for fab-torch flow models on 3D molecular systems
(SoluteInWater with PBC or droplet boundary conditions).

Functions
---------
generate_flow_samples   -- batch-safe sample generation from a fab-torch flow
evaluate_single_model   -- repeated evaluation with full metrics for one flow
evaluate_models         -- compare an arbitrary collection of trained flows
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from .mol_analysis import (
    compute_rdf_3d,
    compute_min_distances,
    compute_mean_pairwise_distances,
    compute_water_geometry,
    density_entropy_3d,
)
from .load import load_model, load_checkpoint, load_md_data, build_target
from .visuals import (
    plot_mol_position_density,
    plot_mol_oo_distances,
    plot_water_geometry,
    plot_mol_energies,
)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _ess(log_w):
    """ESS and ESS/N from log importance weights (robust to -inf/nan)."""
    finite = log_w[np.isfinite(log_w)]
    if len(finite) < 2:
        return 0.0, 0.0
    ess = float(np.exp(2 * logsumexp(finite) - logsumexp(2 * finite)))
    return ess, ess / len(log_w)


def _ms(values):
    """(mean, std) of a list of scalars."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std())


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_flow_samples(flow, target, n_samples, batch_size=256):
    """
    Draw samples from a fab-torch flow model.

    Parameters
    ----------
    flow : trained flow (e.g. WrappedNormFlowModel)
    target : SoluteInWater (or compatible target with
             ``coordinate_transform.forward`` and ``log_prob``)
    n_samples : int
    batch_size : int
        Reduce if running out of memory.

    Returns
    -------
    x_cart : np.ndarray, shape (n_samples, N_ATOMS, 3)
        Cartesian coordinates (nm) in the target's canonical frame.
    log_q : np.ndarray, shape (n_samples,)
        Flow log-density log q(z).
    log_p : np.ndarray, shape (n_samples,)
        Target log-density log p(z)  (= −energy).
    """
    N_ATOMS = target.cartesian_dim // 3
    x_list, lq_list, lp_list = [], [], []

    with torch.no_grad():
        n_done = 0
        while n_done < n_samples:
            n_now = min(batch_size, n_samples - n_done)
            z, lq = flow.sample_and_log_prob((n_now,))
            x_c, _ = target.coordinate_transform.forward(z)
            lp = target.log_prob(z)
            x_list.append(x_c.float().detach().cpu())
            lq_list.append(lq.float().detach().cpu())
            lp_list.append(lp.float().detach().cpu())
            n_done += n_now

    x_cart = torch.cat(x_list).numpy().reshape(n_samples, N_ATOMS, 3)
    log_q  = torch.cat(lq_list).numpy()
    log_p  = torch.cat(lp_list).numpy()
    return x_cart, log_q, log_p


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(
    flow,
    target,
    x_md,
    u_md,
    N_SOLUTE_ATOMS,
    N_WATERS,
    O_IDX,
    H1_IDX,
    H2_IDX,
    L,
    IS_PBC,
    n_samples=5_000,
    n_repeats=5,
    batch_size=256,
    dr=0.005,
    visualize=True,
    print_summary=True,
):
    """
    Evaluate a single flow with ``n_repeats`` independent generation runs.

    Computes mean ± std over repeats for:
      - Solute–water and water–water RDF g(r)
      - Minimum pairwise distances (water–water, solute–water)
      - Water O-H bond lengths and H-O-H angles (pooled across all repeats)
      - Energy statistics (mean, std, finite fraction)
      - Effective sample size (ESS)

    Parameters
    ----------
    flow : trained fab-torch flow
    target : SoluteInWater (or compatible)
    x_md : np.ndarray, shape (N, N_ATOMS, 3) or None
        MD reference Cartesian trajectory (canonical frame).
    u_md : np.ndarray or None
        MD reference energies (= −log p).
    N_SOLUTE_ATOMS, N_WATERS : int
    O_IDX, H1_IDX, H2_IDX : list[int]
    L : float
        Box length (nm).
    IS_PBC : bool
    n_samples : int
        Samples per repeat.
    n_repeats : int
    batch_size : int
    dr : float
        RDF bin width (nm).
    visualize : bool
    print_summary : bool

    Returns
    -------
    results : dict
        Aggregated statistics and per-run arrays.  Key fields:

        * ``g_sw_mean``, ``g_sw_std`` -- solute–water RDF mean and std
        * ``g_ww_mean``, ``g_ww_std`` -- water–water RDF mean and std
        * ``d_ww_pooled``, ``d_sw_pooled`` -- concatenated per-frame min distances
        * ``oh_pooled``, ``hoh_pooled`` -- concatenated bond/angle arrays
        * ``u_gen_pooled`` -- concatenated energies from all repeats
        * ``ess_mean``, ``ess_frac_mean`` -- ESS statistics
        * ``x_gen_runs`` -- list of (n_samples, N_ATOMS, 3) arrays
        * ``md_stats`` -- reference statistics (None if no MD data given)
    """
    _L_rdf = L if IS_PBC else None

    # ------------------------------------------------------------------
    # MD reference statistics (computed once)
    # ------------------------------------------------------------------
    md_stats = None
    if x_md is not None:
        d_ww_md, d_sw_md = compute_min_distances(x_md, O_IDX, L, IS_PBC)
        d_ww_mean_md, d_sw_mean_md = compute_mean_pairwise_distances(x_md, O_IDX, L, IS_PBC)
        r_sw_md, g_sw_md = compute_rdf_3d(x_md, [0], O_IDX, _L_rdf, dr=dr)
        r_ww_md, g_ww_md = compute_rdf_3d(x_md, O_IDX, O_IDX, _L_rdf, dr=dr)
        oh_md, hoh_md = compute_water_geometry(x_md, N_WATERS, O_IDX, H1_IDX, H2_IDX, L, IS_PBC)
        H_md, _ = density_entropy_3d(x_md, O_IDX, L if IS_PBC else None)
        md_stats = {
            "r_sw": r_sw_md, "g_sw": g_sw_md,
            "r_ww": r_ww_md, "g_ww": g_ww_md,
            "d_ww": d_ww_md, "d_sw": d_sw_md,
            "d_ww_mean": d_ww_mean_md, "d_sw_mean": d_sw_mean_md,
            "oh":   oh_md,   "hoh":  hoh_md,
            "density_entropy": H_md,
        }
        if u_md is not None:
            u_md_arr = np.asarray(u_md)
            fin = np.isfinite(u_md_arr)
            md_stats["energy"] = {
                "mean": float(u_md_arr[fin].mean()),
                "std":  float(u_md_arr[fin].std()),
                "finite_frac": float(fin.mean()),
            }

    # ------------------------------------------------------------------
    # Repeated flow generation
    # ------------------------------------------------------------------
    all_g_sw, all_g_ww = [], []
    d_ww_min_scalars, d_sw_min_scalars = [], []
    d_ww_mean_scalars, d_sw_mean_scalars = [], []
    d_ww_arrays, d_sw_arrays = [], []
    oh_arrays, hoh_arrays = [], []
    u_gen_arrays = []
    ess_runs, ess_frac_runs = [], []
    entropy_runs = []
    x_gen_runs = []
    r_sw_ref = None   # same bin centres every repeat (same dr, same L)
    r_ww_ref = None

    for rep in range(n_repeats):
        x_gen, log_q, log_p = generate_flow_samples(
            flow, target, n_samples, batch_size=batch_size,
        )
        x_gen_runs.append(x_gen)

        # RDF
        r_sw, g_sw = compute_rdf_3d(x_gen, [0], O_IDX, _L_rdf, dr=dr)
        r_ww, g_ww = compute_rdf_3d(x_gen, O_IDX, O_IDX, _L_rdf, dr=dr)
        if r_sw_ref is None:
            r_sw_ref, r_ww_ref = r_sw, r_ww
        all_g_sw.append(g_sw)
        all_g_ww.append(g_ww)

        # Distances
        d_ww, d_sw = compute_min_distances(x_gen, O_IDX, L, IS_PBC)
        d_ww_min_scalars.append(float(d_ww.min()))
        d_sw_min_scalars.append(float(d_sw.min()))
        d_ww_arrays.append(d_ww)
        d_sw_arrays.append(d_sw)
        d_ww_mean, d_sw_mean = compute_mean_pairwise_distances(x_gen, O_IDX, L, IS_PBC)
        d_ww_mean_scalars.append(float(d_ww_mean.mean()))
        d_sw_mean_scalars.append(float(d_sw_mean.mean()))

        # Density entropy
        H_rep, _ = density_entropy_3d(x_gen, O_IDX, L if IS_PBC else None)
        entropy_runs.append(H_rep)

        # Geometry
        oh, hoh = compute_water_geometry(
            x_gen, N_WATERS, O_IDX, H1_IDX, H2_IDX, L, IS_PBC,
        )
        oh_arrays.append(oh)
        hoh_arrays.append(hoh)

        # Energy
        u_gen = -log_p
        u_gen_arrays.append(u_gen)
        fin = np.isfinite(u_gen)
        e_mean = float(u_gen[fin].mean()) if fin.any() else float("nan")
        e_std  = float(u_gen[fin].std())  if fin.any() else float("nan")

        # ESS
        log_w = log_p - log_q
        ess, ess_frac = _ess(log_w)
        ess_runs.append(ess)
        ess_frac_runs.append(ess_frac)

        if print_summary:
            print(
                f"  Repeat {rep+1}/{n_repeats}:  "
                f"energy = {e_mean:.1f} ± {e_std:.1f}  "
                f"finite = {fin.mean()*100:.1f}%  "
                f"ESS = {ess:.0f}/{n_samples} ({ess_frac*100:.1f}%)"
            )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    all_g_sw = np.asarray(all_g_sw)
    all_g_ww = np.asarray(all_g_ww)

    ess_mean,      ess_std      = _ms(ess_runs)
    ess_frac_mean, ess_frac_std = _ms(ess_frac_runs)
    d_ww_min_mean, d_ww_min_std = _ms(d_ww_min_scalars)
    d_sw_min_mean, d_sw_min_std = _ms(d_sw_min_scalars)
    d_ww_mean_mean, d_ww_mean_std = _ms(d_ww_mean_scalars)
    d_sw_mean_mean, d_sw_mean_std = _ms(d_sw_mean_scalars)
    entropy_mean, entropy_std = _ms(entropy_runs)

    u_gen_pooled  = np.concatenate(u_gen_arrays)
    fin_pooled    = np.isfinite(u_gen_pooled)
    energy_mean_v = float(u_gen_pooled[fin_pooled].mean()) if fin_pooled.any() else float("nan")
    energy_std_v  = float(u_gen_pooled[fin_pooled].std())  if fin_pooled.any() else float("nan")

    results = {
        "n_repeats":  n_repeats,
        "n_samples":  n_samples,
        "md_stats":   md_stats,
        # RDF
        "r_sw":          r_sw_ref,
        "g_sw_runs":     all_g_sw,
        "g_sw_mean":     all_g_sw.mean(axis=0),
        "g_sw_std":      all_g_sw.std(axis=0),
        "r_ww":          r_ww_ref,
        "g_ww_runs":     all_g_ww,
        "g_ww_mean":     all_g_ww.mean(axis=0),
        "g_ww_std":      all_g_ww.std(axis=0),
        # distances
        "d_ww_min_runs":  np.asarray(d_ww_min_scalars),
        "d_sw_min_runs":  np.asarray(d_sw_min_scalars),
        "d_ww_pooled":    np.concatenate(d_ww_arrays),
        "d_sw_pooled":    np.concatenate(d_sw_arrays),
        "d_ww_min_mean":  d_ww_min_mean,  "d_ww_min_std":  d_ww_min_std,
        "d_sw_min_mean":  d_sw_min_mean,  "d_sw_min_std":  d_sw_min_std,
        "d_ww_mean_runs": np.asarray(d_ww_mean_scalars),
        "d_sw_mean_runs": np.asarray(d_sw_mean_scalars),
        "d_ww_mean_mean": d_ww_mean_mean, "d_ww_mean_std": d_ww_mean_std,
        "d_sw_mean_mean": d_sw_mean_mean, "d_sw_mean_std": d_sw_mean_std,
        # density entropy
        "entropy_runs":  np.asarray(entropy_runs),
        "entropy_mean":  entropy_mean,
        "entropy_std":   entropy_std,
        # geometry
        "oh_pooled":   np.concatenate(oh_arrays),
        "hoh_pooled":  np.concatenate(hoh_arrays),
        # energy
        "u_gen_pooled":  u_gen_pooled,
        "energy_mean":   energy_mean_v,
        "energy_std":    energy_std_v,
        # ESS
        "ess_runs":       np.asarray(ess_runs),
        "ess_frac_runs":  np.asarray(ess_frac_runs),
        "ess_mean":       ess_mean,       "ess_std":       ess_std,
        "ess_frac_mean":  ess_frac_mean,  "ess_frac_std":  ess_frac_std,
        # generated configs
        "x_gen_runs":  x_gen_runs,
    }

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    if visualize:
        _colors = ["black", "steelblue", "orangered", "purple", "darkorange"]

        # Position density (last repeat)
        _ds = [(x_gen_runs[-1], f"Flow (repeat {n_repeats})")]
        if x_md is not None:
            _ds.insert(0, (x_md, "MD reference"))
        plot_mol_position_density(_ds, L, O_IDX, N_SOLUTE_ATOMS, IS_PBC)

        # RDF with ± std band
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (r, g_mean, g_std, title_rdf) in zip(axes, [
            (r_sw_ref, results["g_sw_mean"], results["g_sw_std"], "Solute(0) – Water-O RDF"),
            (r_ww_ref, results["g_ww_mean"], results["g_ww_std"], "Water-O – Water-O RDF"),
        ]):
            if md_stats is not None:
                r_ref = r_sw_ref if title_rdf.startswith("S") else r_ww_ref
                g_ref = md_stats["g_sw"] if title_rdf.startswith("S") else md_stats["g_ww"]
                ax.plot(r_ref, g_ref, lw=2, color=_colors[0], label="MD reference")
            ax.plot(r, g_mean, lw=2, color=_colors[1],
                    label=f"Flow mean ({n_repeats} runs)")
            ax.fill_between(r, np.maximum(g_mean - g_std, 0), g_mean + g_std,
                            color=_colors[1], alpha=0.2, label="Flow ± 1 std")
            ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal gas")
            ax.set_xlabel("r  (nm)", fontsize=12)
            ax.set_ylabel("g(r)", fontsize=12)
            ax.set_title(title_rdf, fontsize=16)
            if IS_PBC:
                ax.set_xlim(0, 0.5 * L)
            ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

        # Min distances
        _ww_d = [(results["d_ww_pooled"], f"Flow ({n_repeats} runs)")]
        _sw_d = [(results["d_sw_pooled"], f"Flow ({n_repeats} runs)")]
        if md_stats is not None:
            _ww_d.insert(0, (md_stats["d_ww"], "MD reference"))
            _sw_d.insert(0, (md_stats["d_sw"], "MD reference"))
        plot_mol_oo_distances(_ww_d, _sw_d)

        # Water geometry
        _oh_ds  = [(results["oh_pooled"],  f"Flow ({n_repeats} runs)", _colors[1])]
        _hoh_ds = [(results["hoh_pooled"], f"Flow ({n_repeats} runs)", _colors[1])]
        if md_stats is not None:
            _oh_ds.insert(0,  (md_stats["oh"],  "MD reference", _colors[0]))
            _hoh_ds.insert(0, (md_stats["hoh"], "MD reference", _colors[0]))
        plot_water_geometry(_oh_ds, _hoh_ds)

        # Energy
        u_ref = u_md if u_md is not None else None
        plot_mol_energies(results["u_gen_pooled"], u_ref,
                          title=f"Energy distribution  ({n_repeats} runs pooled)")

    # ------------------------------------------------------------------
    # Printed summary
    # ------------------------------------------------------------------
    if print_summary:
        _sep = "=" * 72
        print(f"\n{_sep}")
        print("MD reference statistics")
        print(_sep)
        if md_stats is not None:
            e = md_stats.get("energy", {})
            print(f"  energy mean ± std     : {e.get('mean', float('nan')):.2f} ± {e.get('std', float('nan')):.2f}")
            print(f"  energy finite frac    : {e.get('finite_frac', float('nan'))*100:.1f}%")
            print(f"  min d(water-water)    : {md_stats['d_ww'].min():.4f} nm")
            print(f"  min d(solute-water)   : {md_stats['d_sw'].min():.4f} nm")
            print(f"  mean d(water-water)   : {md_stats['d_ww_mean'].mean():.4f} nm")
            print(f"  mean d(solute-water)  : {md_stats['d_sw_mean'].mean():.4f} nm")
            print(f"  density entropy       : {md_stats['density_entropy']:.4f}")
        else:
            print("  (no MD reference provided)")

        print(f"\n{_sep}")
        print(f"Flow statistics averaged over {n_repeats} repeats × {n_samples} samples")
        print(_sep)
        print(f"  energy mean ± std     : {results['energy_mean']:.2f} ± {results['energy_std']:.2f}")
        print(f"  energy finite frac    : {np.isfinite(results['u_gen_pooled']).mean()*100:.1f}%")
        print(f"  ESS                   : {ess_mean:.0f} ± {ess_std:.0f}  "
              f"({ess_frac_mean*100:.2f}% ± {ess_frac_std*100:.2f}%)")
        print(f"  min d(water-water)    : {d_ww_min_mean:.4f} ± {d_ww_min_std:.4f} nm")
        print(f"  min d(solute-water)   : {d_sw_min_mean:.4f} ± {d_sw_min_std:.4f} nm")
        print(f"  mean d(water-water)   : {d_ww_mean_mean:.4f} ± {d_ww_mean_std:.4f} nm")
        print(f"  mean d(solute-water)  : {d_sw_mean_mean:.4f} ± {d_sw_mean_std:.4f} nm")
        print(f"  density entropy       : {entropy_mean:.4f} ± {entropy_std:.4f}")
        print(f"  RDF s-w peak          : {results['g_sw_mean'].max():.3f} ± {results['g_sw_std'][results['g_sw_mean'].argmax()]:.3f}")
        print(f"  RDF w-w peak          : {results['g_ww_mean'].max():.3f} ± {results['g_ww_std'][results['g_ww_mean'].argmax()]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def evaluate_models(
    md_path,
    model_specs,
    N_SOLUTE_ATOMS,
    N_WATERS,
    O_IDX,
    H1_IDX,
    H2_IDX,
    L,
    IS_PBC,
    n_md_compare=5000,
    n_samples=5_000,
    n_repeats=5,
    batch_size=256,
    dr=0.005,
    colors=None,
    device="cpu",
    platform="CPU",
):
    """
    Evaluate and compare multiple flow models.

    Parameters
    ----------
    model_specs : list of dict
        Each dict must contain:

        * ``"flow"``  -- trained fab-torch flow model
        * ``"label"`` -- display label for plots and the summary table

        Example::

            model_specs = [
                {"flow": flow_a, "label": "LGT model"},
                {"flow": flow_b, "label": "Baseline"},
            ]

    x_md : np.ndarray, shape (N, N_ATOMS, 3) or None
    u_md : np.ndarray or None
    N_SOLUTE_ATOMS, N_WATERS : int
    O_IDX, H1_IDX, H2_IDX : list[int]
    L : float
    IS_PBC : bool
    n_samples : int
        Samples per generation repeat.
    n_repeats : int
        Independent generation runs per model.
    batch_size : int
    dr : float
        RDF bin width (nm).
    colors : list[str] or None

    Returns
    -------
    eval_results : dict
        Keyed by label. Each value is a dict with ``"label"`` and
        ``"results"`` (full output of :func:`evaluate_single_model`).
    """
    if colors is None:
        colors = ["black", "steelblue", "orangered", "purple", "darkorange", "crimson"]

    eval_results = {}

    for spec in model_specs:
        run_dir = spec["run_dir"]
        target = build_target(run_dir, md_path, device, platform)
       
        label = spec["label"]
        cfg, flow = load_model(target, run_dir)
        _, checkpoint = load_checkpoint(run_dir, device)
        
        flow.load_state_dict(checkpoint["flow"])
        flow = flow.to(device)
        flow.eval()
        x_md, u_md = load_md_data(md_path, target, n_md_compare, device)
        
        print("\n" + "=" * 72)
        print(f"Model: {label}")
        print("Weights loaded.")
        print("=" * 72)

        results = evaluate_single_model(
            flow=flow,
            target=target,
            x_md=x_md,
            u_md=u_md,
            N_SOLUTE_ATOMS=N_SOLUTE_ATOMS,
            N_WATERS=N_WATERS,
            O_IDX=O_IDX,
            H1_IDX=H1_IDX,
            H2_IDX=H2_IDX,
            L=L,
            IS_PBC=IS_PBC,
            n_samples=n_samples,
            n_repeats=n_repeats,
            batch_size=batch_size,
            dr=dr,
            visualize=False,
            print_summary=True,
        )
        eval_results[label] = {"label": label, "flow": flow, "results": results}

    labels = list(eval_results.keys())

    # ------------------------------------------------------------------
    # Plot 1: Overlaid RDFs with ± std bands
    # ------------------------------------------------------------------
    _rdf_colors = colors[1:]  # reserve colors[0] for MD reference

    for panel, (r_key, g_key, g_std_key, title_rdf) in enumerate([
        ("r_sw", "g_sw_mean", "g_sw_std", "Solute(0) – Water-O RDF"),
        ("r_ww", "g_ww_mean", "g_ww_std", "Water-O – Water-O RDF"),
    ]):
        _, ax = plt.subplots(figsize=(8, 5))

        if x_md is not None:
            md_ref = eval_results[labels[0]]["results"]["md_stats"]
            if md_ref is not None:
                r_ref = md_ref["r_sw"] if panel == 0 else md_ref["r_ww"]
                g_ref = md_ref["g_sw"] if panel == 0 else md_ref["g_ww"]
                ax.plot(r_ref, g_ref, lw=2, color=colors[0], label="MD reference")

        for i, lbl in enumerate(labels):
            res = eval_results[lbl]["results"]
            r   = res[r_key]
            g   = res[g_key]
            gs  = res[g_std_key]
            c   = _rdf_colors[i % len(_rdf_colors)]
            ax.plot(r, g, lw=2, color=c,
                    label=f"{lbl}")

            ax.fill_between(r, np.maximum(g - gs, 0), g + gs, color=c, alpha=0.15)

        ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal gas")
        ax.set_xlabel("r  (nm)", fontsize=12)
        ax.set_ylabel("g(r)", fontsize=12)
        ax.set_title(title_rdf, fontsize=16)
        if IS_PBC:
            ax.set_xlim(0, 0.5 * L)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plot 2: Overlaid min-distance distributions
    # ------------------------------------------------------------------
    ww_dists, sw_dists = [], []
    if x_md is not None:
        md_ref = eval_results[labels[0]]["results"]["md_stats"]
        if md_ref is not None:
            ww_dists.append((md_ref["d_ww"], "MD reference"))
            sw_dists.append((md_ref["d_sw"], "MD reference"))

    for i, lbl in enumerate(labels):
        res = eval_results[lbl]["results"]
        ww_dists.append((res["d_ww_pooled"], lbl))
        sw_dists.append((res["d_sw_pooled"], lbl))

    plot_mol_oo_distances(ww_dists, sw_dists, colors=colors)

    # ------------------------------------------------------------------
    # Plot 3: Overlaid water geometry
    # ------------------------------------------------------------------
    oh_ds, hoh_ds = [], []
    if x_md is not None:
        md_ref = eval_results[labels[0]]["results"]["md_stats"]
        if md_ref is not None:
            oh_ds.append((md_ref["oh"],  "MD reference", colors[0]))
            hoh_ds.append((md_ref["hoh"], "MD reference", colors[0]))

    for i, lbl in enumerate(labels):
        res = eval_results[lbl]["results"]
        c = _rdf_colors[i % len(_rdf_colors)]
        oh_ds.append((res["oh_pooled"],  lbl, c))
        hoh_ds.append((res["hoh_pooled"], lbl, c))

    plot_water_geometry(oh_ds, hoh_ds)

    # ------------------------------------------------------------------
    # Plot 4: Overlaid energy distributions
    # ------------------------------------------------------------------
    u_ref = u_md if u_md is not None else None
    u_flow_all = np.concatenate(
        [eval_results[lbl]["results"]["u_gen_pooled"] for lbl in labels]
    )
    finite = np.isfinite(u_flow_all)
    x_max_e = float(np.percentile(u_flow_all[finite], 99)) if finite.any() else None
    if u_ref is not None:
        u_ref_fin = u_ref[np.isfinite(u_ref)]
        if len(u_ref_fin):
            x_max_e = max(x_max_e or 0.0, float(np.percentile(u_ref_fin, 99)))

    fig, ax = plt.subplots(figsize=(8, 4))
    if u_ref is not None:
        ep = u_ref[u_ref < x_max_e] if x_max_e else u_ref
        ax.hist(ep, bins=80, density=True, alpha=0.35, color=colors[0],
                label=f"MD reference ({len(ep)/len(u_ref)*100:.0f}% shown)")
    for i, lbl in enumerate(labels):
        e = eval_results[lbl]["results"]["u_gen_pooled"]
        ep = e[e < x_max_e] if x_max_e else e
        ax.hist(ep, bins=80, density=True, alpha=0.35,
                color=_rdf_colors[i % len(_rdf_colors)],
                label=f"{lbl} ({len(ep)/len(e)*100:.0f}% shown)")
    ax.set_xlabel("Reduced energy  (−log p)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Energy distribution  ({n_repeats} runs each)", fontsize=16)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    md_ref = eval_results[labels[0]]["results"]["md_stats"]
    col_w = 26

    print("\n" + "=" * (col_w * (len(labels) + 1) + 4))
    print(f"Comparison summary  ({n_repeats} runs × {n_samples} samples per model)")
    print("=" * (col_w * (len(labels) + 1) + 4))

    header_parts = [f"{'Metric':<28}"]
    if md_ref is not None:
        header_parts.append(f"{'MD reference':>{col_w}}")
    for lbl in labels:
        header_parts.append(f"{lbl[:col_w]:>{col_w}}")
    print(" | ".join(header_parts))
    print("-" * (col_w * (len(labels) + 1) + 4))

    def _row(name, md_val, model_vals):
        parts = [f"{name:<28}"]
        if md_ref is not None:
            parts.append(f"{md_val:>{col_w}}")
        for v in model_vals:
            parts.append(f"{v:>{col_w}}")
        print(" | ".join(parts))

    # ESS
    _row(
        "ESS/N (%)",
        "—",
        [f"{eval_results[l]['results']['ess_frac_mean']*100:.2f} ± "
         f"{eval_results[l]['results']['ess_frac_std']*100:.2f}%" for l in labels],
    )

    # Energy
    md_e_str = "—"
    if md_ref is not None and "energy" in md_ref:
        md_e_str = (f"{md_ref['energy']['mean']:.1f} ± "
                    f"{md_ref['energy']['std']:.1f}")
    _row(
        "Energy mean ± std",
        md_e_str,
        [f"{eval_results[l]['results']['energy_mean']:.1f} ± "
         f"{eval_results[l]['results']['energy_std']:.1f}" for l in labels],
    )

    # RDF peak solute-water
    md_sw_peak = f"{md_ref['g_sw'].max():.3f}" if md_ref else "—"
    _row(
        "RDF s-w peak",
        md_sw_peak,
        [f"{eval_results[l]['results']['g_sw_mean'].max():.3f} ± "
         f"{eval_results[l]['results']['g_sw_std'][eval_results[l]['results']['g_sw_mean'].argmax()]:.3f}"
         for l in labels],
    )

    # RDF peak water-water
    md_ww_peak = f"{md_ref['g_ww'].max():.3f}" if md_ref else "—"
    _row(
        "RDF w-w peak",
        md_ww_peak,
        [f"{eval_results[l]['results']['g_ww_mean'].max():.3f} ± "
         f"{eval_results[l]['results']['g_ww_std'][eval_results[l]['results']['g_ww_mean'].argmax()]:.3f}"
         for l in labels],
    )

    # Min distances
    md_dww_str = f"{md_ref['d_ww'].min():.4f}" if md_ref else "—"
    md_dsw_str = f"{md_ref['d_sw'].min():.4f}" if md_ref else "—"
    _row(
        "Min d(water-water) [nm]",
        md_dww_str,
        [f"{eval_results[l]['results']['d_ww_min_mean']:.4f} ± "
         f"{eval_results[l]['results']['d_ww_min_std']:.4f}" for l in labels],
    )
    _row(
        "Min d(solute-water) [nm]",
        md_dsw_str,
        [f"{eval_results[l]['results']['d_sw_min_mean']:.4f} ± "
         f"{eval_results[l]['results']['d_sw_min_std']:.4f}" for l in labels],
    )

    # Mean pairwise distances
    md_dww_mean_str = f"{md_ref['d_ww_mean'].mean():.4f}" if md_ref else "—"
    md_dsw_mean_str = f"{md_ref['d_sw_mean'].mean():.4f}" if md_ref else "—"
    _row(
        "Mean d(water-water) [nm]",
        md_dww_mean_str,
        [f"{eval_results[l]['results']['d_ww_mean_mean']:.4f} ± "
         f"{eval_results[l]['results']['d_ww_mean_std']:.4f}" for l in labels],
    )
    _row(
        "Mean d(solute-water) [nm]",
        md_dsw_mean_str,
        [f"{eval_results[l]['results']['d_sw_mean_mean']:.4f} ± "
         f"{eval_results[l]['results']['d_sw_mean_std']:.4f}" for l in labels],
    )

    # Density entropy
    md_entropy_str = f"{md_ref['density_entropy']:.4f}" if md_ref else "—"
    _row(
        "Density entropy (norm.)",
        md_entropy_str,
        [f"{eval_results[l]['results']['entropy_mean']:.4f} ± "
         f"{eval_results[l]['results']['entropy_std']:.4f}" for l in labels],
    )

    return eval_results

"""
General evaluation helpers for Boltzmann Generator models.

Functions
---------
generate_bg_samples   -- draw samples from a trained BG model
evaluate_single_model -- repeated-generation evaluation with full metrics for one model
evaluate_models       -- load and compare an arbitrary collection of trained models
evaluate_ablation     -- compare Stage 3 models over different overlap-penalty weights
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import utils
from . import visual as vs


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_bg_samples(model, n_gen, n_particles):
    """
    Draw samples from a trained BG model.

    Parameters
    ----------
    model : trained flow model
        Must expose ``.prior``, ``.generator``, and ``._add_fixed_solute``.
    n_gen : int
        Number of samples to generate.
    n_particles : int
        Total particle count (solute + solvent), used to reshape the output.

    Returns
    -------
    x_gen : np.ndarray, shape (n_gen, n_particles, 2)
        Full system coordinates (particle 0 = solute).
    x_solvent_gen : torch.Tensor, shape (n_gen, n_solvent*2)
        Solvent-only flat coordinates on CPU, detached (needed for energy eval).
    log_q : torch.Tensor, shape (n_gen,)
        Per-sample log flow density log q(x) = log p_prior(z) + log|det J|.
        Used to compute importance weights for ESS.
    """
    with torch.no_grad():
        z_gen = model.prior.sample((n_gen,))
        x_solvent_gen, log_det = model.generator(z_gen)
        log_prior = model.prior.log_prob(z_gen)
        log_q = (log_prior - log_det).detach().cpu()
        x_full_flat = model._add_fixed_solute(x_solvent_gen)
    x_gen = x_full_flat.detach().cpu().numpy().reshape(-1, n_particles, 2)
    x_solvent_gen = x_solvent_gen.detach().cpu()
    return x_gen, x_solvent_gen, log_q


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(
    model,
    solute_sys,
    xtraj,
    etraj,
    r_mc,
    gr_mc,
    N_PARTICLES,
    N_SOLVENT,
    L_BOX,
    n_gen=5_000,
    n_repeats=5,
    n_bins=40,
    entropy_bins=50,
    visualize=True,
    print_summary=True,
    mc_stride=5,
    energy_threshold=1000,
):
    """
    Evaluate a single BG model with ``n_repeats`` independent generation runs.

    Computes mean ± std over repeats for:
      - g(r), spatial density entropy, position density histograms
      - min/mean pairwise distances (solute-solvent, solvent-solvent)
      - energy statistics

    Parameters
    ----------
    model : trained BG model
    solute_sys : object
        Full system object; must implement ``get_energy(x)``.
    xtraj : np.ndarray, shape (n_frames, N_PARTICLES, 2)
        MC reference trajectory.
    etraj : array-like
        MC reference energies.
    r_mc, gr_mc : np.ndarray
        MC radial distribution function.
    N_PARTICLES, N_SOLVENT : int
    L_BOX : float
        Box half-width.
    n_gen : int
        Samples per repeat.
    n_repeats : int
        Number of independent generation runs.
    n_bins : int
        RDF histogram bins.
    entropy_bins : int
        2D density histogram bins for entropy.
    visualize : bool
        Produce summary plots.
    print_summary : bool
        Print a statistics table.
    mc_stride : int
        Frame stride for MC position density plot.

    Returns
    -------
    results : dict
        Aggregated statistics, per-run arrays, and generated structures.
        Key fields include ``gr_bg_mean``, ``gr_bg_std``, ``density_entropy_mean``,
        ``energy_mean``, ``x_gen_runs``, and the full ``mc_stats`` reference.
    """
    entropy_box = ((-L_BOX, L_BOX), (-L_BOX, L_BOX))

    # ------------------------------------------------------------------
    # MC reference statistics
    # ------------------------------------------------------------------
    H_mc, hist_mc = utils.density_entropy_2d(xtraj[:, 1:, :], bins=entropy_bins, box=entropy_box)
    mc_fraction_in_box = float(((xtraj > -L_BOX) & (xtraj < L_BOX)).all(axis=-1).mean())

    mc_min_sol_solv, mc_min_solv_solv = utils.distances(xtraj, measure="min")
    mc_mean_sol_solv, mc_mean_solv_solv = utils.distances(xtraj, measure="mean")
    mc_max_sol_solv, mc_max_solv_solv = utils.distances(xtraj, measure="max")

    mc_stats = {
        "fraction_in_box": mc_fraction_in_box,
        "coord_min": float(np.asarray(xtraj).min()),
        "coord_max": float(np.asarray(xtraj).max()),
        "density_entropy_2d_solvent_only": float(H_mc),
        "density_hist_2d_solvent_only": hist_mc,
        "min_solute_solvent": utils.summarize_array(mc_min_sol_solv),
        "min_solvent_solvent": utils.summarize_array(mc_min_solv_solv),
        "mean_solute_solvent": utils.summarize_array(mc_mean_sol_solv),
        "mean_solvent_solvent": utils.summarize_array(mc_mean_solv_solv),
        "max_solute_solvent": utils.summarize_array(mc_max_sol_solv),
        "max_solvent_solvent": utils.summarize_array(mc_max_solv_solv),
        "energy": {
            "mean": float(np.asarray(etraj).mean()),
            "std": float(np.asarray(etraj).std()),
            "min": float(np.asarray(etraj).min()),
            "max": float(np.asarray(etraj).max()),
            "frac_clipped_gt_1e7": float((np.asarray(etraj) > 1e7).mean()),
        },
    }

    # ------------------------------------------------------------------
    # Repeated BG generation
    # ------------------------------------------------------------------
    all_r_bg, all_gr_bg = [], []
    min_sol_solv_runs, min_solv_solv_runs = [], []
    min_sol_solv_arrays, min_solv_solv_arrays = [], []
    mean_sol_solv_runs, mean_solv_solv_runs = [], []
    mean_sol_solv_arrays, mean_solv_solv_arrays = [], []
    max_sol_solv_runs, max_solv_solv_runs = [], []
    max_sol_solv_arrays, max_solv_solv_arrays = [], []
    fraction_in_box_runs, coord_min_runs, coord_max_runs = [], [], []
    entropy_runs, hist_runs = [], []
    energy_runs, bg_energy_arrays = [], []
    ess_runs, ess_frac_runs = [], []
    x_gen_runs = []

    for repeat in range(n_repeats):
        x_gen, _, log_q = generate_bg_samples(model, n_gen, N_PARTICLES)
        x_gen_runs.append(x_gen)

        H_gen, hist_gen = utils.density_entropy_2d(
            x_gen[:, 1:, :], bins=entropy_bins, box=entropy_box,
        )
        entropy_runs.append(float(H_gen))
        hist_runs.append(hist_gen)

        r_bg, gr_bg = utils.compute_gr_solute(x_gen, N_PARTICLES, L_BOX, dim=2, n_bins=n_bins)
        all_r_bg.append(r_bg)
        all_gr_bg.append(gr_bg)

        fraction_in_box_runs.append(float(((x_gen > -L_BOX) & (x_gen < L_BOX)).all(axis=-1).mean()))
        coord_min_runs.append(float(x_gen.min()))
        coord_max_runs.append(float(x_gen.max()))

        bg_min_sol_solv, bg_min_solv_solv = utils.distances(x_gen, measure="min")
        min_sol_solv_runs.append(float(bg_min_sol_solv.min()))
        min_solv_solv_runs.append(float(bg_min_solv_solv.min()))
        min_sol_solv_arrays.append(bg_min_sol_solv)
        min_solv_solv_arrays.append(bg_min_solv_solv)

        bg_mean_sol_solv, bg_mean_solv_solv = utils.distances(x_gen, measure="mean")
        mean_sol_solv_runs.append(float(bg_mean_sol_solv.mean()))
        mean_solv_solv_runs.append(float(bg_mean_solv_solv.mean()))
        mean_sol_solv_arrays.append(bg_mean_sol_solv)
        mean_solv_solv_arrays.append(bg_mean_solv_solv)

        bg_max_sol_solv, bg_max_solv_solv = utils.distances(x_gen, measure="max")
        max_sol_solv_runs.append(float(bg_max_sol_solv.max()))
        max_solv_solv_runs.append(float(bg_max_solv_solv.max()))
        max_sol_solv_arrays.append(bg_max_sol_solv)
        max_solv_solv_arrays.append(bg_max_solv_solv)

        energy_result = utils.eval_energies(solute_sys, x_gen, etraj, verbose=False, energy_threshold=energy_threshold)
        energy_runs.append(energy_result["bg_stats"])
        bg_energy_arrays.append(energy_result["bg_energies"])

        log_w = -energy_result["bg_energies"] - log_q.numpy()
        ess, ess_frac = utils.effective_sample_size(log_w)
        ess_runs.append(ess)
        ess_frac_runs.append(ess_frac)

        if print_summary:
            print(f"\nFinished repeat {repeat + 1}/{n_repeats}")
            print(f"  fraction in box:      {fraction_in_box_runs[-1]:.4f}")
            print(f"  coord min/max:        {x_gen.min():.4f} / {x_gen.max():.4f}")
            print(f"  min solute-solvent:   {bg_min_sol_solv.min():.4f}")
            print(f"  min solvent-solvent:  {bg_min_solv_solv.min():.4f}")
            print(f"  mean solute-solvent:  {bg_mean_sol_solv.mean():.4f}")
            print(f"  mean solvent-solvent: {bg_mean_solv_solv.mean():.4f}")
            print(f"  max solute-solvent:   {bg_max_sol_solv.max():.4f}")
            print(f"  max solvent-solvent:  {bg_max_solv_solv.max():.4f}")
            print(f"  entropy solvent 2D:   {H_gen:.4f}")
            print(f"  energy mean:          {energy_result['bg_stats']['mean']:.4f}")
            print(f"  energy std:           {energy_result['bg_stats']['std']:.4f}")
            print(f"  energy max:           {energy_result['bg_stats']['max']:.4f}")
            print(f"  ESS:                  {ess:.1f} / {n_gen}  ({ess_frac * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    all_r_bg = np.asarray(all_r_bg)
    all_gr_bg = np.asarray(all_gr_bg)
    hist_runs = np.asarray(hist_runs)
    bg_energy_arrays = np.asarray(bg_energy_arrays)
    pooled_bg_energies = bg_energy_arrays.ravel()

    fib_mean, fib_std = utils.mean_and_std(fraction_in_box_runs)
    cmin_mean, cmin_std = utils.mean_and_std(coord_min_runs)
    cmax_mean, cmax_std = utils.mean_and_std(coord_max_runs)
    mss_mean, mss_std = utils.mean_and_std(min_sol_solv_runs)
    mvv_mean, mvv_std = utils.mean_and_std(min_solv_solv_runs)
    mss2_mean, mss2_std = utils.mean_and_std(mean_sol_solv_runs)
    mvv2_mean, mvv2_std = utils.mean_and_std(mean_solv_solv_runs)
    mss3_mean, mss3_std = utils.mean_and_std(max_sol_solv_runs)
    mvv3_mean, mvv3_std = utils.mean_and_std(max_solv_solv_runs)
    ent_mean, ent_std = utils.mean_and_std(entropy_runs)
    hist_mean = hist_runs.mean(axis=0)
    hist_std = hist_runs.std(axis=0)
    ess_mean, ess_std = utils.mean_and_std(ess_runs)
    ess_frac_mean, ess_frac_std = utils.mean_and_std(ess_frac_runs)

    energy_keys = list(energy_runs[0].keys())
    energy_mean = {k: float(np.mean([r[k] for r in energy_runs])) for k in energy_keys}
    energy_std = {k: float(np.std([r[k] for r in energy_runs])) for k in energy_keys}

    results = {
        "n_repeats": n_repeats,
        "n_gen": n_gen,
        "mc_stats": mc_stats,
        # g(r)
        "r_bg_runs": all_r_bg,
        "gr_bg_runs": all_gr_bg,
        "r_bg_mean": all_r_bg.mean(axis=0),
        "r_bg_std": all_r_bg.std(axis=0),
        "gr_bg_mean": all_gr_bg.mean(axis=0),
        "gr_bg_std": all_gr_bg.std(axis=0),
        # density entropy
        "density_entropy_runs": np.asarray(entropy_runs),
        "density_entropy_mean": ent_mean,
        "density_entropy_std": ent_std,
        "hist_bg_runs": hist_runs,
        "hist_bg_mean": hist_mean,
        "hist_bg_std": hist_std,
        "hist_mc": hist_mc,
        # scalar run arrays
        "fraction_in_box_runs": np.asarray(fraction_in_box_runs),
        "coord_min_runs": np.asarray(coord_min_runs),
        "coord_max_runs": np.asarray(coord_max_runs),
        "min_solute_solvent_runs": np.asarray(min_sol_solv_runs),
        "min_solvent_solvent_runs": np.asarray(min_solv_solv_runs),
        "min_solute_solvent_pooled": np.concatenate(min_sol_solv_arrays),
        "min_solvent_solvent_pooled": np.concatenate(min_solv_solv_arrays),
        "mean_solute_solvent_runs": np.asarray(mean_sol_solv_runs),
        "mean_solvent_solvent_runs": np.asarray(mean_solv_solv_runs),
        "mean_solute_solvent_pooled": np.concatenate(mean_sol_solv_arrays),
        "mean_solvent_solvent_pooled": np.concatenate(mean_solv_solv_arrays),
        "max_solute_solvent_runs": np.asarray(max_sol_solv_runs),
        "max_solvent_solvent_runs": np.asarray(max_solv_solv_runs),
        "max_solute_solvent_pooled": np.concatenate(max_sol_solv_arrays),
        "max_solvent_solvent_pooled": np.concatenate(max_solv_solv_arrays),

        # scalar means / stds
        "fraction_in_box_mean": fib_mean, "fraction_in_box_std": fib_std,
        "coord_min_mean": cmin_mean, "coord_min_std": cmin_std,
        "coord_max_mean": cmax_mean, "coord_max_std": cmax_std,
        "min_solute_solvent_mean": mss_mean, "min_solute_solvent_std": mss_std,
        "min_solvent_solvent_mean": mvv_mean, "min_solvent_solvent_std": mvv_std,
        "mean_solute_solvent_mean": mss2_mean, "mean_solute_solvent_std": mss2_std,
        "mean_solvent_solvent_mean": mvv2_mean, "mean_solvent_solvent_std": mvv2_std,
        "max_solute_solvent_mean": mss3_mean, "max_solute_solvent_std": mss3_std,
        "max_solvent_solvent_mean": mvv3_mean, "max_solvent_solvent_std": mvv3_std,
        # energy
        "energy_runs": energy_runs,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "bg_energy_arrays": bg_energy_arrays,
        "pooled_bg_energies": pooled_bg_energies,
        # ESS
        "ess_runs": np.asarray(ess_runs),
        "ess_frac_runs": np.asarray(ess_frac_runs),
        "ess_mean": ess_mean,
        "ess_std": ess_std,
        "ess_frac_mean": ess_frac_mean,
        "ess_frac_std": ess_frac_std,
        # generated structures
        "x_gen_runs": x_gen_runs,
    }

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    if visualize:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r_mc, gr_mc, lw=2, color="steelblue", label="MC reference")
        ax.plot(
            results["r_bg_mean"], results["gr_bg_mean"],
            lw=2, color="orangered", ls="--",
            label=f"BG mean over {n_repeats} runs",
        )
        ax.fill_between(
            results["r_bg_mean"],
            results["gr_bg_mean"] - results["gr_bg_std"],
            results["gr_bg_mean"] + results["gr_bg_std"],
            color="orangered", alpha=0.2, label="BG ± 1 std",
        )
        ax.axhline(1.0, color="k", ls=":", lw=0.8, label="Ideal gas")
        ax.set_xlabel("r  (nm)", fontsize=12)
        ax.set_ylabel("g(r)", fontsize=12)
        ax.set_title("Solute-Solvent g(r): MC vs BG", fontsize=18)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        x_max = np.percentile(etraj, 99) * 2
        vs.plot_energies(
            etraj, pooled_bg_energies, x_max,
            title=f"Energy distribution: MC vs BG pooled over {n_repeats} runs",
        )

        vs.plot_position_density_mc_vs_bg(
            xtraj=xtraj, x_gen=x_gen_runs[-1], l_box=L_BOX, mc_stride=mc_stride,
        )

        vs.plot_pretty_density_histograms(
            hist_mc=hist_mc, hist_bg_mean=hist_mean,
            H_mc=H_mc, H_bg_mean=ent_mean, H_bg_std=ent_std, l_box=L_BOX,
        )

    # ------------------------------------------------------------------
    # Printed summary
    # ------------------------------------------------------------------
    if print_summary:
        print("\n" + "=" * 78)
        print("MC reference statistics")
        print("=" * 78)
        print(f"fraction in box:       {mc_stats['fraction_in_box']:.4f}")
        print(f"coord min/max:         {mc_stats['coord_min']:.4f} / {mc_stats['coord_max']:.4f}")
        print(f"entropy solvent 2D:    {mc_stats['density_entropy_2d_solvent_only']:.4f}")
        print("\nMC distance statistics")
        print("-" * 78)
        for key in ["min_solute_solvent", "min_solvent_solvent",
                    "mean_solute_solvent", "mean_solvent_solvent"]:
            d = mc_stats[key]
            print(
                f"{key:22s}: {d['mean']:.4f} ± {d['std']:.4f} "
                f"(min {d['min']:.4f}, max {d['max']:.4f})"
            )
        print("\nMC energy statistics")
        print("-" * 78)
        for k, v in mc_stats["energy"].items():
            print(f"{k:24s}: {v:.4f}")

        print("\n" + "=" * 78)
        print(f"BG statistics averaged over {n_repeats} runs")
        print("=" * 78)
        print(f"fraction in box:       {fib_mean:.4f} ± {fib_std:.4f}")
        print(f"coord min/max:         {cmin_mean:.4f} / {cmax_mean:.4f}")
        print(f"entropy solvent 2D:    {ent_mean:.4f} ± {ent_std:.4f}")
        print(f"ESS:                   {ess_mean:.1f} ± {ess_std:.1f}  ({ess_frac_mean * 100:.2f}% ± {ess_frac_std * 100:.2f}%)")
        print("\nBG distance statistics")
        print("-" * 78)
        for name, mean_v, std_v in [
            ("min solute-solvent",   mss_mean,  mss_std),
            ("min solvent-solvent",  mvv_mean,  mvv_std),
            ("mean solute-solvent",  mss2_mean, mss2_std),
            ("mean solvent-solvent", mvv2_mean, mvv2_std),
        ]:
            print(f"{name:22s}: {mean_v:.4f} ± {std_v:.4f}")
        print(f"\nBG generated energy statistics  (mean ± std over {n_repeats} runs)")
        print("-" * 78)
        for k in ["mean", "max", "std"]:
            print(f"  energy {k:4s}: {energy_mean[k]:12.4f} ± {energy_std[k]:.4f}")
        print()
        for k in energy_mean:
            if k not in ("mean", "max", "std"):
                print(f"{k:24s}: {energy_mean[k]:.4f} ± {energy_std[k]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Multi-model comparison evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    model_specs,
    BoltzmannGenerator2D,
    solute_sys,
    xtraj,
    etraj,
    r_mc,
    gr_mc,
    N_PARTICLES,
    N_SOLVENT,
    DIM,
    L_BOX,
    n_gen=5_000,
    n_repeats=5,
    n_bins=60,
    entropy_bins=50,
    mc_stride=1,
    device="cpu",
    visualize=True,
    bg_extra_config=None,
    energy_threshold=1000,
    colors=None,
):
    """
    Load and evaluate an arbitrary collection of trained BG models.

    Each model is evaluated independently with
    :func:`evaluate_single_model` (``visualize=False``), then comparison
    plots are produced:

    1. Overlaid g(r) (mean ± std per model)
    2. Position density scatter grid  (MC | model_1 | model_2 | ...)
    3. Solvent-solvent min-distance histograms
    4. Printed summary table

    Parameters
    ----------
    model_specs : list of dict
        Each dict must contain:

        * ``"path"``      -- path to the checkpoint file
        * ``"label"``     -- display label for plots and the summary table
        * ``"w_overlap"`` -- overlap-penalty weight used during training
        * ``"flow_type"`` -- flow architecture string (e.g. ``"spline"``,
          ``"realnvp"``). Required per spec so that different flow types can
          be compared in one call.

        Example::

            model_specs = [
                {
                    "path": "Trained_models/Solute/run_A/model_rnvp_w5_N37_L3",
                    "label": "RealNVP  w=5",
                    "flow_type": "realnvp",
                    "w_overlap": 5.0,
                },
            ]

    BoltzmannGenerator2D : class
        BG wrapper class (from ``Library.boltzmann``).
    solute_sys : object
        System object passed to ``BG.build()`` and used for energy evaluation.
    xtraj : np.ndarray, shape (n_frames, N_PARTICLES, 2)
        MC reference trajectory.
    etraj : array-like
        MC reference energies.
    r_mc, gr_mc : np.ndarray
        MC radial distribution function.
    N_PARTICLES, N_SOLVENT, DIM : int
    L_BOX : float
        Box half-width.
    n_gen : int
        Samples per generation repeat.
    n_repeats : int
        Independent generation runs per model.
    n_bins : int
        RDF histogram bins.
    entropy_bins : int
        2D density histogram bins for entropy.
    mc_stride : int
        Frame stride for MC position density plot.
    device : str
        Torch device string.
    bg_extra_config : dict or None
        Extra keys merged into each model's BG config.

    Returns
    -------
    eval_results : dict
        Keyed by ``"label"``. Each value is a dict with ``"label"``,
        ``"model"``, and ``"results"`` (full output of
        :func:`evaluate_single_model`).
    """

    eval_results = {}

    for spec in model_specs:
        model_path = spec["path"]
        label = spec["label"]
        flow_type = spec["flow_type"]
        w_overlap = spec["w_overlap"]

        print("\n" + "=" * 90)
        print(f"Model  : {label}")
        print(f"Flow   : {flow_type}")
        print(f"Path   : {model_path}")
        print("=" * 90)

        # load_model expects a path without extension (it appends .yml itself);
        # accept paths with or without .pt for convenience
        model_path_base = model_path[:-3] if model_path.endswith('.pt') else model_path
        if not os.path.isfile(model_path_base) and not os.path.isfile(model_path_base + '.pt'):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        model, loss = utils.load_model(model_path_base, BoltzmannGenerator2D, solute_sys, flow_type, w_overlap, device)

        results = evaluate_single_model(
            model=model,
            solute_sys=solute_sys,
            xtraj=xtraj,
            etraj=etraj,
            r_mc=r_mc,
            gr_mc=gr_mc,
            N_PARTICLES=N_PARTICLES,
            N_SOLVENT=N_SOLVENT,
            L_BOX=L_BOX,
            n_gen=n_gen,
            n_repeats=n_repeats,
            n_bins=n_bins,
            entropy_bins=entropy_bins,
            visualize=False,
            print_summary=False,
            energy_threshold=energy_threshold,
        )

        eval_results[label] = {
            "label": label,
            "model": model,
            "loss": loss,
            "results": results,
        }

    labels = list(eval_results.keys())

    # Plot 0: Training losses
    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 3, 4))

    if len(labels) == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        ax = axes[i]

        loss_history = eval_results[label]["loss"]

        # Convert torch tensor to numpy/list if needed
        if hasattr(loss_history, "detach"):
            loss_history = loss_history.detach().cpu().numpy()

        ax.plot(loss_history, label=f"{label} loss")

        final_loss = loss_history[-1]
        ax.axhline(
            y=final_loss,
            ls="--",
            label=f"final = {final_loss:.3f}"
        )

        ax.set_title(f"{label}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.show()
    
    if visualize:
    

        # Plot 1: overlaid g(r) with ± std bands
        vs.visualize_rdf(
            r=[r_mc] + [eval_results[l]["results"]["r_bg_mean"] for l in labels],
            gr=[gr_mc] + [eval_results[l]["results"]["gr_bg_mean"] for l in labels],
            gr_std=[None] + [eval_results[l]["results"]["gr_bg_std"] for l in labels],
            label=["MC reference"] + labels,
            title=f"Solute-Solvent RDF  (mean ± std, {n_repeats} runs each)",
            colors=colors,
        )

        # Plot 2: position density grid  (MC | model_1 | model_2 | ...)
        datasets = [(xtraj[::mc_stride], "MC")] + [
            (eval_results[l]["results"]["x_gen_runs"][-1][::mc_stride], l) for l in labels
        ]
        vs.position_density(datasets, L_BOX, title="Position density: MC and models")

        # Plot 3: min distances (solute-solvent | solvent-solvent), all models vs MC
        mc_sol_solv_arr, mc_solv_solv_arr = utils.distances(xtraj, measure="min")
        vs.visualize_distances(
            ss=[mc_sol_solv_arr] + [eval_results[l]["results"]["min_solute_solvent_pooled"] for l in labels],
            vv=[mc_solv_solv_arr] + [eval_results[l]["results"]["min_solvent_solvent_pooled"] for l in labels],
            label=["MC reference"] + labels,
            sigma=1.1,
            measure="min",
            sub_titles=[
                "Solute-Solvent",
                "Solvent-Solvent",
            ], colors=colors
        )
        # Plot 4: mean distances (solute-solvent | solvent-solvent), all models vs MC
        mc_sol_solv_arr, mc_solv_solv_arr = utils.distances(xtraj, measure="mean")
        vs.visualize_distances(
            ss=[mc_sol_solv_arr] + [eval_results[l]["results"]["mean_solute_solvent_pooled"] for l in labels],
            vv=[mc_solv_solv_arr] + [eval_results[l]["results"]["mean_solvent_solvent_pooled"] for l in labels],
            label=["MC reference"] + labels,
            measure="mean",
            sub_titles=[
                "Solute-Solvent",
                "Solvent-Solvent",
            ], colors=colors
        )
        # Plot 4: max distances (solute-solvent | solvent-solvent), all models vs MC
        mc_sol_solv_arr, mc_solv_solv_arr = utils.distances(xtraj, measure="max")
        vs.visualize_distances(
            ss=[mc_sol_solv_arr] + [eval_results[l]["results"]["max_solute_solvent_pooled"] for l in labels],
            vv=[mc_solv_solv_arr] + [eval_results[l]["results"]["max_solvent_solvent_pooled"] for l in labels],
            label=["MC reference"] + labels,
            measure="max",
            sub_titles=[
                "Solute-Solvent",
                "Solvent-Solvent",
            ], colors=colors
        )

        # Plot 7: energy distributions (all models overlaid)
        all_bg_energies = [eval_results[l]["results"]["pooled_bg_energies"] for l in labels]
        etraj_arr = np.asarray(etraj)
        vs.plot_energies(
            energies=[etraj_arr] + all_bg_energies,
            label=["MC reference"] + labels,
            title=f"Energy distribution: MC vs models  ({n_repeats} runs each)",
            colors=colors,
        )

    # Summary table
    mc_stats = eval_results[labels[0]]["results"]["mc_stats"]
    mc_entropy = float(mc_stats["density_entropy_2d_solvent_only"])

    mc_sol_sol_min_arr, mc_solv_solv_min_arr = utils.distances(xtraj, measure="min")
    mc_min_sol_sol_mean = mc_sol_sol_min_arr.mean()
    mc_min_sol_sol_std = mc_sol_sol_min_arr.std()
    mc_min_solv_solv_mean = mc_solv_solv_min_arr.mean()
    mc_min_solv_solv_std = mc_solv_solv_min_arr.std()
    mc_mean_sol_sol_mean = mc_stats["mean_solute_solvent"]["mean"]
    mc_mean_sol_sol_std = mc_stats["mean_solute_solvent"]["std"]
    mc_mean_solv_solv_mean = mc_stats["mean_solvent_solvent"]["mean"]
    mc_mean_solv_solv_std = mc_stats["mean_solvent_solvent"]["std"]
    mc_max_sol_sol_mean = mc_stats["max_solute_solvent"]["mean"]
    mc_max_sol_sol_std = mc_stats["max_solute_solvent"]["std"]
    mc_max_solv_solv_mean = mc_stats["max_solvent_solvent"]["mean"]
    mc_max_solv_solv_std = mc_stats["max_solvent_solvent"]["std"]

    mc_energy = mc_stats["energy"]

    print("\n" + "=" * 290)
    print(f"Evaluation summary  ({n_repeats} generation runs per model)")
    print("=" * 290)
    print(
        f"{'model':>38s} | "
        f"{'g(r) peak':>9s} | "
        f"{'entropy H':>16s} | "
        f"{'frac phys':>15s} | "
        f"{'ESS (%)':>16s} | "
        f"{'min sol-solv':>17s} | "
        f"{'min solv-solv':>17s} | "
        f"{'mean sol-solv':>17s} | "
        f"{'mean solv-solv':>17s} | "
        f"{'max sol-solv':>17s} | "
        f"{'max solv-solv':>17s} | "
        f"{'energy mean':>22s} | "
        f"{'energy max':>22s} | "
        f"{'energy std':>22s}"
    )
    print("-" * 290)
    print(
        f"{'MC reference':>38s} | "
        f"{float(gr_mc.max()):9.3f} | "
        f"{mc_entropy:16.4f} | "
        f"{'—':>15s} | "
        f"{'—':>16s} | "
        f"{mc_min_sol_sol_mean:7.3f} ± {mc_min_sol_sol_std:<7.3f} | "
        f"{mc_min_solv_solv_mean:7.3f} ± {mc_min_solv_solv_std:<7.3f} | "
        f"{mc_mean_sol_sol_mean:7.3f} ± {mc_mean_sol_sol_std:<7.3f} | "
        f"{mc_mean_solv_solv_mean:7.3f} ± {mc_mean_solv_solv_std:<7.3f} | "
        f"{mc_max_sol_sol_mean:7.3f} ± {mc_max_sol_sol_std:<7.3f} | "
        f"{mc_max_solv_solv_mean:7.3f} ± {mc_max_solv_solv_std:<7.3f} | "
        f"{mc_energy['mean']:>22.2f} | "
        f"{mc_energy['max']:>22.2f} | "
        f"{mc_energy['std']:>22.2f}"
    )
    print("-" * 290)

    for label in labels:
        res = eval_results[label]["results"]
        short_label = label[:38]
        gr_peak = float(res["gr_bg_mean"].max())
        ent_m = res["density_entropy_mean"]
        ent_s = res["density_entropy_std"]
        frac_phys_m = res["energy_mean"].get("frac_physical_lt_1000", float("nan"))
        frac_phys_s = res["energy_std"].get("frac_physical_lt_1000", float("nan"))
        ess_frac_m = res["ess_frac_mean"] * 100
        ess_frac_s = res["ess_frac_std"] * 100
        min_sol_sol_m = res["min_solute_solvent_mean"]
        min_sol_sol_s = res["min_solute_solvent_std"]
        min_solv_solv_m = res["min_solvent_solvent_mean"]
        min_solv_solv_s = res["min_solvent_solvent_std"]
        mean_sol_sol_m = res["mean_solute_solvent_mean"]
        mean_sol_sol_s = res["mean_solute_solvent_std"]
        mean_solv_solv_m = res["mean_solvent_solvent_mean"]
        mean_solv_solv_s = res["mean_solvent_solvent_std"]
        max_sol_sol_m = res["max_solute_solvent_mean"]
        max_sol_sol_s = res["max_solute_solvent_std"]
        max_solv_solv_m = res["max_solvent_solvent_mean"]
        max_solv_solv_s = res["max_solvent_solvent_std"]
        e_mean_m = res["energy_mean"].get("mean", float("nan"))
        e_mean_s = res["energy_std"].get("mean", float("nan"))
        e_max_m = res["energy_mean"].get("max", float("nan"))
        e_max_s = res["energy_std"].get("max", float("nan"))
        e_std_m = res["energy_mean"].get("std", float("nan"))
        e_std_s = res["energy_std"].get("std", float("nan"))
        print(
            f"{short_label:>38s} | "
            f"{gr_peak:9.3f} | "
            f"{ent_m:7.4f} ± {ent_s:<6.4f} | "
            f"{frac_phys_m:6.3f} ± {frac_phys_s:<6.3f} | "
            f"{ess_frac_m:6.2f}% ± {ess_frac_s:<6.2f}% | "
            f"{min_sol_sol_m:7.3f} ± {min_sol_sol_s:<7.3f} | "
            f"{min_solv_solv_m:7.3f} ± {min_solv_solv_s:<7.3f} | "
            f"{mean_sol_sol_m:7.3f} ± {mean_sol_sol_s:<7.3f} | "
            f"{mean_solv_solv_m:7.3f} ± {mean_solv_solv_s:<7.3f} | "
            f"{max_sol_sol_m:7.3f} ± {max_sol_sol_s:<7.3f} | "
            f"{max_solv_solv_m:7.3f} ± {max_solv_solv_s:<7.3f} | "
            f"{e_mean_m:10.2f} ± {e_mean_s:<9.2f} | "
            f"{e_max_m:10.2f} ± {e_max_s:<9.2f} | "
            f"{e_std_m:10.2f} ± {e_std_s:<9.2f}"
        )

    return eval_results


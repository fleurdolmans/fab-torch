import json
import os
import pathlib
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from boltzmann_generators_3d.fab.target_distributions.solute_in_water_LJ import LJParticles
from boltzmann_generators_3d.experiments.logger_setup import setup_logger
from boltzmann_generators_3d.experiments.setup_run_LJ import setup_trainer_and_run_flow


SAVE_DIR = None


def setup_lj_plotter(cfg: DictConfig, target: LJParticles, buffer=None):
    def plot(fab_model, plot_dict):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        figs = []

        L = float(target.box_length_nm)
        n_solvent = int(target.n_solvent)
        n_atoms = target.cartesian_dim // 3
        n_solute = n_atoms - n_solvent
        loss_type = str(cfg.fab.loss_type)

        def wrap(x: torch.Tensor, L: float) -> torch.Tensor:
            return torch.remainder(x, L)

        def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
            L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
            return dx - L_t * torch.round(dx / L_t)

        def pairwise_min_distances_same_group(X: torch.Tensor, L: float):
            B, N, _ = X.shape
            if N < 2:
                return np.full(B, np.inf)

            dmin = torch.full((B,), float("inf"), device=X.device, dtype=X.dtype)
            for i in range(N):
                for j in range(i + 1, N):
                    dij = torch.linalg.norm(mic(X[:, i, :] - X[:, j, :], L), dim=-1)
                    dmin = torch.minimum(dmin, dij)
            return dmin.detach().cpu().numpy()

        def pairwise_min_distances_cross_group(A: torch.Tensor, B: torch.Tensor, L: float):
            d = mic(A[:, :, None, :] - B[:, None, :, :], L)
            d = torch.linalg.norm(d, dim=-1)
            return d.amin(dim=(1, 2)).detach().cpu().numpy()

        def rdf_solute_to_solvent(X_flat: torch.Tensor, L: float, n_solute: int, n_solvent: int, dr: float = 0.005):
            B, D = X_flat.shape
            X = X_flat.reshape(B, -1, 3)

            solute = X[:, :n_solute, :]
            solvent = X[:, n_solute:n_solute + n_solvent, :]

            d = mic(solvent[:, None, :, :] - solute[:, :, None, :], L)
            r = torch.linalg.norm(d, dim=-1).reshape(-1).detach().cpu().numpy()

            r_max = 0.5 * L
            nbins = int(np.floor(r_max / dr))
            edges = np.linspace(0.0, nbins * dr, nbins + 1)
            counts, _ = np.histogram(r, bins=edges)

            r_centers = 0.5 * (edges[:-1] + edges[1:])
            shell_vol = 4.0 * np.pi * (r_centers ** 2) * dr
            V = L ** 3
            rho = n_solvent / V
            expected = B * n_solute * rho * shell_vol

            g_r = counts / np.maximum(expected, 1e-12)
            if len(g_r) > 0:
                g_r[0] = 0.0
            return r_centers, g_r

        def make_centered_for_viz(X_flat: torch.Tensor, L: float):
            X = X_flat.reshape(X_flat.shape[0], -1, 3).clone()

            if hasattr(target.coordinate_transform, "reference_point"):
                ref = target.coordinate_transform.reference_point.to(device=X.device, dtype=X.dtype)  # (1,3)
                ref = ref.unsqueeze(0).expand(X.shape[0], -1, -1)  # (B,1,3)
            else:
                ref = X[:, 0:1, :]  # fallback

            X = mic(X - ref, L)
            return X

        flow_dtype = next(fab_model.flow.parameters()).dtype
        flow_device = next(fab_model.flow.parameters()).device

        if target.eval_mode == "val":
            if target.val_data_i is None or target.val_data_x is None:
                raise ValueError("Need target.val_data_i and target.val_data_x for MD reference.")
            md_i_all = target.val_data_i.reshape(-1, target.internal_dim)
            md_x_all = target.val_data_x.reshape(-1, target.cartesian_dim)
        elif target.eval_mode == "test":
            if target.test_data_i is None or target.test_data_x is None:
                raise ValueError("Need target.test_data_i and target.test_data_x for MD reference.")
            md_i_all = target.test_data_i.reshape(-1, target.internal_dim)
            md_x_all = target.test_data_x.reshape(-1, target.cartesian_dim)
        else:
            raise ValueError(f"Unknown eval_mode: {target.eval_mode}")

        md_i_all = md_i_all.to(device=flow_device, dtype=flow_dtype)
        md_x_all = md_x_all.to(device=flow_device, dtype=flow_dtype)

        n_eval = min(512, md_i_all.shape[0])
        n_diag = min(64, md_i_all.shape[0])
        n_viz = min(4, md_i_all.shape[0])

        # ============================================================
        # BASE TRANSPORT PLOTTER
        # ============================================================
        if loss_type == "base_transport":
            if target.train_data_i is None:
                raise ValueError("Need target.train_data_i for Unmapped/Base samples.")

            base_i_all = target.train_data_i.reshape(-1, target.internal_dim)
            base_i_all = base_i_all.to(device=flow_device, dtype=flow_dtype)

            n_eval = min(512, base_i_all.shape[0], md_i_all.shape[0])
            n_diag = min(64, base_i_all.shape[0], md_i_all.shape[0])
            n_viz = min(4, base_i_all.shape[0], md_i_all.shape[0])

            with torch.no_grad():
                unmapped_i = base_i_all[:n_eval]
                unmapped_x, _ = target.coordinate_transform.forward(unmapped_i)
                unmapped_x = wrap(unmapped_x, L)

                md_i = md_i_all[:n_eval]
                md_x = wrap(md_x_all[:n_eval], L)

                mapped_i, _ = fab_model.flow.forward_map(unmapped_i)
                mapped_x, _ = target.coordinate_transform.forward(mapped_i)
                mapped_x = wrap(mapped_x, L)

                md_lp = target.log_prob(md_i).detach().cpu().numpy()
                unmapped_lp = target.log_prob(unmapped_i).detach().cpu().numpy()
                mapped_lp = target.log_prob(mapped_i).detach().cpu().numpy()

                md_u = -md_lp
                unmapped_u = -unmapped_lp
                mapped_u = -mapped_lp

            print("[LP mean] MD:", float(md_lp.mean()))
            print("[LP mean] Unmapped:", float(unmapped_lp.mean()))
            print("[LP mean] Mapped:", float(mapped_lp.mean()))
            print("[LP mean] gain mapped-unmapped:", float((mapped_lp - unmapped_lp).mean()))

            mdX = md_x[:n_diag].reshape(n_diag, -1, 3)
            unmappedX = unmapped_x[:n_diag].reshape(n_diag, -1, 3)
            mappedX = mapped_x[:n_diag].reshape(n_diag, -1, 3)

            md_solute = mdX[:, :n_solute, :]
            md_solvent = mdX[:, n_solute:, :]
            unmapped_solute = unmappedX[:, :n_solute, :]
            unmapped_solvent = unmappedX[:, n_solute:, :]
            mapped_solute = mappedX[:, :n_solute, :]
            mapped_solvent = mappedX[:, n_solute:, :]

            md_min_ss = pairwise_min_distances_same_group(md_solvent, L)
            unmapped_min_ss = pairwise_min_distances_same_group(unmapped_solvent, L)
            mapped_min_ss = pairwise_min_distances_same_group(mapped_solvent, L)

            md_min_solv_solute = pairwise_min_distances_cross_group(md_solvent, md_solute, L)
            unmapped_min_solv_solute = pairwise_min_distances_cross_group(unmapped_solvent, unmapped_solute, L)
            mapped_min_solv_solute = pairwise_min_distances_cross_group(mapped_solvent, mapped_solute, L)

            r_md, g_md = rdf_solute_to_solvent(md_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)
            r_unmapped, g_unmapped = rdf_solute_to_solvent(unmapped_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)
            r_mapped, g_mapped = rdf_solute_to_solvent(mapped_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)

            fig = plt.figure(figsize=(10, 4))
            plt.hist(md_u, bins=50, alpha=0.45, label="MD")
            plt.hist(unmapped_u, bins=50, alpha=0.45, label="Unmapped")
            plt.hist(mapped_u, bins=50, alpha=0.45, label="Mapped")
            plt.xlabel("Reduced energy")
            plt.ylabel("count")
            plt.title("Energy comparison")
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

            fig = plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(md_lp, bins=50, alpha=0.45, label="MD")
            plt.hist(unmapped_lp, bins=50, alpha=0.45, label="Unmapped")
            plt.hist(mapped_lp, bins=50, alpha=0.45, label="Mapped")
            plt.xlabel("target log_prob")
            plt.ylabel("count")
            plt.title("Target log_prob")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.hist(mapped_lp - unmapped_lp, bins=50, alpha=0.75)
            plt.xlabel("mapped - unmapped target log_prob")
            plt.ylabel("count")
            plt.title("Transport gain")
            plt.tight_layout()
            figs.append(fig)

            fig = plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(r_md, g_md, label="MD")
            plt.plot(r_unmapped, g_unmapped, label="Unmapped")
            plt.plot(r_mapped, g_mapped, label="Mapped")
            plt.xlabel("r (nm)")
            plt.ylabel("g(r)")
            plt.title("Solute-solvent RDF")
            plt.xlim(0, 0.5 * L)
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.hist(md_min_ss, bins=40, alpha=0.45, label="MD")
            plt.hist(unmapped_min_ss, bins=40, alpha=0.45, label="Unmapped")
            plt.hist(mapped_min_ss, bins=40, alpha=0.45, label="Mapped")
            plt.xlabel("min solvent-solvent distance (nm)")
            plt.ylabel("count")
            plt.title("Closest solvent-solvent distance")
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.hist(md_min_solv_solute, bins=40, alpha=0.45, label="MD")
            plt.hist(unmapped_min_solv_solute, bins=40, alpha=0.45, label="Unmapped")
            plt.hist(mapped_min_solv_solute, bins=40, alpha=0.45, label="Mapped")
            plt.xlabel("min solvent-solute distance (nm)")
            plt.ylabel("count")
            plt.title("Closest solvent-solute distance")
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

            md_x_diag = md_x[:n_viz]
            unmapped_x_diag = unmapped_x[:n_viz]
            mapped_x_diag = mapped_x[:n_viz]

            md_viz = make_centered_for_viz(md_x_diag, L).detach().cpu().numpy()
            unmapped_viz = make_centered_for_viz(unmapped_x_diag, L).detach().cpu().numpy()
            mapped_viz = make_centered_for_viz(mapped_x_diag, L).detach().cpu().numpy()

            md_e = md_u[:n_viz]
            unmapped_e = unmapped_u[:n_viz]
            mapped_e = mapped_u[:n_viz]

            def subplot_lj_system(ax, pos, energy, title_str):
                sol = pos[:n_solute]
                solv = pos[n_solute:]
                if len(solv) > 0:
                    s = cfg.target.solvent_sigma_nm ** 2
                    ax.scatter(solv[:, 0], solv[:, 1], solv[:, 2], alpha=0.35, s=s, label="solvent")
                if len(sol) > 0:
                    s = cfg.target.solute_sigma_nm ** 2
                    ax.scatter(sol[:, 0], sol[:, 1], sol[:, 2], s=s, label="solute")
                lim = 0.5 * L
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")
                ax.set_zlabel("z (nm)")
                
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                ax.set_title(f"{title_str}: {energy:.1f}")
                ax.view_init(elev=25, azim=40)
                ax.legend(loc="upper right")

            fig = plt.figure(figsize=(15, 3 * n_viz))
            for k in range(n_viz):
                ax = fig.add_subplot(n_viz, 3, 3 * k + 1, projection="3d")
                subplot_lj_system(ax, md_viz[k], md_e[k], f"MD {k+1}")

                ax = fig.add_subplot(n_viz, 3, 3 * k + 2, projection="3d")
                subplot_lj_system(ax, unmapped_viz[k], unmapped_e[k], f"Unmapped {k+1}")

                ax = fig.add_subplot(n_viz, 3, 3 * k + 3, projection="3d")
                subplot_lj_system(ax, mapped_viz[k], mapped_e[k], f"Mapped {k+1}")

            plt.tight_layout()
            figs.append(fig)

        # ============================================================
        # FORWARD KL AND REVERSE KL PLOTTER
        # ============================================================
        else:
            base_samples, base_logp = fab_model.flow.base(8)
            print("[BASE] sample shape:", tuple(base_samples.shape))
            print("[BASE] log_prob shape:", tuple(base_logp.shape))
            print("[BASE] mean/std:", base_samples.mean().item(), base_samples.std().item())

            if hasattr(fab_model.flow.base, "last_sampling_stats") and fab_model.flow.base.last_sampling_stats is not None:
                print("[BASE stats]", fab_model.flow.base.last_sampling_stats)
            with torch.no_grad():
                md_i = md_i_all[:n_eval]
                md_x = wrap(md_x_all[:n_eval], L)

                flow_i, flow_log_q = fab_model.flow.sample_and_log_prob((n_eval,))
                flow_x, _ = target.coordinate_transform.forward(flow_i)
                flow_x = wrap(flow_x, L)

                md_lp = target.log_prob(md_i).detach().cpu().numpy()
                flow_lp = target.log_prob(flow_i).detach().cpu().numpy()

                md_u = -md_lp
                flow_u = -flow_lp

                print("[debug] flow_i finite:", torch.isfinite(flow_i).all().item())
                print("[debug] flow_i min/max/mean:", float(torch.nan_to_num(flow_i).min()), float(torch.nan_to_num(flow_i).max()), float(torch.nan_to_num(flow_i).mean()))

                print("[debug] flow_x finite:", torch.isfinite(flow_x).all().item())
                print("[debug] flow_x min/max/mean:", float(torch.nan_to_num(flow_x).min()), float(torch.nan_to_num(flow_x).max()), float(torch.nan_to_num(flow_x).mean()))

                flow_lp_t = target.log_prob(flow_i)
                print("[debug] flow_lp finite:", torch.isfinite(flow_lp_t).all().item())
                print("[debug] flow_lp first 8:", flow_lp_t[:8])

                flow_lp = flow_lp_t.detach().cpu().numpy()

            print("[LP mean] MD:", float(md_lp.mean()))
            print("[LP mean] Flow:", float(flow_lp.mean()))
            print("[LP mean] Flow-MD:", float((flow_lp - md_lp).mean()))


            # ------------------------------------------------------------
            # Debugging sanity checks
            # ------------------------------------------------------------
            print("[target sanity] md log_prob[:8]:", target.log_prob(md_i[:8]))
            print("[target sanity] flow log_prob[:8]:", target.log_prob(flow_i[:8]))
            print("[target sanity] base log_prob[:8]:", target.log_prob(base_samples[:8]))

            x_test = md_x[:8]
            i2, ld1 = target.coordinate_transform.inverse(x_test)
            x2, ld2 = target.coordinate_transform.forward(i2)

            dx = mic(x_test.reshape(x_test.shape[0], -1, 3) - x2.reshape(x2.shape[0], -1, 3), L)

            print("[transform sanity] max MIC |x - f(g(x))|:", dx.abs().max().item())
            print("[transform sanity] mean MIC |x - f(g(x))|:", dx.abs().mean().item())
            print("[transform sanity] max |ld1 + ld2|:", (ld1 + ld2).abs().max().item())

            i_test = md_i[:8]
            x_from_i, ld_f = target.coordinate_transform.forward(i_test)
            i_back, ld_b = target.coordinate_transform.inverse(x_from_i)

            print("[transform internal sanity] max |i - g(f(i))|:", (i_test - i_back).abs().max().item())
            print("[transform internal sanity] mean |i - g(f(i))|:", (i_test - i_back).abs().mean().item())
            print("[transform internal sanity] max |ld_f + ld_b|:", (ld_f + ld_b).abs().max().item())

            x_s, lp_s = fab_model.flow.sample_and_log_prob((64,))
            lp_check = fab_model.flow.log_prob(x_s)
            print("[flow density consistency] max |sampled lp - recomputed lp|:",
                (lp_s - lp_check).abs().max().item())
            print("[flow density consistency] mean |sampled lp - recomputed lp|:",
                (lp_s - lp_check).abs().mean().item())

            base_lp_direct = fab_model.flow.base.log_prob(base_samples)
            print("[base consistency] max |returned logp - direct logp|:",
                (base_logp - base_lp_direct).abs().max().item())
            
            with torch.no_grad():
                z = base_samples[:8].clone()
                z_particle = fab_model.flow._flat_to_particle(z)

                print("[layer drift] base min pair distance:",
                    float(np.min(pairwise_min_distances_same_group(z_particle, L))))

                running = z_particle
                for li, layer in enumerate(fab_model.flow.layers):
                    running, _ = layer(running)
                    dmin = pairwise_min_distances_same_group(running, L)
                    print(f"[layer drift] after layer {li+1} min/mean:",
                        float(np.min(dmin)), float(np.mean(dmin)))


            mdX = md_x[:n_diag].reshape(n_diag, -1, 3)
            flowX = flow_x[:n_diag].reshape(n_diag, -1, 3)
            md_solute = mdX[:, :n_solute, :]
            md_solvent = mdX[:, n_solute:, :]
            flow_solute = flowX[:, :n_solute, :]
            flow_solvent = flowX[:, n_solute:, :]

            md_min_ss = pairwise_min_distances_same_group(md_solvent, L)
            flow_min_ss = pairwise_min_distances_same_group(flow_solvent, L)

            md_min_solv_solute = pairwise_min_distances_cross_group(md_solvent, md_solute, L)
            flow_min_solv_solute = pairwise_min_distances_cross_group(flow_solvent, flow_solute, L)

            print("[Distances] MD min solvent-solvent mean/median/min:",
                float(np.mean(md_min_ss)),
                float(np.median(md_min_ss)),
                float(np.min(md_min_ss)))

            print("[Distances] Flow min solvent-solvent mean/median/min:",
                float(np.mean(flow_min_ss)),
                float(np.median(flow_min_ss)),
                float(np.min(flow_min_ss)))

            print("[Distances] MD min solvent-solute mean/median/min:",
                float(np.mean(md_min_solv_solute)),
                float(np.median(md_min_solv_solute)),
                float(np.min(md_min_solv_solute)))

            print("[Distances] Flow min solvent-solute mean/median/min:",
                float(np.mean(flow_min_solv_solute)),
                float(np.median(flow_min_solv_solute)),
                float(np.min(flow_min_solv_solute)))

            mdX = md_x[:n_diag].reshape(n_diag, -1, 3)
            flowX = flow_x[:n_diag].reshape(n_diag, -1, 3)

            r_md, g_md = rdf_solute_to_solvent(md_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)
            r_flow, g_flow = rdf_solute_to_solvent(flow_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)

            fig = plt.figure(figsize=(10, 4))
            plt.hist(md_u, bins=50, alpha=0.45, label="MD")
            # plt.hist(flow_u, bins=50, alpha=0.45, label="Flow")
            flow_u_finite = flow_u[np.isfinite(flow_u)]
            if flow_u_finite.size > 0:
                plt.hist(flow_u_finite, bins=50, alpha=0.45, label="Flow")
            else:
                print("[plot] flow_u has no finite values")
            plt.xlabel("Reduced energy")
            plt.ylabel("count")
            plt.title("Energy comparison")
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

            fig = plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(md_lp, bins=50, alpha=0.45, label="MD")
            plt.hist(flow_lp, bins=50, alpha=0.45, label="Flow")
            plt.xlabel("target log_prob")
            plt.ylabel("count")
            plt.title("Target log_prob")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.hist(md_i.detach().cpu().numpy().ravel(), bins=100, alpha=0.45, label="MD")
            plt.hist(flow_i.detach().cpu().numpy().ravel(), bins=100, alpha=0.45, label="Flow")
            plt.xlabel("solvent coordinate value")
            plt.ylabel("count")
            plt.title("Solvent coordinate marginals")
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

            fig = plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(r_md, g_md, label="MD")
            plt.plot(r_flow, g_flow, label="Flow")
            plt.xlabel("r (nm)")
            plt.ylabel("g(r)")
            plt.title("Solute-solvent RDF")
            plt.xlim(0, 0.5 * L)
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.hist(md_min_ss, bins=40, alpha=0.45, label="MD")
            plt.hist(flow_min_ss, bins=40, alpha=0.45, label="Flow")
            plt.xlabel("min solvent-solvent distance (nm)")
            plt.ylabel("count")
            plt.title("Closest solvent-solvent distance")
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.hist(md_min_solv_solute, bins=40, alpha=0.45, label="MD")
            plt.hist(flow_min_solv_solute, bins=40, alpha=0.45, label="Flow")
            plt.xlabel("min solvent-solute distance (nm)")
            plt.ylabel("count")
            plt.title("Closest solvent-solute distance")
            plt.legend()
            plt.tight_layout()
            figs.append(fig)

            md_x_diag = md_x[:n_viz]
            flow_x_diag = flow_x[:n_viz]

            md_viz = make_centered_for_viz(md_x_diag, L).detach().cpu().numpy()
            flow_viz = make_centered_for_viz(flow_x_diag, L).detach().cpu().numpy()

            md_e = md_u[:n_viz]
            flow_e = flow_u[:n_viz]

            def subplot_lj_system(ax, pos, energy, title_str):
                sol = pos[:n_solute]
                solv = pos[n_solute:]
                if len(solv) > 0:
                    ax.scatter(solv[:, 0], solv[:, 1], solv[:, 2], alpha=0.35, s=20, label="solvent")
                if len(sol) > 0:
                    ax.scatter(sol[:, 0], sol[:, 1], sol[:, 2], s=20, label="solute")
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")
                ax.set_zlabel("z (nm)")
                ax.set_title(f"{title_str}: {energy:.1f}")
                ax.view_init(elev=25, azim=40)
                ax.legend(loc="upper right")

            fig = plt.figure(figsize=(10, 3 * n_viz))
            for k in range(n_viz):
                ax = fig.add_subplot(n_viz, 2, 2 * k + 1, projection="3d")
                subplot_lj_system(ax, md_viz[k], md_e[k], f"MD {k+1}")

                ax = fig.add_subplot(n_viz, 2, 2 * k + 2, projection="3d")
                subplot_lj_system(ax, flow_viz[k], flow_e[k], f"Flow {k+1}")

            plt.tight_layout()
            figs.append(fig)
        return figs

    return plot


def pick_system_keys(d: dict, keys: list) -> dict:
    if d is None:
        return None
    tgt = d.get("target", d)
    return {k: tgt.get(k) for k in keys if k in tgt}


def load_json(path):
    if not path:
        return None
    p = pathlib.Path(path).with_suffix(".json")
    if not p.exists():
        return None
    with p.open("r") as f:
        return json.load(f)


def overwrite_cfg(cfg, system_cfgs):
    if not system_cfgs:
        return cfg

    by_name = {name: sys for name, sys in system_cfgs}

    train_cfg = by_name.get("train")
    val_cfg = by_name.get("val")
    test_cfg = by_name.get("test")

    # Use train as the runtime/system config if present, otherwise first available.
    base_cfg = train_cfg if train_cfg is not None else next(iter(by_name.values()))

    # Only require compatibility, not full equality.
    compatible_keys = [
        "cartesian_dim",
        "temperature",
        "box_length_nm",
        "nonbonded_cutoff_nm",
        "n_solvent",
        "solvent_sigma_nm",
        "solvent_epsilon_kjmol",
        "solvent_mass_amu",
        "solute_mass_amu",
        "switch_nm",
        "constrain_solutes",
        "solute_solute_interaction",
        "grid_spacing_nm",
        "solid"

    ]

    for name, sys_cfg in by_name.items():
        for key in compatible_keys:
            if key in base_cfg and key in sys_cfg and base_cfg[key] != sys_cfg[key]:
                raise ValueError(
                    f"Incompatible train/val/test configs for transport: "
                    f"{name} differs on '{key}' "
                    f"(train/base={base_cfg[key]!r}, {name}={sys_cfg[key]!r})"
                )

    # Merge only the base/train runtime config into cfg.target.
    return OmegaConf.merge(cfg, OmegaConf.create({"target": base_cfg}))

def _run(cfg: DictConfig) -> None:
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    global SAVE_DIR
    SAVE_DIR = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(SAVE_DIR, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    logger = setup_logger(cfg, SAVE_DIR)

    platform_list = ["Reference", "CPU", "OpenCL", "CUDA", "None"]
    if cfg.target.platform_name == "CUDA":
        platform_properties = {"Precision": "mixed", "DeviceIndex": "0"}
    elif cfg.target.platform_name in platform_list:
        platform_properties = None
    else:
        raise NotImplementedError(cfg.target.platform_name)

    target = LJParticles(
        dim=cfg.target.cartesian_dim,
        n_solvent=cfg.target.n_solvent,
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
        box_length_nm=cfg.target.box_length_nm,
        solvent_sigma_nm=cfg.target.solvent_sigma_nm,
        solvent_epsilon_kjmol=cfg.target.solvent_epsilon_kjmol,
        solvent_mass_amu=cfg.target.solvent_mass_amu,
        solute_positions_nm=cfg.target.solute_positions_nm,
        solute_sigma_nm=cfg.target.solute_sigma_nm,
        solute_epsilon_kjmol=cfg.target.solute_epsilon_kjmol,
        solute_mass_amu=cfg.target.solute_mass_amu,
        switch_nm=cfg.target.switch_nm,
        cutoff_nm=cfg.target.nonbonded_cutoff_nm,
        constrain_solutes=cfg.target.constrain_solutes,
        solute_solute_interaction=cfg.target.solute_solute_interaction,
        seed=cfg.target.seed,
        grid_spacing_nm=cfg.target.grid_spacing_nm,
        platform_name=cfg.target.platform_name,
        platform_properties=platform_properties,
        transform_version=cfg.target.transform_version,
        curriculum_type=cfg.target.curriculum_type,
        curriculum_lambda=cfg.target.curriculum_lambda,
        curriculum_soft_energy_cut=cfg.target.curriculum_soft_energy_cut,
        solid=cfg.target.solid,
    )

    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    
    

    setup_trainer_and_run_flow(cfg, setup_lj_plotter, target)


@hydra.main(config_path="./config/", config_name="SoluteInSolvent", version_base="1.1")
def run(cfg: DictConfig) -> None:
    MD_KEYS = [
        "cartesian_dim",
        "temperature",
        "box_length_nm",
        "nonbonded_cutoff_nm",
        "n_solvent",
        "solvent_sigma_nm",
        "solvent_epsilon_kjmol",
        "solvent_mass_amu",
        "solute_positions_nm",
        "solute_epsilon_kjmol",
        "solute_mass_amu",
        "switch_nm",
        "constrain_solutes",
        "solute_solute_interaction",
        "grid_spacing_nm",
        "solid"
    ]

    train_data_config = load_json(cfg.target.train_samples_path)
    val_data_config = load_json(cfg.target.val_samples_path)
    test_data_config = load_json(cfg.target.test_samples_path)

    system_cfgs = []
    for name, dc in [("train", train_data_config), ("val", val_data_config), ("test", test_data_config)]:
        if dc is None:
            continue
        sys_part = pick_system_keys(dc, MD_KEYS)
        if not sys_part:
            raise ValueError(f"{name} config JSON exists but has none of the expected keys: {MD_KEYS}")
        system_cfgs.append((name, sys_part))

    cfg = overwrite_cfg(cfg, system_cfgs)
    print(OmegaConf.to_yaml(cfg))
    _run(cfg)


if __name__ == "__main__":
    run()
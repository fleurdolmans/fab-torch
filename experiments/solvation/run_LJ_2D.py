import json
import os
import pathlib
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fab.target_distributions.solute_in_water_LJ_2D import LJParticles2D
from experiments.logger_setup import setup_logger
from experiments.setup_run_LJ_2D import setup_trainer_and_run_flow


SAVE_DIR = None


def setup_lj_plotter_2d(cfg, target: LJParticles2D, buffer=None):
    def plot(fab_model, plot_dict):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        figs = []

        L = float(target.box_length_nm)
        n_solvent = int(target.n_solvent)
        n_particles = target.n_particles
        n_solute = target.n_solute
        loss_type = str(cfg.fab.loss_type)

        def wrap_unit(x):
            return torch.remainder(x, 1.0)

        def mic(dx, L):
            L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
            return dx - L_t * torch.round(dx / L_t)

        def pairwise_min_distances_same_group(X, L):
            B, N, _ = X.shape
            if N < 2:
                return np.full(B, np.inf)

            dmin = torch.full((B,), float("inf"), device=X.device, dtype=X.dtype)
            for i in range(N):
                for j in range(i + 1, N):
                    dij = torch.linalg.norm(mic(X[:, i, :] - X[:, j, :], L), dim=-1)
                    dmin = torch.minimum(dmin, dij)
            return dmin.detach().cpu().numpy()

        def pairwise_min_distances_cross_group(A, Bgrp, L):
            d = mic(A[:, :, None, :] - Bgrp[:, None, :, :], L)
            d = torch.linalg.norm(d, dim=-1)
            return d.amin(dim=(1, 2)).detach().cpu().numpy()

        def rdf_solute_to_solvent(X_flat, L, n_solute, n_solvent, dr=0.005):
            B, D = X_flat.shape
            X = X_flat.reshape(B, -1, 2)

            solute = X[:, :n_solute, :]
            solvent = X[:, n_solute:n_solute + n_solvent, :]

            d = mic(solvent[:, None, :, :] - solute[:, :, None, :], L)
            r = torch.linalg.norm(d, dim=-1).reshape(-1).detach().cpu().numpy()

            r_max = 0.5 * L
            nbins = int(np.floor(r_max / dr))
            edges = np.linspace(0.0, nbins * dr, nbins + 1)
            counts, _ = np.histogram(r, bins=edges)

            r_centers = 0.5 * (edges[:-1] + edges[1:])
            shell_area = 2.0 * np.pi * r_centers * dr
            area = L ** 2
            rho = n_solvent / area
            expected = B * n_solute * rho * shell_area

            g_r = counts / np.maximum(expected, 1e-12)
            if len(g_r) > 0:
                g_r[0] = 0.0
            return r_centers, g_r

        def make_centered_for_viz(X_flat, L):
            X = X_flat.reshape(X_flat.shape[0], -1, 2).clone()
            ref = X[:, 0:1, :]
            X = mic(X - ref, L)
            return X

        flow_dtype = next(fab_model.flow.parameters()).dtype
        flow_device = next(fab_model.flow.parameters()).device

        if target.eval_mode == "val":
            md_i_all = target.val_data_i.reshape(-1, target.internal_dim)
            md_x_all = target.val_data_x.reshape(-1, target.cartesian_dim)
        elif target.eval_mode == "test":
            md_i_all = target.test_data_i.reshape(-1, target.internal_dim)
            md_x_all = target.test_data_x.reshape(-1, target.cartesian_dim)
        else:
            raise ValueError(f"Unknown eval_mode: {target.eval_mode}")

        md_i_all = md_i_all.to(device=flow_device, dtype=flow_dtype)
        md_x_all = md_x_all.to(device=flow_device, dtype=flow_dtype)

        n_eval = min(512, md_i_all.shape[0])
        n_diag = min(64, md_i_all.shape[0])
        n_viz = min(4, md_i_all.shape[0])

        with torch.no_grad():
            md_i = md_i_all[:n_eval]
            md_x = md_x_all[:n_eval]

            flow_i, flow_log_q = fab_model.flow.sample_and_log_prob((n_eval,))
            flow_x, _ = target.coordinate_transform.forward(flow_i)

            md_lp = target.log_prob(md_i).detach().cpu().numpy()
            flow_lp = target.log_prob(flow_i).detach().cpu().numpy()

            md_u = -md_lp
            flow_u = -flow_lp

        mdX = md_x[:n_diag].reshape(n_diag, -1, 2)
        flowX = flow_x[:n_diag].reshape(n_diag, -1, 2)

        md_solute = mdX[:, :n_solute, :]
        md_solvent = mdX[:, n_solute:, :]
        flow_solute = flowX[:, :n_solute, :]
        flow_solvent = flowX[:, n_solute:, :]

        md_min_ss = pairwise_min_distances_same_group(md_solvent, L)
        flow_min_ss = pairwise_min_distances_same_group(flow_solvent, L)

        md_min_solv_solute = pairwise_min_distances_cross_group(md_solvent, md_solute, L)
        flow_min_solv_solute = pairwise_min_distances_cross_group(flow_solvent, flow_solute, L)

        r_md, g_md = rdf_solute_to_solvent(md_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)
        r_flow, g_flow = rdf_solute_to_solvent(flow_x, L, n_solute=n_solute, n_solvent=n_solvent, dr=0.005)

        fig = plt.figure(figsize=(10, 4))
        plt.hist(md_u, bins=50, alpha=0.45, label="MD")
        plt.hist(flow_u[np.isfinite(flow_u)], bins=50, alpha=0.45, label="Flow")
        plt.xlabel("Reduced energy")
        plt.ylabel("count")
        plt.title("Energy comparison")
        plt.legend()
        plt.tight_layout()
        figs.append(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(md_lp, bins=50, alpha=0.45, label="MD")
        plt.hist(flow_lp[np.isfinite(flow_lp)], bins=50, alpha=0.45, label="Flow")
        plt.xlabel("target log_prob")
        plt.ylabel("count")
        plt.title("Target log_prob")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(md_i.detach().cpu().numpy().ravel(), bins=100, alpha=0.45, label="MD")
        plt.hist(flow_i.detach().cpu().numpy().ravel(), bins=100, alpha=0.45, label="Flow")
        plt.xlabel("internal coordinate value")
        plt.ylabel("count")
        plt.title("Internal coordinate marginals")
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
                ax.scatter(solv[:, 0], solv[:, 1], alpha=0.35, s=20, label="solvent")
            if len(sol) > 0:
                ax.scatter(sol[:, 0], sol[:, 1], s=40, label="solute")

            lim = 0.5 * L
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            ax.set_title(f"{title_str}: {energy:.1f}")
            ax.legend(loc="upper right")

        fig = plt.figure(figsize=(10, 3 * n_viz))
        for k in range(n_viz):
            ax = fig.add_subplot(n_viz, 2, 2 * k + 1)
            subplot_lj_system(ax, md_viz[k], md_e[k], f"MD {k+1}")

            ax = fig.add_subplot(n_viz, 2, 2 * k + 2)
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

    target = LJParticles2D(
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
        box_length_nm=cfg.target.box_length_nm,
        solvent_sigma_nm=cfg.target.solvent_sigma_nm,
        solvent_epsilon_kjmol=cfg.target.solvent_epsilon_kjmol,
        solute_sigma_nm=cfg.target.solute_sigma_nm,
        solute_epsilon_kjmol=cfg.target.solute_epsilon_kjmol,
        n_solute=1,
        transform_version=cfg.target.transform_version,
    )

    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    
    

    setup_trainer_and_run_flow(cfg, setup_lj_plotter_2d, target)


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
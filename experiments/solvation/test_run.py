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

import numpy as np
import torch
import random
import pathlib
import os
import json

from experiments.logger_setup import setup_logger
from fab.target_distributions.solute_in_water import SoluteInWater

def run_transform_test_droplet(cfg):
    """
    Transform test for droplet systems. Checks that:
      1) inverse() runs and returns finite values
      2) roundtrip X -> I -> X_rec matches X up to PBC images (MIC)
      3) energy(X) == energy(X_rec) when both are canonicalized consistently for OpenMM PBC
    """
    # --- Build target exactly like training does ---
    platform_properties = None
    if cfg.target.platform_name == "CUDA":
        platform_properties = {"Precision": "mixed", "DeviceIndex": "0"}

    device = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
    print("device:", device)

    target = SoluteInWater(
        solute_pdb_path=cfg.target.solute_pdb_path,
        solute_xml_path=cfg.target.solute_xml_path,
        solute_inpcrd_path=cfg.target.solute_inpcrd_path,
        solute_prmtop_path=cfg.target.solute_prmtop_path,
        dim=cfg.target.cartesian_dim,
        num_solvent_molecules=cfg.target.num_solvent_molecules,
        temperature=cfg.target.temperature,
        energy_cut=cfg.target.energy_cut,
        energy_max=cfg.target.energy_max,
        n_threads=1,  # deterministic debug
        train_samples_path=cfg.target.train_samples_path,
        val_samples_path=cfg.target.val_samples_path,
        test_samples_path=cfg.target.test_samples_path,
        eval_mode="val",
        device=device,
        logger=None,
        save_dir=".",
        boundary_condition=cfg.target.boundary_condition,
        box_length_nm=cfg.target.box_length_nm,
        nonbonded_cutoff_nm=cfg.target.nonbonded_cutoff_nm,
        rigid_water=cfg.target.rigid_water,
        internal_constraints=cfg.target.internal_constraints,
        external_constraints=cfg.target.external_constraints,
        constraint_radius=cfg.target.constraint_radius,
        constraint_force=cfg.target.constraint_force,
        platform_name=cfg.target.platform_name,
        platform_properties=platform_properties,
    )

    if cfg.target.boundary_condition != "droplet":
        raise ValueError(f"This test is for droplet only, got boundary_condition={cfg.target.boundary_condition}")
    
    if target.val_data_x is None:
        print("No val frames loaded. Nothing to test.")
        return
    
    # Number of Test Frames
    B = 128
    X = target.val_data_x.reshape(-1, target.cartesian_dim)[:B].to(device)

    print(f"Testing {X.shape[0]} frames...\n")

    print("\n================ TRANSFORM TEST ================\n")
    # ------------------------------------------------
    # 1) inverse does not crash + finite
    # ------------------------------------------------
    try:
        with torch.no_grad():
            I, logdet = target.coordinate_transform.inverse(X)
        print("✓ inverse() ran successfully")
    except Exception as e:
        print("✗ inverse() FAILED")
        print(e)
        return

    if torch.isfinite(I).all() and torch.isfinite(logdet).all():
        print("✓ inverse outputs finite values")
    else:
        print("✗ inverse produced NaN/Inf")
        return
    # ------------------------------------------------
    # 2) Roundtrip consistency
    # ------------------------------------------------
    with torch.no_grad():
        X_rec, _ = target.coordinate_transform.forward(I)

        # X_coord is the "canonical" Cartesian that cartesian_to_z() uses internally:
        # typically: made whole, centered on atom0, and rotated (if your transform does rotation).
        _, _, X_coord, _ = target.coordinate_transform.cartesian_to_z(X, setup=False)

        err_coord = (X_rec - X_coord).abs().max().item()
        print(f"Roundtrip max abs error vs X_coord (nm): {err_coord:.3e}")

        # Optional: if your transform restores translation/rotation to raw frame, you can also check this.
        err_raw = (X_rec - X).abs().max().item()
        print(f"Roundtrip max abs error vs raw X (nm):     {err_raw:.3e}")

        # Decide which one is the actual criterion:
        # - If your transform removes translation (and doesn't restore), err_coord should be tiny.
        # - If you restore translation exactly, err_raw can be tiny too.
        ok = err_coord < 1e-6

    print("✓ roundtrip OK" if ok else "✗ roundtrip FAILED")

    # ------------------------------------------------
    # 3) Energy consistency
    # ------------------------------------------------
    with torch.no_grad():
        # IMPORTANT: never feed *centered* coords to OpenMM unless you consistently do so for both.
        # For droplet systems, energy is NOT translationally invariant if you have any external constraints.
        # Even without external constraints, it *should* be invariant, but if you used any centering during inverse,
        # compare energies on the same representation used for roundtrip (X_coord vs X_rec).

        # Use the same representation as the roundtrip criterion:
        Ux = -target.p.log_prob_x(X_coord)     # energy on canonical coords
        Urec = -target.p.log_prob_x(X_rec)     # energy on reconstructed coords

        diff = Urec - Ux
        mean_diff = diff.mean().item()
        max_abs_diff = diff.abs().max().item()

        print("[DEBUG] Ux(coord) mean/max/min:", Ux.mean().item(), Ux.max().item(), Ux.min().item(), flush=True)
        print("[DEBUG] Urec      mean/max/min:", Urec.mean().item(), Urec.max().item(), Urec.min().item(), flush=True)
        print("[DEBUG] ΔU        mean/maxabs/min:",
              mean_diff, max_abs_diff, diff.min().item(), flush=True)

    print(f"Energy difference U(X_rec) - U(X_coord): mean={mean_diff:.3e}, maxabs={max_abs_diff:.3e}")

    # Droplet energy consistency tolerance:
    # - If you have no external constraints and use double precision, 1e-3 kBT is often achievable.
    # - If you have constraints / mixed precision / etc, you may need 1e-2 kBT.
    tol = 1e-3
    if max_abs_diff < tol:
        print("✓ energy consistency OK")
    else:
        print("✗ energy mismatch detected")

    print("\n================ TEST DONE =================\n")

def run_test_pbc(cfg: DictConfig) -> None:
    # Seeds
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    # Platform props (copy of your logic)
    platform_list = ["Reference", "CPU", "OpenCL", "CUDA", "None"]
    if cfg.target.platform_name == "CUDA":
        platform_properties = {
            "Precision": "mixed",
            "DeviceIndex": "0",
        }
    elif cfg.target.platform_name in platform_list:
        platform_properties = None
    else:
        raise NotImplementedError(
            f"Platform {cfg.target.platform_name} not implemented."
        )

    device = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
    print("[TEST_PBC] device:", device, flush=True)

    # --- sanity: require pbc ---
    if cfg.target.boundary_condition != "pbc":
        raise ValueError("[TEST_PBC] cfg.target.boundary_condition must be 'pbc'.")

    # Build target
    target = SoluteInWater(
        solute_pdb_path=cfg.target.solute_pdb_path,
        solute_xml_path=cfg.target.solute_xml_path,
        solute_inpcrd_path=cfg.target.solute_inpcrd_path,
        solute_prmtop_path=cfg.target.solute_prmtop_path,
        dim=cfg.target.cartesian_dim,
        num_solvent_molecules=cfg.target.num_solvent_molecules,
        temperature=cfg.target.temperature,
        energy_cut=cfg.target.energy_cut,
        energy_max=cfg.target.energy_max,
        n_threads=1,  # deterministic debug
        train_samples_path=cfg.target.train_samples_path,
        val_samples_path=cfg.target.val_samples_path,
        test_samples_path=cfg.target.test_samples_path,
        eval_mode="val",
        device=device,
        logger=None,
        save_dir=".",
        boundary_condition=cfg.target.boundary_condition,
        box_length_nm=cfg.target.box_length_nm,
        nonbonded_cutoff_nm=cfg.target.nonbonded_cutoff_nm,
        rigid_water=cfg.target.rigid_water,
        internal_constraints=cfg.target.internal_constraints,
        external_constraints=cfg.target.external_constraints,
        constraint_radius=cfg.target.constraint_radius,
        constraint_force=cfg.target.constraint_force,
        platform_name=cfg.target.platform_name,
        platform_properties=platform_properties,
    )
    print("atom order", target.system.atoms[:20])

    if cfg.target.boundary_condition != "pbc":
        raise ValueError(f"This test is for PBC only, got boundary_condition={cfg.target.boundary_condition}")

    if target.val_data_x is None:
        print("No val frames loaded. Nothing to test.")
        return


    # Take a small batch
    B = 64
    X = target.val_data_x.reshape(-1, target.cartesian_dim)[:B].to(device)
    print(f"Testing {X.shape[0]} frames...\n", flush=True)

    L = float(cfg.target.box_length_nm)
    n_solute = 3
    n_waters = int(cfg.target.num_solvent_molecules)

    print("\n================ PBC TEST ================\n", flush=True)

    # ------------------------------------------------
    # 1) preprocessing does not crash + finite
    # ------------------------------------------------
    try:
        with torch.no_grad():
            Xp = target.coordinate_transform.forward(X)
        print("✓ coordinate_transform.forward() ran successfully", flush=True)
    except Exception as e:
        print("✗ coordinate_transform.forward() FAILED", flush=True)
        print(e, flush=True)
        return

    if torch.isfinite(Xp).all():
        print("✓ preprocess outputs finite values", flush=True)
    else:
        print("✗ preprocess produced NaN/Inf", flush=True)
        return

    # reshape for geometry
    N = Xp.shape[1] // 3
    Xp3 = Xp.view(-1, N, 3)

    # ------------------------------------------------
    # 2) water O-H bonds sanity
    # ------------------------------------------------
    with torch.no_grad():
        # water block starts at n_solute, each water: O,H,H
        start = n_solute
        # collect all O-H distances into one big array (B * n_waters * 2)
        dists = []
        for w in range(n_waters):
            i = start + 3 * w
            O = Xp3[:, i + 0, :]
            H1 = Xp3[:, i + 1, :]
            H2 = Xp3[:, i + 2, :]
            d1 = torch.norm(target.coordinate_transform.mic(H1 - O, L), dim=-1)
            d2 = torch.norm(target.coordinate_transform.mic(H2 - O, L), dim=-1)
            dists.append(d1)
            dists.append(d2)
        d = torch.cat(dists, dim=0)  # (2 * B * n_waters,)

        d_cpu = d.detach().cpu().numpy()
        print(
            f"[OH] mean={d_cpu.mean():.6f} std={d_cpu.std():.6f} "
            f"p1={np.quantile(d_cpu, 0.01):.6f} p99={np.quantile(d_cpu, 0.99):.6f} "
            f"min={d_cpu.min():.6f} max={d_cpu.max():.6f}  (nm)",
            flush=True
        )

    # ------------------------------------------------
    # 3) energies finite
    # ------------------------------------------------
    with torch.no_grad():
        U0 = -target.p.log_prob_x(Xp)
        print(
            f"[ENERGY] U0(kBT) mean={U0.mean().item():.3f} "
            f"max={U0.max().item():.3f} min={U0.min().item():.3f}",
            flush=True
        )

        # random uniform translation
        B, D = X.shape
        N = D // 3
        X3 = X.view(B, N, 3)
        shift = torch.rand((B, 1, 3), device=X.device, dtype=X.dtype) * L
        Xshift = target.coordinate_transform.wrap(X3 + shift, L).view(B, D)

        Xp_shift = target.coordinate_transform.forward(Xshift)
        U1 = -target.p.log_prob_x(Xp_shift)

        dU = (U1 - U0).abs()
        print(
            f"[ΔU] |U(preprocess(shift(X)))-U(preprocess(X))| "
            f"mean={dU.mean().item():.3e} max={dU.max().item():.3e} (kBT)",
            flush=True
        )


    # tolerance: should be extremely small if everything consistent
    tol = 1e-4
    if dU.max().item() < tol:
        print("✓ translation invariance OK", flush=True)
    else:
        print("✗ translation invariance mismatch (could be constraints or preprocessing differences)", flush=True)

    print("\n================ TEST DONE =================\n", flush=True)
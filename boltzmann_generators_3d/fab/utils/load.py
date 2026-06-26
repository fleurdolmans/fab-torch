import os
import sys
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc, animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf
from boltzmann_generators_3d.fab.target_distributions.solute_in_water import SoluteInWater
import h5py


from boltzmann_generators_3d.experiments.make_flow import (
    make_shared_water_spline_flow_nf,
    make_perm_equi_spline_flow_nf,
    make_perm_equi_joint_spline_flow_nf,
    make_perm_equi_sfic_flow_nf,
    make_spherical_circular_rqs_flow_nf,
    make_coupled_rqs_flow_nf,
    make_realnvp_flow_nf,
    make_perm_equi_torus_flow_nf,
    make_circ_rqs_torus_flow_nf,
    make_perm_equi_sfic_torus_flow_nf,
    make_perm_equi_gps_flow_nf
)
def load_config(run_dir):
    for _candidate in [
        os.path.join(run_dir, "config.yaml"),
        os.path.join(run_dir, ".hydra", "config.yaml"),
    ]:
        if os.path.exists(_candidate):
            config_path = _candidate
            break
    else:
        raise FileNotFoundError(f"config.yaml not found under {run_dir}")

    cfg = OmegaConf.load(config_path)
    return cfg

def build_target(run_dir, md_path, device="cpu", platform="CPU"):
    cfg = load_config(run_dir)

    _save_dir = os.path.join(run_dir, "viz_metrics")
    os.makedirs(_save_dir, exist_ok=True)
    cfg.target.solute_pdb_path = "/Users/fleurdolmans/Documents/UvA/Master_AI/Thesis/HDD/data/molecules/solutes/so2.pdb"
    cfg.target.solute_xml_path = "/Users/fleurdolmans/Documents/UvA/Master_AI/Thesis/HDD/data/molecules/solutes/so2.xml"

    target = SoluteInWater(
        solute_pdb_path=str(cfg.target.solute_pdb_path),
        solute_xml_path=str(OmegaConf.select(cfg, "target.solute_xml_path", default=None) or ""),
        solute_inpcrd_path=str(OmegaConf.select(cfg, "target.solute_inpcrd_path", default=None) or ""),
        solute_prmtop_path=str(OmegaConf.select(cfg, "target.solute_prmtop_path", default=None) or ""),
        dim=int(cfg.target.cartesian_dim),
        num_solvent_molecules=int(cfg.target.num_solvent_molecules),
        temperature=float(cfg.target.temperature),
        energy_cut=float(cfg.target.energy_cut),
        energy_max=float(cfg.target.energy_max),
        n_threads=int(OmegaConf.select(cfg, "target.n_threads", default=1)),
        val_samples_path=md_path,
        device=device,
        save_dir=_save_dir,
        boundary_condition=str(OmegaConf.select(cfg, "target.boundary_condition", default="pbc")),
        box_length_nm=float(OmegaConf.select(cfg, "target.box_length_nm", default=2.5)),
        nonbonded_cutoff_nm=float(OmegaConf.select(cfg, "target.nonbonded_cutoff_nm", default=1.0)),
        rigid_water=bool(OmegaConf.select(cfg, "target.rigid_water", default=False)),
        internal_constraints=str(OmegaConf.select(cfg, "target.internal_constraints", default="none")),
        external_constraints=bool(OmegaConf.select(cfg, "target.external_constraints", default=False)),
        constraint_radius=float(OmegaConf.select(cfg, "target.constraint_radius", default=1.0)),
        constraint_force=float(OmegaConf.select(cfg, "target.constraint_force", default=10000.0)),
        platform_name=platform,
        energy_mode=str(OmegaConf.select(cfg, "training.energy_mode", default="full")),
        transform_version=str(OmegaConf.select(cfg, "target.transform.version", default=None)),
        canonical_sorting=bool(OmegaConf.select(cfg, "target.transform.canonical_sorting", default=False)),
    )
    if OmegaConf.select(cfg, "training.use_64_bit", default=False):
        torch.set_default_dtype(torch.float64)
        target = target.double()
    return target

def load_model(target, run_dir):
    cfg = load_config(run_dir)
    # Dispatch on cfg.flow.type — mirrors setup_run.py
    _ft = cfg.flow.type
    if _ft == "shared-water-spline-nf":
        flow = make_shared_water_spline_flow_nf(cfg, target)
    elif _ft == "perm-equi-gps-nf":
        flow = make_perm_equi_gps_flow_nf(cfg, target)
    elif _ft == "perm-equi-spline-nf":
        flow = make_perm_equi_spline_flow_nf(cfg, target)
    elif _ft == "perm-equi-joint-spline-nf":
        flow = make_perm_equi_joint_spline_flow_nf(cfg, target)
    elif _ft == "spherical-circ-rqs-nf":
        flow = make_spherical_circular_rqs_flow_nf(cfg, target)
    elif _ft == "coupled-rqs-nf":
        flow = make_coupled_rqs_flow_nf(cfg, target)
    elif _ft == "realnvp-nf":
        flow = make_realnvp_flow_nf(cfg, target)
    elif _ft == "perm-equi-sfic-nf":
        flow = make_perm_equi_sfic_flow_nf(cfg, target)
    elif _ft == "perm-equi-torus-nf":
        flow = make_perm_equi_torus_flow_nf(cfg, target)
    elif _ft == "circ-rqs-torus-nf":
        flow = make_circ_rqs_torus_flow_nf(cfg, target)
    elif _ft == "perm-equi-sfic-torus-nf":
        flow = make_perm_equi_sfic_torus_flow_nf(cfg, target)
    else:
        raise NotImplementedError(f"flow.type '{_ft}' not handled. Add it to this cell.")

    return cfg, flow

def load_checkpoint(run_dir, device="cpu"):
    chkpts_dir = os.path.join(run_dir, "model_checkpoints")
    assert os.path.isdir(chkpts_dir), f"model_checkpoints/ not found in {run_dir}"

    iter_dirs = []
    for _e in os.scandir(chkpts_dir):
        if _e.is_dir():
            _m = re.search(r"iter_(\d+)$", _e.name)
            if _m:
                iter_dirs.append((_e.path, int(_m.group(1))))

    # Locate the highest-numbered iter_* checkpoint
    chkpt_dir, chkpt_iter = max(iter_dirs, key=lambda x: x[1])
    model_path = os.path.join(chkpt_dir, "model.pt")
    print(f"Loading iteration {chkpt_iter} checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return model_path, checkpoint


def load_md_data(md_path, target, n_md_compare=5000, device="cpu"):
    """
    Load MD reference data.

    Priority:
    1. If target.val_data_i exists, use it and transform it to the LGT canonical frame.
    2. Otherwise, try to load md_path from .pt/.pth/.h5/.npz.

    Returns
    -------
    x_md : np.ndarray or torch.Tensor
        MD coordinates.
    u_md : np.ndarray or torch.Tensor
        MD energies / negative log probabilities.
    """

    n_atoms = target.cartesian_dim // 3

    # ------------------------------------------------------------
    # Preferred path: use target validation data and canonicalize it
    # ------------------------------------------------------------
    if getattr(target, "val_data_i", None) is not None:
        with torch.no_grad():
            _val_i = target.val_data_i[:n_md_compare].float().to(device)

            x_md_canonical, _ = target.coordinate_transform.forward(_val_i)
            lp_md = target.log_prob(_val_i).cpu()

        x_md = x_md_canonical.float().cpu().view(-1, n_atoms, 3).numpy()
        u_md = (-lp_md.float()).numpy()

        print(
            f"MD reference : {x_md.shape} "
            f"(LGT canonical frame, first {len(x_md)} frames)"
        )
        print(f"Energy  mean={u_md.mean():.2f}  std={u_md.std():.2f}")

        return x_md, u_md

    # ------------------------------------------------------------
    # Fallback path: load from md_path
    # ------------------------------------------------------------
    md_path = Path(md_path)

    if not md_path.exists():
        print(f"No MD reference data found at: {md_path}")
        return None, None

    suffix = md_path.suffix.lower()

    if suffix in [".pt", ".pth"]:
        md_data = torch.load(md_path, map_location=device)

        x_md = md_data["x"]
        u_md = md_data["u"]

        if n_md_compare is not None:
            x_md = x_md[:n_md_compare]
            u_md = u_md[:n_md_compare]

        print(f"Loaded PyTorch MD reference: x={x_md.shape}, u={u_md.shape}")
        return x_md, u_md

    elif suffix == ".npz":
        md_data = np.load(md_path)

        print("Available NPZ keys:", md_data.files)

        x_key = "x"
        u_key = "u"

        x_md = md_data[x_key]
        u_md = md_data[u_key]

        if n_md_compare is not None:
            x_md = x_md[:n_md_compare]
            u_md = u_md[:n_md_compare]

        print(f"Loaded NPZ MD reference: x={x_md.shape}, u={u_md.shape}")
        return x_md, u_md

    elif suffix in [".h5", ".hdf5"]:
        with h5py.File(md_path, "r") as f:
            print("Available HDF5 datasets:")

            datasets = {}

            def collect_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj.shape
                    print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

            f.visititems(collect_dataset)

            # Change these if your printed HDF5 keys are different
            possible_x_keys = ["x", "positions", "coordinates", "trajectory", "coords"]
            possible_u_keys = ["u", "energy", "energies", "potential_energy"]

            x_key = next((k for k in possible_x_keys if k in datasets), None)
            u_key = next((k for k in possible_u_keys if k in datasets), None)

            if x_key is None:
                raise KeyError(
                    "Could not find coordinate dataset in HDF5 file. "
                    "Check the printed dataset names and set x_key manually."
                )

            x_md = f[x_key][:]

            if u_key is not None:
                u_md = f[u_key][:]
            else:
                print("No energy dataset found in HDF5 file; setting u_md=None.")
                u_md = None

        if n_md_compare is not None:
            x_md = x_md[:n_md_compare]
            if u_md is not None:
                u_md = u_md[:n_md_compare]

        print(
            f"Loaded HDF5 MD reference: x={x_md.shape}, "
            f"u={None if u_md is None else u_md.shape}"
        )

        return x_md, u_md

    else:
        raise ValueError(f"Unsupported MD file format: {suffix}")
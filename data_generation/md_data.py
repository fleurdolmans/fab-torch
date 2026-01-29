import hydra
from omegaconf import DictConfig, OmegaConf

import os

import openmm as mm
from openmm import unit, app
import numpy as np

from mdtraj.reporters import HDF5Reporter
from sys import stdout
import json
import pathlib

import mdtraj as md

from fab.target_distributions.solute_in_water import TriatomicInWaterSys


def create_md_sim(cfg: DictConfig):
    """
    Running a simulation using the TriatomicInWaterSys class from fab.target_distributions.h2o_in_h2o,
    to create MD data for a triatomic solute in water solvent.
    
    See also:
    http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html#saving-the-results
    """
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raise error when solvent is not water
    if cfg.solvent_name != "water":
        raise NotImplementedError("Currently only water solvent is supported.")

    # Set platform properties based on the selected platform
    platform_list = ["Reference", "CPU", "OpenCL", "CUDA", "None"]
    if cfg.platform_name == "CUDA":
        platform_properties = {
        "Precision": "mixed",   # Best speed/accuracy tradeoff
        "DeviceIndex": "0",         # Pick GPU 0
    }
    elif cfg.platform_name in platform_list:
        platform_properties = None
    else:
        raise NotImplementedError(f"Platform {cfg.platform_name} not implemented. Either use 'Reference', 'CPU', 'CUDA', 'OpenCL' or 'None'")
    
    # Initialize the TriatomicInWaterSys class with the necessary parameters:
    # 3 atoms in solute, 3 atoms in solvent, 4 solvent molecules. 3 dimensions per atom (xyz)
    dim = 3 * (3 + 3 * cfg.num_solvent_molecules)
    system = TriatomicInWaterSys(
        cfg.solute_pdb_path,
        cfg.solute_xml_path,
        cfg.solute_inpcrd_path,
        cfg.solute_prmtop_path,
        dim,
        cfg.external_constraints,
        cfg.internal_constraints,
        cfg.rigid_water,
        cfg.constraint_radius,
        cfg.constraint_force,
    )

    # Create a simulation object: Set up the simulation object with the system, integrator, and initial positions
    integrator = mm.LangevinMiddleIntegrator(
        cfg.temperature * unit.kelvin, 1.0 / unit.picosecond, cfg.femtoseconds_per_timestep * unit.femtosecond
    )
    # integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    sim = app.Simulation(
        system.topology,
        system.system,
        integrator,
        mm.Platform.getPlatformByName(cfg.platform_name),
        platform_properties,
    )
    print("OpenMM platform:", sim.context.getPlatform().getName())
    if sim.context.getPlatform().getName() == "CUDA":
        print("CUDA properties:", sim.context.getPlatform().getPropertyNames())

    sim.context.setPositions(system.positions)
    # Minimize energy: Perform an energy minimization to remove any irregularities in the initial configuration
    sim.minimizeEnergy()
    # Add stdout reporter
    sim.reporters.append(
        app.statedatareporter.StateDataReporter(
            stdout,
            cfg.report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            elapsedTime=True,
            totalSteps=cfg.equi_steps + cfg.burnin_steps + cfg.num_steps,
        )
    )

    # Equilibrate: Equilibrate the system with a short run, allowing the solvent to relax around
    # the central water molecule
    sim.step(cfg.equi_steps)
    # Run the burn-in simulation: Run the simulation for a desired number of steps, discarding the
    # first few steps to allow the system to reach equilibrium
    sim.step(cfg.burnin_steps)

    # Saving data
    cnstrnts = (
        f"_ec{cfg.external_constraints}_r{cfg.constraint_radius:.1f}_fc{cfg.constraint_force}_"
        f"ic{cfg.internal_constraints}_rw{cfg.rigid_water}"
    )
    filename = f"traj_{cfg.solute_name}In{cfg.solvent_name}_{cfg.simulation_version}.h5"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cartesian_dim"] = dim
    with open((out_dir / filename).with_suffix(".json"), "w") as f:
        json.dump(cfg_dict, f, indent=4)
    
    # Add reporters to save trajectory and state data at specified intervals
    sim.reporters.append(HDF5Reporter(str(out_dir / filename), cfg.save_interval))
    sim.reporters.append(app.PDBReporter(str((out_dir / filename).with_suffix(".pdb")), cfg.save_interval))
    
    sim.reporters.append(
        app.statedatareporter.StateDataReporter(
            str(out_dir / cfg.diagnostics_filename),
            cfg.save_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )

    # Run the production simulation: Finally, run the simulation for a desired number of steps:
    sim.step(cfg.num_steps)


def plot_md_diagnostics(out_dir: str | pathlib.Path, diagnostics_filename: str , show: bool = False):
    """
    Plot MD diagnostics (potential energy and temperature over time) from the MD simulation report.
    Saves a plot to save_dir if provided, and shows the plot if show=True.
    """
    import matplotlib.pyplot as plt
    # Load data of this run from disk.
    out_dir = pathlib.Path(out_dir)
    report_path = pathlib.Path(out_dir / diagnostics_filename)

    # Load data of this run from disk.
    with open(report_path, "r") as f:
        report = f.read()

    steps, energies, temps = [], [], []
    for r, line in enumerate(report.split("\n")[1:]):  # skip header
        if not line:
            continue
        step, energy, temp = line.split(",")
        if r == 0:
            initial_step = int(float(step))
        steps.append(int(float(step)) - initial_step)
        energies.append(float(energy))
        temps.append(float(temp))

    fig, ax1 = plt.subplots()
    ax1.plot(steps, energies, label="Potential energy")
    ax1.set_ylabel("Potential energy (kJ/mol)")
    ax1.set_ylim(min(energies) * 1.05, 0)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps, temps, label="Temperature")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend(loc="upper right")

    ax1.set_xlabel("Saved frame index (relative steps)")
    plt.tight_layout()

    if out_dir is not None:
        out_dir = out_dir / "md_energy_temperature.png"
        fig.savefig(out_dir, dpi=300)
        print(f"Saved MD diagnostics plot to {out_dir}")

    if show:
        plt.show()
    
    plt.close(fig)

def validate_md(out_dir: str | pathlib.Path, diagnostics_filename: str , traj_path: str | pathlib.Path) -> dict:
    """ 
    Validate the MD simulation results for a triatomic solute in water solvent.
    Checks include:
    - Existence of trajectory and log files
    - Parsing and statistics of potential energy and temperature
    - Basic geometry checks for solute molecule (bond lengths, angles)
    - Anchor atom behavior if external constraints are used
    Results are outputed in a validation report (JSON), saved in out_dir 
    """

    out_dir = pathlib.Path(out_dir)
    md_log = pathlib.Path(out_dir / diagnostics_filename)

    # Where to save the validation result
    validation_json = out_dir / f"validation_{cfg.solute_name}In{cfg.solvent_name}_{cfg.simulation_version}.json"

    report: dict = {
        "status": "unknown",
        "warnings": [],
        "errors": [],
        "md_log_stats": {},
        "trajectory_stats": {},
        "topology_head": None,
        "solute_selection": {},
        "bond_lengths_frame0_A": [],
        "anchor_stats": {},
    }

    def _finalize_and_save(status: str):
        report["status"] = status
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(validation_json, "w") as f:
            json.dump(report, f, indent=2)

    try:
        if not md_log.exists():
            raise FileNotFoundError(f"Missing MD log file: {md_log}")

        # Parse run_statistics.txt
        steps, pes, temps = [], [], []
        with open(md_log, "r") as f:
            for k, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if k == 0 and "Potential Energy" in line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                step, pe, temp = parts[0], parts[1], parts[2]
                steps.append(int(float(step)))
                pes.append(float(pe))
                temps.append(float(temp))

        if len(temps) == 0:
            raise ValueError(f"Could not parse any samples from {md_log}")

        pes = np.asarray(pes, dtype=np.float64)
        temps = np.asarray(temps, dtype=np.float64)

        def _assert_finite(name: str, arr: np.ndarray):
            if not np.isfinite(arr).all():
                bad = np.where(~np.isfinite(arr))[0][:10]
                raise ValueError(f"{name} contains NaN/inf at indices {bad.tolist()} in {md_log}")

        _assert_finite("Potential energy", pes)
        _assert_finite("Temperature", temps)

        report["md_log_stats"] = {
            "n_samples": int(len(temps)),
            "temperature_K": {
                "mean": float(temps.mean()),
                "std": float(temps.std()),
                "max": float(temps.max()),
                "min": float(temps.min()),
            },
            "potential_energy_kJmol": {
                "mean": float(pes.mean()),
                "std": float(pes.std()),
                "max": float(pes.max()),
                "min": float(pes.min()),
            },
            "step_first": int(steps[0]),
            "step_last": int(steps[-1]),
        }

        # Explosion detector (keep as warning or error — your choice)
        if temps.max() > 5000 or temps.mean() > 2000:
            report["errors"].append(
                "MD appears to have exploded: temperature is unphysically large "
                f"(T_mean={temps.mean():.2f} K, T_max={temps.max():.2f} K)."
            )

        # Load trajectory 
        if traj_path.suffix.lower() == ".h5":
            traj = md.load_hdf5(str(traj_path))
        else:
            traj = md.load(str(traj_path))

        top = traj.topology

        report["trajectory_stats"] = {
            "frames": int(traj.n_frames),
            "atoms": int(traj.n_atoms),
        }

        # Store a small topology preview (like your printed table.head())
        table, bonds_df = top.to_dataframe()
        # Keep this compact (head only), and JSON-friendly
        report["topology_head"] = table.head(6).to_dict(orient="records")

        # Solute selections 
        if cfg.solute_name == "water":
            solute = top.select("resid 0")
            o_idx = top.select("resid 0 and element O")
            h_idx = top.select("resid 0 and element H")

            if len(o_idx) == 1 and len(h_idx) == 2:
                o = int(o_idx[0])
                h1, h2 = map(int, h_idx)
                

                angle_triplet = np.array([[h1, o, h2]])
                angle_rad = md.compute_angles(traj[0], angle_triplet)[0][0]
                angle_deg = np.degrees(angle_rad)

                report["solute_angle_frame0_deg"] = {
                    "atoms": f"H({h1})-O({o})-H({h2})",
                    "angle_deg": float(angle_deg),
                }

                # TIP3P equilibrium ≈ 104.5°
                if angle_deg < 90 or angle_deg > 120:
                    report["warnings"].append(
                        f"Water angle suspicious: {angle_deg:.2f}° (expected ~104.5°)"
                    )
            

        elif cfg.solute_name == "so2":
            solute = top.select("not water")
            s_idx = top.select("not water and element S")
            o_idx = top.select("not water and element O")

            if len(s_idx) == 1 and len(o_idx) == 2:
                s = int(s_idx[0])
                o1, o2 = map(int, o_idx)

                angle_triplet = np.array([[o1, s, o2]])  # angle at S
                angle_rad = md.compute_angles(traj[0], angle_triplet)[0][0]
                angle_deg = np.degrees(angle_rad)

                report["solute_angle_frame0_deg"] = {
                    "atoms": f"O({o1})-S({s})-O({o2})",
                    "angle_deg": float(angle_deg),
                }

                # sanity check
                if angle_deg < 90 or angle_deg > 150:
                    report["warnings"].append(
                        f"SO2 angle suspicious: {angle_deg:.2f}° (expected ~119°)"
                    )
            else:
                report["warnings"].append(
                    f"Could not uniquely identify SO2 atoms: S={len(s_idx)}, O={len(o_idx)}"
                )
        else:
            raise NotImplementedError(f"Solute name {cfg.solute_name} not implemented in validate_md().")

        solute_indices = set(map(int, solute)) if len(solute) > 0 else set()

        # Bond length sanity (frame 0)
        bond_pairs = []
        bond_labels = []
        for bond in top.bonds:
            i = bond.atom1.index
            j = bond.atom2.index
            if (len(solute_indices) == 0) or (i in solute_indices or j in solute_indices):
                bond_pairs.append([i, j])
                bond_labels.append((bond.atom1.name, bond.atom2.name, i, j))

        if bond_pairs:
            d_nm = md.compute_distances(traj[0], bond_pairs)[0]
            d_A = d_nm * 10.0

            for (n1, n2, i, j), dist in zip(bond_labels, d_A):
                report["bond_lengths_frame0_A"].append(
                    {"atom1": f"{n1}({i})", "atom2": f"{n2}({j})", "distance_A": float(dist)}
                )


        # Anchor/centering sanity
        a0 = list(top.atoms)[0]
        report["anchor_stats"]["atom0"] = {
            "index": int(a0.index),
            "name": str(a0.name),
            "element": str(a0.element.symbol) if a0.element is not None else None,
            "residue": str(a0.residue),
        }

        if cfg.external_constraints:
            a0_xyz = traj.xyz[:, 0, :]
            r0 = np.linalg.norm(a0_xyz, axis=1)
            report["anchor_stats"]["distance_from_origin_nm"] = {
                "mean": float(r0.mean()),
                "std": float(r0.std()),
                "max": float(r0.max()),
                "min": float(r0.min()),
            }
            if r0.max() > 0.2:  # 2 Å
                report["warnings"].append(
                    "Anchoring check: atom0 is not staying near origin "
                    f"(max |r| = {r0.max():.3f} nm)."
                )

        # Decide pass/fail based on errors
        if report["errors"]:
            _finalize_and_save("fail")
            # If you want validation to stop the pipeline, raise:
            raise ValueError("MD validation failed. See JSON report for details.")
        else:
            _finalize_and_save("pass")
            return report

    except Exception as e:
        # Capture unexpected exceptions in the JSON too
        report["errors"].append(f"Exception: {type(e).__name__}: {e}")
        _finalize_and_save("error")
        raise



@hydra.main(config_path="./config/", config_name="make_md_data", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.create_md:
        create_md_sim(cfg)
        print("MD data creation completed. MD data saved in ", cfg.out_dir)
    
    # Plot MD diagnostics.
    if cfg.plot.md_diagnostics:
        plot_md_diagnostics(out_path=cfg.out_dir, diagnostics_filename=cfg.diagnostics_filename, show=cfg.plot.show)

    # Validate trajectory file
    if cfg.validate_md:
        traj_h5 = cfg.out_dir / f"traj_{cfg.solute_name}In{cfg.solvent_name}_{cfg.simulation_version}.h5"
        traj_pdb = cfg.out_dir / f"traj_{cfg.solute_name}In{cfg.solvent_name}_{cfg.simulation_version}.pdb"
        
        if traj_h5.exists():
            traj_path = traj_h5
        elif traj_pdb.exists():
            traj_path = traj_pdb
        else:
            raise FileNotFoundError(
                f"No trajectory found. Expected {traj_pdb.name} or {traj_h5.name} in {out_dir}."
            )
        validate_md(report_path=diagnostics_dir, traj_path=traj_path)
        print("MD data validation completed.")


if __name__ == "__main__":
    main()


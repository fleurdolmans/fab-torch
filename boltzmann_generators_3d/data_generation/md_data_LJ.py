import hydra
from omegaconf import DictConfig, OmegaConf

import os

import math
import openmm as mm
from openmm import unit, app
import numpy as np

from mdtraj.reporters import HDF5Reporter
from sys import stdout
import json
import pathlib

import mdtraj as md

from boltzmann_generators_3d.fab.target_distributions.solute_in_water_LJ import LJParticlesSys
from openmmtools.integrators import GradientDescentMinimizationIntegrator

def _make_simulation(topology, system, integrator, platform_name="CUDA", device_index=0):
    platform_list = ["Reference", "CPU", "OpenCL", "CUDA"]
    if platform_name == "CUDA":
        platform_properties = {
            "Precision": "mixed",
            "DeviceIndex": str(device_index),
        }
    elif platform_name in platform_list:
        platform_properties = None
    else:
        raise NotImplementedError(
            f"Unsupported platform: {platform_name}. "
            "Use one of Reference, CPU, OpenCL, CUDA."
        )

    return app.Simulation(
        topology,
        system,
        integrator,
        mm.Platform.getPlatformByName(platform_name),
        platform_properties,
    )


def _copy_state(src_sim: app.Simulation, dst_sim: app.Simulation, copy_velocities=True):
    state = src_sim.context.getState(getPositions=True, getVelocities=copy_velocities)
    dst_sim.context.setPositions(state.getPositions())
    if copy_velocities:
        dst_sim.context.setVelocities(state.getVelocities())

def create_md_sim(cfg: DictConfig):
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = LJParticlesSys(
        n_solvent=cfg.n_solvent,
        box_length_nm=cfg.box_length_nm,
        solvent_sigma_nm=cfg.solvent_sigma_nm,
        solvent_epsilon_kjmol=cfg.solvent_epsilon_kjmol,
        solvent_mass_amu=cfg.solvent_mass_amu,
        solute_positions_nm=cfg.solute_positions_nm,
        solute_sigma_nm=cfg.solute_sigma_nm,
        solute_epsilon_kjmol=cfg.solute_epsilon_kjmol,
        solute_mass_amu=cfg.solute_mass_amu,
        switch_nm=cfg.switch_nm,
        cutoff_nm=cfg.nonbonded_cutoff_nm,
        constrain_solutes=cfg.constrain_solutes,
        solute_solute_interaction=cfg.solute_solute_interaction,
        seed=cfg.seed,
        grid_spacing_nm=cfg.grid_spacing_nm,
    )

    # -------------------------------------------------
    # Stage 0: create initial context and set positions
    # -------------------------------------------------
    init_integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    sim = _make_simulation(
        topology=system.topology,
        system=system.system,
        integrator=init_integrator,
        platform_name=cfg.platform_name,
        device_index=cfg.device_index,
    )
    sim.context.setPositions(system.positions)

    print("OpenMM platform:", sim.context.getPlatform().getName())

    # -------------------------------------------------
    # Stage 1: L-BFGS minimization
    # OpenMM LocalEnergyMinimizer uses L-BFGS
    # -------------------------------------------------
    print("Stage 1/5: L-BFGS minimization")
    sim.minimizeEnergy()

    # -------------------------------------------------
    # Stage 2: 5000 steps of gradient descent
    # -------------------------------------------------
    print("Stage 2/5: gradient descent minimization")
    gd_integrator = GradientDescentMinimizationIntegrator(
        initial_step_size=0.01 * unit.angstrom
    )
    gd_sim = _make_simulation(
        topology=system.topology,
        system=system.system,
        integrator=gd_integrator,
        platform_name=cfg.platform_name,
        device_index=cfg.device_index,
    )
    _copy_state(sim, gd_sim, copy_velocities=False)
    gd_sim.step(cfg.gradient_steps)
    # -------------------------------------------------
    # Stage 3: Nosé–Hoover thermalization
    # SI: 50,000 steps at dt = 0.01 fs
    # -------------------------------------------------
    print("Stage 3/5: Nosé–Hoover thermalization")
    nh_integrator = mm.NoseHooverIntegrator(
        cfg.temperature * unit.kelvin,
        cfg.nose_hoover_collision_ps / unit.picosecond,
        0.01 * unit.femtosecond,
    )
    nh_sim = _make_simulation(
        topology=system.topology,
        system=system.system,
        integrator=nh_integrator,
        platform_name=cfg.platform_name,
        device_index=cfg.device_index,
    )
    _copy_state(gd_sim, nh_sim, copy_velocities=False)
    nh_sim.context.setVelocitiesToTemperature(cfg.temperature * unit.kelvin, cfg.seed)
    nh_sim.step(cfg.nh_steps)

    # -------------------------------------------------
    # Stage 4: Langevin thermalization
    # SI: 50,000 fs at dt = 0.1 fs = 500,000 steps
    # tau = 0.2 ps -> gamma = 5 ps^-1
    # -------------------------------------------------
    print("Stage 4/5: Langevin thermalization")
    gamma = 1.0 / (0.2 * unit.picosecond)
    langevin_burnin = mm.LangevinMiddleIntegrator(
        cfg.temperature * unit.kelvin,
        gamma,
        0.1 * unit.femtosecond,
    )
    burnin_sim = _make_simulation(
        topology=system.topology,
        system=system.system,
        integrator=langevin_burnin,
        platform_name=cfg.platform_name,
        device_index=cfg.device_index,
    )
    _copy_state(nh_sim, burnin_sim, copy_velocities=True)
    burnin_sim.step(cfg.burnin_steps)

    # -------------------------------------------------
    # Stage 5: production
    # SI: 10.5 ns at dt = 1.5 fs = 7,000,000 steps
    # save every 7000 steps
    # -------------------------------------------------
    print("Stage 5/5: production")
    prod_integrator = mm.LangevinMiddleIntegrator(
        cfg.temperature * unit.kelvin,
        gamma,
        1.5 * unit.femtosecond,
    )
    prod_sim = _make_simulation(
        topology=system.topology,
        system=system.system,
        integrator=prod_integrator,
        platform_name=cfg.platform_name,
        device_index=cfg.device_index,
    )
    _copy_state(burnin_sim, prod_sim, copy_velocities=True)

    total_prod_steps = cfg.production_steps
    save_interval = cfg.save_interval

    traj_filename = out_dir / "traj_lj_particles.h5"
    diagnostics_filename = out_dir / "run_statistics.txt"

    prod_sim.reporters.append(HDF5Reporter(str(traj_filename), save_interval))
    prod_sim.reporters.append(
        app.StateDataReporter(
            str(diagnostics_filename),
            save_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    prod_sim.reporters.append(
        app.StateDataReporter(
            stdout,
            save_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            elapsedTime=True,
            totalSteps=total_prod_steps,
        )
    )

    prod_sim.step(total_prod_steps)

    cfg_dict = {
        **OmegaConf.to_container(cfg, resolve=True),
        "cartesian_dim": 3 * system.topology.getNumAtoms(),
    }
    with open(out_dir / "traj_lj_particles.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)


def plot_md_diagnostics(out_dir: str | pathlib.Path, diagnostics_filename: str, show: bool = False):
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    report_path = out_dir / diagnostics_filename

    with open(report_path, "r") as f:
        report = f.read()

    steps, energies, temps = [], [], []
    for r, line in enumerate(report.split("\n")[1:]):
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
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps, temps, label="Temperature")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend(loc="upper right")

    ax1.set_xlabel("Saved frame index (relative steps)")
    plt.tight_layout()

    out_path = out_dir / "md_energy_temperature.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved MD diagnostics plot to {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def validate_lj_md(cfg: DictConfig, project_name: str = "lj_particles") -> dict:
    out_dir = pathlib.Path(cfg.out_dir)
    md_log = out_dir / cfg.diagnostics_filename
    traj_h5 = out_dir / "traj_lj_particles.h5"
    validation_json = out_dir / f"validation_{project_name}.json"

    report = {
        "status": "unknown",
        "warnings": [],
        "errors": [],
        "md_log_stats": {},
        "trajectory_stats": {},
        "solute_displacement_nm": {},
        "min_distance_nm_frame0": None,
        "density_estimate": {},
    }

    def _save(status: str):
        report["status"] = status
        with open(validation_json, "w") as f:
            json.dump(report, f, indent=2)

    try:
        if not md_log.exists():
            raise FileNotFoundError(f"Missing diagnostics file: {md_log}")
        if not traj_h5.exists():
            raise FileNotFoundError(f"Missing trajectory file: {traj_h5}")

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
                steps.append(int(float(parts[0])))
                pes.append(float(parts[1]))
                temps.append(float(parts[2]))

        if len(temps) == 0:
            raise ValueError("Could not parse any samples from diagnostics file.")

        pes = np.asarray(pes, dtype=np.float64)
        temps = np.asarray(temps, dtype=np.float64)

        if not np.isfinite(pes).all():
            raise ValueError("Potential energy contains NaN/Inf.")
        if not np.isfinite(temps).all():
            raise ValueError("Temperature contains NaN/Inf.")

        report["md_log_stats"] = {
            "n_samples": int(len(temps)),
            "step_first": int(steps[0]),
            "step_last": int(steps[-1]),
            "temperature_K": {
                "mean": float(temps.mean()),
                "std": float(temps.std()),
                "min": float(temps.min()),
                "max": float(temps.max()),
            },
            "potential_energy_kJmol": {
                "mean": float(pes.mean()),
                "std": float(pes.std()),
                "min": float(pes.min()),
                "max": float(pes.max()),
            },
        }

        if temps.max() > 5000 or temps.mean() > 2000:
            report["errors"].append(
                f"Simulation appears unstable: T_mean={temps.mean():.2f} K, T_max={temps.max():.2f} K."
            )

        traj = md.load_hdf5(str(traj_h5))
        report["trajectory_stats"] = {
            "frames": int(traj.n_frames),
            "atoms": int(traj.n_atoms),
        }

        expected_atoms = len(cfg.solute_positions_nm) + int(cfg.n_solvent)
        if traj.n_atoms != expected_atoms:
            report["errors"].append(
                f"Atom count mismatch: trajectory has {traj.n_atoms}, expected {expected_atoms}."
            )

        if traj.unitcell_lengths is None:
            report["errors"].append("Trajectory has no periodic box information.")
        else:
            volumes_nm3 = np.prod(traj.unitcell_lengths, axis=1)
            mean_volume_nm3 = float(volumes_nm3.mean())
            number_density = float(cfg.n_solvent / mean_volume_nm3)
            report["density_estimate"] = {
                "n_solvent": int(cfg.n_solvent),
                "mean_volume_nm3": mean_volume_nm3,
                "number_density_per_nm3": number_density,
            }

        n_solute = len(cfg.solute_positions_nm)
        if n_solute > 0:
            solute_xyz = traj.xyz[:, :n_solute, :]
            ref = traj.xyz[0, :n_solute, :]
            disp = np.linalg.norm(solute_xyz - ref[None, :, :], axis=-1)
            report["solute_displacement_nm"] = {
                "mean": float(disp.mean()),
                "max": float(disp.max()),
            }

            if cfg.constrain_solutes and disp.max() > 1e-4:
                report["warnings"].append("Solutes were meant to be fixed, but small displacement was detected.")

        # Minimum image convention for frame 0
        xyz0 = traj.xyz[0]
        L = np.asarray(traj.unitcell_lengths[0]) if traj.unitcell_lengths is not None else None
        dmin = np.inf

        for i in range(traj.n_atoms):
            for j in range(i + 1, traj.n_atoms):
                dr = xyz0[i] - xyz0[j]
                if L is not None:
                    dr -= L * np.round(dr / L)
                d = np.linalg.norm(dr)
                dmin = min(dmin, d)

        report["min_distance_nm_frame0"] = float(dmin)

        if dmin < 0.15:
            report["warnings"].append(f"Very small pair distance in frame 0: {dmin:.3f} nm")

        if report["errors"]:
            _save("fail")
            raise ValueError("LJ MD validation failed. See JSON report.")
        else:
            _save("pass")
            return report

    except Exception as e:
        report["errors"].append(f"Exception: {type(e).__name__}: {e}")
        _save("error")
        raise


def load_json(path):
    if not path:
        return None

    p = pathlib.Path(path).with_suffix(".json")
    if not p.exists():
        return None

    with p.open("r") as f:
        return json.load(f)


@hydra.main(config_path="./config/", config_name="make_lj_md_data", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.create_md:
        create_md_sim(cfg)
        print("LJ MD data creation completed. Data saved in", cfg.out_dir)

    if cfg.plot.md_diagnostics:
        plot_md_diagnostics(
            out_dir=cfg.out_dir,
            diagnostics_filename=cfg.diagnostics_filename,
            show=cfg.plot.show,
        )

    if cfg.validate_md:
        project_name = "lj_particles"

        if not cfg.create_md:
            cfg_from_json = load_json(pathlib.Path(cfg.out_dir) / "traj_lj_particles.json")
            if cfg_from_json is None:
                raise FileNotFoundError(f"Config JSON not found in {cfg.out_dir}.")
            cfg = OmegaConf.create(cfg_from_json)

        validate_lj_md(cfg, project_name=project_name)
        print("LJ MD validation completed.")


if __name__ == "__main__":
    main()

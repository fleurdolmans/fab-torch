import hydra
from omegaconf import DictConfig, OmegaConf

import os

import openmm as mm
from openmm import unit, app

from mdtraj.reporters import HDF5Reporter
from sys import stdout
import json
import pathlib

from fab.target_distributions.solute_in_water import TriatomicInWaterSys

def plot_md_diagnostics(report_path: str | pathlib.Path, save_dir: str | pathlib.Path | None = None, show: bool = False):
    import matplotlib.pyplot as plt
    # Load data of this run from disk.
    report_path = pathlib.Path(report_path)

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

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "md_energy_temperature.png"
        fig.savefig(out_path, dpi=300)
        print(f"Saved MD diagnostics plot to {out_path}")

    if show:
        plt.show()
    
    plt.close(fig)


def run_md_sim(cfg: DictConfig):
    """
    Running a simulation using the TriatomicInWaterSys class from fab.target_distributions.h2o_in_h2o.
    
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
    filename = (
        f"{cfg.solute_name}In{cfg.solvent_name}_dim{int(dim)}_temp{cfg.temperature}_eq{cfg.equi_steps}_burn{cfg.burnin_steps}"
        f"_steps{cfg.num_steps}_fpt{cfg.femtoseconds_per_timestep}_every{cfg.save_interval}{cnstrnts}.h5"
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cartesian_dim"] = dim
    with open((out_dir / filename).with_suffix(".json"), "w") as f:
        json.dump(cfg_dict, f, indent=4)
    
    # Add reporters to save trajectory and state data at specified intervals
    sim.reporters.append(HDF5Reporter(str(out_dir / filename), cfg.save_interval))
    sim.reporters.append(app.PDBReporter(str(out_dir / f"traj_{cfg.solute_name}In{cfg.solvent_name}.pdb"), cfg.save_interval))
    
    sim.reporters.append(
        app.statedatareporter.StateDataReporter(
            str(out_dir / "last_md_run_start.txt"),
            cfg.save_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )

    # Run the production simulation: Finally, run the simulation for a desired number of steps:
    sim.step(cfg.num_steps)


@hydra.main(config_path="./config/", config_name="make_md_data", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.run_md:
        run_md_sim(cfg)
        print("MD data creation completed. MD data saved in ", cfg.out_dir)
    
    # Plot MD diagnostics.
    if cfg.plot.md_diagnostics:
        plot_md_diagnostics(report_path=pathlib.Path(cfg.out_dir) / "last_md_run_start.txt", save_dir=cfg.plot.save_dir, show=cfg.plot.show)


if __name__ == "__main__":
    main()


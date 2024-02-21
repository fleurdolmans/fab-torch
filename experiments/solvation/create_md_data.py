import hydra
from omegaconf import DictConfig, OmegaConf


from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from mdtraj.reporters import HDF5Reporter
from sys import stdout
import json
import pathlib

from fab.target_distributions.h2o_in_h2o import WaterInWaterBox


def run_md_sim(cfg: DictConfig):
    """
    Running a simulation using the WaterInWaterBox class from fab.target_distributions.h2o_in_h2o.
    
    See also:
    http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html#saving-the-results
    """
    out_dir = pathlib.Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3 atoms in solute, 3 atoms in solvent, 4 solvent molecules. 3 dimensions per atom (xyz)
    dim = 3 * (3 + 3 * cfg.num_solvent_molecules)
    system = WaterInWaterBox(
        cfg.solute_pdb_path,
        dim,
        cfg.external_constraints,
        cfg.internal_constraints,
        cfg.rigidwater,
    )

    # Create a simulation object: Set up the simulation object with the system, integrator, and initial positions
    integrator = mm.LangevinMiddleIntegrator(
        cfg.temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond
    )
    # integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    sim = app.Simulation(
        system.topology,
        system.system,
        integrator,
        platform=mm.Platform.getPlatformByName("Reference"),
    )
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

    # For MDTraj
    cnstrnts = f"_ec{cfg.external_constraints}_ic{cfg.internal_constraints}_rw{cfg.rigidwater}"
    filename = (
        f"output_dim{int(dim)}_temp{int(cfg.temperature)}_eq{int(cfg.equi_steps)}_burn{int(cfg.burnin_steps)}"
        f"_steps{int(cfg.num_steps)}_every{int(cfg.save_interval)}{cnstrnts}.h5"
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cartesian_dim"] = dim
    with open((out_dir / filename).with_suffix(".json"), "w") as f:
        json.dump(cfg_dict, f, indent=4)
    sim.reporters.append(HDF5Reporter(str(out_dir / filename), cfg.save_interval))
    sim.reporters.append(
        app.statedatareporter.StateDataReporter(
            str(out_dir / "last_md_run_data.txt"),
            cfg.save_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    # Run the production simulation: Finally, run the simulation for a desired number of steps:
    sim.step(cfg.num_steps)

    # # Final state
    # state = sim.context.getState(getPositions=True)
    # positions = state.getPositions(True).value_in_unit(unit.nanometer)
    # print(positions)
    # app.PDBFile.writeFile(sim.topology, positions, open('output.pdb', 'w'))

    import matplotlib.pyplot as plt
    with open(out_dir / "last_md_run_data.txt", "r") as f:
        report = f.read()
        steps, energies, temps = [], [], []
        for r, line in enumerate(report.split("\n")[1:]):
            if len(line) == 0:
                continue
            step, energy, temp = line.split(",")
            if r == 0:
                initial_step = int(float(step))
            steps.append(int(float(step)) - initial_step)
            energies.append(float(energy))
            temps.append(float(temp))

        fig, ax1 = plt.subplots()
        ax1.plot(steps, energies, label="Potential energy")
        ax1.set_ylabel("MD energy estimate (kJ/mol)")
        ax1.set_ylim(min(energies) * 1.05, 0)
        plt.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(steps, temps, label="Temperature", color="orange")
        ax2.set_ylabel("MD temperature estimate (K)")
        plt.legend(loc="upper right")

        plt.xlabel("MD sample index")
        plt.show()
        plt.close()


@hydra.main(config_path="./config/", config_name="make_md_data.yaml", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_md_sim(cfg)


if __name__ == "__main__":
    main()


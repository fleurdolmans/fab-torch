from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from mdtraj.reporters import HDF5Reporter
from sys import stdout
import pathlib

from fab.target_distributions.h2o_in_h2o import WaterInWaterBox


if __name__ == "__main__":
    """
    Running a simulation using the WaterInWaterBox class from fab.target_distributions.h2o_in_h2o.
    
    See also:
    http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html#saving-the-results
    """
    solvent_pdb_path = "/home/timsey/HDD/data/molecules/solvents/water.pdb"
    out_dir = pathlib.Path("/home/timsey/HDD/data/molecules/md/")
    out_dir.mkdir(parents=True, exist_ok=True)

    dim = 3 * (3 + 3 * 4)  # 3 atoms in solute, 3 atoms in solvent, 4 solvent molecules. 3 dimensions per atom (xyz)
    temperature = 300.0  # Kelvin
    equi_steps = 1e3  # Steps for equilibration
    burnin_steps = 1e5  # Steps for burn-in
    num_steps = 1e7  # Simulation steps
    report_interval = 1e3  # Report to stdout every n steps
    save_interval = 10  # Save positions every m steps

    system = WaterInWaterBox(solvent_pdb_path, dim)

    # Create a simulation object: Set up the simulation object with the system, integrator, and initial positions
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond
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
            report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            elapsedTime=True,
            totalSteps=equi_steps + burnin_steps + num_steps,
        )
    )
    # Equilibrate: Equilibrate the system with a short run, allowing the solvent to relax around
    # the central water molecule
    sim.step(equi_steps)
    # Run the burn-in simulation: Run the simulation for a desired number of steps, discarding the
    # first few steps to allow the system to reach equilibrium
    sim.step(burnin_steps)
    # Add file reporter
    # sim.reporters.append(
    #     app.pdbreporter.PDBReporter(
    #         out_dir / (
    #             f"output_dim{int(dim)}_temp{int(temperature)}_eq{int(equi_steps)}_burn{int(burnin_steps)}"
    #             f"_steps{int(num_steps)}_every{int(save_interval)}.pdb"
    #         ),
    #         save_interval,
    #     )
    # )
    # For MDTraj
    sim.reporters.append(
        HDF5Reporter(
            str(out_dir / (
                f"output_dim{int(dim)}_temp{int(temperature)}_eq{int(equi_steps)}_burn{int(burnin_steps)}"
                f"_steps{int(num_steps)}_every{int(save_interval)}.h5"
            )),
            save_interval,
        )
    )
    # Run the production simulation: Finally, run the simulation for a desired number of steps:
    sim.step(num_steps)

    # # Final state
    # state = sim.context.getState(getPositions=True)
    # positions = state.getPositions(True).value_in_unit(unit.nanometer)
    # print(positions)
    # app.PDBFile.writeFile(sim.topology, positions, open('output.pdb', 'w'))

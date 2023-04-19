import torch
from torch import nn
import numpy as np
import tempfile
import pathlib

from fab.target_distributions.base import TargetDistribution
from fab.transforms.global_3point_spherical_transform import Global3PointSphericalTransform

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import TestSystem, get_data_filename


class WaterInWaterBox(TestSystem):

    """Water box with water micro-solvation system.

    Parameters
    ----------
    dim: dimensionality of system (num_atoms x 3).

    """
    def __init__(self, dim, **kwargs):
        TestSystem.__init__(self, **kwargs)
        # http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        # Two methods: either create system from pdb and FF with forcefield.createSystems() or use prmtop and crd files,
        #  as in the openmmtools testsystems examples:
        #  https://openmmtools.readthedocs.io/en/stable/_modules/openmmtools/testsystems.html#AlanineDipeptideImplicit

        # prmtop + crd example:
        # prmtop_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")
        # crd_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.crd")
        #
        # # Initialize system.
        # prmtop = app.AmberPrmtopFile(prmtop_filename)
        # system = prmtop.createSystem(
        #     implicitSolvent=app.OBC1, constraints=constraints, nonbondedCutoff=None, hydrogenMass=hydrogenMass
        # )
        # # Extract topology
        # topology = prmtop.topology
        # # Read positions.
        # inpcrd = app.AmberInpcrdFile(crd_filename)
        # positions = inpcrd.getPositions(asNumpy=True)
        # self.topology, self.system, self.positions = topology, system, positions

        # TODO: Other parameters? HydrogenMass, cutoffs, etc.?
        # TODO: Add radial restraint force to keep solvation shells close
        # TODO: Add implicit solvent to the system
        self.num_atoms_per_solute = 3   # Water
        self.num_atoms_per_solvent = 3  # Water
        self.num_solvent_molecules = (dim - self.num_atoms_per_solute) // (self.num_atoms_per_solvent * 3)

        # pdb example:
        # Initial solute molecule
        pdb = app.PDBFile("/home/timsey/HDD/data/molecules/solvents/water.pdb")
        # NOTE: Add solute force field if not water!
        # Add solvent
        modeller = app.modeller.Modeller(pdb.topology, pdb.positions)
        forcefield = app.ForceField("amber14/tip3p.xml")  # tip3pfb
        # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’
        modeller.addSolvent(forcefield, model='tip3p', numAdded=self.num_solvent_molecules)
        # Create system
        self.system = forcefield.createSystem(modeller.topology)

        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()
        # self.topology.atoms() yields the atom order, which is OHH OHH OHH etc.
        # This is the order in which the coordinates are stored in the positions array.
        self.atoms = [atom.name for atom in self.topology.atoms()]


class H2OinH2O(nn.Module, TargetDistribution):
    def __init__(
        self,
        dim=3 * (3 + 3 * 8),  # 3 atoms in solute, 3 atoms in solvent, 8 solvent molecules. 3 dimensions per atom (xyz)
        temperature=300,
        energy_cut=1.0e8,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        energy_max=1.0e20,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        n_threads=4,
        data_path=None,
        device="cpu",
    ):
        """
        Boltzmann distribution of H2O in H2O.
        :param temperature: Temperature of the system
        :param energy_cut: Value after which the energy is logarithmically scaled
        :param energy_max: Maximum energy allowed, higher energies are cut
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :param data_path: Path to the data used for coordinate transform
        """
        super(H2OinH2O, self).__init__()

        self.dim = dim

        # Steps to take:
        # 1. Load topology of solute.
        # 2. Solvate the solute.
        # 3. Add the solute and solvent force fields.
        # 4. Add the implicit solvent force field / external potential term.

        system = WaterInWaterBox(dim)

        # TODO: Do we want to run a quick simulation to get a sense of the coordinate magnitude for normalisation?
        #  If so, we can run a quick simulation here, as in aldp.py to obtain `transform_data`.
        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            traj_sim = app.Simulation(
                system.topology,
                system.system,
                mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                platform=mm.Platform.getPlatformByName("Reference"),
            )
            traj_sim.context.setPositions(system.positions)
            traj_sim.minimizeEnergy()
            state = traj_sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = pathlib.Path(tempfile.gettempdir())
            data_path = tmp_dir / f"h2o_in_h2o_{dim}.pt"

            torch.save(torch.tensor(position.reshape(1, dim).astype(np.float64)), data_path)
            del traj_sim

        data_path = pathlib.Path(data_path)
        assert data_path.suffix == ".pt", "Data path must be a .pt file"
        transform_data = torch.load(data_path)
        assert transform_data.shape[-1] == dim, (
            f"Data shape ({transform_data.shape}) does not match number of "
            f"coordinates in current system ({dim})."
        )

        self.coordinate_transform = Global3PointSphericalTransform(system, transform_data.to(device))

        if n_threads > 1:
            self.p = bg.distributions.TransformedBoltzmannParallel(
                system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
                n_threads=n_threads,
            )
        else:
            # Need to define sim, since the non-parallel version does not take a system as input (parallel builds the
            #  sim from system in exactly this way).
            sim = app.Simulation(
                system.topology,
                system.system,
                mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                mm.Platform.getPlatformByName("Reference"),
            )

            self.p = bg.distributions.TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
            )

    def log_prob(self, x: torch.tensor):
        # Add global potential for implicit solvent here?
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}
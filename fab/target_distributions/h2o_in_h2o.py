import torch
from torch import nn

from fab.target_distributions.base import TargetDistribution
from fab.transforms.water_transform import WaterCoordinateTransform

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import TestSystem, get_data_filename


class WaterInWaterBox(TestSystem):

    """Water box with water micro-solvation system.

    Parameters
    ----------
    constraints : optional, default=openmm.app.HBonds
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.

    """
    def __init__(self, **kwargs):
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
        self.num_atoms_per_solute = 3   # Water
        self.num_atoms_per_solvent = 3  # Water
        self.num_solvent_molecules = 8

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


class H2OinH2O(nn.Module, TargetDistribution):
    def __init__(
        self,
        temperature=1000,
        energy_cut=1.0e8,
        energy_max=1.0e20,
        n_threads=4,
    ):
        """
        Boltzmann distribution of Alanine dipeptide
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        """
        super(H2OinH2O, self).__init__()

        # Steps to take:
        # 1. Load topology of solute.
        # 2. Solvate the solute.
        # 3. Add the solute and solvent force fields.
        # 4. Add the implicit solvent force field / external potential term.

        system = WaterInWaterBox()

        # TODO: Do we want to run a quick simulation to get a sense of the coordinate magnitude for normalisation?
        #  If so, we can run a quick simulation here, as in aldp.py to obtain `transform_data`.

        self.coordinate_transform = WaterCoordinateTransform(system)

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
import torch
from torch import nn
import numpy as np

from fab.target_distributions.base import TargetDistribution
from fab.transforms.water_transform import WaterCoordinateTransform

import boltzgen as bg
import normflows as nf
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from openmmtools.testsystems import TestSystem, get_data_filename
import mdtraj
import tempfile


class WaterInWaterBox(TestSystem):

    """Water box with water micro-solvation system.

    Parameters
    ----------
    constraints : optional, default=openmm.app.HBonds
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.

    """
    def __init__(self, constraints=app.HBonds, hydrogenMass=None, **kwargs):
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

        # pdb example:
        # Initial solute molecule
        pdb = app.PDBFile("/home/timsey/HDD/data/molecules/solvents/water.pdb")
        # Add force field for molecule
        forcefield = app.ForceField("amber14/tip3pfb.xml")
        self.system = forcefield.createSystem(pdb.topology)
        # Add solvent
        modeller = app.modeller.Modeller(pdb.topology, pdb.positions)
        # TODO: Does this mutate self.system as well?
        modeller.addSolvent(self.system, model='tip3p', numAdded=8)  # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’

        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()


class H2OinH2O(nn.Module, TargetDistribution):
    def __init__(
        self,
        data_path=None,
        temperature=1000,
        energy_cut=1.0e8,
        energy_max=1.0e20,
        n_threads=4,
        transform="z_matrix",
        ind_circ_dih=[],
        shift_dih=False,
        shift_dih_params={"hist_bins": 100},
        default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env="vacuum",
    ):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
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
        sim = None

        # TODO: Transform should do:
        #  Z --> X:
        #   Add [0, 0, 0] coordinate of the solute.
        #   Divide coordinates into blocks corresponding to water molecules.
        #   Reorder based on the oxygen atom bond distances. This ensures molecular permutation invariance.
        #    NOTE: This means we cannot use different base distributions for different types of bonds, because the
        #     order with get mixed. Although we can do different things for the oxygen and hydrogen, since we're
        #     fixing the block size. This works in general if we have a single type of solvent, and assume the solute
        #     is the first block.
        #   Maybe also reorder the hydrogen atoms based on the bond distance to their oxygen atom.
        #   Starting from the first atom, revert coordinates to Cartesian.
        #   Keep track of the det log Jacobian for all transformations: polar to Cartesian, scaling, rotations.
        #    Q: What about reordering? Should we take the determinant of the permutation matrix? 1 or -1.
        #  X --> Z:
        #   Same, but now do the inverse coordinate transform after the blocking and reordering. The reordering works
        #   both directions, because we can determine Cartesian distance easily from both Z and X coordinates.

        self.coordinate_transform = WaterCoordinateTransform(
            transform_data,
            ndim,
            z_matrix,
            cart_indices,
            mode=transform,
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,)

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
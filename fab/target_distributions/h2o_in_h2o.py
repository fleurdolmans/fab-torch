import torch
from torch import nn
import numpy as np

from fab.target_distributions.base import TargetDistribution

import boltzgen as bg
import normflows as nf
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from openmmtools.testsystems import TestSystem, get_data_filename
import mdtraj
import tempfile


class WaterCoordinateTransform(nf.flows.Flow):
    """
    Coordinate transform for Boltzmann generators, see
    https://science.sciencemag.org/content/365/6457/eaaw1147
    The code of this function was taken from
    https://github.com/maccallumlab/BoltzmannGenerator
    Meaning of forward and backward pass are switched to meet
    convention of normflows package
    """
    def __init__(self, data, n_dim, z_matrix, backbone_indices,
                 mode='mixed', ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}):
        """
        Constructor
        :param data: Data used to initialize transformation
        :param n_dim: Number of dimensions in original space
        :param z_matrix: Defines which atoms to represent in internal coordinates
        :param backbone_indices: Indices of atoms of backbone, will be left in
        cartesian coordinates or are the last to be converted to internal coordinates
        :param mode: Mode of the coordinate transform, can be mixed or internal
        """
        super().__init__()

    def forward(self, z):
        # Transform Z --> X. Return x and log det Jacobian.
        raise NotImplementedError()
        return x, log_det

    def inverse(self, x):
        # Transform X --> Z. Return z and log det Jacobian.
        raise NotImplementedError()
        return z, log_det


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


# Ferry's code for converting between xyz and z-matrix coordinates.
def unit_vector(vector):
    """" Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_dist(v1, v2):
    return np.linalg.norm(np.subtract(v1, v2))


def get_angle(atom1, atom2, atom3):
    """" Returns the angle between three atoms in radian."""
    v1 = np.subtract(atom2, atom1)
    v2 = np.subtract(atom2, atom3)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_dihedral(atom1, atom2, atom3, atom4):
    """" Returns the dihidral between four atoms in radian."""

    b0 = np.subtract(atom1, atom2)  # intentionally different from the rest
    b1 = np.subtract(atom3, atom2)
    b2 = np.subtract(atom4, atom3)

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def rotation_matrix(axis, angle):
    """
    Euler-Rodrigues formula for rotation matrix
    """
    # Normalize the axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def disp_vector(atom1, atom2, atom3, zmat):
    """Find the cartesian coordinates of the atom"""
    distance, angle, dihedral = zmat
    dihedral = dihedral

    # Vector pointing from atom3 to atom2
    a = np.subtract(atom2, atom3)

    # Vector pointing from atom1 to atom3
    b = np.subtract(atom3, atom1)

    # Vector of length distance pointing from atom1 to atom2
    d = distance * a / np.linalg.norm(a)

    # Vector normal to plane defined by atom1, atom2, atom3
    normal = np.cross(a, b)

    # Rotate d by the angle around the normal to the plane defined by atom1, atom2, atom3
    d = np.dot(rotation_matrix(normal, angle), d)

    # Rotate d around a by the dihedral
    d = np.dot(rotation_matrix(a, dihedral), d)
    return d


def cartesian_to_zmat(coords):
    """
    Converts cartesian coordinates to the zmatrix

    Args:
        coords (list): list of frames in the xyz format.

    returns:
        zmatrix (list)
    """
    zmat = []

    for frame in coords:  # TODO: Order should be determined by bond distance then?
        tmp = []
        for i, atom in enumerate(frame):
            if i == 0:  # the first atom
                pass  # TODO: This should just correspond to [0, 0, 0], so maybe we subtract the first atom from all the others?
            elif i == 1:  # the second atom; just distance
                dist = get_dist(frame[0], frame[1])
                tmp.append(np.asarray([dist]))
            elif i == 2:  # the third atom; distance and angle
                dist = get_dist(frame[0], frame[2])
                angle = get_angle(frame[1], frame[0], frame[i])
                tmp.append(np.asarray([dist, angle]))
            elif i > 2:  # everything after the first three atoms
                dist = get_dist(frame[0], frame[i])
                angle = get_angle(frame[1], frame[0], frame[i])
                dihedral = get_dihedral(frame[2], frame[1], frame[0], frame[i])
                tmp.append(np.asarray([dist, angle, dihedral]))
        zmat.append(np.asarray(tmp))
    return np.asarray(zmat)


def zmat_to_cartesian(zmat):  # Z --> X
    # TODO: Jacobian is the Jacobian of 3D polar coordinate transform. r^2 sin((azimuthal angle).
    #  Taking the log, this apparently gives:
    #  jac = -torch.sum(
    #       2 * torch.log(bonds) + torch.log(torch.abs(torch.sin(angles))), dim=1
    #  )
    #  Here we can identify bonds with r, and angles with the azimuthal angle. Curious why the dihedral then
    #  actually corresponds to theta (which is the horizontal angle, and also the second coordinate usually).
    #  Also, while the polar transform has this determinant for polar --> cartesian, the BG code seems to use it
    #  for cartesian --> polar (or actually X --> Z). Oh actually, there is a minus sign, and the
    #  log det jac of the inverse transform = -log det jac of the forward. So this is correct. One can to through all
    #  the computations for cartesian --> polar transform (which are much more annoying) and find the same thing.
    #  But do note that their InternalCoordinateTransform.inverse() does not seem to use this? Or no, it does, but
    #  only for the 4 cartesian coordinates, see reconstruct_cart(). Weird.
    #  NOTE: Don't forget the Jacobian of scaling!
    coords = []
    for frame in zmat:  # TODO: Order should be determined by bond distance then?
        tmp = []  # placeholder for coordinates of current frame
        for i in range(len(frame) + 1):
            if i == 0:  # add first atom
                tmp.append([0, 0, 0])  # TODO: Cartesian first atom should then also correspond to [0, 0, 0] probably.
            elif i == 1:  # add second atom
                distance = frame[0][0]
                tmp.append([distance, 0, 0])
            elif i == 2:  # add third atom
                atom1, atom2 = tmp[:2]
                distance, angle = frame[1]
                v1 = unit_vector(np.subtract(atom2, atom1))
                d = distance * v1
                disp = np.dot(rotation_matrix([0, 0, 1], angle), d)  # normal is the z-axis
                new_atom = np.add(atom1, disp)
                tmp.append(list(new_atom))
            elif i > 2:  # fourth atom and onward
                disp = disp_vector(tmp[2], tmp[1], tmp[0], frame[i - 1])
                new_atom = np.add(tmp[0], disp)
                tmp.append(list(new_atom))
        coords.append(tmp)
    return coords

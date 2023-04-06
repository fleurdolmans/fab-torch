import torch
from torch import nn
import numpy as np

from fab.target_distributions.base import TargetDistribution

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
import mdtraj
import tempfile

import normflows as nf
import multiprocessing as mp
import math


class AldpBoltzmann(nn.Module, TargetDistribution):
    def __init__(
        self,
        data_path=None,
        temperature=1000,
        energy_cut=1.0e8,
        energy_max=1.0e20,
        n_threads=4,
        transform="internal",
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
        super(AldpBoltzmann, self).__init__()

        # Define molecule parameters
        ndim = 66
        if transform == "mixed":
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19]),
            ]
            cart_indices = [6, 8, 9, 10, 14]
        elif transform == "internal":
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19]),
            ]
            cart_indices = [8, 6, 14]

        # System setup
        if env == "vacuum":
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == "implicit":
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError("This environment is not implemented.")
        sim = app.Simulation(
            system.topology,
            system.system,
            mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
            mm.Platform.getPlatformByName("Reference"),
        )

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
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + "/aldp.pt"
            torch.save(torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path)

            del traj_sim

        if data_path[-2:] == "h5":
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == "pt":
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError("Loading data or this format is not implemented.")

        # Set distribution
        # bg.distributions.TransformedBoltzmann() calls the bg.flows.CoordinateTransform.forward() when computing
        #  energy of a flow sample (Z to X).
        #  CoordinateTransform subclasses nf.flows.Flow, presumably because a coordinate transform = invertible = flow.
        #  CoordinateTransform.forward() therefore calls inverse() of the underlying transform, e.g.
        #  bg.internal.CompleteInternalCoordinateTransform, which indeed does the transformation from Z to X
        #  (accordingly, forward() of bg.internal.CompleteInternalCoordinateTransform goes X to Z).
        # self.coordinate_transform = bg.flows.CoordinateTransform(
        #     transform_data,
        #     ndim,
        #     z_matrix,
        #     cart_indices,
        #     mode=transform,
        #     ind_circ_dih=ind_circ_dih,
        #     shift_dih=shift_dih,
        #     shift_dih_params=shift_dih_params,
        #     default_std=default_std,
        # )
        #
        # if n_threads > 1:
        #     self.p = bg.distributions.TransformedBoltzmannParallel(
        #         system,
        #         temperature,
        #         energy_cut=energy_cut,
        #         energy_max=energy_max,
        #         transform=self.coordinate_transform,
        #         n_threads=n_threads,
        #     )
        # else:
        #     self.p = bg.distributions.TransformedBoltzmann(
        #         sim.context,
        #         temperature,
        #         energy_cut=energy_cut,
        #         energy_max=energy_max,
        #         transform=self.coordinate_transform,
        #     )

        self.coordinate_transform = CoordinateTransform(
            transform_data,
            ndim,
            z_matrix,
            cart_indices,
            mode=transform,
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
        )

        if n_threads > 1:
            self.p = TransformedBoltzmannParallel(
                system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
                n_threads=n_threads,
            )
        else:
            self.p = TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
            )

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}





# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# Boltzmann Generator package stuff for understanding how coordinate transforms are implemented.
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------





class CoordinateTransform(nf.flows.Flow):
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
        # if mode == 'mixed':
        #     self.transform = MixedTransform(n_dim, backbone_indices, z_matrix, data,
        #                                           ind_circ_dih, shift_dih, shift_dih_params,
        #                                           default_std)
        if mode == 'internal':
            self.transform = CompleteInternalCoordinateTransform(n_dim, z_matrix,
                                                backbone_indices, data, ind_circ_dih,
                                                shift_dih, shift_dih_params, default_std)
        else:
            raise NotImplementedError('This mode is not implemented.')

    def forward(self, z):
        z_, log_det = self.transform.inverse(z)  # Z --> X
        return z_, log_det

    def inverse(self, z):
        z_, log_det = self.transform.forward(z)  # X --> Z
        return z_, log_det


class Scaling(nf.flows.Flow):
    """
    Applys a scaling factor
    """
    def __init__(self, mean, log_scale):
        """
        Constructor
        :param means: The mean of the previous layer
        :param log_scale: The log of the scale factor to apply
        """
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_parameter('log_scale', torch.nn.Parameter(log_scale))

    def forward(self, z):
        scale = torch.exp(self.log_scale)
        z_ = (z-self.mean) * scale + self.mean
        logdet = torch.log(scale) * self.mean.shape[0]
        return z_, logdet

    def inverse(self, z):
        scale = torch.exp(self.log_scale)
        z_ = (z-self.mean) / scale + self.mean
        logdet = -torch.log(scale) * self.mean.shape[0]
        return z_, logdet


class AddNoise(nf.flows.Flow):
    """
    Adds a small amount of Gaussian noise
    """
    def __init__(self, log_std):
        """
        Constructor
        :param log_std: The log standard deviation of the noise
        """
        super().__init__()
        self.register_parameter('log_std', torch.nn.Parameter(log_std))

    def forward(self, z):
        eps = torch.randn_like(z)
        z_ = z + torch.exp(self.log_std) * eps
        logdet = torch.zeros(z_.shape[0])
        return z_, logdet

    def inverse(self, z):
        return self.forward(z)


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise NotImplementedError()


def calc_bonds(ind1, ind2, coords):
    """Calculate bond lengths

    Parameters
    ----------
    ind1 : torch.LongTensor
        A n_bond x 3 tensor of indices for the coordinates of particle 1
    ind2 : torch.LongTensor
        A n_bond x 3 tensor of indices for the coordinates of particle 2
    coords : torch.tensor
        A n_batch x n_coord tensor of flattened input coordinates
    """
    p1 = coords[:, ind1]
    p2 = coords[:, ind2]
    return torch.norm(p2 - p1, dim=2)


def calc_angles(ind1, ind2, ind3, coords):
    b = coords[:, ind1]
    c = coords[:, ind2]
    d = coords[:, ind3]
    bc = b - c
    bc = bc / torch.norm(bc, dim=2, keepdim=True)
    cd = d - c
    cd = cd / torch.norm(cd, dim=2, keepdim=True)
    cos_angle = torch.sum(bc * cd, dim=2)
    angle = torch.acos(cos_angle)
    return angle


def calc_dihedrals(ind1, ind2, ind3, ind4, coords):
    a = coords[:, ind1]
    b = coords[:, ind2]
    c = coords[:, ind3]
    d = coords[:, ind4]

    b0 = a - b
    b1 = c - b
    b1 = b1 / torch.norm(b1, dim=2, keepdim=True)
    b2 = d - c

    v = b0 - torch.sum(b0 * b1, dim=2, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=2, keepdim=True) * b1
    x = torch.sum(v * w, dim=2)
    b1xv = torch.cross(b1, v, dim=2)
    y = torch.sum(b1xv * w, dim=2)
    angle = torch.atan2(y, x)
    return -angle


def reconstruct_cart(cart, ref_atoms, bonds, angles, dihs):
    # Get the positions of the 4 reconstructing atoms
    p1 = cart[:, ref_atoms[:, 0], :]
    p2 = cart[:, ref_atoms[:, 1], :]
    p3 = cart[:, ref_atoms[:, 2], :]

    bonds = bonds.unsqueeze(2)
    angles = angles.unsqueeze(2)
    dihs = dihs.unsqueeze(2)

    # Compute the log jacobian determinant.
    jac = torch.sum(
        2 * torch.log(torch.abs(bonds.squeeze(2)))
        + torch.log(torch.abs(torch.sin(angles.squeeze(2)))),
        dim=1,
    )

    # Reconstruct the position of p4
    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=2)
    n = n / torch.norm(n, dim=2, keepdim=True)
    nn = torch.cross(v1, n, dim=2)
    nn = nn / torch.norm(nn, dim=2, keepdim=True)

    n = n * torch.sin(dihs)
    nn = nn * torch.cos(dihs)

    v3 = n + nn
    v3 = v3 / torch.norm(v3, dim=2, keepdim=True)
    v3 = v3 * bonds * torch.sin(angles)

    v1 = v1 / torch.norm(v1, dim=2, keepdim=True)
    v1 = v1 * bonds * torch.cos(angles)

    # Store the final position in x
    new_cart = p1 + v3 - v1

    return new_cart, jac


class InternalCoordinateTransform(Transform):
    def __init__(self, dims, z_indices=None, cart_indices=None, data=None,
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}):
        super().__init__()
        self.dims = dims
        with torch.no_grad():
            # Setup indexing.
            self._setup_indices(z_indices, cart_indices)
            self._validate_data(data)
            # Setup the mean and standard deviations for each internal coordinate.
            #  These are computed by transforming data X --> Z and then computing means and stds.
            #  These values are then used to unnormalise the internal coordinates in the Z --> X direction,
            #   so that the resulting output fits the values in the Boltzmann distribution. Note that the learned
            #   values will still be off initially, as the current normalisation only makes it so the flow needs to
            #   output closer-to-unit values (the flow still needs to learn to actually do this).
            transformed, _ = self._fwd(data)
            # Normalize
            self.default_std = default_std
            self.ind_circ_dih = ind_circ_dih
            self._setup_mean_bonds(transformed)
            transformed[:, self.bond_indices] -= self.mean_bonds
            self._setup_std_bonds(transformed)
            transformed[:, self.bond_indices] /= self.std_bonds
            self._setup_mean_angles(transformed)
            transformed[:, self.angle_indices] -= self.mean_angles
            self._setup_std_angles(transformed)
            transformed[:, self.angle_indices] /= self.std_angles
            self._setup_mean_dih(transformed)
            transformed[:, self.dih_indices] -= self.mean_dih
            self._fix_dih(transformed)
            self._setup_std_dih(transformed)
            transformed[:, self.dih_indices] /= self.std_dih
            if shift_dih:
                val = torch.linspace(-math.pi, math.pi,
                                     shift_dih_params['hist_bins'])
                for i in self.ind_circ_dih:
                    dih = transformed[:, self.dih_indices[i]]
                    dih = dih * self.std_dih[i] + self.mean_dih[i]
                    dih = (dih + math.pi) % (2 * math.pi) - math.pi
                    hist = torch.histc(dih, bins=shift_dih_params['hist_bins'],
                                       min=-math.pi, max=math.pi)
                    self.mean_dih[i] = val[torch.argmin(hist)] + math.pi
                    dih = (dih - self.mean_dih[i]) / self.std_dih[i]
                    dih = (dih + math.pi) % (2 * math.pi) - math.pi
                    transformed[:, self.dih_indices[i]] = dih
            scale_jac = -(
                    torch.sum(torch.log(self.std_bonds))
                    + torch.sum(torch.log(self.std_angles))
                    + torch.sum(torch.log(self.std_dih))
            )
            self.register_buffer("scale_jac", scale_jac)

    def forward(self, x, context=None):
        trans, jac = self._fwd(x)
        trans[:, self.bond_indices] -= self.mean_bonds
        trans[:, self.bond_indices] /= self.std_bonds
        trans[:, self.angle_indices] -= self.mean_angles
        trans[:, self.angle_indices] /= self.std_angles
        trans[:, self.dih_indices] -= self.mean_dih
        self._fix_dih(trans)
        trans[:, self.dih_indices] /= self.std_dih
        return trans, jac + self.scale_jac

    def _fwd(self, x):
        x = x.clone()
        # we can do everything in parallel...
        inds1 = self.inds_for_atom[self.rev_z_indices[:, 1]]
        inds2 = self.inds_for_atom[self.rev_z_indices[:, 2]]
        inds3 = self.inds_for_atom[self.rev_z_indices[:, 3]]
        inds4 = self.inds_for_atom[self.rev_z_indices[:, 0]]

        # Calculate the bonds, angles, and torions for a batch.
        bonds = calc_bonds(inds1, inds4, coords=x)
        angles = calc_angles(inds2, inds1, inds4, coords=x)
        dihedrals = calc_dihedrals(inds3, inds2, inds1, inds4, coords=x)

        jac = -torch.sum(
            2 * torch.log(bonds) + torch.log(torch.abs(torch.sin(angles))), dim=1
        )

        # Replace the cartesian coordinates with internal coordinates.
        x[:, inds4[:, 0]] = bonds
        x[:, inds4[:, 1]] = angles
        x[:, inds4[:, 2]] = dihedrals
        return x, jac

    def inverse(self, x, context=None):
        # Gather all of the atoms represented as cartesisan coordinates.
        n_batch = x.shape[0]
        cart = x[:, self.init_cart_indices].view(n_batch, -1, 3)

        # Setup the log abs det jacobian
        jac = x.new_zeros(x.shape[0])
        self.angle_loss = torch.zeros_like(jac)

        # Loop over all of the blocks, where all of the atoms in each block
        # can be built in parallel because they only depend on atoms that
        # are already cartesian. `atoms_to_build` lists the `n` atoms
        # that can be built as a batch, where the indexing refers to the
        # original atom order. `ref_atoms` has size n x 3, where the indexing
        # refers to the position in `cart`, rather than the original order.
        for block in self.rev_blocks:
            atoms_to_build = block[:, 0]
            ref_atoms = block[:, 1:]

            # Get all of the bonds by retrieving the appropriate columns and
            # un-normalizing.
            bonds = (
                    x[:, 3 * atoms_to_build]
                    * self.std_bonds[self.atom_to_stats[atoms_to_build]]
                    + self.mean_bonds[self.atom_to_stats[atoms_to_build]]
            )

            # Get all of the angles by retrieving the appropriate columns and
            # un-normalizing.
            angles = (
                    x[:, 3 * atoms_to_build + 1]
                    * self.std_angles[self.atom_to_stats[atoms_to_build]]
                    + self.mean_angles[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the dihedrals by retrieving the appropriate columns and
            # un-normalizing.
            dihs = (
                    x[:, 3 * atoms_to_build + 2]
                    * self.std_dih[self.atom_to_stats[atoms_to_build]]
                    + self.mean_dih[self.atom_to_stats[atoms_to_build]]
            )

            # Compute angle loss
            self.angle_loss = self.angle_loss + self._periodic_angle_loss(angles)
            self.angle_loss = self.angle_loss + self._periodic_angle_loss(dihs)

            # Fix the dihedrals to lie in [-pi, pi].
            dihs = torch.where(dihs < math.pi, dihs + 2 * math.pi, dihs)
            dihs = torch.where(dihs > math.pi, dihs - 2 * math.pi, dihs)

            # Compute the cartesian coordinates for the newly placed atoms.
            new_cart, cart_jac = reconstruct_cart(cart, ref_atoms, bonds, angles, dihs)
            jac = jac + cart_jac

            # Concatenate the cartesian coordinates for the newly placed
            # atoms onto the full set of cartesian coordiantes.
            cart = torch.cat([cart, new_cart], dim=1)
        # Permute cart back into the original order and flatten.
        cart = cart[:, self.rev_perm_inv]
        cart = cart.view(n_batch, -1)
        return cart, jac - self.scale_jac

    def _setup_mean_bonds(self, x):
        mean_bonds = torch.mean(x[:, self.bond_indices], dim=0)
        self.register_buffer("mean_bonds", mean_bonds)

    def _setup_std_bonds(self, x):
        # Adding 1e-4 might help for numerical stability but results in some
        # dimensions being not properly normalised e.g. bond lengths
        # which can have stds of the order 1e-7
        # The flow will then have to fit to a very concentrated dist
        if x.shape[0] > 1:
            std_bonds = torch.std(x[:, self.bond_indices], dim=0)
        else:
            std_bonds = torch.ones_like(self.mean_bonds) \
                        * self.default_std['bond']
        self.register_buffer("std_bonds", std_bonds)

    def _setup_mean_angles(self, x):
        mean_angles = torch.mean(x[:, self.angle_indices], dim=0)
        self.register_buffer("mean_angles", mean_angles)

    def _setup_std_angles(self, x):
        if x.shape[0] > 1:
            std_angles = torch.std(x[:, self.angle_indices], dim=0)
        else:
            std_angles = torch.ones_like(self.mean_angles) \
                         * self.default_std['angle']
        self.register_buffer("std_angles", std_angles)

    def _setup_mean_dih(self, x):
        sin = torch.mean(torch.sin(x[:, self.dih_indices]), dim=0)
        cos = torch.mean(torch.cos(x[:, self.dih_indices]), dim=0)
        mean_dih = torch.atan2(sin, cos)
        self.register_buffer("mean_dih", mean_dih)

    def _fix_dih(self, x):
        dih = x[:, self.dih_indices]
        dih = (dih + math.pi) % (2 * math.pi) - math.pi
        x[:, self.dih_indices] = dih

    def _setup_std_dih(self, x):
        if x.shape[0] > 1:
            std_dih = torch.std(x[:, self.dih_indices], dim=0)
        else:
            std_dih = torch.ones_like(self.mean_dih) \
                      * self.default_std['dih']
            std_dih[self.ind_circ_dih] = 1.
        self.register_buffer("std_dih", std_dih)

    def _validate_data(self, data):
        if data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_dim = data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

    def _setup_indices(self, z_indices, cart_indices):
        n_atoms = self.dims // 3
        ind_for_atom = torch.zeros(n_atoms, 3, dtype=torch.long)
        for i in range(n_atoms):
            ind_for_atom[i, 0] = 3 * i
            ind_for_atom[i, 1] = 3 * i + 1
            ind_for_atom[i, 2] = 3 * i + 2
        self.register_buffer("inds_for_atom", ind_for_atom)

        sorted_z_indices = topological_sort(z_indices)
        sorted_z_indices = [
            [item[0], item[1][0], item[1][1], item[1][2]] for item in sorted_z_indices
        ]
        rev_z_indices = list(reversed(sorted_z_indices))

        mod = [item[0] for item in sorted_z_indices]
        modified_indices = []
        for index in mod:
            modified_indices.extend(self.inds_for_atom[index])
        bond_indices = list(modified_indices[0::3])  # First index mod 3 is index for bond distance
        angle_indices = list(modified_indices[1::3])  # Second index mod 3 is index for angle
        dih_indices = list(modified_indices[2::3])  # Third index mod 3 is index for dihedral

        self.register_buffer("modified_indices", torch.LongTensor(modified_indices))
        self.register_buffer("bond_indices", torch.LongTensor(bond_indices))
        self.register_buffer("angle_indices", torch.LongTensor(angle_indices))
        self.register_buffer("dih_indices", torch.LongTensor(dih_indices))
        self.register_buffer("sorted_z_indices", torch.LongTensor(sorted_z_indices))
        self.register_buffer("rev_z_indices", torch.LongTensor(rev_z_indices))

        #
        # Setup indexing for reverse pass.
        #
        # First, create an array that maps from an atom index into mean_bonds, std_bonds, etc.
        atom_to_stats = torch.zeros(n_atoms, dtype=torch.long)
        for i, j in enumerate(mod):
            atom_to_stats[j] = i
        self.register_buffer("atom_to_stats", atom_to_stats)

        # Next create permutation vector that is used in the reverse pass. This maps
        # from the original atom indexing to the order that the cartesian coordinates
        # will be built in. This will be filled in as we go.
        rev_perm = torch.zeros(n_atoms, dtype=torch.long)
        self.register_buffer("rev_perm", rev_perm)
        # Next create the inverse of rev_perm. This will be filled in as we go.
        rev_perm_inv = torch.zeros(n_atoms, dtype=torch.long)
        self.register_buffer("rev_perm_inv", rev_perm_inv)

        # Create the list of columns that form our initial cartesian coordintes.
        init_cart_indices = self.inds_for_atom[cart_indices].view(-1)
        self.register_buffer("init_cart_indices", init_cart_indices)

        # Update our permutation vectors for the initial cartesian atoms.
        for i, j in enumerate(cart_indices):
            self.rev_perm[i] = torch.as_tensor(j, dtype=torch.long)
            self.rev_perm_inv[j] = torch.as_tensor(i, dtype=torch.long)

        # Break Z into blocks, where all of the atoms within a block can be built
        # in parallel, because they only depend on already-cartesian atoms.
        all_cart = set(cart_indices)
        current_cart_ind = i + 1
        blocks = []
        while sorted_z_indices:
            next_z_indices = []
            next_cart = set()
            block = []
            for atom1, atom2, atom3, atom4 in sorted_z_indices:
                if (atom2 in all_cart) and (atom3 in all_cart) and (atom4 in all_cart):
                    # We can build this atom from existing cartesian atoms, so we add
                    # it to the list of cartesian atoms available for the next block.
                    next_cart.add(atom1)

                    # Add this atom to our permutation marices.
                    self.rev_perm[current_cart_ind] = atom1
                    self.rev_perm_inv[atom1] = current_cart_ind
                    current_cart_ind += 1

                    # Next, we convert the indices for atoms2-4 from their normal values
                    # to the appropriate indices to index into the cartesian array.
                    atom2_mod = self.rev_perm_inv[atom2]
                    atom3_mod = self.rev_perm_inv[atom3]
                    atom4_mod = self.rev_perm_inv[atom4]

                    # Finally, we append this information to the current block.

                    block.append([atom1, atom2_mod, atom3_mod, atom4_mod])
                else:
                    # We can't build this atom from existing cartesian atoms,
                    # so put it on the list for next time.
                    next_z_indices.append([atom1, atom2, atom3, atom4])
            sorted_z_indices = next_z_indices
            all_cart = all_cart.union(next_cart)
            block = torch.as_tensor(block, dtype=torch.long)
            blocks.append(block)
        self.rev_blocks = blocks

    def _periodic_angle_loss(self, angles):
        """
        Penalizes angles outside the range [-pi, pi]

        Prevents violating invertibility in internal coordinate transforms.
        Computes

            L = (a-pi) ** 2 for a > pi
            L = (a+pi) ** 2 for a < -pi

        and returns the sum over all angles per batch.
        """
        zero = torch.zeros(1, 1, dtype=angles.dtype).to(angles.device)
        positive_loss = torch.sum(torch.where(angles > math.pi, angles - math.pi, zero) ** 2, dim=-1)
        negative_loss = torch.sum(torch.where(angles < -math.pi, angles + math.pi, zero) ** 2, dim=-1)
        return positive_loss + negative_loss


def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted


class CompleteInternalCoordinateTransform(nn.Module):
    def __init__(
            self,
            n_dim,
            z_mat,
            cartesian_indices,
            data,
            ind_circ_dih=[],
            shift_dih=False,
            shift_dih_params={'hist_bins': 100},
            default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}
    ):
        super().__init__()
        # cartesian indices are the atom indices of the atoms that are not
        # represented in internal coordinates but are left as cartesian
        # e.g. for 22 atoms it could be [4, 5, 6, 8, 14, 15, 16, 18]
        self.n_dim = n_dim
        self.len_cart_inds = len(cartesian_indices)
        assert self.len_cart_inds == 3

        # Create our internal coordinate transform
        self.ic_transform = InternalCoordinateTransform(
            n_dim, z_mat, cartesian_indices, data, ind_circ_dih,
            shift_dih, shift_dih_params, default_std
        )

        # permute puts the cartesian coords first then the internal ones
        # permute_inv does the opposite
        permute = torch.zeros(n_dim, dtype=torch.long)
        permute_inv = torch.zeros(n_dim, dtype=torch.long)
        all_ind = cartesian_indices + [row[0] for row in z_mat]
        for i, j in enumerate(all_ind):
            permute[3 * i + 0] = torch.as_tensor(3 * j + 0, dtype=torch.long)
            permute[3 * i + 1] = torch.as_tensor(3 * j + 1, dtype=torch.long)
            permute[3 * i + 2] = torch.as_tensor(3 * j + 2, dtype=torch.long)
            permute_inv[3 * j + 0] = torch.as_tensor(3 * i + 0, dtype=torch.long)
            permute_inv[3 * j + 1] = torch.as_tensor(3 * i + 1, dtype=torch.long)
            permute_inv[3 * j + 2] = torch.as_tensor(3 * i + 2, dtype=torch.long)
        self.register_buffer("permute", permute)
        self.register_buffer("permute_inv", permute_inv)

        data = data[:, self.permute]
        b1, b2, angle = self._convert_last_internal(data[:, :3 * self.len_cart_inds])
        self.register_buffer("mean_b1", torch.mean(b1))
        self.register_buffer("mean_b2", torch.mean(b2))
        self.register_buffer("mean_angle", torch.mean(angle))
        if b1.shape[0] > 1:
            self.register_buffer("std_b1", torch.std(b1))
            self.register_buffer("std_b2", torch.std(b2))
            self.register_buffer("std_angle", torch.std(angle))
        else:
            self.register_buffer("std_b1", b1.new_ones([]) * default_std['bond'])
            self.register_buffer("std_b2", b2.new_ones([]) * default_std['bond'])
            self.register_buffer("std_angle", angle.new_ones([]) * default_std['angle'])
        scale_jac = -(torch.log(self.std_b1) + torch.log(self.std_b2) + torch.log(self.std_angle))
        self.register_buffer("scale_jac", scale_jac)

    def forward(self, x):  # X --> Z
        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])

        # Run transform to internal coordinates.
        x, new_jac = self.ic_transform.forward(x)
        jac = jac + new_jac

        # Permute to put PCAs first.
        x = x[:, self.permute]

        # Split off the PCA coordinates and internal coordinates
        int_coords = x[:, 3 * self.len_cart_inds:]

        # Compute last internal coordinates
        b1, b2, angle = self._convert_last_internal(x[:, :3 * self.len_cart_inds])
        jac = jac - torch.log(b2)
        # Normalize
        b1 -= self.mean_b1
        b1 /= self.std_b1
        b2 -= self.mean_b2
        b2 /= self.std_b2
        angle -= self.mean_angle
        angle /= self.std_angle
        jac = jac + self.scale_jac

        # Merge everything back together.
        x = torch.cat([b1[:, None], b2[:, None], angle[:, None]] + [int_coords], dim=1)

        return x, jac

    def inverse(self, x):  # Z --> X
        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])

        # Separate the internal coordinates
        b1, b2, angle = x[:, 0], x[:, 1], x[:, 2]
        int_coords = x[:, 3 * self.len_cart_inds - 6:]

        # Reconstruct first three atoms
        b1 = b1 * self.std_b1 + self.mean_b1
        b2 = b2 * self.std_b2 + self.mean_b2
        angle = angle * self.std_angle + self.mean_angle
        jac = jac - self.scale_jac
        cart_coords = x.new_zeros(x.shape[0], 3 * self.len_cart_inds)
        cart_coords[:, 3] = b1
        cart_coords[:, 6] = b2 * torch.cos(angle)
        cart_coords[:, 7] = b2 * torch.sin(angle)
        jac = jac + torch.log(b2)

        # Merge everything back together
        x = torch.cat([cart_coords] + [int_coords], dim=1)

        # Permute back into atom order
        x = x[:, self.permute_inv]

        # Run through inverse internal coordinate transform
        x, new_jac = self.ic_transform.inverse(x)
        jac = jac + new_jac

        return x, jac

    def _convert_last_internal(self, x):
        p1 = x[:, :3]
        p2 = x[:, 3:6]
        p3 = x[:, 6:9]
        p21 = p2 - p1
        p31 = p3 - p1
        b1 = torch.norm(p21, dim=1)
        b2 = torch.norm(p31, dim=1)
        cos_angle = torch.sum((p21) * (p31), dim=1) / b1 / b2
        angle = torch.acos(cos_angle)
        return b1, b2, angle


class Boltzmann(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterface.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)


class TransformedBoltzmann(nn.Module):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max, transform):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        """
        super().__init__()
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterface.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

        self.transform = transform

    def log_prob(self, z):
        z, log_det = self.transform(z)  # Z --> X
        return -self.norm_energy(z) + log_det


class BoltzmannParallel(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution using OpenMM to get energy and forces and processes the
    batch of states in parallel
    """
    def __init__(self, system, temperature, energy_cut, energy_max, n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, OpenMMEnergyInterfaceParallel.var_init,
                            (system, temperature))

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)


class TransformedBoltzmannParallel(nn.Module):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces and processes the batch of
    states in parallel
    """
    def __init__(self, system, temperature, energy_cut, energy_max, transform,
                 n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        super().__init__()
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, OpenMMEnergyInterfaceParallel.var_init,
                            (system, temperature))

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0],
            self.energy_cut, self.energy_max)

        self.transform = transform

    def log_prob(self, z):
        z_, log_det = self.transform(z)
        return -self.norm_energy(z_) + log_det


# Gas constant in kJ / mol / K
R = 8.314e-3


class OpenMMEnergyInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule / unit.mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule / unit.mole / unit.nanometer
                    )
                    / kBT
                )
                forces[i, :] = torch.from_numpy(-f)
        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


class OpenMMEnergyInterfaceParallel(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def var_init(sys, temp):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(sys.topology, sys.system,
                             mm.LangevinIntegrator(temp * unit.kelvin,
                                                   1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                             platform=mm.Platform.getPlatformByName('Reference'))
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        force = force.reshape(n_dim * 3)
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(
            OpenMMEnergyInterfaceParallel.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


def regularize_energy(energy, energy_cut, energy_max):
    # Cast inputs to same type
    energy_cut = energy_cut.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut
    )
    energy = torch.where(energy_finite, energy,
                         torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))
    return energy
import torch
from torch import nn
import numpy as np

import normflows as nf
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
        x, log_det = z_to_cartesian(z)
        return x, log_det

    def inverse(self, x):
        # Transform X --> Z. Return z and log det Jacobian.
        z, log_det = cartesian_to_z(x)
        return z, log_det


def cartesian_to_z(x):
    """
    Transform Cartesian coordinates to internal coordinates.
    :param x: Cartesian coordinates: n_batch x n_atoms x 3
    :return: Internal coordinates: n_batch x n_atoms x 3, and log det Jacobian
    """

    jac = x.new_zeros(x.shape[0])  # Setup log det Jacobian of the coordinate transform.

    # We define an order in coordinates, so that the system representation is invariant to permutations
    # of molecule labels. This is done by setting the solute oxygen atom to the origin, and sorting the remaining
    # oxygen atoms by their distance to the solute oxygen. The hydrogen atoms are sorted by their distance to the
    # azimuthal axis. We assume the rows are ordered as: [O, H, H, O, H, H, ...].
    x_centered = x - x[:, 0:1, :]  # Center x around the solute oxygen, which now has coordinates [0, 0, 0].

    oxygen_dists = x_centered[:, ::3, ]

    return z, log_det
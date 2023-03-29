import torch
from torch import nn
import numpy as np
import math

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
    def __init__(self, system):
        """
        Constructor
        :param system: TestSystem object with topology, system, positions, num_atoms_per_solute,
        num_atoms_per_solvent, and num_solvent_molecules.
        """
        super().__init__()

    def forward(self, z):
        # Transform Z --> X. Return x and log det Jacobian.
        x, log_det_jac = z_to_cartesian(z)
        return x, log_det_jac

    def inverse(self, x):
        # Transform X --> Z. Return z and log det Jacobian.
        z, log_det_jac = cartesian_to_z(x)
        return z, log_det_jac


def cartesian_to_z(x):
    """
    Transform Cartesian coordinates to internal coordinates.
    :param x: Cartesian coordinates: n_batch x n_atoms x 3
    :return: Spherical coordinates: n_batch x n_atoms x 3, and log det Jacobian
    """

    # Setup rotation axes
    z_axis = x.new_zeros(x.shape[0], 3)
    z_axis[:, 2] = 1
    y_axis = x.new_zeros(x.shape[0], 3)
    y_axis[:, 1] = 1

    # Set the first solute atom to the origin, and the second atom to the z-axis. The third atom is aligned with the
    #  yz-plane. We then define all other atoms w.r.t. this reference frame, allowing us to use 3D spherical
    #  coordinates. Note that this is equivalent to using planar angles between the solute-molecule-plane and the plane
    #  formed by the first two solute atoms and any other atom in the system.
    x = setup_coordinate_system(x, z_axis, y_axis)

    solute_atom0 = x[:, 0, :]
    if not torch.isclose(torch.zeros(x.shape[0], 3), solute_atom0).all():
        raise ValueError(
            "Expected solute atom0 to be at origin, but norm mismatch (n_batch): {}".format(
                torch.norm(torch.zeros(x.shape[0], 3) - solute_atom0, dim=1)
            )
        )

    z = x.new_zeros(x.shape)
    log_det_jac = x.new_zeros(x.shape[0])  # n_batch

    for atom_num in range(x.shape[1]):
        atom_coords = x[:, atom_num, :]

        if atom_num == 0:  # solute reference: [0, 0, 0] in z
            if not torch.isclose(torch.zeros(x.shape[0], 3), atom_coords).all():
                raise ValueError(
                    "Expected first atom to be at origin, but norm mismatch (n_batch): {}".format(
                        torch.norm(torch.zeros(x.shape[0], 3) - atom_coords, dim=1)
                    )
                )
        elif atom_num == 1:  # axis-aligned atom [r, 0, 0]
            r = torch.norm(atom_coords, dim=-1)  # Vector norm
            # This should match the distance along z axis, since the vector should be aligned with z-axis.
            if not torch.isclose(r, atom_coords[:, -1]).all():
                raise ValueError(
                    "Expected atom to be aligned with z-axis, but norm mismatch (n_batch): {}".format(
                        torch.abs(r - torch.norm(atom_coords, dim=1))
                    )
                )
            if not torch.isclose(unit_vector(z_axis), unit_vector(atom_coords), atol=1e-5).all():
                raise ValueError(
                    "Expected atom to be aligned with z-axis, but vector mismatch (n_batch): {}".format(
                        torch.abs(unit_vector(z_axis) - unit_vector(atom_coords))
                    )
                )
            z[:, atom_num, 0] = r
        elif atom_num == 2:  # r and phi
            r = torch.norm(atom_coords, dim=-1)
            phi, _ = get_angle_and_normal(z_axis, solute_atom0, atom_coords)
            z[:, atom_num, 0] = r
            z[:, atom_num, 1] = phi
        else:  # r, phi, theta
            r = torch.norm(atom_coords, dim=-1)
            phi, _ = get_angle_and_normal(z_axis, solute_atom0, atom_coords)
            solute_atom0 = x[:, 0, :]
            solute_atom1 = x[:, 1, :]
            solute_atom2 = x[:, 2, :]  # TODO: make this a dummy if the solute does not define a plane
            theta = get_theta(solute_atom0, solute_atom1, solute_atom2, atom_coords, phi)
            z[:, atom_num, 0] = r
            z[:, atom_num, 1] = phi
            z[:, atom_num, 2] = theta

    log_det_jac += None  # TODO

    return z, log_det_jac


def z_to_cartesian(z):
    """
    Transform Spherical coordinates to Cartesian coordinates.
    :param z: Spherical coordinates: n_batch x n_atoms x 3
    :return: Cartesian coordinates: n_batch x n_atoms x 3, and log det Jacobian
    """

    # Setup rotation axes
    z_axis = z.new_zeros(z.shape[0], 3)
    z_axis[:, 2] = 1
    x_axis = z.new_zeros(z.shape[0], 3)
    x_axis[:, 0] = 1

    x = z.new_zeros(z.shape)
    log_det_jac = z.new_zeros(z.shape[0])  # n_batch

    # # TODO: Flow output should have 6 degrees of freedom masked out: these correspond to the orientation we choose
    # #  for the first three atoms.
    # mask = z.new_ones(z.shape)
    # mask[:, 0, :] = 0
    # mask[:, 1, 1:] = 0
    # mask[:, 2, 2] = 0
    # z = mask * z

    for atom_num in range(z.shape[1]):
        atom_coords = z[:, atom_num, :]

        if atom_num == 0:  # solute_atom0 reference: [0, 0, 0] in x
            if not torch.isclose(torch.zeros(z.shape[0], 3), atom_coords).all():
                raise ValueError(
                    "Expected first atom to be at origin, but norm mismatch (n_batch): {}".format(
                        torch.norm(torch.zeros(z.shape[0], 3) - atom_coords, dim=1)
                    )
                )
            continue
        elif atom_num == 1:  # axis-aligned atom: e.g., [0, 0, r] in x
            # phi and theta should be 0 for this atom
            if not torch.isclose(torch.zeros(z.shape[0], 2), atom_coords[:, 1:]).all():
                raise ValueError(
                    "Expected first atom to be axis aligned, but norm mismatch (n_batch): {}".format(
                        torch.norm(torch.zeros(z.shape[0], 2) - atom_coords[:, 1:], dim=1)
                    )
                )
            r = atom_coords[:, 0]
            x[:, atom_num, 2] = r
        elif atom_num == 2:  # distance and angle: reconstruct x
            # theta should be 0 for this atom
            if not torch.isclose(torch.zeros(z.shape[0], 1), atom_coords[:, 2:]).all():
                raise ValueError(
                    "Expected third atom to have theta = 0, but norm mismatch (n_batch): {}".format(
                        torch.norm(torch.zeros(z.shape[0], 1) - atom_coords[:, 2:], dim=1)
                    )
                )
            r = atom_coords[:, 0]  # atom-to-reconstruct r
            phi = atom_coords[:, 1]  # atom-to-reconstruct phi
            # Define normal for rotation: for X --> Z we took the x > 0 normal when rotating towards the
            #  alignment axis. Now we rotate away from the axis, so we take the negative angle.
            #  Alternatively, we can take the positive angle and the normal with x < 0.
            phi_rotation = rotation_matrix(x_axis, -phi)
            # Apply rotation to z-axis
            xyz = torch.einsum('bij,bj -> bi', phi_rotation, z_axis)
            x[:, atom_num, :] = xyz
        else:
            # We always use the first molecule as the anchor for the angles.
            r = atom_coords[:, 0]  # atom-to-reconstruct r
            phi = atom_coords[:, 1]  # atom-to-reconstruct phi
            theta = atom_coords[:, 2]  # atom-to-reconstruct dihedral
            # Step 1: rotate the z-axis vector around the positive x-axis (by convention) by -phi
            # NOTE: would prefer +phi, but we defined phi the other direction, and code for
            #  computing theta now depends on this convention.
            phi_rotation = rotation_matrix(x_axis, -phi)
            xyz = torch.einsum('bij,bj -> bi', phi_rotation, z_axis)
            # Step 2: rotate this vector around the positive z-axis (by convention) by theta (NOT -theta).
            theta_rotation = rotation_matrix(z_axis, theta)
            xyz = torch.einsum('bij,bj -> bi', theta_rotation, xyz)
            x[:, atom_num, :] = xyz

    log_det_jac += None  # TODO

    return x, log_det_jac


def setup_coordinate_system(x, z_axis, y_axis):
    """
    Sets up the global coordinate system that we transform into 3D spherical coordinates.

    The solute oxygen atom is set to the origin: this defines the internal coordinate r.
    The first hydrogen atom is aligned with the z-axis: this defines the internal coordinate phi.
    The second hydrogen atom is aligned with the yz-plane: this defines the internal coordinate theta.

    :param x: Cartesian coordinates: n_batch x n_atoms x 3
    :return: Internal coordinates: n_batch x n_atoms x 3, and log det Jacobian of the initial transformation.
    """

    x_centered = x - x[:, 0:1, :]  # Center x around the solute oxygen, which now has coordinates [0, 0, 0].

    solute_atom0 = x_centered[:, 0, :]  # e.g., oxygen atom: n_batch x 3 at [0, 0, 0], defines r.
    solute_atom1 = x_centered[:, 1, :]  # e.g., first hydrogen; will become [r, 0, 0], defines phi.
    solute_atom2 = x_centered[:, 2, :]  # e.g., second hydrogen; will become [r, phi, 0], defines theta.

    # First define phi w.r.t. the z-axis. This means we rotate solute_atom1 to align with the z-axis.
    # E.g. we want to align the first H atom with the z-axis.
    # The rotation should happen along the normal given by the z-O-H plane (for water).
    phi_rad, phi_axis = get_angle_and_normal(z_axis, solute_atom0, solute_atom1)
    phi_rotation = rotation_matrix(phi_rad, phi_axis)
    x_phi = torch.einsum('bij,bnj -> bni', phi_rotation, x_centered)

    # Now define theta w.r.t. the yz-plane. THis means we rotate solute_atom2 to align with the yz-plane.
    # We want to rotate around the z-axis, so that the first H atom stays fixed in place.
    # So to find the angle, we project the second H atom onto the xy-plane, and find its angle with the y-axis.
    # To project an atom onto a plane, we in general project the atom onto the plane normal,
    #  then subtract this component form the original to find the projection onto the plane.
    # However, we have an easy solution: projecting onto the xy-plane is just setting the z-component to 0.
    # Here we use the plane projection, because it means we don't have to mutate an existing object.
    xy_proj = solute_atom2 - unit_vector(z_axis) * torch.sum(z_axis * solute_atom2, dim=1, keepdim=True)
    # Angle between atom_to_align and projection, through origin (solute_atom0)
    theta_rad, _ = get_angle_and_normal(y_axis, solute_atom0, xy_proj, to_yz_plane=True)
    # We rotate around the alignment axis (z-axis) to put the molecule into the yz-plane
    theta_rotation = rotation_matrix(z_axis, theta_rad)
    # x rotated into the yz plane
    x = torch.einsum('bij,bnj -> bni', theta_rotation, x_phi)

    return x


def unit_vector(vector):
    """"
    Returns the unit vector of the vector.
    """
    return vector / torch.norm(vector, dim=-1, keepdim=True)


def get_angle_and_normal(atom1, atom2, atom3, to_yz_plane=False):
    """"
    Returns the angle between three atoms in radian, and the axis of rotation.

    atom2 is the atom located at vertex where we want to know the angle.

    When trying to align axes: atom1 = alignment axis, atom3 = axis to align. This ensures
    that the normal (axis of rotation) has the correct orientation w.r.t. the rotation angle.
    """
    v1 = atom2 - atom1
    v2 = atom2 - atom3
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    # atom3 x atom 1, e.g.: H x z^
    cross = torch.cross(v2_u, v1_u, dim=-1)  # normal vector
    dot = torch.sum(v1_u * v2_u, dim=-1)
    rads = torch.arccos(torch.clip(dot, -1.0, 1.0))

    # We need to fix the rotation axis orientation, so that we know how to reconstruct X from the angle
    #  information in Z. So we pick the convention that the rotation is the normal with x > 0.
    # This means that we sometimes flip the convention, so we need to adjust the angle appropriately.
    #  That is, when we find a normal with x < 0, we negate it, and adjust the angle as: rad = 2pi - rad.

    # Note that this can mess up if we are rotating vectors into the yz-plane, since the rotation axis has x=0 there.
    #  If so, we want to use the z > 0 vector as the normal.
    if to_yz_plane:
        if not torch.isclose(cross[:, 0], cross.new_zeros(cross.shape[0])).any():
            raise ValueError("Convention is to rotate normal with x!=0 if possible, but `to_yz_plane` was True.")
        sign = torch.sign(cross[:, 2])  # Flip direction if z < 0
    else:
        if torch.isclose(cross[:, 0], cross.new_zeros(cross.shape[0])).any():
            raise ValueError("Found rotation around axis with x=0. Set `to_yz_plane` to True.")
        sign = torch.sign(cross[:, 0])  # Flip direction if x < 0

    cross = sign.unsqueeze(-1) * cross
    # Adjust angles:
    #  This evaluates to: 2pi - rads if sign = -1, else: 0 + rads
    rads = 2 * math.pi * (1 - sign) / 2 + sign * rads

    return rads, cross


def get_theta(atom1, atom2, atom3, atom4, phi):
    """
    Returns the theta (3D spherical coordinates) between four atoms in radian.

    Theta is defined as the rotation of atom4 around the (atom2 - atom1) vector (with z > 0),
    with theta=0 in the plane defined by atom1, atom2 and atom3.

    NOTE: We need phi to determine in which half-volume (e.g., y > 0 or y < 0 if phi is defined w.r.t. z > 0)
     we find ourselves, as the rotation with theta rad is taken w.r.t. the opposite axis, depending (e.g.,
     the y > 0 axis if phi in [0, pi], but the y < 0 axis if phi in [pi, 2pi]). This is a bit annoying, but
     it's a result of doing the azimuthal angle first. In standard spherical coordinates phi is in [0, pi] and
     theta is in [0, 2pi], but in our case the opposite is true. We cannot take theta in [0, 2pi], because
     this will give a double cover of the r-sphere.
    NOTE: In principle, we could determine phi from the angle_vector (atom4 - atom1) again, rather than passing it
     as an argument.
    """
    # Rotation axis definition for theta (e.g., z-axis in our cases).
    rotation_axis = unit_vector(atom2 - atom1)
    # rotation_axis and plane_axis define the plane (e.g., O-H1 and O-H2)
    plane_axis = unit_vector(atom3 - atom1)
    # Get vector that lies in the theta=0 plane defined by the 3 atoms: plane_axis.
    # Project this vector onto the plane defined by the rotation axis (e.g., onto xy-plane)
    mag_along_normal = torch.sum(rotation_axis * plane_axis, dim=-1, keepdim=True)
    in_both_planes = plane_axis - unit_vector(rotation_axis) * mag_along_normal
    # Project the atom4 - atom1 vector onto the rotation_axis plane as well. We care about the
    #  angle between this vector and the in_both_planes vector around the rotation_axis.
    angle_vector = unit_vector(atom4 - atom1)
    #     print("angle_vector", (atom4 - atom1)[0])
    mag_along_normal = torch.sum(rotation_axis * angle_vector, dim=-1, keepdim=True)
    in_rot_plane = angle_vector - unit_vector(rotation_axis) * mag_along_normal
    # Find angle through inner product: this value being negative corresponds to phi > pi
    inner = torch.sum(unit_vector(in_both_planes) * unit_vector(in_rot_plane), dim=-1)
    theta = torch.arccos(inner)

    # NOTE: we define the rotation axis as (atom2 - atom1), which is orthogonal to the plane in which
    #  in_both_planes and in_rot_plane live. The computed angle corresponds to a rotation around either this
    #  axis, or its negation, depending the relative orientation of in_both_planes and in_rot_plane. This
    #  orientation can be determined with their cross-product, which is the actual axis of rotation!
    # Since we want the axis of rotation to be (atom2 - atom1), we need to check whether the actual axis
    #  aligns with this, and if not, change the rotation angle accordingly.
    cross = torch.cross(in_both_planes, in_rot_plane, dim=-1)
    # Does this axis align with the rotation_axis: +1 if aligned, -1 if opposite
    norm_sign = torch.sum(unit_vector(rotation_axis) * unit_vector(cross), dim=-1)  # n_batch
    #  This evaluates to: 2pi - theta if sign = -1, else: 0 + theta
    theta = 2 * math.pi * (1 - norm_sign) / 2 + norm_sign * theta

    # Now some magic to make the theta angle work out correctly. We need to treat every xy-quadrant separately.
    # If y > 0, x < 0; we need: new_theta = theta
    # If y > 0, x > 0; we need: new_theta = theta - 2pi
    # If y < 0, x > 0; we need: new_theta = theta - pi
    # If y < 0, x < 0; we need: new_theta = theta - pi
    # phi in [0, pi] means y > 0, phi in [pi, 2pi] means y < 0.
    if not ((phi > phi.new_ones(phi.shape) * math.pi).long() == (inner < 0)).all():
        raise ValueError("Given angle phi does not match atom4 vector orientation.")
    y_sign = -1 * torch.sign(phi - phi.new_ones(phi.shape) * math.pi)  # +1 if y > 0, -1 if y < 0
    x_sign = torch.sign(atom4[:, 0])  # +1 if x > 0, -1 if x < 0
    y_comp = (y_sign - 1) / 2 * math.pi  # -pi if y < 0
    xy_comp = ((y_sign + 1) / 2) * ((x_sign + 1) / 2) * -2 * math.pi  # -2pi if y > 0 and x > 0
    theta = theta + xy_comp + y_comp
    return theta


def rotation_matrix(rotation_axis, rotation_rad):
    # Euler-Rodrigues
    a = torch.cos(rotation_rad / 2)
    # orthogonal to both alignment and to-align atom.
    rot_axis = unit_vector(rotation_axis)
    # Why -rot_axis? Not on Wikipedia, but works (+ does not work, because it rotates in the wrong direction)
    xyz_rot = -rot_axis * torch.sin(rotation_rad / 2).unsqueeze(-1)
    b = xyz_rot[:, 0]
    c = xyz_rot[:, 1]
    d = xyz_rot[:, 2]

    rot_matrix = torch.stack([
        torch.stack([a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], dim=1),
        torch.stack([2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)], dim=1),
        torch.stack([2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c], dim=1)
    ], dim=2)

    return rot_matrix


def perm_parity(lst):
    # TODO: This is not used currently.
    """
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    """
    parity = 1
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity


def order_solvent_molecules(x, atoms_per_solute_mol, atoms_per_solvent_mol):
    # TODO: This is not used currently, as this is not actually doing anything in the
    #  Z --> X direction. What to do for permutation invariance in that setting?
    """
    Order oxygen atoms by their distance to the solute oxygen atom.

    :param x: Cartesian coordinates: n_batch x n_atoms x 3
    :param atoms_per_solute_mol: Number of atoms per solute molecule
    :param atoms_per_solvent_mol: Number of atoms per solvent molecule

    :return:
        Ordered Cartesian coordinates: n_batch x n_atoms x 3
        Parity of the permutation: n_batch
    """

    # Reference coordinates: e.g., the oxygen atoms
    ref_coords = x[:, atoms_per_solute_mol::atoms_per_solvent_mol, :]  # n_batch x num_atoms

    # Distances to anchor at origin
    ref_dists = torch.norm(ref_coords, dim=2)

    # Order reference atoms by distance to origin
    ref_permute = torch.argsort(ref_dists, dim=1) * atoms_per_solvent_mol

    # Start building full permutation tensor.
    #  First step is to permute reference atoms (e.g., oxygens) together with their hydrogens.
    #  Then we can start permuting the remaining atoms (e.g., hydrogens) within the molecules, if necessary.
    full_permute = torch.zeros(x.shape[:2])

    # Solute molecule is fixed; reflect this in permutation tensor.
    num_hydrogen = atoms_per_solvent_mol - 1
    for j in range(1, num_hydrogen + 1, 1):
        full_permute[:, j] = j

    # Inefficient way to build full permutation tensor
    for full_perm, ref_perm in zip(full_permute, ref_permute):
        for i, ref_num in enumerate(ref_perm):
            # ref_permute skips solute molecule, so index this back in
            ind = atoms_per_solute_mol + i * atoms_per_solvent_mol  # index in permutation tensor
            atom_num = atoms_per_solute_mol + ref_num  # atom number that should be at this index
            full_perm[ind] = atom_num
            for j in range(1, num_hydrogen + 1, 1):
                full_perm[ind + j] = atom_num + j

    b = x.shape[0]
    n = x.shape[1]
    flat_x = x.reshape(b * n, -1)
    flat_permute = (full_permute + (torch.arange(0, b) * n).unsqueeze(-1)).flatten().unsqueeze(-1).long()
    x_ordered = flat_x[flat_permute].reshape(b, n, flat_x.shape[-1])

    parity = torch.tensor([perm_parity(perm) for perm in full_permute.clone()])

    return x_ordered, parity
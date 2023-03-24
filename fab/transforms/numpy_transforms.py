import numpy as np


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
    """Returns the dihedral between four atoms in radian."""

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
    #  The normal is the x-axis in our case!
    d = np.dot(rotation_matrix(normal, angle), d)

    # Rotate d around a by the dihedral
    #  The vector a is the z-axis in our case!
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
    return np.asarray(coords)

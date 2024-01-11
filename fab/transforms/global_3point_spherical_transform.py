import torch
import torch.nn.functional as F
import math

import normflows as nf


def stable_inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Global3PointSphericalTransform(nf.flows.Flow):
    """
    See https://www.overleaf.com/read/tydmgmdzkzkh#fa4b70 for definitions of the transformation, and explanation of
    the various symbols + Jacobian determinant.

    This transform is used to transform Cartesian coordinates to internal coordinates and back. The reference frame
    is determined by the first three atoms in the system (e.g., the solute). The first atom is placed at the origin,
    the second atom is placed along the z-axis, and the third atom is placed in the yz-plane. All other atoms are
    placed relative to these three atoms. Atoms are described in terms of distance from the origin r > 0, angle with
    the z-axis phi in [0, 2pi], and angle with the yz-plane theta in [-pi/2, pi/2]. The transformation is invertible
    and differentiable.
    """

    def __init__(self, system=None, transform_data=None):
        """
        Constructor
        :param transform_data: Data used to set up coordinate scale for r. Must be a single frame of shape (1, ).
        """
        super().__init__()
        if system is not None:
            self.system = system
            self.atom_order = system.atoms
        else:
            print("No molecular system specified: presumably testing...?")
        self.transform_data = transform_data  # shape = 1 x n_atoms . 3 = 1 x ndim
        # self.device = transform_data.device
        assert transform_data.shape[0] == 1, "Data used for setting up coordinate transform must be a single frame."
        with torch.no_grad():
            z, _, _, _ = self.cartesian_to_z(transform_data, setup=True)
            self._setup_scale_r(z)  # Rescale r (by mean r between solute center and solvent atoms)
            self._setup_scale_phi(z)  # Rescale phi (by 2pi)
            self._setup_scale_theta(z)  # Rescale theta (by pi)
            # TODO: Do we need means as well? We set our coordinate system around (0, 0, 0) by default, so probably
            #  not.

    def forward(self, z):
        # TODO: For water in water atom types (e.g. OpenMM indices) matches Flow types (OHH OHH OHH etc). But maybe
        #  not in general for other systems. `z_to_cartesian` assumes OHH order, so this is what the Flow should use.
        #  If this is not the order OpenMM expects, then we should reorder the atoms after transforming to Cartesian.
        # Transform Z --> X. Return x and log det Jacobian.
        # Sometimes on cpu and sometimes on gpu, so we need to make sure the scales are on the right device.
        self.scale_r = self.scale_r.to(z.device)
        self.scale_phi = self.scale_phi.to(z.device)
        self.scale_theta = self.scale_theta.to(z.device)
        x, log_det_jac = self.z_to_cartesian(z)
        return x, log_det_jac

    def inverse(self, x):
        # TODO: For water in water atom types (e.g. OpenMM indices) matches Flow types (OHH OHH OHH etc). But maybe
        #  not in general for other systems. `z_to_cartesian` assumes OHH order, so this is what the Flow should use.
        #  If this is not the order OpenMM expects, then we should reorder the atoms before transforming to Z.
        # Transform X --> Z. Return z and log det Jacobian.
        # Sometimes on cpu and sometimes on gpu, so we need to make sure the scales are on the right device.
        self.scale_r = self.scale_r.to(x.device)
        self.scale_phi = self.scale_phi.to(x.device)
        self.scale_theta = self.scale_theta.to(x.device)
        z, log_det_jac, _, _ = self.cartesian_to_z(x)
        return z, log_det_jac

    def _setup_scale_r(self, z):
        # Some dofs are removed, so radial coordinates are at index 0, 1, 3, and every third index after.
        scale_r = z.new_ones(1)  # Scale 1 best? Since softplus already changes the necessary resolution of fr.

        # Or scale by the mean / max of the output values of the flow for the radial coordinates.
        # We have that r = softplus(fr), where fr is the flow output. This ensures that r is positive. To improve
        #  stability, we can scale fr such that the flow output is closer to unity. Thus, we scale by some mean / max of
        #  softplus^(-1)(r). This value can be negative, so we take the absolute value appropriately (in that case,
        #  the flow outputs values closer to -1 instead of 1, but that is fine).
        # mean r between solute oxygen and all other oxygens in the system
        # radial_coords = torch.cat((z[:, 0:2], z[0:1, 3::3]), dim=1)
        # scale_r = torch.max(radial_coords.abs(), dim=-1)[0]
        # scale_r = torch.mean(radial_coords, dim=-1).abs()
        self.register_buffer("scale_r", scale_r)

    def _setup_scale_phi(self, z):
        # TODO: With Circular Flow, we can make sure phi and theta are periodic. Currently we have [0, pi]
        #  by default, which means that we can use this scale to scale up the phi to the [0, 2pi] range. Theta we
        #  can leave untouched, except that we do have to shift it to the [-pi/2, pi/2] range. We can do this in the
        #  actual z_to_cartesian() function.
        scale_phi = z.new_ones(1) * 2
        self.register_buffer("scale_phi", scale_phi)

    def _setup_scale_theta(self, z):
        scale_theta = z.new_ones(1)
        self.register_buffer("scale_theta", scale_theta)

    def setup_coordinate_system(self, x):
        """
        Set up the global coordinate system. Essentially just
        :param x: Cartesian coordinates: n_batch x n_atoms . 3

        :return:
        x: Centralised Cartesian coordinates: n_batch x n_atoms x 3
        x_coord: Flattened centralised Cartesian coordinates: n_batch x n_atoms . 3
        z_axis: definition of z-axis (coordinates): n_batch x 3
        y_axis: definition of y-axis (coordinates): n_batch x 3
        """
        x = x.reshape(x.shape[0], -1, 3)

        # Setup rotation axes
        z_axis = x.new_zeros(x.shape[0], 3)
        z_axis[:, 2] = 1
        y_axis = x.new_zeros(x.shape[0], 3)
        y_axis[:, 1] = 1

        # Set the first solute atom to the origin, and the second atom to the z-axis. The third atom is aligned with the
        #  yz-plane. We then define all other atoms w.r.t. this reference frame, allowing us to use 3D spherical
        #  coordinates. Note that this is equivalent to using planar angles between the solute-molecule-plane and
        #  the plane formed by the first two solute atoms and any other atom in the system.
        # x is then the original x coordinates in the new coordinate system.
        x = self.rotate_into_global_coordinate_system(x, z_axis, y_axis)

        x_coord = x.reshape(x.shape[0], -1)  # The original x coordinates in the new coordinate system, but flattened.
        return x, x_coord, z_axis, y_axis

    def cartesian_to_z(self, x, setup=False):
        """
        Transform Cartesian coordinates to internal coordinates.
        :param x: Cartesian coordinates: n_batch x n_atoms . 3
        :param setup: If True, use to set up coordinate scale for r. x must then be a single frame of shape (1, ndim),
            where ndim = n_atoms . 3.
        :return: Spherical coordinates: n_batch x (n_atoms . 3 - 6), and log det Jacobian
        """

        if setup:
            assert x.shape[0] == 1, "Data used for setting up coordinate transform must be a single frame."

        x, x_coord, z_axis, y_axis = self.setup_coordinate_system(x)

        solute_atom0 = x[:, 0, :]
        if not torch.isclose(x.new_zeros(x.shape[0], 3), solute_atom0).all():
            raise ValueError(
                "Expected solute atom0 to be at origin, but norm mismatch (n_batch): {}".format(
                    torch.norm(x.new_zeros(x.shape[0], 3) - solute_atom0, dim=1)
                )
            )

        z = x.new_zeros(x.shape)
        log_det_jac = x.new_zeros(x.shape[0])  # n_batch

        unnorm_z = z.clone()
        unnorm_z.requires_grad = False

        for atom_num in range(x.shape[1]):
            atom_coords = x[:, atom_num, :]

            if atom_num == 0:  # solute reference: [0, 0, 0] in z
                if not torch.isclose(x.new_zeros(x.shape[0], 3), atom_coords).all():
                    raise ValueError(
                        "Expected first atom to be at origin, but norm mismatch (n_batch): {}".format(
                            torch.norm(x.new_zeros(x.shape[0], 3) - atom_coords, dim=1)
                        )
                    )
            elif atom_num == 1:  # axis-aligned atom [r, 0, 0]
                if not torch.isclose(unit_vector(z_axis), unit_vector(atom_coords), atol=1e-5).all():
                    raise ValueError(
                        "Expected atom to be aligned with z-axis, but vector mismatch (n_batch): {}".format(
                            torch.abs(unit_vector(z_axis) - unit_vector(atom_coords))
                        )
                    )
                r = torch.norm(atom_coords, dim=-1)  # Vector norm
                fr_scaled = stable_inverse_softplus(r)  # Inverse softplus: this is f_r * s_r
                # This should match the distance along z axis, since the vector should be aligned with z-axis.
                if not torch.isclose(r, atom_coords[:, -1]).all():
                    raise ValueError(
                        "Expected atom to be aligned with z-axis, but norm mismatch (n_batch): {}".format(
                            torch.abs(r - torch.norm(atom_coords, dim=1))
                        )
                    )
                unnorm_z[:, atom_num, 0] = fr_scaled
                if not setup:  # Setup determines scaling, so don't try to scale here.
                    # Normalise
                    fr = fr_scaled / self.scale_r  # This is f_r, i.e., the flow output
                    # Log det Jacobian contribution of scaling.
                    log_det_jac += -1 * (  # Notation chosen for consistency with 2D and 3D case
                        torch.log(self.scale_r)  # s_r
                    )
                else:
                    fr = fr_scaled

                # Assign
                z[:, atom_num, 0] = fr
                # Contribution of inverse softplus: reciprocal of sigmoid.
                log_det_jac += -1 * (torch.log(torch.sigmoid(fr_scaled)))  # sigmoid(f_r * s_r)

            elif atom_num == 2:  # r and phi
                r = torch.norm(atom_coords, dim=-1)
                fr_scaled = stable_inverse_softplus(r)  # Inverse softplus: this is f_s * s_r
                phi, _ = get_angle_and_normal(z_axis, solute_atom0, atom_coords)  # this is f_phi * s_phi
                unnorm_z[:, atom_num, 0] = fr_scaled
                unnorm_z[:, atom_num, 1] = phi
                if not setup:
                    # Normalise
                    fr = fr_scaled / self.scale_r
                    fphi = phi / self.scale_phi
                    # Log det Jacobian contribution of scaling.
                    log_det_jac += -1 * (torch.log(self.scale_r) + torch.log(self.scale_phi))
                else:
                    fr = fr_scaled
                    fphi = phi
                # Transformation without scaling
                log_det_jac += -1 * (
                    torch.log(r)  # r = softplus(f_r * s_r) = norm(x, y, z)
                    + torch.log(torch.sigmoid(fr_scaled))  # sigmoid(f_r * s_r)
                )
                # Assign
                z[:, atom_num, 0] = fr
                z[:, atom_num, 1] = fphi

            else:  # r, phi, theta
                r = torch.norm(atom_coords, dim=-1)
                fr_scaled = stable_inverse_softplus(r)  # Inverse softplus: this is f_r * s_r
                phi, _ = get_angle_and_normal(z_axis, solute_atom0, atom_coords)  # this is f_phi * s_phi
                solute_atom0 = x[:, 0, :]
                solute_atom1 = x[:, 1, :]
                solute_atom2 = x[:, 2, :]  # TODO: make this a dummy if the solute does not define a plane
                # this is f_theta * s_theta
                theta = get_theta(solute_atom0, solute_atom1, solute_atom2, atom_coords, phi)
                unnorm_z[:, atom_num, 0] = fr_scaled
                unnorm_z[:, atom_num, 1] = phi
                unnorm_z[:, atom_num, 2] = theta
                if not setup:
                    # Normalise
                    fr = fr_scaled / self.scale_r
                    fphi = phi / self.scale_phi
                    ftheta = theta / self.scale_theta
                    # Log det Jacobian contribution of scaling.
                    log_det_jac += -1 * (
                            torch.log(self.scale_r) + torch.log(self.scale_phi) + torch.log(self.scale_theta)
                    )
                else:
                    fr = fr_scaled
                    fphi = phi
                    ftheta = theta

                # Transformation without scaling
                log_det_jac += -1 * (
                    2 * torch.log(r)
                    + torch.log(torch.abs(torch.sin(phi)))  # r = softplus(f_r * s_r) = norm(x, y, z)
                    + torch.log(torch.sigmoid(fr_scaled))  # phi = f_phi * s_phi  # sigmoid(f_r * s_r)
                )
                # Assign
                z[:, atom_num, 0] = fr
                z[:, atom_num, 1] = fphi
                z[:, atom_num, 2] = ftheta

        # Reshape z from n_batch x n_atoms x 3 to n_batch x n_dim
        z = z.reshape(z.shape[0], -1)
        unnorm_z = unnorm_z.reshape(unnorm_z.shape[0], -1)

        # We have lost 6 degrees of freedom: we will remove these in the internal coordinate representation, but keep
        #  them explicit in Cartesian space.
        # Internal coordinates are ordered as: r, phi, theta per atom. The chosen orientation sets all three to 0 for
        #  the first atom, phi and theta to 0 for the second atom, and theta to 0 for the third atom. We therefore
        #  remove these by slicing z.
        z = torch.cat((z[:, 3:4], z[:, 6:8], z[:, 9:]), dim=1)
        unnorm_z = torch.cat((unnorm_z[:, 3:4], unnorm_z[:, 6:8], unnorm_z[:, 9:]), dim=1)
        assert z.shape[1] == len(x[0].flatten()) - 6, (
            "Expected internal coordinates to have 6 fewer dofs than Cartesian."
        )

        return z, log_det_jac, x_coord, unnorm_z

    def z_to_cartesian(self, z):
        """  # TODO: internal coordinates have 6 fewer dofs than Cartesian.
        Transform Spherical coordinates to Cartesian coordinates.
        :param z: Spherical coordinates: n_batch x (n_atoms . 3 - 6).
        :return: Cartesian coordinates: n_batch x n_atoms . 3, and log det Jacobian

        OpenMM expects specific atom types at specific indices. For water in water, this order is OHH OHH OHH etc.
        """

        # We add the 6 degrees of freedom back in here. 0s for the first atom, 0 for phi and theta for the second atom,
        #  and 0 for theta for the third atom.
        internal_atom1 = torch.zeros_like(z)[:, :3]
        internal_atom2 = torch.cat((z[:, 0:1], torch.zeros_like(z)[:, :2]), dim=1)
        internal_atom3 = torch.cat((z[:, 1:3], torch.zeros_like(z)[:, :1]), dim=1)
        internal_rest = z[:, 3:]
        z = torch.cat((internal_atom1, internal_atom2, internal_atom3, internal_rest), dim=1)

        # Reshape z from n_batch x n_dim to n_batch x n_atoms x 3
        z = z.reshape(z.shape[0], -1, 3)

        # Setup rotation axes
        z_axis = z.new_zeros(z.shape[0], 3)
        z_axis[:, 2] = 1
        x_axis = z.new_zeros(z.shape[0], 3)
        x_axis[:, 0] = 1

        x = z.new_zeros(z.shape)
        log_det_jac = z.new_zeros(z.shape[0])  # n_batch

        # # Flow output should have 6 degrees of freedom masked out: these correspond to the orientation we choose
        # # for the first three atoms.
        # mask = z.new_ones(z.shape)
        # mask[:, 0, :] = 0
        # mask[:, 1, 1:] = 0
        # mask[:, 2, 2] = 0
        # z = mask * z

        for atom_num in range(z.shape[1]):
            atom_coords = z[:, atom_num, :]

            if atom_num == 0:  # solute_atom0 reference: [0, 0, 0] in x
                if not torch.isclose(z.new_zeros(z.shape[0], 3), atom_coords).all():
                    raise ValueError(
                        "Expected first atom to be at origin, but norm mismatch (n_batch): {}".format(
                            torch.norm(z.new_zeros(z.shape[0], 3) - atom_coords, dim=1)
                        )
                    )
                continue
            elif atom_num == 1:  # axis-aligned atom: e.g., [0, 0, r] in x
                # phi and theta should be 0 for this atom
                if not torch.isclose(z.new_zeros(z.shape[0], 2), atom_coords[:, 1:]).all():
                    raise ValueError(
                        "Expected first atom to be axis aligned, but norm mismatch (n_batch): {}".format(
                            torch.norm(z.new_zeros(z.shape[0], 2) - atom_coords[:, 1:], dim=1)
                        )
                    )
                fr = atom_coords[:, 0]  # this is f_r
                fr_scaled = fr * self.scale_r  # Scale back: this is f_r * s_r
                r = F.softplus(fr_scaled)  # Fix r to lie in [0, inf]: This is softplus(f_r * s_r)
                x[:, atom_num, 2] = r
                # Jacobian determinant contribution: d/d(f_r) softplus(f_r * s_r) = sigmoid(f_r * s_r) * s_r
                log_det_jac += torch.log(torch.sigmoid(fr_scaled)) + torch.log(self.scale_r)  # sigmoid(f_r * s_r) * s_r
            elif atom_num == 2:  # distance and angle: reconstruct x
                # theta should be 0 for this atom
                if not torch.isclose(z.new_zeros(z.shape[0], 1), atom_coords[:, 2:]).all():
                    raise ValueError(
                        "Expected third atom to have theta = 0, but norm mismatch (n_batch): {}".format(
                            torch.norm(z.new_zeros(z.shape[0], 1) - atom_coords[:, 2:], dim=1)
                        )
                    )
                fr = atom_coords[:, 0]  # This is f_r
                fphi = atom_coords[:, 1]  # This is f_phi
                # Scaling
                fr_scaled = fr * self.scale_r  # Fix r to lie in [0, inf]: This is f_r * s_r
                r = F.softplus(fr_scaled)  # This is r = softplus(f_r * s_r)
                phi = fphi * self.scale_phi  # This is phi = f_phi * s_phi
                # Fix phi to lie in [0, 2pi]. We assume the flow output maps to this range. This does not
                #  affect the Jacobian.
                # TODO: This should in principle do nothing if we're using the Circular Flow.
                phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
                phi = torch.where(phi > 2 * math.pi, phi - 2 * math.pi, phi)
                # Define normal for rotation: for X --> Z direction, we took the x > 0 normal when rotating towards the
                #  alignment axis. Now we rotate away from the axis, so we take the negative angle.
                #  Alternatively, we can take the positive angle and the normal with x < 0.
                phi_rotation = rotation_matrix(x_axis, -phi)
                xyz = torch.einsum("bij,bj -> bi", phi_rotation, z_axis * r.unsqueeze(-1))  # Apply rotation to z-axis
                # x coordinate should be 0 for this atom
                if not torch.isclose(z.new_zeros(z.shape[0], 1), xyz[:, :1]).all():
                    raise ValueError(
                        "Expected third atom to have x = 0, but norm mismatch (n_batch): {}".format(
                            torch.norm(z.new_zeros(z.shape[0], 1) - xyz[:, 2:], dim=1)
                        )
                    )
                x[:, atom_num, :] = xyz
                # Jacobian determinant contribution
                log_det_jac += (
                    torch.log(r)  # r = softplus(f_r * s_r) = norm(x, y, z)
                    + torch.log(self.scale_r)  # Scaling contribution for r: s_r
                    + torch.log(self.scale_phi)  # Scaling contribution for phi: s_phi  # sigmoid(f_r * s_r)
                    + torch.log(torch.sigmoid(fr_scaled))
                )
            else:
                # We always use the first molecule as the anchor for the angles.
                fr = atom_coords[:, 0]  # atom-to-reconstruct r
                fphi = atom_coords[:, 1]  # atom-to-reconstruct phi
                ftheta = atom_coords[:, 2]  # atom-to-reconstruct dihedral
                # Scaling
                fr_scaled = fr * self.scale_r  # This is r = softplus(f_r * s_r)
                r = F.softplus(fr_scaled)  # Fix r to lie in [0, inf]: This is f_r * s_r
                phi = fphi * self.scale_phi  # This is phi = f_phi * s_phi
                theta = ftheta * self.scale_theta  # This is theta = f_theta * s_theta
                # Fix phi to lie in [0, 2pi]. This does not affect the Jacobian.
                # TODO: This should in principle do nothing if we're using the Circular Flow. E.g., we should see no
                #  phi values < 0 or > pi here.
                phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
                phi = torch.where(phi > 2 * math.pi, phi - 2 * math.pi, phi)
                # Fix theta to lie in [-pi/2, pi/2]. This does not affect the Jacobian.
                # TODO: This should in principle only shift from [0, pi] to [-pi/2, pi/2] if we're using the
                #  Circular Flow. It should not create folds. E.g., we should see no theta values < 0 or > pi here.
                theta = torch.where(theta < -math.pi / 2, theta + math.pi, theta)
                theta = torch.where(theta > math.pi / 2, theta - math.pi, theta)

                # Step 1: rotate the z-axis vector around the positive x-axis (by convention) by -phi
                # NOTE: would (for cleanliness) prefer +phi, but we defined phi the other direction, and code for
                #  computing theta now depends on this convention. In any case, this is equivalent to rotating by
                #  phi around the negative x-axis.
                phi_rotation = rotation_matrix(x_axis, -phi)
                xyz = torch.einsum("bij,bj -> bi", phi_rotation, z_axis * r.unsqueeze(-1))
                # Step 2: rotate this vector around the positive z-axis (by convention) by theta (NOT -theta).
                theta_rotation = rotation_matrix(z_axis, theta)
                xyz = torch.einsum("bij,bj -> bi", theta_rotation, xyz)
                x[:, atom_num, :] = xyz
                # Jacobian determinant contribution
                log_det_jac += (
                    2 * torch.log(r)  # r^2: r = softplus(f_r * s_r) = norm(x, y, z)
                    + torch.log(torch.abs(torch.sin(phi)))  # theta rotation Jacobian bit
                    + torch.log(self.scale_r)  # Scaling from r: s_r
                    + torch.log(self.scale_phi)  # here: phi = f_phi * s_phi, so scaling from phi: s_phi
                    + torch.log(self.scale_theta)  # Scaling from theta: s_theta
                    + torch.log(torch.sigmoid(fr_scaled))  # sigmoid(f_r * s_r)
                )

        # Reshape to n_batch x n_atoms . 3
        x = x.reshape(x.shape[0], -1)

        return x, log_det_jac

    @staticmethod
    def rotate_into_global_coordinate_system(x, z_axis, y_axis):
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

        # First define phi w.r.t. the z-axis. This means we rotate solute_atom1 to align with the z-axis.
        # E.g. we want to align the first H atom with the z-axis.
        # The rotation should happen along the normal given by the z-O-H plane (for water).
        # TODO: This is the culprit for the error when the first solute hydrogen has y==0 exactly.
        phi_rad, phi_axis = get_angle_and_normal(z_axis, solute_atom0, solute_atom1, align_first_solute_h=True)
        phi_rotation = rotation_matrix(phi_axis, phi_rad)
        x_phi = torch.einsum("bij,bnj -> bni", phi_rotation, x_centered)

        # Now define theta w.r.t. the yz-plane. This means we rotate solute_atom2 to align with the yz-plane.
        # We want to rotate around the z-axis, so that the first H atom stays fixed in place.
        # So to find the angle, we project the second H atom onto the xy-plane, and find its angle with the y-axis.
        # To project an atom onto a plane, we in general project the atom onto the plane normal,
        #  then subtract this component form the original to find the projection onto the plane.
        # (we also have an easy solution available: projecting onto the xy-plane is just setting the z-component to 0)
        # Here we use the plane projection, because it means we don't have to mutate an existing object.
        solute_atom2 = x_phi[:, 2, :]  # e.g., second hydrogen; will become [r, phi, 0], defines theta.
        xy_proj = solute_atom2 - unit_vector(z_axis) * torch.sum(z_axis * solute_atom2, dim=1, keepdim=True)
        # Angle between atom_to_align and projection, through origin (solute_atom0)
        if not torch.isclose(solute_atom0, torch.zeros_like(solute_atom0)).all():
            raise ValueError("Solute atom0 is not at the origin.")
        theta_rad, _ = get_angle_and_normal(y_axis, solute_atom0, xy_proj, to_yz_plane=True)
        # We rotate around the alignment axis (z-axis) to put the molecule into the yz-plane
        theta_rotation = rotation_matrix(z_axis, theta_rad)
        # x rotated into the yz plane
        x = torch.einsum("bij,bnj -> bni", theta_rotation, x_phi)

        return x


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / torch.norm(vector, dim=-1, keepdim=True)


def get_angle_and_normal(atom1, atom2, atom3, to_yz_plane=False, align_first_solute_h=False):
    """
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
        # TODO: What if x == 0? Then we still need to pick a convention for the rotation axis, but how do we make sure
        #  this is consistent? See the below check. This situation occurs when the first solute hydrogen has
        #  y == 0, which can happen, although it should be rare. Seems to happen when setting up the coordinate system
        #  sometimes.
        # TODO: Can't we just remove the check? It's actually fine if x=0 for the rotation axis, since this is a valid
        #  axis to rotate around for aligning the first solute hydrogen with the z-axis. The only problem may be with
        #  the reverse transformation, where we would need to know how exactly to invert this (set a convention).
        #  Here we are in a situation where the rotation axis has z=0 and x=0, so we can use y>0 as our convention.
        if torch.isclose(cross[:, 0], cross.new_zeros(cross.shape[0])).any():
            # Exception: both x and z are 0, so use y-axis as rotation axis. Convention is to use the y>0 axis.
            sign = torch.sign(cross[:, 1])  # Flip direction if y < 0
            # TODO: Actually, this signing is unnecessary, since this exception should only happen when aligning the
            #  first solute hydrogen during coordinate setup. We can just rotate however we want, since we never have
            #  to reverse this rotation (this atom is always set to align to the z-axis in the reverse transform).
            if not align_first_solute_h:
                raise ValueError("Found rotation around axis with x=0 outside of coordinate setup.")
            # cross[:, 0] += 1e-8  # Increases error, and doesn't fully fix the problem when non-solute atom has y=0.
        else:
            # Standard setting: use x value for defining rotation direction.
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
    #  Equivalent to setting z = 0 for rotation around z-axis.
    mag_along_normal = torch.sum(rotation_axis * plane_axis, dim=-1, keepdim=True)
    # Note: the below is essentially the y-unit vector in our setting, but with sign depending on the position of atom3.
    in_both_planes = unit_vector(plane_axis - unit_vector(rotation_axis) * mag_along_normal)
    # Project the atom4 - atom1 vector onto the rotation_axis plane as well. We care about the
    #  angle between this vector and the in_both_planes vector around the rotation_axis.
    angle_vector = unit_vector(atom4 - atom1)
    mag_along_normal = torch.sum(rotation_axis * angle_vector, dim=-1, keepdim=True)
    in_rot_plane = angle_vector - unit_vector(rotation_axis) * mag_along_normal
    # Find angle through inner product: this value being negative corresponds to phi > pi
    inner = torch.sum(in_both_planes * unit_vector(in_rot_plane), dim=-1)
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

    # Now some sign magic to make the theta angle work out correctly. We need to treat every xy-quadrant separately.
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
    rot_matrix = torch.stack(
        [
            torch.stack([a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], dim=1),
            torch.stack([2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)], dim=1),
            torch.stack([2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c], dim=1),
        ],
        dim=2,
    )

    return rot_matrix


if __name__ == "__main__":
    n_batch = 1024
    n_atoms = 27
    t_x = torch.randn(1, n_atoms * 3)
    t = Global3PointSphericalTransform(transform_data=t_x)
    print(
        "\nScaling params for (r, phi, theta): ({:.3f}, {:.3f}, {:.3f})\n".format(
            t.scale_r.item(), t.scale_phi.item(), t.scale_theta.item()
        )
    )

    x = torch.randn(n_batch, n_atoms * 3)
    print("Cartesian dim: {}".format(x.shape))
    z, jac_xi, x_coord, _ = t.cartesian_to_z(x)
    print("Internal dim: {}".format(z.shape))
    x_recon, jac_ix = t.z_to_cartesian(z)
    print("\nJacobian Cart --> Internal: {:.5f}".format(jac_xi.mean().item()))
    print("Jacobian Internal --> Cart: {:.5f}".format(jac_ix.mean().item()))

    print("\nOriginal Cart:", x_coord.reshape(x_coord.shape[0], -1, 3)[0, :7, :])
    print("Reconstruction:", x_recon.reshape(x_recon.shape[0], -1, 3)[0, :7, :])

    errs = torch.abs(x_coord - x_recon)
    norm_err = torch.norm(x_coord - x_recon, dim=-1)
    print("\nMax error (xyz): {:.5f}".format(errs.max().item()))
    print("Mean error (xyz): {:.8f}".format(errs.mean().item()))
    print("Min error (xyz): {:.8f}".format(errs.min().item()))
    print("Max error norm (r): {:.5f}".format(norm_err.max().item()))
    print("Mean error norm (r): {:.8f}".format(norm_err.mean().item()))
    print("Min error norm (r): {:.8f}".format(norm_err.min().item()))
    if not torch.isclose(x_coord, x_recon, atol=1e-3).all():
        raise ValueError("Max reconstruction error > 1e-3...")

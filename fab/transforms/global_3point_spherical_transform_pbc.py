import torch
import torch.nn.functional as F
import math

import normflows as nf

def stable_inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Global3PointSphericalTransformPBC(nf.flows.Flow):
    """
    Global 3-point spherical transform with PBC handling. This is a global coordinate transform that maps Cartesian coordinates 
    to a 3D spherical coordinate system defined by the first three atoms in the system (the solute). The first solute atom is set 
    to the origin, the first solute hydrogen is aligned with the z-axis, and the second solute hydrogen is aligned with the yz-plane.
    This allows us to use 3D spherical coordinates (r, phi, theta) for all other atoms in the system, which are then transformed by the 
    flow. We also handle PBCs by making sure that we always take the minimum image convention when defining our coordinate system and when 
    reconstructing Cartesian coordinates from internal coordinates. This means that we always define our coordinate system and reconstruct 
    Cartesian coordinates in a way that respects the periodicity of the box, ensuring that we do not have any discontinuities or artifacts due to PBCs.
    """

    def __init__(self, system=None, transform_data=None, box_length_nm=2.5):
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
        
        # box length in nm of the pbc cubic box
        self.box_length_nm = box_length_nm

        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3    


        self._stats = {
            "seam_x": 0,
            "seam_z": 0,
            "degenerate": 0,
            "calls": 0,
            "B": 0,
        }
        
        # self.device = transform_data.device
        assert transform_data.shape[0] == 1, "Data used for setting up coordinate transform must be a single frame."
        with torch.no_grad():
            z, _, _, _= self.cartesian_to_z(transform_data, setup=True)
            self._setup_scale_r(z)
            self._setup_scale_phi(z)
            self._setup_offset_phi(z)
            self._setup_scale_theta(z)
    
    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0

    def get_stats(self):
        out = dict(self._stats)
        self.reset_stats()
        return out
    
    def _begin_stats(self, B: int):
        self._stats["calls"] += 1
        self._stats["B"] += int(B)

    def _record(self, name: str, mask: torch.Tensor):
        self._stats[name] += int(mask.sum().item())
        

    def forward(self, z):
        # For water-in-water atom types (e.g. OpenMM indices) matches Flow types (OHH OHH OHH etc). But maybe
        #  not in general for other systems. `z_to_cartesian` assumes OHH order, so this is what the Flow should use.
        #  If this is not the order OpenMM expects, then we should reorder the atoms after transforming to Cartesian.
        # TODO: Check this for SO2. S should be the first atom.
        # Transform I --> X. Return x and log det Jacobian.
        # Sometimes on cpu and sometimes on gpu, so we need to make sure the scales are on the right device.
        self.scale_r = self.scale_r.to(z.device)
        self.scale_phi = self.scale_phi.to(z.device)
        self.offset_phi = self.offset_phi.to(z.device)
        self.scale_theta = self.scale_theta.to(z.device)
        x, log_det_jac = self.z_to_cartesian(z)
        return x, log_det_jac

    def inverse(self, x):
        # For water-in-water atom types (e.g. OpenMM indices) matches Flow types (OHH OHH OHH etc). But maybe
        #  not in general for other systems. `z_to_cartesian` assumes OHH order, so this is what the Flow should use.
        #  If this is not the order OpenMM expects, then we should reorder the atoms before transforming to I.
        # TODO: Check this for SO2. S should be the first atom.
        # Transform X --> I. Return z and log det Jacobian.
        # Sometimes on cpu and sometimes on gpu, so we need to make sure the scales are on the right device.
        self.scale_r = self.scale_r.to(x.device)
        self.scale_phi = self.scale_phi.to(x.device)
        self.offset_phi = self.offset_phi.to(x.device)
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
        # With Circular Flow, we can make sure phi and theta are periodic. Currently, we have that the flow outputs
        # values in [-pi, pi], which means that we need to shift phi to get the desired [0, 2pi] range. We do not do
        # this with a scale operation though, so we can leave the phi scale on 1.
        scale_phi = z.new_ones(1)
        self.register_buffer("scale_phi", scale_phi)

    def _setup_offset_phi(self, z):
        # With Circular Flow, we can make sure phi and theta are periodic. Currently, we have that the flow outputs
        # values in [-pi, pi], which means that we need to shift phi to get the desired [0, 2pi] range. We do this
        # by applying an offset of pi.
        offset_phi = z.new_ones(1) * math.pi
        self.register_buffer("offset_phi", offset_phi)

    def _setup_scale_theta(self, z):
        # With Circular Flow, we can make sure phi and theta are periodic. Currently, we have that the flow outputs
        # values in [-pi, pi], which means that we need to half this to get values in the desired theta range of
        # [-pi/2, pi/2].
        scale_theta = z.new_ones(1) / 2
        self.register_buffer("scale_theta", scale_theta)


    def setup_coordinate_system(self, x, setup=False):
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
        x = self.rotate_into_global_coordinate_system(x, z_axis, y_axis, setup=setup)

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

        B = x.shape[0]
        if not setup:
            self._begin_stats(B)

        # x: (B, N, 3), centered on atom0, but NOT rotated
        x, x_coord, _, _ = self.setup_coordinate_system(x, setup=setup)

        # sanity: atom0 at origin
        if not torch.isclose(x[:, 0, :], x.new_zeros(B, 3)).all():
            raise ValueError("Expected atom0 at origin after setup_coordinate_system.")

        N = x.shape[1]
        z = x.new_zeros(B, N, 3)
        log_det_jac = x.new_zeros(B)

        # small eps to avoid log(0) / division by zero
        eps = 1e-12 if x.dtype == torch.float64 else 1e-9

        # atom0 is fixed -> all zeros in z[:,0,:] and contributes nothing
        # atoms 1..N-1: convert xyz -> (r, phi, theta)
        v = x[:, 1:, :]                       # (B, N-1, 3)
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

        r = torch.sqrt(vx * vx + vy * vy + vz * vz).clamp_min(eps)  # (B, N-1)

        # polar angle theta in [0, pi]
        cos_theta = (vz / r).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)  # (B, N-1)

        # azimuth phi in [0, 2pi)
        phi = torch.atan2(vy, vx)                 # (-pi, pi]
        phi = torch.where(phi < 0, phi + 2 * math.pi, phi)

        # r parameterization via softplus
        fr_scaled = stable_inverse_softplus(r)  # (B, N-1)  this is f_r * scale_r

        if not setup:
            fr = fr_scaled / self.scale_r
            fphi = (phi - self.offset_phi) / self.scale_phi
            ftheta = theta / self.scale_theta

            # scaling contributions
            log_det_jac += -(torch.log(self.scale_r) + torch.log(self.scale_phi) + torch.log(self.scale_theta)) * (N - 1)
        else:
            fr = fr_scaled
            fphi = phi
            ftheta = theta

        # store per-atom internal coords
        z[:, 1:, 0] = fr
        z[:, 1:, 1] = fphi
        z[:, 1:, 2] = ftheta

        # Jacobian of xyz -> (fr,phi,theta): |J| = r^2 * sin(theta)
        # r -> fr_scaled via inverse softplus: dr/ d(fr_scaled) = sigmoid(fr_scaled)
        # so xyz -> (fr_scaled,phi,theta): |J| = r^2 * sin(theta) * sigmoid(fr_scaled)
        # logdet for inverse (x->z) is negative of forward:
        log_det_jac += -(
            2.0 * torch.log(r) +
            torch.log(torch.sin(theta).clamp_min(eps)) +
            torch.log(torch.sigmoid(fr_scaled))
        ).sum(dim=1)

        # flatten and drop atom0 (translation removed only)
        z = z[:, 1:, :].reshape(B, -1)

        unnorm_z = x.new_zeros(B, N, 3)
        unnorm_z[:, 1:, 0] = fr_scaled
        unnorm_z[:, 1:, 1] = phi
        unnorm_z[:, 1:, 2] = theta
        unnorm_z = unnorm_z[:, 1:, :].reshape(B, -1)

        return z, log_det_jac, x_coord, unnorm_z

    def z_to_cartesian(self, z):
        """
        Inverse of the no-global-rotation spherical transform.

        z layout: (B, (N-1)*3) for atoms 1..N-1:
            [f_r, f_phi, f_theta] per atom (in that order)

        Reconstruct in lab frame:
            r     = softplus(f_r * scale_r)                (r > 0)
            phi   = f_phi * scale_phi + offset_phi         (wrap to [0, 2pi))
            theta = f_theta * scale_theta                  (polar angle, ideally in [0, pi])
            x = r sin(theta) cos(phi)
            y = r sin(theta) sin(phi)
            z = r cos(theta)

        Atom0 is fixed at origin (translation removed only).
        """
        B = z.shape[0]
        N = self.n_atoms  # total atoms

        # reshape z -> (B, N-1, 3)
        z = z.view(B, N - 1, 3)
        fr = z[..., 0]
        fphi = z[..., 1]
        ftheta = z[..., 2]

        # ensure buffers on correct device
        self.scale_r = self.scale_r.to(z.device)
        self.scale_phi = self.scale_phi.to(z.device)
        self.offset_phi = self.offset_phi.to(z.device)
        self.scale_theta = self.scale_theta.to(z.device)

        # --- inverse scaling / parameterization ---
        fr_scaled = fr * self.scale_r
        r = F.softplus(fr_scaled)

        phi = fphi * self.scale_phi + self.offset_phi
        phi = torch.remainder(phi, 2 * math.pi)  # [0, 2pi)

        theta = ftheta * self.scale_theta
        # optional: keep theta away from exact 0 or pi (numerical stability)
        eps = 1e-12 if r.dtype == torch.float64 else 1e-9
        theta = theta.clamp(eps, math.pi - eps)

        # --- spherical -> Cartesian ---
        sin_t = torch.sin(theta)
        vx = r * sin_t * torch.cos(phi)
        vy = r * sin_t * torch.sin(phi)
        vz = r * torch.cos(theta)

        # assemble x with atom0 at origin
        x = z.new_zeros(B, N, 3)
        x[:, 1:, 0] = vx
        x[:, 1:, 1] = vy
        x[:, 1:, 2] = vz

        # --- log|det J| for forward map z->x ---
        # per atom: |J| = (dr/d(fr_scaled)) * (d(fr_scaled)/d(fr)) * (dphi/dfphi) * (dtheta/dftheta) * r^2 sin(theta)
        # dr/d(fr_scaled) = sigmoid(fr_scaled)
        # so logdet = log(sigmoid(fr_scaled)) + log(scale_r) + log(scale_phi) + log(scale_theta) + 2 log(r) + log(sin(theta))
        log_det = (
            torch.log(torch.sigmoid(fr_scaled)) +
            torch.log(self.scale_r) +
            torch.log(self.scale_phi) +
            torch.log(self.scale_theta) +
            2.0 * torch.log(r.clamp_min(eps)) +
            torch.log(sin_t.clamp_min(eps))
        ).sum(dim=1)

        return x.reshape(B, -1), log_det

    def make_water_whole(self, x, L, n_solute=3):
        """
        Make each water molecule whole by MIC-wrapping H positions relative to O.
        x: (B, N, 3)
        L: (1,1,3) or (B,1,3)
        """
        xs = x[:, n_solute:, :]                 # solvent part (B, Ns, 3)
        Ns = xs.shape[1]
        assert Ns % self.n_atoms_per_mol == 0


        # Reshape into (B, n_mol, atoms_per_mol, 3)
        mols = xs.view(xs.shape[0], Ns // self.n_atoms_per_mol, self.n_atoms_per_mol, 3)

        # Reference atom = first atom in each molecule
        ref = mols[:, :, 0, :]                          # (B, n_mol, 3)

        # All other atoms
        others = mols[:, :, 1:, :]                      # (B, n_mol, atoms_per_mol-1, 3)

        # Displacements relative to reference
        d = others - ref.unsqueeze(2)

        # MIC wrap relative to reference
        d = d - L * torch.round(d / L)

        # Rebuild molecule
        mols[:, :, 1:, :] = ref.unsqueeze(2) + d

        # Flatten back
        xs_whole = mols.view(xs.shape[0], Ns, 3)

        x_out = x.clone()
        x_out[:, n_solute:, :] = xs_whole
        return x_out
    
 


    def rotate_into_global_coordinate_system(self, x, z_axis, y_axis, setup=False):
        """
        NO global rotation, NO MIC, NO wrapping, NO make_water_whole.
        Only remove translation by centering on atom0.

        x: (B, N, 3)
        returns: centered x (B, N, 3)
        """
        # center on solute atom0
        x_centered = x - x[:, 0:1, :]

        # sanity
        if not torch.isclose(x_centered[:, 0, :], torch.zeros_like(x_centered[:, 0, :])).all():
            raise ValueError("Atom0 is not at the origin after centering.")

        return x_centered


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

    if cross.type() == torch.DoubleTensor:
        eps_x = 1e-12
        eps_deg = 1e-12
    else:
        eps_x = 1e-9
        eps_deg = 1e-9

    # True degeneracy: axis undefined (vectors parallel)
    cn = torch.linalg.norm(cross, dim=-1)
    deg = cn < eps_deg
    if deg.any():
        # deterministic fallback: angle 0, arbitrary axis
        # (axis won't matter because angle=0)
        cross = cross.clone()
        cross[deg] = torch.tensor([1.0, 0.0, 0.0], device=cross.device, dtype=cross.dtype)
        rads = rads.clone()
        rads[deg] = 0.0

    # We need to fix the rotation axis orientation, so that we know how to reconstruct X from the angle
    #  information in I. So we pick the convention that the rotation is the normal with x > 0.
    # This means that we sometimes flip the convention, so we need to adjust the angle appropriately.
    #  That is, when we find a normal with x < 0, we negate it, and adjust the angle as: rad = 2pi - rad.

    # Note that this can mess up if we are rotating vectors into the yz-plane, since the rotation axis has x=0 there.
    #  If so, we want to use the z > 0 vector as the normal.

    if to_yz_plane: 
        # Orientation using z; if z ~ 0 fall back to y
        sign = torch.sign(cross[:, 2])          # primary sign test (z-component)
        seam   = torch.abs(cross[:, 2]) < eps_x   # detect seam (z ≈ 0)
        sign = torch.where(seam,
                        torch.sign(cross[:, 1]),  # fallback sign test (y-component)
                        sign)
    else:
        # What if x == 0? Then we still need to pick a convention for the rotation axis, but how do we make sure
        #  this is consistent? See the below check. This situation occurs when the first solute hydrogen has
        #  y == 0, which can happen, although it should be rare. Seems to happen when setting up the coordinate system
        #  sometimes.
        # Can't we just remove the check? It's actually fine if x=0 for the rotation axis, since this is a valid
        #  axis to rotate around for aligning the first solute hydrogen with the z-axis. The only problem may be with
        #  the reverse transformation, where we would need to know how exactly to invert this (set a convention).
        #  Here we are in a situation where the rotation axis has z=0 and x=0, so we can use y>0 as our convention.
    
        # Standard: use x; if x ~ 0 (seam) fall back to y
        sign = torch.sign(cross[:, 0])
        seam = torch.abs(cross[:, 0]) < eps_x
        sign = torch.where(seam, torch.sign(cross[:, 1]), sign)
    
    # Avoid sign == 0
    sign = torch.where(sign == 0, torch.ones_like(sign), sign) 
    cross = sign.unsqueeze(-1) * cross
    # Adjust angles:
    #  This evaluates to: 2pi - rads if sign = -1, else: 0 + rads
    rads = 2 * math.pi * (1 - sign) / 2 + sign * rads

    return rads, cross, seam, deg
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
    #  axis, or its negation, depending on the relative orientation of in_both_planes and in_rot_plane. This
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
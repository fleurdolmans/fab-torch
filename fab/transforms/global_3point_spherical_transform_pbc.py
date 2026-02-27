import torch
import math
import torch.nn.functional as F
import normflows as nf

from fab.transforms.global_3point_spherical_transform import (
    Global3PointSphericalTransform,
    get_angle_and_normal,
    rotation_matrix,
    unit_vector,
    get_theta,
    stable_inverse_softplus,   # only if you call it directly
)

class PBCGlobal3PointSphericalTransform(nf.flows.Flow):
    """
    Same as Global3PointSphericalTransform, but:
      - uses MIC for all relative vectors (PBC-consistent)
      - wraps final Cartesian output to [0, L)
    """
    def __init__(self, L: float, system=None, transform_data=None):
        super().__init__()
        self.L = float(L)
        if system is not None:
            self.system = system
            self.atom_order = system.atoms
        self.transform_data = transform_data
        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // 3

        self._stats = {
            "seam_x": 0,
            "seam_z": 0,
            "degenerate": 0,
            "calls": 0,
            "B": 0,
        }

        assert transform_data.shape[0] == 1
        with torch.no_grad():
            z, _, _, _ = self.cartesian_to_z(transform_data, setup=True)
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

    def _L_tensor(self, ref: torch.Tensor, L) -> torch.Tensor:
        return torch.as_tensor(L, device=ref.device, dtype=ref.dtype)

    def wrap(self, x: torch.Tensor, L) -> torch.Tensor:
        L_t = self._L_tensor(x, L)
        return torch.remainder(x, L_t)

    def mic(self, dx: torch.Tensor, L) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def _setup_scale_r(self, z):
        scale_r = z.new_ones(1)
        self.register_buffer("scale_r", scale_r)

    def _setup_scale_phi(self, z):
        scale_phi = z.new_ones(1)
        self.register_buffer("scale_phi", scale_phi)

    def _setup_offset_phi(self, z):
        offset_phi = z.new_ones(1) * math.pi
        self.register_buffer("offset_phi", offset_phi)

    def _setup_scale_theta(self, z):
        scale_theta = z.new_ones(1) / 2
        self.register_buffer("scale_theta", scale_theta)

    # --- PBC-aware coordinate setup ---
    def rotate_into_global_coordinate_system(self, x, z_axis, y_axis, setup=False):
        """
        PBC change: center with MIC instead of plain subtraction.
        """
        # x: (B, N, 3) in [0, L)
        # center relative to atom0 using MIC so all coords are continuous
        x_centered = self.mic(x - x[:, 0:1, :], self.L)

        solute_atom0 = x_centered[:, 0, :]
        solute_atom1 = x_centered[:, 1, :]

        # same as your droplet version from here
        phi_rad, phi_axis, _, _ = get_angle_and_normal(
            z_axis, solute_atom0, solute_atom1, align_first_solute_h=True
        )
        phi_rotation = rotation_matrix(phi_axis, phi_rad)
        x_phi = torch.einsum("bij,bnj -> bni", phi_rotation, x_centered)

        solute_atom2 = x_phi[:, 2, :]
        xy_proj = solute_atom2 - unit_vector(z_axis) * torch.sum(z_axis * solute_atom2, dim=1, keepdim=True)
        theta_rad, _, _, _ = get_angle_and_normal(y_axis, solute_atom0, xy_proj, to_yz_plane=True)
        theta_rotation = rotation_matrix(z_axis, theta_rad)
        x_out = torch.einsum("bij,bnj -> bni", theta_rotation, x_phi)
        return x_out

    def setup_coordinate_system(self, x, setup=False):
        x = x.reshape(x.shape[0], -1, 3)
        z_axis = x.new_zeros(x.shape[0], 3); z_axis[:, 2] = 1
        y_axis = x.new_zeros(x.shape[0], 3); y_axis[:, 1] = 1
        x = self.rotate_into_global_coordinate_system(x, z_axis, y_axis, setup=setup)
        x_coord = x.reshape(x.shape[0], -1)
        return x, x_coord, z_axis, y_axis

    def cartesian_to_z(self, x, setup=False):
        """
        Important: before doing anything, wrap x into [0, L).
        """
        x = self.wrap(x, self.L)
        B = x.shape[0]
        N = x.shape[1] // 3
        X = x.view(B, N, 3)
        L_t = self._L_tensor(X, self.L)

        # solute whole
        a0 = X[:, 0, :]
        X[:, 1, :] = a0 + self.mic(X[:, 1, :] - a0, L_t)
        X[:, 2, :] = a0 + self.mic(X[:, 2, :] - a0, L_t)

        # waters whole: indices 3 + 3*w : O, H, H
        start = 3
        for w in range(self.n_waters):     # make sure self.n_waters exists; else compute
            i = start + 3*w
            O  = X[:, i, :]
            X[:, i+1, :] = O + self.mic(X[:, i+1, :] - O, L_t)
            X[:, i+2, :] = O + self.mic(X[:, i+2, :] - O, L_t)

        x = X.view(B, -1)
        
        # then use the same logic as your droplet version, but it will call the MIC-aware setup above
        return Global3PointSphericalTransform.cartesian_to_z(self, x, setup=setup)

    def z_to_cartesian(self, z):
        x, log_det = Global3PointSphericalTransform.z_to_cartesian(self, z)
        # x is centered around atom0 at origin; put it into the box
        X = x.view(x.shape[0], -1, 3)   # (B, N, 3)
        B, N, _ = X.shape
        L_t = self._L_tensor(X, self.L)
        
        # place atom0 at box center (or any anchor you choose)
        center = 0.5 * L_t
        X = X + center                  # translation only

        # optionally wrap after translation (not strictly required for OpenMM)
        X = self.wrap(X, L_t)

        # --- make solute whole relative to atom0 ---
        a0 = X[:, 0, :]
        X[:, 1, :] = a0 + self.mic(X[:, 1, :] - a0, L_t)
        X[:, 2, :] = a0 + self.mic(X[:, 2, :] - a0, L_t)

        # --- make each water whole relative to its O ---
        start = 3
        for w in range(self.n_waters):
            i = start + 3 * w
            O = X[:, i, :]
            X[:, i + 1, :] = O + self.mic(X[:, i + 1, :] - O, L_t)
            X[:, i + 2, :] = O + self.mic(X[:, i + 2, :] - O, L_t)

        return X.reshape(B, -1), log_det

    def forward(self, z):
        self.scale_r = self.scale_r.to(z.device)
        self.scale_phi = self.scale_phi.to(z.device)
        self.offset_phi = self.offset_phi.to(z.device)
        self.scale_theta = self.scale_theta.to(z.device)
        x, log_det_jac = self.z_to_cartesian(z)
        return x, log_det_jac

    def inverse(self, x):
        self.scale_r = self.scale_r.to(x.device)
        self.scale_phi = self.scale_phi.to(x.device)
        self.offset_phi = self.offset_phi.to(x.device)
        self.scale_theta = self.scale_theta.to(x.device)
        z, log_det, _, _ = self.cartesian_to_z(x, setup=False)
        return z, log_det
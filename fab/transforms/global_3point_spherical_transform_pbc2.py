import torch
import math
import normflows as nf

import torch
import math
import normflows as nf

import torch
import math
import normflows as nf


# class PBCGlobal3PointSphericalTransform2(nf.flows.Flow):
#     """
#     PBC + rigid-water-friendly coordinate transform.

#     Internal coords i:
#       - solute: shape only -> (r1, r2, theta)                  -> 3 dims
#       - waters: sorted by O distance to solute atom 0
#           each water: O position relative to solute origin     -> 3 dims
#                     + rotation vector omega                    -> 3 dims

#     forward(i)  : i -> Cartesian x (flattened)
#     inverse(x)  : Cartesian x -> i
#     """
#     def __init__(self, L: float, system=None, transform_data=None, internal_dim=None):
#         super().__init__()
#         self.L = float(L)
#         self.system = system
#         self.transform_data = transform_data

#         assert transform_data is not None and transform_data.shape[0] == 1
#         self.n_atoms = transform_data.shape[1] // 3
#         self.n_solute = 3
#         self.n_atoms_per_mol = 3
#         self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
#         assert self.n_solute + 3 * self.n_waters == self.n_atoms

#         with torch.no_grad():
#             x0 = transform_data.reshape(1, self.n_atoms, 3).clone()
#             w0_start = self.n_solute
#             ref = x0[:, w0_start:w0_start + 3, :]   # [O,H,H]
#             ref = ref[0]
#             ref_O = ref[0:1, :]
#             ref_rel = ref - ref_O
#             self.ref_H1 = ref_rel[1].clone()
#             self.ref_H2 = ref_rel[2].clone()

#             ref_cent = ref_rel.mean(dim=0, keepdim=True)
#             self.ref_water_kabsch = (ref_rel - ref_cent).clone()

#         self.internal_dim = internal_dim
#         print("internal_dim in transform:", self.internal_dim)

#     # ---------- PBC helpers ----------
#     def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
#         return torch.as_tensor(L, device=x.device, dtype=x.dtype)

#     def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
#         L_t = self._L_tensor(dx, L)
#         return dx - L_t * torch.round(dx / L_t)

#     def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
#         L_t = self._L_tensor(x, self.L)
#         return x - L_t * torch.floor(x / L_t)

#     def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
#         x0 = x_sol[:, 0:1, :]
#         d = self.mic(x_sol - x0, self.L)
#         return x0 + d

#     def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
#         O = x_w[:, 0:1, :]
#         d = self.mic(x_w - O, self.L)
#         return O + d

#     # ---------- SO(3) maps ----------
#     def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
#         B = w.shape[0]
#         theta = torch.linalg.norm(w, dim=1, keepdim=True)
#         I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)

#         small = theta[:, 0] < 1e-8
#         big = ~small

#         R = I.clone()

#         if big.any():
#             th = theta[big]
#             k = w[big] / th
#             kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

#             K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
#             K[:, 0, 1] = -kz
#             K[:, 0, 2] =  ky
#             K[:, 1, 0] =  kz
#             K[:, 1, 2] = -kx
#             K[:, 2, 0] = -ky
#             K[:, 2, 1] =  kx

#             ct = torch.cos(th).view(-1, 1, 1)
#             st = torch.sin(th).view(-1, 1, 1)
#             Ibig = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(big.sum(), 3, 3)

#             R[big] = Ibig + st * K + (1.0 - ct) * (K @ K)

#         if small.any():
#             K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
#             ws = w[small]
#             K[:, 0, 1] = -ws[:, 2]
#             K[:, 0, 2] =  ws[:, 1]
#             K[:, 1, 0] =  ws[:, 2]
#             K[:, 1, 2] = -ws[:, 0]
#             K[:, 2, 0] = -ws[:, 1]
#             K[:, 2, 1] =  ws[:, 0]
#             R[small] = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0) + K

#         return R

#     def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
#         B = R.shape[0]
#         trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
#         cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
#         theta = torch.acos(cos_theta)

#         w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

#         small = theta < 1e-6
#         near_pi = torch.abs(theta - math.pi) < 1e-4
#         mid = ~(small | near_pi)

#         if small.any():
#             A = 0.5 * (R[small] - R[small].transpose(1, 2))
#             w[small, 0] = A[:, 2, 1]
#             w[small, 1] = A[:, 0, 2]
#             w[small, 2] = A[:, 1, 0]

#         if mid.any():
#             th = theta[mid]
#             denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
#             axis = torch.stack([
#                 (R[mid, 2, 1] - R[mid, 1, 2]) / denom,
#                 (R[mid, 0, 2] - R[mid, 2, 0]) / denom,
#                 (R[mid, 1, 0] - R[mid, 0, 1]) / denom,
#             ], dim=1)
#             w[mid] = axis * th.unsqueeze(1)

#         if near_pi.any():
#             Rp = R[near_pi]
#             th = theta[near_pi]

#             diag = torch.stack([
#                 Rp[:, 0, 0],
#                 Rp[:, 1, 1],
#                 Rp[:, 2, 2],
#             ], dim=1)

#             axis = torch.sqrt(((diag + 1.0) / 2.0).clamp_min(0.0))

#             axis0 = axis[:, 0].clone()
#             axis1 = axis[:, 1].clone()
#             axis2 = axis[:, 2].clone()

#             axis1 = torch.where(
#                 axis0 > 1e-6,
#                 (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis0.clamp_min(1e-12)),
#                 axis1,
#             )
#             axis2 = torch.where(
#                 axis0 > 1e-6,
#                 (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis0.clamp_min(1e-12)),
#                 axis2,
#             )

#             axis0 = torch.where(
#                 (axis0 <= 1e-6) & (axis1 > 1e-6),
#                 (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis1.clamp_min(1e-12)),
#                 axis0,
#             )
#             axis2 = torch.where(
#                 (axis0 <= 1e-6) & (axis1 > 1e-6),
#                 (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis1.clamp_min(1e-12)),
#                 axis2,
#             )

#             axis0 = torch.where(
#                 (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
#                 (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis2.clamp_min(1e-12)),
#                 axis0,
#             )
#             axis1 = torch.where(
#                 (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
#                 (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis2.clamp_min(1e-12)),
#                 axis1,
#             )

#             axis = torch.stack([axis0, axis1, axis2], dim=1)
#             axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
#             w[near_pi] = axis * th.unsqueeze(1)

#         return w

#     def so3_logdet_exp(self, w: torch.Tensor) -> torch.Tensor:
#         theta = torch.linalg.norm(w, dim=1).clamp_min(1e-12)
#         half = 0.5 * theta
#         return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

#     # ---------- Kabsch ----------
#     @staticmethod
#     def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#         """
#         X: (B,3,3), Y: (B,3,3), centered
#         Returns R such that X @ R ≈ Y
#         """
#         C = X.transpose(1, 2) @ Y
#         U, S, Vh = torch.linalg.svd(C)
#         V = Vh.transpose(1, 2)
#         Ut = U.transpose(1, 2)

#         det = torch.det(V @ Ut)
#         D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
#         D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
#         return V @ D @ Ut

#     def angle_abc(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         ba = a - b
#         bc = c - b
#         ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         cosang = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
#         return torch.acos(cosang)

#     # ---------- Sorting helper ----------
#     def _extract_and_sort_waters(self, x_rel: torch.Tensor):
#         """
#         x_rel: (B, N, 3) relative to solute atom 0
#         Returns:
#           O_sorted:     (B, W, 3)
#           water_sorted: (B, W, 3, 3) where each water is [O,H,H]
#         """
#         water_list = []
#         O_list = []
#         for k in range(self.n_waters):
#             s = self.n_solute + 3 * k
#             w = x_rel[:, s:s+3, :]            # (B,3,3)
#             w = self.make_whole_water(w)
#             water_list.append(w)
#             O_list.append(w[:, 0, :])

#         water_all = torch.stack(water_list, dim=1)   # (B, W, 3, 3)
#         O_all = torch.stack(O_list, dim=1)           # (B, W, 3)

#         r = torch.linalg.norm(O_all, dim=-1)         # (B, W)
#         perm = torch.argsort(r, dim=1)               # (B, W)

#         O_sorted = torch.gather(
#             O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3)
#         )
#         water_sorted = torch.gather(
#             water_all,
#             1,
#             perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
#         )
#         return O_sorted, water_sorted

#     def inverse(self, x: torch.Tensor):
#         """
#         x: (B, 3*N) flattened Cartesian
#         Returns:
#           i: (B, internal_dim)
#           logdet_xi: (B,)
#         """
#         B = x.shape[0]
#         x = x.view(B, self.n_atoms, 3)

#         # Make solute whole and center on solute atom 0
#         sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
#         origin = sol[:, 0:1, :]                     # (B,1,3)
#         x_rel = self.mic(x - origin, self.L)        # (B,N,3)

#         # Solute internal shape
#         v1 = x_rel[:, 1, :]
#         v2 = x_rel[:, 2, :]

#         r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
#         r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
#         theta = self.angle_abc(x_rel[:, 1, :], x_rel[:, 0, :], x_rel[:, 2, :]).unsqueeze(1)

#         pieces = [r1, r2, theta]
#         logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

#         # Sort waters by O distance to solute
#         O_sorted, water_sorted = self._extract_and_sort_waters(x_rel)

#         Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)
#         for k in range(self.n_waters):
#             w = water_sorted[:, k, :, :]      # (B,3,3)
#             O = O_sorted[:, k, :]             # (B,3)

#             w_rel = w - w[:, 0:1, :]
#             w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
#             R = self.kabsch_rotation(Yref, w_cent)
#             omega = self.rotmat_to_rotvec(R)

#             pieces += [O, omega]

#         i = torch.cat(pieces, dim=1)
#         return i, logdet

#     def forward(self, i: torch.Tensor):
#         """
#         i: (B, internal_dim)
#         Returns:
#           x: (B, 3*N) flattened Cartesian in [0,L)
#           logdet_ix: (B,)
#         """
#         B = i.shape[0]
#         assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

#         # Solute shape
#         r1 = i[:, 0:1].clamp_min(1e-6)
#         r2 = i[:, 1:2].clamp_min(1e-6)
#         theta = i[:, 2:3].clamp(1e-3, math.pi - 1e-3)

#         center = 0.5 * self.L
#         x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

#         # Canonical solute placement
#         x[:, 0, :] = center

#         x[:, 1, 0] = center + r1[:, 0]
#         x[:, 1, 1] = center
#         x[:, 1, 2] = center

#         x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
#         x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
#         x[:, 2, 2] = center

#         idx = 3
#         logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

#         H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
#         H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

#         # Reconstruct waters in sorted order
#         for k in range(self.n_waters):
#             O = i[:, idx:idx+3]
#             omega = i[:, idx+3:idx+6]
#             idx += 6

#             R = self.rotvec_to_rotmat(omega)

#             O_abs = center + O
#             atom0 = self.n_solute + 3 * k

#             x[:, atom0 + 0, :] = O_abs

#             H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
#             H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

#             x[:, atom0 + 1, :] = O_abs + H1
#             x[:, atom0 + 2, :] = O_abs + H2

#         x = self.wrap_0L(x)
#         return x.view(B, -1), logdet

class PBCGlobal3PointSphericalTransform2(nf.flows.Flow):
    """
    PBC + rigid-water-friendly coordinate transform.

    Internal coords i:
      - solute: v1 = x1-x0, v2 = x2-x0  (each MIC wrapped)  -> 6 dims
      - each water: O position relative to solute origin (MIC) -> 3 dims
                  + rotation vector omega (axis-angle)        -> 3 dims
        (H positions are reconstructed from O + omega and a fixed reference geometry)

    forward(i)  : i -> Cartesian x (flattened)
    inverse(x)  : Cartesian x -> i
    """
    def __init__(self, L: float, system=None, transform_data=None, internal_dim=None):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.transform_data = transform_data

        assert transform_data is not None and transform_data.shape[0] == 1
        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        # Build a reference rigid-water geometry in the "water body frame":
        # O at origin; two H vectors defined relative to O. We pull this from transform_data.
        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()  # (1,N,3)
            # Use first water (water index 0) from transform_data as reference
            w0_start = self.n_solute
            ref = x0[:, w0_start:w0_start+3, :]  # (1,3,3) = [O,H,H] in OpenMM's OHH ordering
            ref = ref[0]  # (3,3)
            ref_O = ref[0:1, :]
            ref_rel = ref - ref_O  # O at 0
            # Store reference H vectors (3,)
            self.ref_H1 = ref_rel[1].clone()
            self.ref_H2 = ref_rel[2].clone()

            # Also store reference water points for Kabsch (centered on centroid) to infer rotation from data
            ref_cent = ref_rel.mean(dim=0, keepdim=True)  # (1,3)
            self.ref_water_kabsch = (ref_rel - ref_cent).clone()  # (3,3)

        # Internal dimension: solute(6) + waters(6 each)
        # self.internal_dim = 6 + 6 * self.n_waters
        self.internal_dim = internal_dim
        # print("internal_dim  in transform:", self.internal_dim )

    # ---------- PBC helpers ----------
    def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
        # broadcastable tensor of L
        return torch.as_tensor(L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        # wrap positions into [0, L)

        L_t = self._L_tensor(x, self.L)
        # wrapped = torch.remainder(x, L_t)
        wrapped = x - L_t * torch.floor(x / L_t)
        return wrapped

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        # x_sol: (B,3,3)
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0, self.L)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        # x_w: (B,3,3) with ordering [O,H,H]
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O, self.L)
        return O + d

    # ---------- SO(3) maps ----------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (B,3) rotation vector (axis * angle)
        returns R: (B,3,3)
        """
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)  # (B,1)
        k = w / theta  # (B,3)

        kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
        K = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
        K[:, 0, 1] = -kz
        K[:, 0, 2] =  ky
        K[:, 1, 0] =  kz
        K[:, 1, 2] = -kx
        K[:, 2, 0] = -ky
        K[:, 2, 1] =  kx

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
        ct = torch.cos(theta).view(B, 1, 1)
        st = torch.sin(theta).view(B, 1, 1)

        R = I + st * K + (1.0 - ct) * (K @ K)
        return R

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: (B,3,3)
        returns w: (B,3)
        """
        # Robust log map for SO(3)
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)  # (B,)

        w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

        # For small angles, use first-order approximation
        small = theta < 1e-6
        if small.any():
            # vee(R - R^T)/2
            Rt = R[small].transpose(1, 2)
            A = (R[small] - Rt) * 0.5
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        # For general case:
        big = ~small
        if big.any():
            th = theta[big]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            wx = (R[big, 2, 1] - R[big, 1, 2]) / denom
            wy = (R[big, 0, 2] - R[big, 2, 0]) / denom
            wz = (R[big, 1, 0] - R[big, 0, 1]) / denom
            axis = torch.stack([wx, wy, wz], dim=1)
            w[big] = axis * th.unsqueeze(1)

        # Optional: keep theta in [0, pi] already guaranteed by acos.
        return w

    def so3_logdet_exp(self, w: torch.Tensor) -> torch.Tensor:
        """
        log |det J| for the SO(3) exponential map w (R^3) -> R (SO(3))
        Used as a measure correction if you want rotation vectors to represent Haar measure.
        Returns (B,) logdet contribution.
        """
        theta = torch.linalg.norm(w, dim=1).clamp_min(1e-12)  # (B,)
        half = 0.5 * theta
        # 2*log(sin(θ/2)/(θ/2))
        val = 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))
        # At theta→0, limit is 0; above formula is stable with clamp_min.
        return val

    # ---------- Kabsch for water orientation from data ----------
    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3) current points (centered)
        Y: (B,3,3) reference points (centered)
        Returns R: (B,3,3) such that X @ R ≈ Y
        """
        C = X.transpose(1, 2) @ Y
        U, S, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)  # (B,)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

        R = V @ D @ Ut
        return R

    # ---------- nf API ----------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N) flattened Cartesian
        Returns:
          i: (B, internal_dim)
          logdet_xi: (B,)
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Make solute whole and center on solute atom 0
        sol = self.make_whole_solute(x[:, :3, :])  # (B,3,3)
        origin = sol[:, 0:1, :]                    # (B,1,3)
        x_rel = self.mic(x - origin, self.L)       # (B,N,3) relative to solute atom0

        # Solute internal: two MIC bond vectors from atom0
        v1 = x_rel[:, 1, :]  # already mic-wrapped
        v2 = x_rel[:, 2, :]

        pieces = [v1, v2]

        logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

        # Waters: O position + rotation vector omega
        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)  # (B,3,3)
        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s+3, :]               # (B,3,3) [O,H,H] in solute-centered frame
            w = self.make_whole_water(w)         # ensure H's are whole w.r.t O

            O = w[:, 0, :]                       # (B,3)

            # Infer orientation by Kabsch from reference geometry
            w_rel = w - w[:, 0:1, :]             # O at 0
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
            # R = self.kabsch_rotation(w_cent, Yref)  # (B,3,3)
            R = self.kabsch_rotation(Yref, w_cent)

            omega = self.rotmat_to_rotvec(R)     # (B,3)

            pieces += [O, omega]

            # Optional SO(3) exp-map Jacobian correction
            # logdet = logdet + self.so3_logdet_exp(omega)

        i = torch.cat(pieces, dim=1)  # (B, 6 + 6*n_waters)
        return i, logdet

    def forward(self, i: torch.Tensor):
        """
        i: (B, internal_dim)
        Returns:
          x: (B, 3*N) flattened Cartesian in [0,L)
          logdet_ix: (B,) such that log p_X = log p_I - logdet_ix (nf convention depends on usage)
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # Unpack solute
        v1 = i[:, 0:3]
        v2 = i[:, 3:6]

        # Place solute atom0 at box center (nice gauge choice for OpenMM)
        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        # Waters
        idx = 6
        logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)  # (1,3,1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        for k in range(self.n_waters):
            O = i[:, idx:idx+3]           # (B,3)
            omega = i[:, idx+3:idx+6]     # (B,3)
            idx += 6

            R = self.rotvec_to_rotmat(omega)  # (B,3,3)

            # Oxygen position is relative to solute atom0 at center
            O_abs = center + O
            x[:, self.n_solute + 3*k + 0, :] = O_abs

            # Rotate reference H vectors
            # (B,3,3) @ (B,3,1) -> (B,3,1) -> (B,3)
            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            x[:, self.n_solute + 3*k + 1, :] = O_abs + H1
            x[:, self.n_solute + 3*k + 2, :] = O_abs + H2

            # logdet = logdet + self.so3_logdet_exp(omega)

        # Wrap into [0,L)
        x = self.wrap_0L(x)

        return x.view(B, -1), logdet

# class PBCGlobal3PointSphericalTransform2(nf.flows.Flow):
#     """
#     PBC + rigid-water-friendly coordinate transform.

#     Internal coords i:
#       - solute: shape only -> (r1, r2, theta)                  -> 3 dims
#       - waters: 
#           each water: O position relative to solute origin     -> 3 dims
#                     + rotation vector omega                    -> 3 dims
#         (H positions are reconstructed from O + omega and a fixed reference geometry)

#     forward(i)  : i -> Cartesian x (flattened)
#     inverse(x)  : Cartesian x -> i
#     """
#     def __init__(self, L: float, system=None, transform_data=None, internal_dim=None):
#         super().__init__()
#         self.L = float(L)
#         self.system = system
#         self.transform_data = transform_data

#         assert transform_data is not None and transform_data.shape[0] == 1
#         self.n_atoms = transform_data.shape[1] // 3
#         self.n_solute = 3
#         self.n_atoms_per_mol = 3
#         self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
#         assert self.n_solute + 3 * self.n_waters == self.n_atoms

#         # Build a reference rigid-water geometry in the "water body frame":
#         # O at origin; two H vectors defined relative to O. We pull this from transform_data.
#         with torch.no_grad():
#             x0 = transform_data.reshape(1, self.n_atoms, 3).clone()  # (1,N,3)
#             # Use first water (water index 0) from transform_data as reference
#             w0_start = self.n_solute
#             ref = x0[:, w0_start:w0_start+3, :]  # (1,3,3) = [O,H,H] in OpenMM's OHH ordering
#             ref = ref[0]  # (3,3)
#             ref_O = ref[0:1, :]
#             ref_rel = ref - ref_O  # O at 0
#             # Store reference H vectors (3,)
#             self.ref_H1 = ref_rel[1].clone()
#             self.ref_H2 = ref_rel[2].clone()

#             # Also store reference water points for Kabsch (centered on centroid) to infer rotation from data
#             ref_cent = ref_rel.mean(dim=0, keepdim=True)  # (1,3)
#             self.ref_water_kabsch = (ref_rel - ref_cent).clone()  # (3,3)

#         # Internal dimension: solute(6) + waters(6 each)
#         self.internal_dim = internal_dim
#         print("internal_dim  in transform:", self.internal_dim )

#     # ---------- PBC helpers ----------
#     def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
#         # broadcastable tensor of L
#         return torch.as_tensor(L, device=x.device, dtype=x.dtype)

#     def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
#         L_t = self._L_tensor(dx, L)
#         return dx - L_t * torch.round(dx / L_t)

#     def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
#         # wrap positions into [0, L)

#         L_t = self._L_tensor(x, self.L)
#         # wrapped = torch.remainder(x, L_t)
#         wrapped = x - L_t * torch.floor(x / L_t)
#         return wrapped

#     def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
#         # x_sol: (B,3,3)
#         x0 = x_sol[:, 0:1, :]
#         d = self.mic(x_sol - x0, self.L)
#         return x0 + d

#     def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
#         # x_w: (B,3,3) with ordering [O,H,H]
#         O = x_w[:, 0:1, :]
#         d = self.mic(x_w - O, self.L)
#         return O + d

#     # ---------- SO(3) maps ----------
#     def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
#         """
#         w: (B,3) rotation vector (axis * angle)
#         returns R: (B,3,3)
#         """
#         B = w.shape[0]
#         theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)  # (B,1)
#         k = w / theta  # (B,3)

#         kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
#         K = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
#         K[:, 0, 1] = -kz
#         K[:, 0, 2] =  ky
#         K[:, 1, 0] =  kz
#         K[:, 1, 2] = -kx
#         K[:, 2, 0] = -ky
#         K[:, 2, 1] =  kx

#         I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
#         ct = torch.cos(theta).view(B, 1, 1)
#         st = torch.sin(theta).view(B, 1, 1)

#         R = I + st * K + (1.0 - ct) * (K @ K)
#         return R

#     def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
#         """
#         R: (B,3,3)
#         returns w: (B,3)
#         """
#         # Robust log map for SO(3)
#         B = R.shape[0]
#         trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
#         cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
#         theta = torch.acos(cos_theta)  # (B,)

#         w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

#         # For small angles, use first-order approximation
#         small = theta < 1e-6
#         if small.any():
#             # vee(R - R^T)/2
#             Rt = R[small].transpose(1, 2)
#             A = (R[small] - Rt) * 0.5
#             w[small, 0] = A[:, 2, 1]
#             w[small, 1] = A[:, 0, 2]
#             w[small, 2] = A[:, 1, 0]

#         # For general case:
#         big = ~small
#         if big.any():
#             th = theta[big]
#             denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
#             wx = (R[big, 2, 1] - R[big, 1, 2]) / denom
#             wy = (R[big, 0, 2] - R[big, 2, 0]) / denom
#             wz = (R[big, 1, 0] - R[big, 0, 1]) / denom
#             axis = torch.stack([wx, wy, wz], dim=1)
#             w[big] = axis * th.unsqueeze(1)

#         return w

#     def so3_logdet_exp(self, w: torch.Tensor) -> torch.Tensor:
#         """
#         log |det J| for the SO(3) exponential map w (R^3) -> R (SO(3))
#         Used as a measure correction if you want rotation vectors to represent Haar measure.
#         Returns (B,) logdet contribution.
#         """
#         theta = torch.linalg.norm(w, dim=1).clamp_min(1e-12)  # (B,)
#         half = 0.5 * theta
#         # 2*log(sin(θ/2)/(θ/2))
#         val = 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))
#         # At theta→0, limit is 0; above formula is stable with clamp_min.
#         return val

#     # ---------- Kabsch for water orientation from data ----------
#     @staticmethod
#     def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#         """
#         X: (B,3,3) current points (centered)
#         Y: (B,3,3) reference points (centered)
#         Returns R: (B,3,3) such that X @ R ≈ Y
#         """
#         C = X.transpose(1, 2) @ Y
#         U, S, Vh = torch.linalg.svd(C)
#         V = Vh.transpose(1, 2)
#         Ut = U.transpose(1, 2)

#         det = torch.det(V @ Ut)  # (B,)
#         D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
#         D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

#         R = V @ D @ Ut
#         return R
    
#     def angle_abc(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         """
#         Angle ABC in radians.
#         a,b,c: (B,3)
#         returns: (B,)
#         """
#         ba = a - b
#         bc = c - b
#         ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         cosang = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
#         return torch.acos(cosang)

#     def inverse(self, x: torch.Tensor):
#         """
#         x: (B, 3*N) flattened Cartesian
#         Returns:
#           i: (B, internal_dim)
#           logdet_xi: (B,)
#         """
#         B = x.shape[0]
#         x = x.view(B, self.n_atoms, 3)

#         # Make solute whole and center on solute atom 0
#         sol = self.make_whole_solute(x[:, :3, :])  # (B,3,3)
#         origin = sol[:, 0:1, :]                    # (B,1,3)
#         x_rel = self.mic(x - origin, self.L)       # (B,N,3) relative to solute atom0

#         # # Solute internal: two MIC bond vectors from atom0
#         # v1 = x_rel[:, 1, :]  # already mic-wrapped
#         # v2 = x_rel[:, 2, :]

#         # pieces = [v1, v2]

#         # Solute internal shape: r1, r2, theta
#         v1 = x_rel[:, 1, :]                      # S -> O1
#         v2 = x_rel[:, 2, :]                      # S -> O2

#         r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)   # (B,1)
#         r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)   # (B,1)
#         theta = self.angle_abc(x_rel[:, 1, :], x_rel[:, 0, :], x_rel[:, 2, :]).unsqueeze(1)  # (B,1)

#         pieces = [r1, r2, theta]

#         logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

#         # Waters: O position + rotation vector omega
#         Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)  # (B,3,3)
#         for k in range(self.n_waters):
#             s = self.n_solute + 3 * k
#             w = x_rel[:, s:s+3, :]               # (B,3,3) [O,H,H] in solute-centered frame
#             w = self.make_whole_water(w)         # ensure H's are whole w.r.t O

#             O = w[:, 0, :]                       # (B,3)

#             # Infer orientation by Kabsch from reference geometry
#             w_rel = w - w[:, 0:1, :]             # O at 0
#             w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
#             # R = self.kabsch_rotation(w_cent, Yref)  # (B,3,3)
#             R = self.kabsch_rotation(Yref, w_cent)

#             omega = self.rotmat_to_rotvec(R)     # (B,3)

#             pieces += [O, omega]

#             # Optional SO(3) exp-map Jacobian correction
#             # logdet = logdet + self.so3_logdet_exp(omega)

#         i = torch.cat(pieces, dim=1)  # (B, 6 + 6*n_waters)
#         return i, logdet

#     def forward(self, i: torch.Tensor):
#         """
#         i: (B, internal_dim)
#         Returns:
#           x: (B, 3*N) flattened Cartesian in [0,L)
#           logdet_ix: (B,) such that log p_X = log p_I - logdet_ix (nf convention depends on usage)
#         """
#         B = i.shape[0]
#         assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

#         # # Unpack solute
#         # v1 = i[:, 0:3]
#         # v2 = i[:, 3:6]

#         # # Place solute atom0 at box center
#         # center = 0.5 * self.L
#         # x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
#         # x[:, 0, :] = center
#         # x[:, 1, :] = center + v1
#         # x[:, 2, :] = center + v2

#         # Unpack solute shape
#         r1 = i[:, 0:1].clamp_min(1e-6)          # (B,1)
#         r2 = i[:, 1:2].clamp_min(1e-6)          # (B,1)
#         theta = i[:, 2:3].clamp(1e-3, math.pi - 1e-3)   # (B,1)

#         center = 0.5 * self.L
#         x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

#         # S at center
#         x[:, 0, :] = center

#         # O1 on +x
#         x[:, 1, 0] = center + r1[:, 0]
#         x[:, 1, 1] = center
#         x[:, 1, 2] = center

#         # O2 in xy-plane
#         x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
#         x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
#         x[:, 2, 2] = center

#         # Waters
#         # idx = 6
#         idx = 3
#         logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

#         H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)  # (1,3,1)
#         H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

#         for k in range(self.n_waters):
#             O = i[:, idx:idx+3]           # (B,3)
#             omega = i[:, idx+3:idx+6]     # (B,3)
#             idx += 6

#             R = self.rotvec_to_rotmat(omega)  # (B,3,3)

#             # Oxygen position is relative to solute atom0 at center
#             O_abs = center + O
#             x[:, self.n_solute + 3*k + 0, :] = O_abs

#             # Rotate reference H vectors
#             # (B,3,3) @ (B,3,1) -> (B,3,1) -> (B,3)
#             H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
#             H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

#             x[:, self.n_solute + 3*k + 1, :] = O_abs + H1
#             x[:, self.n_solute + 3*k + 2, :] = O_abs + H2

#             # logdet = logdet + self.so3_logdet_exp(omega)

#         # Wrap into [0,L)
#         x = self.wrap_0L(x)

#         return x.view(B, -1), logdet


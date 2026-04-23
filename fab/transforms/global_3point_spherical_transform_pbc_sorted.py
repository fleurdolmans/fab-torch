import torch
import math
import normflows as nf


class PBCGlobal3PointSphericalTransformSorted(nf.flows.Flow):
    """
    PBC + rigid-water-friendly coordinate transform.

    Internal coords i:
      - solute: v1 = x1-x0, v2 = x2-x0  (MIC wrapped)         -> 6 dims
      - waters: sorted by O distance to solute atom 0
          each water: O position relative to solute origin    -> 3 dims
                    + rotation vector omega                   -> 3 dims

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

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()
            w0_start = self.n_solute
            ref = x0[:, w0_start:w0_start + 3, :]   # [O,H,H]
            ref = ref[0]
            ref_O = ref[0:1, :]
            ref_rel = ref - ref_O
            self.ref_H1 = ref_rel[1].clone()
            self.ref_H2 = ref_rel[2].clone()

            ref_cent = ref_rel.mean(dim=0, keepdim=True)
            self.ref_water_kabsch = (ref_rel - ref_cent).clone()

        # v1,v2 solute + 6 per water
        self.internal_dim = 6 + 6 * self.n_waters if internal_dim is None else internal_dim
        print("internal_dim in transform:", self.internal_dim)

    # ---------- PBC helpers ----------
    def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
        return torch.as_tensor(L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L_t = self._L_tensor(x, self.L)
        return x - L_t * torch.floor(x / L_t)

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0, self.L)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O, self.L)
        return O + d

    # ---------- SO(3) maps ----------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True)
        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)

        small = theta[:, 0] < 1e-8
        big = ~small

        R = I.clone()

        if big.any():
            th = theta[big]
            k = w[big] / th
            kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

            K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
            K[:, 0, 1] = -kz
            K[:, 0, 2] =  ky
            K[:, 1, 0] =  kz
            K[:, 1, 2] = -kx
            K[:, 2, 0] = -ky
            K[:, 2, 1] =  kx

            ct = torch.cos(th).view(-1, 1, 1)
            st = torch.sin(th).view(-1, 1, 1)
            Ibig = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(big.sum(), 3, 3)

            R[big] = Ibig + st * K + (1.0 - ct) * (K @ K)

        if small.any():
            K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
            ws = w[small]
            K[:, 0, 1] = -ws[:, 2]
            K[:, 0, 2] =  ws[:, 1]
            K[:, 1, 0] =  ws[:, 2]
            K[:, 1, 2] = -ws[:, 0]
            K[:, 2, 0] = -ws[:, 1]
            K[:, 2, 1] =  ws[:, 0]
            R[small] = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0) + K

        return R

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        near_pi = torch.abs(theta - math.pi) < 1e-4
        mid = ~(small | near_pi)

        if small.any():
            A = 0.5 * (R[small] - R[small].transpose(1, 2))
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        if mid.any():
            th = theta[mid]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            axis = torch.stack([
                (R[mid, 2, 1] - R[mid, 1, 2]) / denom,
                (R[mid, 0, 2] - R[mid, 2, 0]) / denom,
                (R[mid, 1, 0] - R[mid, 0, 1]) / denom,
            ], dim=1)
            w[mid] = axis * th.unsqueeze(1)

        if near_pi.any():
            Rp = R[near_pi]
            th = theta[near_pi]

            diag = torch.stack([Rp[:, 0, 0], Rp[:, 1, 1], Rp[:, 2, 2]], dim=1)
            axis = torch.sqrt(((diag + 1.0) / 2.0).clamp_min(0.0))

            axis0 = axis[:, 0].clone()
            axis1 = axis[:, 1].clone()
            axis2 = axis[:, 2].clone()

            axis1 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis1,
            )
            axis2 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis1.clamp_min(1e-12)),
                axis0,
            )
            axis2 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis1.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis2.clamp_min(1e-12)),
                axis0,
            )
            axis1 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis2.clamp_min(1e-12)),
                axis1,
            )

            axis = torch.stack([axis0, axis1, axis2], dim=1)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
            w[near_pi] = axis * th.unsqueeze(1)

        return w

    # ---------- Kabsch ----------
    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3), Y: (B,3,3), centered
        Returns R such that X @ R ≈ Y
        """
        C = X.transpose(1, 2) @ Y
        U, S, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return V @ D @ Ut

    # ---------- Sorting helper ----------
    def _extract_and_sort_waters(self, x_rel: torch.Tensor):
        """
        x_rel: (B, N, 3) relative to solute atom 0

        Returns
        -------
        O_sorted:     (B, W, 3)
        water_sorted: (B, W, 3, 3) where each water is [O,H,H]
        """
        water_list = []
        O_list = []

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s+3, :]      # (B,3,3)
            w = self.make_whole_water(w)
            water_list.append(w)
            O_list.append(w[:, 0, :])

        water_all = torch.stack(water_list, dim=1)   # (B,W,3,3)
        O_all = torch.stack(O_list, dim=1)           # (B,W,3)

        r = torch.linalg.norm(O_all, dim=-1)         # (B,W)
        perm = torch.argsort(r, dim=1, stable=True)  # canonical sort by O radius

        O_sorted = torch.gather(
            O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3)
        )
        water_sorted = torch.gather(
            water_all,
            1,
            perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        )
        return O_sorted, water_sorted

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
        x_rel = self.mic(x - origin, self.L)       # (B,N,3)

        # Solute internal: v1, v2
        v1 = x_rel[:, 1, :]
        v2 = x_rel[:, 2, :]

        pieces = [v1, v2]
        logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

        # Sort waters by O distance to solute origin
        O_sorted, water_sorted = self._extract_and_sort_waters(x_rel)

        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            w = water_sorted[:, k, :, :]   # (B,3,3)
            O = O_sorted[:, k, :]          # (B,3)

            w_rel = w - w[:, 0:1, :]
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
            R = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(R)

            pieces += [O, omega]

        i = torch.cat(pieces, dim=1)  # (B, 6 + 6*n_waters)
        return i, logdet

    def forward(self, i: torch.Tensor):
        """
        i: (B, internal_dim)
        Returns:
          x: (B, 3*N) flattened Cartesian in [0,L)
          logdet_ix: (B,)
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # Unpack solute v1, v2
        v1 = i[:, 0:3]
        v2 = i[:, 3:6]

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # Canonical solute placement
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        idx = 6
        logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        # Reconstruct waters in sorted canonical order
        for k in range(self.n_waters):
            O = i[:, idx:idx+3]
            omega = i[:, idx+3:idx+6]
            idx += 6

            R = self.rotvec_to_rotmat(omega)

            O_abs = center + O
            atom0 = self.n_solute + 3 * k

            x[:, atom0 + 0, :] = O_abs

            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            x[:, atom0 + 1, :] = O_abs + H1
            x[:, atom0 + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet

class PBCGlobal3PointSphericalTransformSorted2(nf.flows.Flow):
    """
    PBC + rigid-water-friendly coordinate transform.

    Internal coords i:
      - solute: shape only -> (r1, r2, theta)                  -> 3 dims
      - waters: sorted by O distance to solute atom 0
          each water: O position relative to solute origin     -> 3 dims
                    + rotation vector omega                    -> 3 dims

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

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()
            w0_start = self.n_solute
            ref = x0[:, w0_start:w0_start + 3, :]   # [O,H,H]
            ref = ref[0]
            ref_O = ref[0:1, :]
            ref_rel = ref - ref_O
            self.ref_H1 = ref_rel[1].clone()
            self.ref_H2 = ref_rel[2].clone()

            ref_cent = ref_rel.mean(dim=0, keepdim=True)
            self.ref_water_kabsch = (ref_rel - ref_cent).clone()

        self.internal_dim = internal_dim
        print("internal_dim in transform:", self.internal_dim)

    # ---------- PBC helpers ----------
    def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
        return torch.as_tensor(L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L_t = self._L_tensor(x, self.L)
        return x - L_t * torch.floor(x / L_t)

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0, self.L)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O, self.L)
        return O + d

    # ---------- SO(3) maps ----------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True)
        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)

        small = theta[:, 0] < 1e-8
        big = ~small

        R = I.clone()

        if big.any():
            th = theta[big]
            k = w[big] / th
            kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

            K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
            K[:, 0, 1] = -kz
            K[:, 0, 2] =  ky
            K[:, 1, 0] =  kz
            K[:, 1, 2] = -kx
            K[:, 2, 0] = -ky
            K[:, 2, 1] =  kx

            ct = torch.cos(th).view(-1, 1, 1)
            st = torch.sin(th).view(-1, 1, 1)
            Ibig = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(big.sum(), 3, 3)

            R[big] = Ibig + st * K + (1.0 - ct) * (K @ K)

        if small.any():
            K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
            ws = w[small]
            K[:, 0, 1] = -ws[:, 2]
            K[:, 0, 2] =  ws[:, 1]
            K[:, 1, 0] =  ws[:, 2]
            K[:, 1, 2] = -ws[:, 0]
            K[:, 2, 0] = -ws[:, 1]
            K[:, 2, 1] =  ws[:, 0]
            R[small] = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0) + K

        return R

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        near_pi = torch.abs(theta - math.pi) < 1e-4
        mid = ~(small | near_pi)

        if small.any():
            A = 0.5 * (R[small] - R[small].transpose(1, 2))
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        if mid.any():
            th = theta[mid]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            axis = torch.stack([
                (R[mid, 2, 1] - R[mid, 1, 2]) / denom,
                (R[mid, 0, 2] - R[mid, 2, 0]) / denom,
                (R[mid, 1, 0] - R[mid, 0, 1]) / denom,
            ], dim=1)
            w[mid] = axis * th.unsqueeze(1)

        if near_pi.any():
            Rp = R[near_pi]
            th = theta[near_pi]

            diag = torch.stack([
                Rp[:, 0, 0],
                Rp[:, 1, 1],
                Rp[:, 2, 2],
            ], dim=1)

            axis = torch.sqrt(((diag + 1.0) / 2.0).clamp_min(0.0))

            axis0 = axis[:, 0].clone()
            axis1 = axis[:, 1].clone()
            axis2 = axis[:, 2].clone()

            axis1 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis1,
            )
            axis2 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis1.clamp_min(1e-12)),
                axis0,
            )
            axis2 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis1.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis2.clamp_min(1e-12)),
                axis0,
            )
            axis1 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis2.clamp_min(1e-12)),
                axis1,
            )

            axis = torch.stack([axis0, axis1, axis2], dim=1)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
            w[near_pi] = axis * th.unsqueeze(1)

        return w

    def so3_logdet_exp(self, w: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(w, dim=1).clamp_min(1e-12)
        half = 0.5 * theta
        return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

    # ---------- Kabsch ----------
    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3), Y: (B,3,3), centered
        Returns R such that X @ R ≈ Y
        """
        C = X.transpose(1, 2) @ Y
        U, S, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return V @ D @ Ut

    def angle_abc(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ba = a - b
        bc = c - b
        ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cosang = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cosang)

    # ---------- Sorting helper ----------
    def _extract_and_sort_waters(self, x_rel: torch.Tensor):
        """
        x_rel: (B, N, 3) relative to solute atom 0
        Returns:
          O_sorted:     (B, W, 3)
          water_sorted: (B, W, 3, 3) where each water is [O,H,H]
        """
        water_list = []
        O_list = []
        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s+3, :]            # (B,3,3)
            w = self.make_whole_water(w)
            water_list.append(w)
            O_list.append(w[:, 0, :])

        water_all = torch.stack(water_list, dim=1)   # (B, W, 3, 3)
        O_all = torch.stack(O_list, dim=1)           # (B, W, 3)

        r = torch.linalg.norm(O_all, dim=-1)         # (B, W)
        perm = torch.argsort(r, dim=1)               # (B, W)

        O_sorted = torch.gather(
            O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3)
        )
        water_sorted = torch.gather(
            water_all,
            1,
            perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        )
        return O_sorted, water_sorted

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
        sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
        origin = sol[:, 0:1, :]                     # (B,1,3)
        x_rel = self.mic(x - origin, self.L)        # (B,N,3)

        # Solute internal shape
        v1 = x_rel[:, 1, :]
        v2 = x_rel[:, 2, :]

        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
        theta = self.angle_abc(x_rel[:, 1, :], x_rel[:, 0, :], x_rel[:, 2, :]).unsqueeze(1)

        pieces = [r1, r2, theta]
        logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

        # Sort waters by O distance to solute
        O_sorted, water_sorted = self._extract_and_sort_waters(x_rel)

        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)
        for k in range(self.n_waters):
            w = water_sorted[:, k, :, :]      # (B,3,3)
            O = O_sorted[:, k, :]             # (B,3)

            w_rel = w - w[:, 0:1, :]
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
            R = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(R)

            pieces += [O, omega]

        i = torch.cat(pieces, dim=1)
        return i, logdet

    def forward(self, i: torch.Tensor):
        """
        i: (B, internal_dim)
        Returns:
          x: (B, 3*N) flattened Cartesian in [0,L)
          logdet_ix: (B,)
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # Solute shape
        r1 = i[:, 0:1].clamp_min(1e-6)
        r2 = i[:, 1:2].clamp_min(1e-6)
        theta = i[:, 2:3].clamp(1e-3, math.pi - 1e-3)

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # Canonical solute placement
        x[:, 0, :] = center

        x[:, 1, 0] = center + r1[:, 0]
        x[:, 1, 1] = center
        x[:, 1, 2] = center

        x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
        x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
        x[:, 2, 2] = center

        idx = 3
        logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        # Reconstruct waters in sorted order
        for k in range(self.n_waters):
            O = i[:, idx:idx+3]
            omega = i[:, idx+3:idx+6]
            idx += 6

            R = self.rotvec_to_rotmat(omega)

            O_abs = center + O
            atom0 = self.n_solute + 3 * k

            x[:, atom0 + 0, :] = O_abs

            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            x[:, atom0 + 1, :] = O_abs + H1
            x[:, atom0 + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet



class PBCFixedSoluteTransformSorted(nf.flows.Flow):
    """
    PBC + fully fixed solute + rigid-water transform, with proper inverse-frame alignment.

    Solute:
      - translation fixed at box center in forward()
      - rotation fixed to canonical frame from transform_data
      - internal geometry fixed

    Internal coords i:
      - waters only, sorted by O distance to canonical solute atom 0
          each water:
              O position in canonical solute frame   -> 3 dims
              rotation vector omega                  -> 3 dims

    forward(i):  i -> Cartesian x (flattened)
    inverse(x):  Cartesian x -> i

    Important:
      - inverse() first aligns the whole configuration to the CURRENT solute frame,
        then expresses waters in the canonical solute frame.
      - forward() reconstructs the fixed canonical solute at box center and places
        waters in that same canonical frame.
      - Because waters are sorted canonically, forward(inverse(x)) returns the
        canonically ordered solvent, not the original water indexing.
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

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()[0]

            # -------------------------
            # Fixed canonical solute
            # -------------------------
            sol = x0[:self.n_solute].clone()  # (3,3)

            v1 = sol[1] - sol[0]
            v2 = sol[2] - sol[0]

            e1 = v1 / torch.norm(v1).clamp_min(1e-12)
            tmp = v2 - torch.dot(v2, e1) * e1
            e2 = tmp / torch.norm(tmp).clamp_min(1e-12)
            e3 = torch.cross(e1, e2, dim=0)

            R = torch.stack([e1, e2, e3], dim=1)  # columns are basis vectors

            sol_centered = sol - sol[0]
            sol_fixed = (R.T @ sol_centered.T).T   # == sol_centered @ R for row vectors
            self.solute_fixed = sol_fixed.clone()  # canonical solute geometry

            # -------------------------
            # Reference rigid water
            # -------------------------
            w0_start = self.n_solute
            ref = x0[w0_start:w0_start + 3, :]   # [O,H,H]
            ref_O = ref[0:1, :]
            ref_rel = ref - ref_O

            self.ref_H1 = ref_rel[1].clone()
            self.ref_H2 = ref_rel[2].clone()

            ref_cent = ref_rel.mean(dim=0, keepdim=True)
            self.ref_water_kabsch = (ref_rel - ref_cent).clone()

        self.internal_dim = 6 * self.n_waters if internal_dim is None else internal_dim
        assert self.internal_dim == 6 * self.n_waters, (
            self.internal_dim, 6 * self.n_waters
        )

    # ---------- PBC helpers ----------
    def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
        return torch.as_tensor(L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L_t = self._L_tensor(x, self.L)
        return x - L_t * torch.floor(x / L_t)

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0, self.L)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O, self.L)
        return O + d

    # ---------- Solute frame ----------
    def solute_frame(self, sol: torch.Tensor) -> torch.Tensor:
        """
        sol: (B,3,3), whole current solute coordinates

        Returns:
          Rcur: (B,3,3) with columns [e1, e2, e3], matching the convention
                used to build self.solute_fixed in __init__.

        For row-vector coordinates x, canonical coordinates are x_canon = x @ Rcur.
        """
        v1 = sol[:, 1, :] - sol[:, 0, :]
        v2 = sol[:, 2, :] - sol[:, 0, :]

        e1 = v1 / torch.linalg.norm(v1, dim=1, keepdim=True).clamp_min(1e-12)
        tmp = v2 - (v2 * e1).sum(dim=1, keepdim=True) * e1
        e2 = tmp / torch.linalg.norm(tmp, dim=1, keepdim=True).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=1)

        Rcur = torch.stack([e1, e2, e3], dim=2)  # columns are basis vectors
        return Rcur

    # ---------- SO(3) maps ----------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True)
        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)

        small = theta[:, 0] < 1e-8
        big = ~small

        R = I.clone()

        if big.any():
            th = theta[big]
            k = w[big] / th
            kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

            K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
            K[:, 0, 1] = -kz
            K[:, 0, 2] =  ky
            K[:, 1, 0] =  kz
            K[:, 1, 2] = -kx
            K[:, 2, 0] = -ky
            K[:, 2, 1] =  kx

            ct = torch.cos(th).view(-1, 1, 1)
            st = torch.sin(th).view(-1, 1, 1)
            Ibig = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(big.sum(), 3, 3)

            R[big] = Ibig + st * K + (1.0 - ct) * (K @ K)

        if small.any():
            K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
            ws = w[small]
            K[:, 0, 1] = -ws[:, 2]
            K[:, 0, 2] =  ws[:, 1]
            K[:, 1, 0] =  ws[:, 2]
            K[:, 1, 2] = -ws[:, 0]
            K[:, 2, 0] = -ws[:, 1]
            K[:, 2, 1] =  ws[:, 0]
            R[small] = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0) + K

        return R

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        near_pi = torch.abs(theta - math.pi) < 1e-4
        mid = ~(small | near_pi)

        if small.any():
            A = 0.5 * (R[small] - R[small].transpose(1, 2))
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        if mid.any():
            th = theta[mid]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            axis = torch.stack([
                (R[mid, 2, 1] - R[mid, 1, 2]) / denom,
                (R[mid, 0, 2] - R[mid, 2, 0]) / denom,
                (R[mid, 1, 0] - R[mid, 0, 1]) / denom,
            ], dim=1)
            w[mid] = axis * th.unsqueeze(1)

        if near_pi.any():
            Rp = R[near_pi]
            th = theta[near_pi]

            diag = torch.stack([Rp[:, 0, 0], Rp[:, 1, 1], Rp[:, 2, 2]], dim=1)
            axis = torch.sqrt(((diag + 1.0) / 2.0).clamp_min(0.0))

            axis0 = axis[:, 0].clone()
            axis1 = axis[:, 1].clone()
            axis2 = axis[:, 2].clone()

            axis1 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis1,
            )
            axis2 = torch.where(
                axis0 > 1e-6,
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis0.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 0, 1] + Rp[:, 1, 0]) / (4.0 * axis1.clamp_min(1e-12)),
                axis0,
            )
            axis2 = torch.where(
                (axis0 <= 1e-6) & (axis1 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis1.clamp_min(1e-12)),
                axis2,
            )

            axis0 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 0, 2] + Rp[:, 2, 0]) / (4.0 * axis2.clamp_min(1e-12)),
                axis0,
            )
            axis1 = torch.where(
                (axis0 <= 1e-6) & (axis1 <= 1e-6) & (axis2 > 1e-6),
                (Rp[:, 1, 2] + Rp[:, 2, 1]) / (4.0 * axis2.clamp_min(1e-12)),
                axis1,
            )

            axis = torch.stack([axis0, axis1, axis2], dim=1)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
            w[near_pi] = axis * th.unsqueeze(1)

        return w

    # ---------- Kabsch ----------
    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3), Y: (B,3,3), centered
        Returns R such that X @ R ≈ Y
        """
        C = X.transpose(1, 2) @ Y
        U, S, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return V @ D @ Ut

    # ---------- Sorting helper ----------
    def _extract_and_sort_waters(self, x_rel: torch.Tensor):
        """
        x_rel: (B, N, 3), already in canonical solute frame and relative to solute atom 0

        Returns
        -------
        O_sorted:     (B, W, 3)
        water_sorted: (B, W, 3, 3) where each water is [O,H,H]
        """
        water_list = []
        O_list = []

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s + 3, :]
            w = self.make_whole_water(w)
            water_list.append(w)
            O_list.append(w[:, 0, :])

        water_all = torch.stack(water_list, dim=1)   # (B,W,3,3)
        O_all = torch.stack(O_list, dim=1)           # (B,W,3)

        r = torch.linalg.norm(O_all, dim=-1)         # (B,W)
        perm = torch.argsort(r, dim=1, stable=True)

        O_sorted = torch.gather(
            O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3)
        )
        water_sorted = torch.gather(
            water_all,
            1,
            perm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        )
        return O_sorted, water_sorted

    # ---------- nf API ----------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N) flattened Cartesian
        Returns:
          i: (B, 6*n_waters)
          logdet_xi: (B,)
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # 1) make current solute whole
        sol = self.make_whole_solute(x[:, :self.n_solute, :])   # (B,3,3)

        # 2) current solute origin
        origin = sol[:, 0:1, :]                                 # (B,1,3)

        # 3) all atoms relative to current solute atom 0
        x_rel = self.mic(x - origin, self.L)                    # (B,N,3)

        # 4) rotate into canonical solute frame
        Rcur = self.solute_frame(sol)                           # (B,3,3)
        x_rel = x_rel @ Rcur                                    # row-vector convention

        pieces = []
        logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

        O_sorted, water_sorted = self._extract_and_sort_waters(x_rel)

        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            w = water_sorted[:, k, :, :]   # (B,3,3)
            O = O_sorted[:, k, :]          # (B,3)

            w_rel = w - w[:, 0:1, :]
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)

            # R maps reference water -> current water in canonical frame
            R = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(R)

            pieces += [O, omega]

        i = torch.cat(pieces, dim=1)
        return i, logdet

    def forward(self, i: torch.Tensor):
        """
        i: (B, 6*n_waters)
        Returns:
          x: (B, 3*N) flattened Cartesian in [0,L)
          logdet_ix: (B,)
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # fixed canonical solute at box center
        sol = self.solute_fixed.to(i.device, i.dtype)
        x[:, :self.n_solute, :] = center + sol

        idx = 0
        logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        for k in range(self.n_waters):
            O = i[:, idx:idx + 3]
            omega = i[:, idx + 3:idx + 6]
            idx += 6

            R = self.rotvec_to_rotmat(omega)

            O_abs = center + O
            atom0 = self.n_solute + 3 * k

            x[:, atom0 + 0, :] = O_abs

            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            x[:, atom0 + 1, :] = O_abs + H1
            x[:, atom0 + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet
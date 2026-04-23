import torch
import math
import normflows as nf
import torch.nn.functional as F


class PBCFixedSoluteRadialWaterTransform(nf.flows.Flow):
    """
    Fixed-solute transform with spherical water oxygen coordinates.

    Internal coordinates per water:
        [u_r, u_theta, u_phi, omega_x, omega_y, omega_z]

    where
        r     = r_min + softplus(u_r)
        theta = pi * sigmoid(u_theta)
        phi   = 2*pi*sigmoid(u_phi) - pi

    Solute is fully fixed:
        - translation fixed
        - rotation fixed
        - bond lengths fixed
        - bond angle fixed

    Water H positions are reconstructed from:
        O position + rotation vector omega

    Optional:
        sort waters by radius in inverse() to reduce permutation noise.
    """

    def __init__(
        self,
        L: float,
        system=None,
        transform_data=None,
        r_min: float = 0.28,
        sort_waters_by_radius: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.r_min = float(r_min)
        self.sort_waters_by_radius = bool(sort_waters_by_radius)
        self.eps = float(eps)

        assert transform_data is not None and transform_data.shape[0] == 1

        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3)[0]  # (N,3)

            # -------------------------
            # Fixed solute geometry
            # -------------------------
            sol = x0[:3].clone()  # [S,O,O]

            v1 = sol[1] - sol[0]
            v2 = sol[2] - sol[0]

            e1 = v1 / torch.norm(v1)
            tmp = v2 - torch.dot(v2, e1) * e1
            e2 = tmp / torch.norm(tmp)
            e3 = torch.cross(e1, e2, dim=0)

            R = torch.stack([e1, e2, e3], dim=1)  # (3,3)
            sol_centered = sol - sol[0]
            sol_fixed = (R.T @ sol_centered.T).T  # canonical frame

            self.register_buffer("solute_fixed", sol_fixed)

            # -------------------------
            # Reference rigid water
            # -------------------------
            w0 = x0[self.n_solute:self.n_solute + 3].clone()  # [O,H,H]
            w0_rel = w0 - w0[0]

            self.register_buffer("ref_H1", w0_rel[1].clone())
            self.register_buffer("ref_H2", w0_rel[2].clone())

            ref_cent = w0_rel.mean(dim=0, keepdim=True)
            self.register_buffer("ref_water_kabsch", (w0_rel - ref_cent).clone())

        # 6 dims per water
        self.internal_dim = 6 * self.n_waters

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _L_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(dx)
        return dx - L * torch.round(dx / L)

    def wrap(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(x)
        return x - L * torch.floor(x / L)

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O)
        return O + d

    @staticmethod
    def inverse_softplus(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # inverse of softplus on y > 0
        y = torch.clamp(y, min=eps)
        return torch.log(torch.expm1(y).clamp_min(eps))

    @staticmethod
    def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def spherical_to_cartesian(
        self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        r, theta, phi: (B,)
        returns xyz: (B,3)
        """
        st = torch.sin(theta)
        x = r * st * torch.cos(phi)
        y = r * st * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def cartesian_to_spherical(self, xyz: torch.Tensor):
        """
        xyz: (B,3)
        returns r, theta, phi each (B,)
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.linalg.norm(xyz, dim=-1).clamp_min(self.eps)
        theta = torch.acos((z / r).clamp(-1.0 + self.eps, 1.0 - self.eps))
        phi = torch.atan2(y, x)
        return r, theta, phi

    # ------------------------------------------------------------------
    # SO(3)
    # ------------------------------------------------------------------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
        k = w / theta

        K = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
        ct = torch.cos(theta).view(B, 1, 1)
        st = torch.sin(theta).view(B, 1, 1)

        return I + st * K + (1.0 - ct) * (K @ K)

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))

        w = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        if small.any():
            A = (R[small] - R[small].transpose(1, 2)) * 0.5
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        big = ~small
        if big.any():
            th = theta[big]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            w[big, 0] = (R[big, 2, 1] - R[big, 1, 2]) / denom
            w[big, 1] = (R[big, 0, 2] - R[big, 2, 0]) / denom
            w[big, 2] = (R[big, 1, 0] - R[big, 0, 1]) / denom
            w[big] *= th.unsqueeze(1)

        return w

    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3) current points (centered)
        Y: (B,3,3) reference points (centered)
        Returns R: (B,3,3) such that X @ R ≈ Y
        """
        C = X.transpose(1, 2) @ Y
        U, _, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

        R = V @ D @ Ut
        return R

    # ------------------------------------------------------------------
    # inverse: x -> i
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N)
        returns:
            i: (B, 6*n_waters)
            logdet: zeros
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # solute origin is atom 0; solute orientation/shape ignored because fixed
        sol = x[:, :3, :]
        origin = sol[:, 0:1, :]
        x_rel = self.mic(x - origin)

        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        O_list = []
        omg_list = []
        r_list = []

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s + 3, :]         # (B,3,3)
            w = self.make_whole_water(w)

            O = w[:, 0, :]                   # (B,3)

            # orientation
            w_rel = w - w[:, 0:1, :]
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
            R = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(R)

            O_list.append(O)
            omg_list.append(omega)
            r_list.append(torch.linalg.norm(O, dim=-1))

        O_all = torch.stack(O_list, dim=1)        # (B,W,3)
        omg_all = torch.stack(omg_list, dim=1)    # (B,W,3)
        r_all = torch.stack(r_list, dim=1)        # (B,W)

        if self.sort_waters_by_radius:
            perm = torch.argsort(r_all, dim=1)    # (B,W)
            O_all = torch.gather(O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
            omg_all = torch.gather(omg_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
            r_all = torch.gather(r_all, 1, perm)

        # convert O -> unconstrained spherical
        r, theta, phi = self.cartesian_to_spherical(O_all.reshape(B * self.n_waters, 3))

        u_r = self.inverse_softplus(r - self.r_min, eps=self.eps)
        u_theta = self.logit(theta / math.pi, eps=self.eps)
        u_phi = self.logit((phi + math.pi) / (2.0 * math.pi), eps=self.eps)

        sph = torch.stack([u_r, u_theta, u_phi], dim=1).view(B, self.n_waters, 3)

        i = torch.cat([sph, omg_all], dim=-1).reshape(B, 6 * self.n_waters)
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return i, logdet

    # ------------------------------------------------------------------
    # forward: i -> x
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        i: (B, 6*n_waters)
        returns:
            x: (B, 3*N)
            logdet: zeros
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        center = torch.as_tensor(0.5 * self.L, device=i.device, dtype=i.dtype)

        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # fixed solute
        sol = self.solute_fixed.to(i.device, i.dtype)
        x[:, :3, :] = center + sol

        w = i.view(B, self.n_waters, 6)
        sph = w[:, :, :3]
        omg = w[:, :, 3:]

        u_r = sph[:, :, 0].reshape(-1)
        u_theta = sph[:, :, 1].reshape(-1)
        u_phi = sph[:, :, 2].reshape(-1)

        r = self.r_min + F.softplus(u_r)
        theta = math.pi * torch.sigmoid(u_theta)
        phi = 2.0 * math.pi * torch.sigmoid(u_phi) - math.pi

        O_rel = self.spherical_to_cartesian(r, theta, phi).view(B, self.n_waters, 3)
        O_abs = center + O_rel

        R = self.rotvec_to_rotmat(omg.reshape(B * self.n_waters, 3))

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1).expand(B * self.n_waters, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1).expand(B * self.n_waters, 3, 1)

        H1 = (R @ H1_ref).squeeze(-1).view(B, self.n_waters, 3)
        H2 = (R @ H2_ref).squeeze(-1).view(B, self.n_waters, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs[:, k, :]
            x[:, s + 1, :] = O_abs[:, k, :] + H1[:, k, :]
            x[:, s + 2, :] = O_abs[:, k, :] + H2[:, k, :]

        x = self.wrap(x)
        logdet = torch.zeros(B, device=i.device, dtype=i.dtype)
        return x.view(B, -1), logdet

import math
import torch
import torch.nn.functional as F
import normflows as nf


class PBCFixedSoluteSequentialOOTransform(nf.flows.Flow):
    """
    Fixed-solute + rigid-water transform with partial water-water overlap encoding.

    Internal coords per water:
        [u_r_clear, u_theta, u_phi, omega_x, omega_y, omega_z]

    Decoding is sequential:
        r_k = lower_bound_k(direction_k, previous_Os) + softplus(u_r_clear_k)

    where lower_bound_k enforces:
        - minimum solute-water radius r_min
        - minimum O-O distance oo_min to all previously placed waters

    This keeps the transform invertible (for states satisfying the constraint)
    and partially encodes solvent-solvent exclusion in the representation.
    """

    def __init__(
        self,
        L: float,
        system=None,
        transform_data=None,
        r_min: float = 0.28,          # minimum solute-water O radius
        oo_min: float = 0.22,         # minimum O-O distance encoded in transform
        sort_waters_by_radius: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.r_min = float(r_min)
        self.oo_min = float(oo_min)
        self.sort_waters_by_radius = bool(sort_waters_by_radius)
        self.eps = float(eps)

        assert transform_data is not None and transform_data.shape[0] == 1

        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3)[0]

            # -------------------------
            # Fixed canonical solute
            # -------------------------
            sol = x0[:3].clone()

            v1 = sol[1] - sol[0]
            v2 = sol[2] - sol[0]

            e1 = v1 / torch.norm(v1)
            tmp = v2 - torch.dot(v2, e1) * e1
            e2 = tmp / torch.norm(tmp)
            e3 = torch.cross(e1, e2, dim=0)

            R = torch.stack([e1, e2, e3], dim=1)
            sol_centered = sol - sol[0]
            sol_fixed = (R.T @ sol_centered.T).T

            self.register_buffer("solute_fixed", sol_fixed)

            # -------------------------
            # Reference rigid water
            # -------------------------
            w0 = x0[self.n_solute:self.n_solute + 3].clone()   # [O,H,H]
            w0_rel = w0 - w0[0]

            self.register_buffer("ref_H1", w0_rel[1].clone())
            self.register_buffer("ref_H2", w0_rel[2].clone())

            ref_cent = w0_rel.mean(dim=0, keepdim=True)
            self.register_buffer("ref_water_kabsch", (w0_rel - ref_cent).clone())

        self.internal_dim = 6 * self.n_waters

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def _L_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(dx)
        return dx - L * torch.round(dx / L)

    def wrap(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(x)
        return x - L * torch.floor(x / L)

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O)
        return O + d

    @staticmethod
    def inverse_softplus(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return torch.log(torch.expm1(y).clamp_min(eps))

    @staticmethod
    def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def spherical_to_cartesian(
        self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        st = torch.sin(theta)
        x = r * st * torch.cos(phi)
        y = r * st * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def cartesian_to_spherical(self, xyz: torch.Tensor):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.linalg.norm(xyz, dim=-1).clamp_min(self.eps)
        theta = torch.acos((z / r).clamp(-1.0 + self.eps, 1.0 - self.eps))
        phi = torch.atan2(y, x)
        return r, theta, phi

    # ------------------------------------------------------------------
    # SO(3)
    # ------------------------------------------------------------------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
        k = w / theta

        K = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
        ct = torch.cos(theta).view(B, 1, 1)
        st = torch.sin(theta).view(B, 1, 1)

        return I + st * K + (1.0 - ct) * (K @ K)

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))

        w = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        if small.any():
            A = (R[small] - R[small].transpose(1, 2)) * 0.5
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        big = ~small
        if big.any():
            th = theta[big]
            denom = (2.0 * torch.sin(th)).clamp_min(1e-12)
            w[big, 0] = (R[big, 2, 1] - R[big, 1, 2]) / denom
            w[big, 1] = (R[big, 0, 2] - R[big, 2, 0]) / denom
            w[big, 2] = (R[big, 1, 0] - R[big, 0, 1]) / denom
            w[big] *= th.unsqueeze(1)

        return w

    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        C = X.transpose(1, 2) @ Y
        U, _, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

        return V @ D @ Ut

    # ------------------------------------------------------------------
    # radial lower bound from previous waters
    # ------------------------------------------------------------------
    def ray_lower_bound(self, u: torch.Tensor, O_prev: torch.Tensor) -> torch.Tensor:
        """
        u:      (B,3) unit direction for current water
        O_prev: (B,K,3) already-placed oxygen positions relative to solute center
        returns:
            lb: (B,) lower bound on radius along ray u to avoid all previous waters
        """
        B = u.shape[0]
        lb = torch.full((B,), self.r_min, device=u.device, dtype=u.dtype)

        if O_prev.shape[1] == 0:
            return lb

        # For each previous oxygen p, solve |r u - p| >= d
        # Quadratic in r:
        #   r^2 - 2 a r + (|p|^2 - d^2) >= 0,   a = u·p
        # Forbidden interval if discriminant > 0:
        #   r in [a - sqrt(discr), a + sqrt(discr)]
        # So safe lower bound is upper root.
        d = torch.as_tensor(self.oo_min, device=u.device, dtype=u.dtype)

        p = O_prev                                  # (B,K,3)
        a = (u[:, None, :] * p).sum(dim=-1)         # (B,K)
        p2 = (p * p).sum(dim=-1)                    # (B,K)

        discr = a * a - (p2 - d * d)                # (B,K)
        valid = discr > 0.0

        upper = a + torch.sqrt(torch.clamp(discr, min=0.0))   # (B,K)
        upper = torch.where(valid, upper, torch.zeros_like(upper))
        upper = torch.clamp(upper, min=0.0)

        lb = torch.maximum(lb, upper.max(dim=1).values)
        return lb

    # ------------------------------------------------------------------
    # inverse: x -> i
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N)
        returns:
            i: (B, 6*n_waters)
            logdet: zeros
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # use solute atom 0 as origin; solute itself is ignored because fixed
        sol = x[:, :3, :]
        origin = sol[:, 0:1, :]
        x_rel = self.mic(x - origin)

        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        O_list = []
        omg_list = []
        r_list = []

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s + 3, :]
            w = self.make_whole_water(w)

            O = w[:, 0, :]

            w_rel = w - w[:, 0:1, :]
            w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
            R = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(R)

            O_list.append(O)
            omg_list.append(omega)
            r_list.append(torch.linalg.norm(O, dim=-1))

        O_all = torch.stack(O_list, dim=1)       # (B,W,3)
        omg_all = torch.stack(omg_list, dim=1)   # (B,W,3)
        r_all = torch.stack(r_list, dim=1)       # (B,W)

        # fixed ordering for invertibility
        if self.sort_waters_by_radius:
            perm = torch.argsort(r_all, dim=1)
            O_all = torch.gather(O_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
            omg_all = torch.gather(omg_all, 1, perm.unsqueeze(-1).expand(-1, -1, 3))

        # Sequentially convert O positions into constrained radial coords
        sph_parts = []
        O_prev = torch.zeros((B, 0, 3), device=x.device, dtype=x.dtype)

        for k in range(self.n_waters):
            O_k = O_all[:, k, :]                              # (B,3)
            r, theta, phi = self.cartesian_to_spherical(O_k)

            u = O_k / r.unsqueeze(1).clamp_min(self.eps)
            lb = self.ray_lower_bound(u, O_prev)              # (B,)
            extra = r - lb

            # Must be positive if data respects encoded oo_min
            if torch.any(extra <= 0):
                # Clamp to preserve numerical stability; exact invertibility then fails
                # only for inputs violating the encoded constraint.
                extra = torch.clamp(extra, min=self.eps)

            u_r = self.inverse_softplus(extra, eps=self.eps)
            u_theta = self.logit(theta / math.pi, eps=self.eps)
            u_phi = self.logit((phi + math.pi) / (2.0 * math.pi), eps=self.eps)

            sph_parts.append(torch.stack([u_r, u_theta, u_phi], dim=1))
            O_prev = torch.cat([O_prev, O_k[:, None, :]], dim=1)

        sph = torch.stack(sph_parts, dim=1)                  # (B,W,3)

        i = torch.cat([sph, omg_all], dim=-1).reshape(B, 6 * self.n_waters)
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return i, logdet

    # ------------------------------------------------------------------
    # forward: i -> x
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        i: (B, 6*n_waters)
        returns:
            x: (B, 3*N)
            logdet: zeros
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        center = torch.as_tensor(0.5 * self.L, device=i.device, dtype=i.dtype)

        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # fixed solute
        sol = self.solute_fixed.to(i.device, i.dtype)
        x[:, :3, :] = center + sol

        w = i.view(B, self.n_waters, 6)
        sph = w[:, :, :3]
        omg = w[:, :, 3:]

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        O_prev = torch.zeros((B, 0, 3), device=i.device, dtype=i.dtype)

        O_abs_all = []
        H1_all = []
        H2_all = []

        for k in range(self.n_waters):
            u_r = sph[:, k, 0]
            u_theta = sph[:, k, 1]
            u_phi = sph[:, k, 2]

            theta = math.pi * torch.sigmoid(u_theta)
            phi = 2.0 * math.pi * torch.sigmoid(u_phi) - math.pi

            # direction first
            r_dummy = torch.ones_like(theta)
            u = self.spherical_to_cartesian(r_dummy, theta, phi)   # unit direction

            lb = self.ray_lower_bound(u, O_prev)
            r = lb + F.softplus(u_r)

            O_rel = self.spherical_to_cartesian(r, theta, phi)     # (B,3)
            O_abs = center + O_rel
            O_abs_all.append(O_abs)

            # reconstruct rigid water
            R = self.rotvec_to_rotmat(omg[:, k, :])                # (B,3,3)
            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            H1_all.append(O_abs + H1)
            H2_all.append(O_abs + H2)

            O_prev = torch.cat([O_prev, O_rel[:, None, :]], dim=1)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs_all[k]
            x[:, s + 1, :] = H1_all[k]
            x[:, s + 2, :] = H2_all[k]

        x = self.wrap(x)
        logdet = torch.zeros(B, device=i.device, dtype=i.dtype)
        return x.view(B, -1), logdet
import torch
import math
from torch import nn

try:
    import normflows as nf
    _BaseFlow = nf.flows.Flow
except Exception:
    _BaseFlow = nn.Module

from fab.utils.manifold_utils import wrap_to_pi, ManifoldState

class SFICTransform(nf.flows.Flow):
    """
    Solute-frame internal coordinate transform for a triatomic solute + rigid water 
    solvents in a fixed cubic box with PBC.

    REQUIREMENT: The simulation MUST use rigid water (rigid_water=True and
    internal_constraints='hbonds' in the OpenMM system). H positions are
    reconstructed in forward() from a fixed reference O-H geometry rotated by
    the water's orientation. If the simulation uses flexible water, the H
    fluctuations are silently discarded by inverse(), causing systematic energy
    errors on generated samples.

    Internal coords i (all unconstrained real-valued for the flow):
      - solute shape (3 dims):
          z_r1    = log(|S-O1|)                  unconstrained, maps R -> R+
          z_r2    = log(|S-O2|)                  unconstrained, maps R -> R+
          z_theta = logit(angle(O1,S,O2) / pi)   unconstrained, maps R -> (0,pi)
      - per water (6 dims):
          z_O   = arctanh(2 * O_canon / L)       unconstrained, maps R^3 -> (-L/2,L/2)^3
          omega = rotation vector (axis * angle)  unconstrained R^3

    All internal coords are unconstrained real numbers suitable for a standard
    Gaussian-base spline flow.

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

        # Build reference rigid-water geometry expressed in the canonical solute frame
        # of the reference configuration (transform_data).
        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()  # (1,N,3)

            # Canonical rotation for the reference configuration
            sol_ref = x0[0, :self.n_solute, :]          # (3,3) solute atoms
            origin_ref = sol_ref[0:1, :]                 # (1,3) = atom0
            v1_ref = sol_ref[1] - sol_ref[0]             # atom1 - atom0
            v2_ref = sol_ref[2] - sol_ref[0]             # atom2 - atom0
            e1_ref = v1_ref / v1_ref.norm().clamp_min(1e-12)
            v2_perp_ref = v2_ref - (v2_ref * e1_ref).sum() * e1_ref
            e2_ref = v2_perp_ref / v2_perp_ref.norm().clamp_min(1e-12)
            e3_ref = torch.cross(e1_ref, e2_ref, dim=0)
            # R_sol_ref: rows are canonical basis vectors; R_sol_ref @ v_lab = v_canonical
            R_sol_ref = torch.stack([e1_ref, e2_ref, e3_ref], dim=0)  # (3,3)

            # Reference water in canonical solute frame
            w0_start = self.n_solute
            ref_water_abs = x0[0, w0_start:w0_start+3, :]           # (3,3) [O,H1,H2] absolute
            ref_water_rel = ref_water_abs - origin_ref               # (3,3) relative to atom0
            ref_water_canon = (R_sol_ref @ ref_water_rel.T).T        # (3,3) in canonical frame

            ref_O_canon = ref_water_canon[0:1, :]                    # (1,3)
            ref_rel_canon = ref_water_canon - ref_O_canon            # O at origin, canonical frame

            # Reference H vectors in canonical solute frame (3,)
            self.ref_H1 = ref_rel_canon[1].clone()
            self.ref_H2 = ref_rel_canon[2].clone()

            # Centered reference water for Kabsch in canonical solute frame (3,3)
            ref_cent = ref_rel_canon.mean(dim=0, keepdim=True)       # (1,3)
            self.ref_water_kabsch = (ref_rel_canon - ref_cent).clone()

        # Internal dimension: solute(3: r1, r2, theta) + waters(6 each: O_canon + omega)
        self.internal_dim = internal_dim
        print("internal_dim  in transform:", self.internal_dim)

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
    
    def angle_abc(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Angle ABC in radians.
        a,b,c: (B,3)
        returns: (B,)
        """
        ba = a - b
        bc = c - b
        ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cosang = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cosang)

    def compute_R_sol(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Per-sample rotation matrix mapping lab frame to canonical solute frame.

        Canonical frame: v1 along +x, component of v2 perpendicular to v1 along +y.

        v1, v2: (B, 3)
        Returns R_sol: (B, 3, 3) satisfying R_sol[b] @ v_lab_col = v_canonical_col
        """
        eps = 1e-12
        e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(eps)            # (B,3)
        v2_perp = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
        e2 = v2_perp / v2_perp.norm(dim=-1, keepdim=True).clamp_min(eps)  # (B,3)
        e3 = torch.cross(e1, e2, dim=-1)                                   # (B,3)
        # Rows of R_sol are the canonical basis vectors expressed in lab frame
        return torch.stack([e1, e2, e3], dim=1)                            # (B,3,3)

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

        # # Solute internal: two MIC bond vectors from atom0
        # v1 = x_rel[:, 1, :]  # already mic-wrapped
        # v2 = x_rel[:, 2, :]

        # pieces = [v1, v2]

        # Solute internal shape: r1, r2, theta
        v1 = x_rel[:, 1, :]                      # S -> O1
        v2 = x_rel[:, 2, :]                      # S -> O2

        r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)   # (B,1)
        r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)   # (B,1)
        theta = self.angle_abc(x_rel[:, 1, :], x_rel[:, 0, :], x_rel[:, 2, :]).unsqueeze(1)  # (B,1)

        eps = 1e-7

        # Reparameterise solute DOF to unconstrained reals
        # r1, r2 > 0  ->  z_r = log(r)   (maps R -> R+, avoids positivity clamp)
        # theta in (0,pi) -> z_theta = logit(theta/pi)  (maps R -> (0,pi))
        r1_c = r1[:, 0].clamp_min(eps)
        r2_c = r2[:, 0].clamp_min(eps)
        theta_c = theta[:, 0].clamp(eps, math.pi - eps)

        z_r1 = torch.log(r1_c).unsqueeze(1)                              # (B,1)
        z_r2 = torch.log(r2_c).unsqueeze(1)                              # (B,1)
        z_theta = torch.log(theta_c / (math.pi - theta_c)).unsqueeze(1)  # (B,1)

        pieces = [z_r1, z_r2, z_theta]

        # Total solute logdet = spherical Jacobian (x -> (r1,r2,theta))
        #                     + reparameterisation Jacobian ((r1,r2,theta) -> (z_r1,z_r2,z_theta))
        # Spherical:  -(2*log(r1) + 2*log(r2) + log(sin(theta)))
        # Reparam:    -log(r1) - log(r2) + log(pi) - log(theta) - log(pi - theta)
        # Combined:   -(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
        logdet = -(
            3.0 * torch.log(r1_c) +
            3.0 * torch.log(r2_c) +
            torch.log(theta_c.sin().clamp_min(eps)) +
            torch.log(theta_c) +
            torch.log((math.pi - theta_c).clamp_min(eps)) -
            math.log(math.pi)
        )

        # Rotation matrix mapping lab frame -> canonical solute frame
        R_sol = self.compute_R_sol(v1, v2)  # (B,3,3)

        # Waters: z_O (arctanh-reparameterised O in canonical frame) + rotation vector omega
        Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)  # (B,3,3)
        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s+3, :]                  # (B,3,3) [O,H,H] in lab frame rel. to atom0
            w = self.make_whole_water(w)             # ensure H's are whole w.r.t O

            # Transform water atoms to canonical solute frame
            # w_canon[b,k,:] = R_sol[b] @ w[b,k,:]  (column-vector convention)
            w_canon = torch.bmm(R_sol, w.transpose(1, 2)).transpose(1, 2)  # (B,3,3)

            O_canon = w_canon[:, 0, :]               # (B,3) O in canonical frame, in (-L/2, L/2)^3

            # Reparameterise O: O_canon in (-L/2, L/2)^3 -> z_O in R^3 via arctanh
            # z_O = arctanh(2 * O_canon / L);  handles box boundary as hard wall.
            frac = (2.0 / self.L) * O_canon
            frac = frac.clamp(-1.0 + eps, 1.0 - eps)
            z_O = torch.arctanh(frac)                # (B,3)

            # Logdet contribution: log|d(z_O)/d(O_canon)| = sum_i log(2/L) - log(1 - frac_i^2)
            logdet = logdet + (
                3.0 * (math.log(2.0) - math.log(self.L)) -
                torch.log((1.0 - frac ** 2).clamp_min(eps)).sum(dim=-1)
            )

            # Local water geometry in canonical frame for Kabsch
            w_rel_canon = w_canon - w_canon[:, 0:1, :]
            w_cent_canon = w_rel_canon - w_rel_canon.mean(dim=1, keepdim=True)

            R_water = self.kabsch_rotation(w_cent_canon, Yref)  # (B,3,3)
            omega = self.rotmat_to_rotvec(R_water)              # (B,3)

            # SO(3) exp-map measure correction: -log|det J_exp(omega)|
            logdet = logdet - self.so3_logdet_exp(omega)

            pieces += [z_O, omega]

        i = torch.cat(pieces, dim=1)  # (B, 3 + 6*n_waters)
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

        # # Unpack solute
        # v1 = i[:, 0:3]
        # v2 = i[:, 3:6]

        # # Place solute atom0 at box center
        # center = 0.5 * self.L
        # x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
        # x[:, 0, :] = center
        # x[:, 1, :] = center + v1
        # x[:, 2, :] = center + v2

        # Unpack solute shape: all inputs are unconstrained reals
        z_r1    = i[:, 0:1]   # (B,1)
        z_r2    = i[:, 1:2]   # (B,1)
        z_theta = i[:, 2:3]   # (B,1)

        r1    = torch.exp(z_r1)                         # > 0
        r2    = torch.exp(z_r2)                         # > 0
        theta = math.pi * torch.sigmoid(z_theta)        # in (0, pi)

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # S at center
        x[:, 0, :] = center

        # O1 on +x axis of canonical frame
        x[:, 1, 0] = center + r1[:, 0]
        x[:, 1, 1] = center
        x[:, 1, 2] = center

        # O2 in xy-plane of canonical frame
        x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
        x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
        x[:, 2, 2] = center

        # Combined logdet: spherical |J_{i->x}| = r1^3 * r2^3 * sin(theta)
        # plus inverse reparameterisation Jacobians for r1, r2, theta
        # d(r)/d(z_r) = r  ->  log|J| += log(r)  (×2 for r1, r2 after combined: 3 each)
        # d(theta)/d(z_theta) = pi * sigmoid * (1-sigmoid) = theta*(pi-theta)/pi
        # Total: 3*log(r1) + 3*log(r2) + log(sin(theta)) + log(theta) + log(pi-theta) - log(pi)
        eps = 1e-7
        logdet = (
            3.0 * torch.log(r1[:, 0]) +
            3.0 * torch.log(r2[:, 0]) +
            torch.log(theta[:, 0].sin().clamp_min(eps)) +
            torch.log(theta[:, 0].clamp_min(eps)) +
            torch.log((math.pi - theta[:, 0]).clamp_min(eps)) -
            math.log(math.pi)
        )

        # Waters
        idx = 3

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)  # (1,3,1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        for k in range(self.n_waters):
            z_O   = i[:, idx:idx+3]       # (B,3)  unconstrained
            omega = i[:, idx+3:idx+6]     # (B,3)
            idx += 6

            R = self.rotvec_to_rotmat(omega)  # (B,3,3)

            # Map z_O -> O_canon in (-L/2, L/2)^3 via tanh
            O_canon = (self.L / 2.0) * torch.tanh(z_O)   # (B,3)

            # Logdet: log|d(O_canon)/d(z_O)| = sum_j [log(L/2) + log(1 - tanh^2(z_O_j))]
            frac = 2.0 * O_canon / self.L   # = tanh(z_O), in (-1,1)
            logdet = logdet + (
                3.0 * math.log(self.L / 2.0) +
                torch.log((1.0 - frac ** 2).clamp_min(eps)).sum(dim=-1)
            )

            # SO(3) exp-map measure correction: +log|det J_exp(omega)|
            logdet = logdet + self.so3_logdet_exp(omega)

            # Oxygen position in canonical solute frame, then placed at center
            O_abs = center + O_canon
            x[:, self.n_solute + 3*k + 0, :] = O_abs

            # Rotate reference H vectors
            # (B,3,3) @ (B,3,1) -> (B,3,1) -> (B,3)
            H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            x[:, self.n_solute + 3*k + 1, :] = O_abs + H1
            x[:, self.n_solute + 3*k + 2, :] = O_abs + H2

        # Wrap into [0,L)
        x = self.wrap_0L(x)

        return x.view(B, -1), logdet


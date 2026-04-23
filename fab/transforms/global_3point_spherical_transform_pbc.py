import torch
import math
import normflows as nf
import torch.nn.functional as F
from torch import nn

# If you use normflows, keep this inheritance.
# Otherwise replace nf.flows.Flow with nn.Module.
try:
    import normflows as nf
    _BaseFlow = nf.flows.Flow
except Exception:
    _BaseFlow = nn.Module


from dataclasses import dataclass
from typing import Dict, Tuple



# ============================================================
# Basic manifold helpers
# ============================================================

def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def torus_add(theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """T^n group operation in angle coordinates."""
    return wrap_to_pi(theta + delta)


def torus_sub(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Difference on T^n in principal branch."""
    return wrap_to_pi(theta_a - theta_b)


def hat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (..., 3)
    returns skew matrix (..., 3, 3)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([O,   -wz,  wy], dim=-1),
        torch.stack([wz,   O,  -wx], dim=-1),
        torch.stack([-wy, wx,   O], dim=-1),
    ], dim=-2)
    return K


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map so(3) -> SO(3)
    w: (..., 3)
    R: (..., 3, 3)
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (..., 1)
    K = hat(w)

    I = torch.eye(3, device=w.device, dtype=w.dtype)
    I = I.expand(w.shape[:-1] + (3, 3))

    theta2 = theta ** 2
    A = torch.where(
        theta > eps,
        torch.sin(theta) / theta,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
    )
    B = torch.where(
        theta > eps,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
    )

    A = A.unsqueeze(-1)
    B = B.unsqueeze(-1)
    return I + A * K + B * (K @ K)


def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Principal logarithm SO(3) -> so(3) ~ R^3
    Only used locally inside the flow; the state itself stays on SO(3).
    R: (..., 3, 3)
    w: (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    A = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack([
        A[..., 2, 1],
        A[..., 0, 2],
        A[..., 1, 0],
    ], dim=-1)

    small = theta < 1e-6
    w = torch.zeros(R.shape[:-2] + (3,), device=R.device, dtype=R.dtype)

    if small.any():
        w[small] = vee[small]

    big = ~small
    if big.any():
        th = theta[big]
        scale = th / torch.sin(th).clamp_min(eps)
        w[big] = vee[big] * scale.unsqueeze(-1)

    return w


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """
    Polar projection to nearest rotation matrix.
    Useful after numerical noise.
    M: (..., 3, 3)
    """
    U, _, Vh = torch.linalg.svd(M)
    V = Vh.transpose(-1, -2)
    R = U @ V.transpose(-1, -2)

    det = torch.det(R)
    bad = det < 0
    if bad.any():
        U_fix = U.clone()
        U_fix[bad, :, -1] *= -1.0
        R = U_fix @ V.transpose(-1, -2)
    return R


# ============================================================
# Structured state container
# ============================================================

@dataclass
class ManifoldState:
    # solute bond vectors anchored at atom 0
    solute: torch.Tensor   # (B, 6), [v1(3), v2(3)]
    # torus angles for oxygen positions relative to atom 0
    tau: torch.Tensor      # (B, W, 3), each in [-pi, pi)
    # water orientations as rotation matrices
    R: torch.Tensor        # (B, W, 3, 3)


# ============================================================
# Transform: Cartesian <-> translation-reduced manifold state
# ============================================================

class PBCRigidWaterTorusSO3Transform(nn.Module):
    """
    Translation-reduced transform for:
      - one flexible triatomic solute
      - rigid OHH waters
      - cubic PBC box of edge length L

    Forward/inverse are bijective on the intended physical support:
      - solute bonded vectors remain in the principal MIC branch
      - water geometry is rigid and labeled OHH
      - water oxygen positions are torus-valued relative to atom 0

    State:
      solute : R^6
      tau    : T^(3W)
      R      : SO(3)^W
    """

    def __init__(self, L: float, transform_data: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.L = float(L)
        self.eps = float(eps)

        assert transform_data.ndim == 2 and transform_data.shape[0] == 1
        n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_water = 3
        self.n_waters = (n_atoms - self.n_solute) // self.n_atoms_per_water
        self.n_atoms = n_atoms

        # Reference water geometry from first water in transform_data
        with torch.no_grad():
            x0 = transform_data.reshape(1, n_atoms, 3).clone()
            w0 = x0[:, self.n_solute:self.n_solute + 3, :][0]   # (3,3), OHH
            O = w0[0:1]
            rel = w0 - O

            ref_H1 = rel[1].clone()
            ref_H2 = rel[2].clone()

            ref_frame = self.water_frame_from_rel_positions(
                torch.stack([
                    torch.zeros_like(ref_H1),
                    ref_H1,
                    ref_H2,
                ], dim=0).unsqueeze(0)
            )[0]

        self.register_buffer("ref_H1", ref_H1)
        self.register_buffer("ref_H2", ref_H2)
        self.register_buffer("ref_frame", ref_frame)

    def _L_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(dx)
        return dx - L * torch.round(dx / L)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L_tensor(x)
        return x - L * torch.floor(x / L)

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

    def water_frame_from_rel_positions(self, w_rel: torch.Tensor) -> torch.Tensor:
        """
        w_rel: (B,3,3), OHH with O at origin
        returns frame with columns [e1, e2, n]
        """
        u1 = w_rel[:, 1, :]
        u2 = w_rel[:, 2, :]

        e1 = self._normalize(u1)
        n = self._normalize(torch.cross(u1, u2, dim=-1))
        e2 = self._normalize(torch.cross(n, e1, dim=-1))
        return torch.stack([e1, e2, n], dim=-1)

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O)
        return O + d

    def cartesian_rel_to_tau(self, rel: torch.Tensor) -> torch.Tensor:
        """
        rel in principal MIC branch, shape (...,3), approximately in [-L/2,L/2)
        tau in [-pi, pi)
        """
        return wrap_to_pi((2.0 * math.pi / self.L) * rel)

    def tau_to_cartesian_rel(self, tau: torch.Tensor) -> torch.Tensor:
        """
        principal branch representative in [-L/2,L/2)
        """
        return (self.L / (2.0 * math.pi)) * tau

    def inverse(self, x: torch.Tensor) -> Tuple[ManifoldState, torch.Tensor]:
        """
        Cartesian -> manifold state
        x: (B, 3N)
        returns state, logdet
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Translation gauge: anchor solute atom 0 as origin
        sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
        origin = sol[:, 0:1, :]
        x_rel = self.mic(x - origin)

        # Flexible solute in anchored bond-vector coordinates
        v1 = x_rel[:, 1, :]
        v2 = x_rel[:, 2, :]
        solute = torch.cat([v1, v2], dim=-1)

        taus = []
        Rs = []

        F_ref = self.ref_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w_rel_to_sol = x_rel[:, s:s + 3, :]      # (B,3,3), OHH

            # make water whole relative to O
            w = self.make_whole_water(w_rel_to_sol)

            O_rel = w[:, 0, :]                       # oxygen rel to anchored solute atom 0
            tau = self.cartesian_rel_to_tau(O_rel)

            # rigid orientation as SO(3) element
            w_rel = w - w[:, 0:1, :]
            F_cur = self.water_frame_from_rel_positions(w_rel)
            R = F_cur @ F_ref.transpose(-1, -2)
            R = project_to_so3(R)

            taus.append(tau)
            Rs.append(R)

        tau = torch.stack(taus, dim=1)              # (B,W,3)
        R = torch.stack(Rs, dim=1)                  # (B,W,3,3)

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return ManifoldState(solute=solute, tau=tau, R=R), logdet

    def forward(self, state: ManifoldState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manifold state -> Cartesian
        returns x_flat, logdet
        """
        B = state.solute.shape[0]
        v1 = state.solute[:, 0:3]
        v2 = state.solute[:, 3:6]

        center = torch.full((B, 3), 0.5 * self.L, device=v1.device, dtype=v1.dtype)

        x = torch.zeros((B, self.n_atoms, 3), device=v1.device, dtype=v1.dtype)
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        H1_ref = self.ref_H1.to(v1.device, v1.dtype)
        H2_ref = self.ref_H2.to(v1.device, v1.dtype)

        for k in range(self.n_waters):
            O_rel = self.tau_to_cartesian_rel(state.tau[:, k, :])     # principal branch rep
            Rk = state.R[:, k, :, :]

            O_abs = center + O_rel
            H1 = torch.einsum("bij,j->bi", Rk, H1_ref)
            H2 = torch.einsum("bij,j->bi", Rk, H2_ref)

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1
            x[:, s + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return x.view(B, -1), logdet

class PBCGlobal3PointSphericalTransform(_BaseFlow):
    """
    Fixed-box PBC-safe transform for:
      - one flexible triatomic solute (atoms 0,1,2)
      - rigid water solvents in OHH ordering

    Internal coordinates:
      solute:
        v1 = MIC(x1 - x0)                          -> 3
        v2 = MIC(x2 - x0)                          -> 3

      each water:
        O_rel   = MIC(O - x0)                      -> 3
        omega   = rigid-water rotation vector      -> 3

    Total internal dim:
        6 + 6 * n_waters

    Properties
    ----------
    - removes only global translation (atom 0 anchored in forward)
    - keeps global rotation (important for fixed-box PBC)
    - preserves water ordering (permutation symmetry can be handled by the flow)
    - locally invertible almost everywhere
    - not globally bijective because:
        * rotvec has SO(3) branch cut at angle pi
        * MIC has measure-zero half-box ambiguities
    """

    def __init__(
        self,
        L: float,
        system=None,
        transform_data=None,
        internal_dim=None,
        eps: float = 1e-8,
        oxygen_r_min: float = 0.20,
        oxygen_r_scale: float = 0.25,
    ):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.transform_data = transform_data
        self.eps = eps

        # New: oxygen radial parameterization hyperparameters
        self.oxygen_r_min = float(oxygen_r_min)
        self.oxygen_r_scale = float(oxygen_r_scale)

        assert transform_data is not None and transform_data.shape[0] == 1
        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_water = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_water
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        expected_dim = 6 + 6 * self.n_waters
        self.internal_dim = expected_dim if internal_dim is None else internal_dim
        assert self.internal_dim == expected_dim, (self.internal_dim, expected_dim)

        # Build reference rigid-water geometry from transform_data
        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3).clone()
            w0 = x0[:, self.n_solute:self.n_solute + 3, :][0]   # (3,3), OHH
            O = w0[0:1]
            w_rel = w0 - O

            self.ref_H1 = w_rel[1].clone()
            self.ref_H2 = w_rel[2].clone()

            ref_cent = w_rel.mean(dim=0, keepdim=True)
            self.ref_water_centered = (w_rel - ref_cent).clone()

        ref_w_rel = torch.stack(
            [
                torch.zeros_like(self.ref_H1),
                self.ref_H1,
                self.ref_H2,
            ],
            dim=0,
        ).unsqueeze(0)

        self.ref_water_frame = self.water_frame_from_rel_positions(ref_w_rel)[0]

    # ------------------------------------------------------------------
    # PBC helpers
    # ------------------------------------------------------------------
    def oxygen_latent_to_cartesian(self, z_O: torch.Tensor):
        """
        Map latent oxygen vector z_O in R^3 to Cartesian oxygen displacement O_rel in R^3.

        z_O:   (B,3)
        O_rel: (B,3)

        O_rel = (r_min + r_scale * ||z_O||) * z_O / ||z_O||
        """
        rho = torch.linalg.norm(z_O, dim=-1, keepdim=True).clamp_min(self.eps)   # (B,1)
        u = z_O / rho                                                             # (B,3)
        r = self.oxygen_r_min + self.oxygen_r_scale * rho                         # (B,1)
        O_rel = r * u                                                             # (B,3)

        # log |det dO/dz| for radial map in 3D:
        # det = (r / rho)^2 * dr/drho, with dr/drho = oxygen_r_scale
        logdet = (
            2.0 * torch.log((r / rho).clamp_min(self.eps)) +
            math.log(self.oxygen_r_scale)
        ).squeeze(-1)  # (B,)

        return O_rel, logdet
    


    def oxygen_cartesian_to_latent(self, O_rel: torch.Tensor):
        """
        Inverse of oxygen_latent_to_cartesian.

        O_rel: (B,3)
        z_O:   (B,3)
        """
        r = torch.linalg.norm(O_rel, dim=-1, keepdim=True).clamp_min(self.eps)   # (B,1)
        u = O_rel / r                                                             # (B,3)

        # invert r = r_min + r_scale * rho
        rho = ((r - self.oxygen_r_min) / self.oxygen_r_scale).clamp_min(self.eps) # (B,1)
        z_O = rho * u                                                             # (B,3)

        # inverse logdet = - forward logdet
        logdet_inv = -(
            2.0 * torch.log((r / rho).clamp_min(self.eps)) +
            math.log(self.oxygen_r_scale)
        ).squeeze(-1)  # (B,)

        return z_O, logdet_inv
    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

    def water_frame_from_rel_positions(self, w_rel: torch.Tensor) -> torch.Tensor:
        """
        Build a deterministic labeled body frame for water from relative coordinates.

        Parameters
        ----------
        w_rel : (B,3,3)
            Water coordinates relative to oxygen, in OHH ordering:
            w_rel[:,0,:] = 0
            w_rel[:,1,:] = H1 - O
            w_rel[:,2,:] = H2 - O

        Returns
        -------
        F : (B,3,3)
            Frame matrix with columns [e1, e2, n], acting on column vectors.
        """
        u1 = w_rel[:, 1, :]   # H1 - O
        u2 = w_rel[:, 2, :]   # H2 - O

        e1 = self._normalize(u1)
        n = self._normalize(torch.cross(u1, u2, dim=1))
        e2 = self._normalize(torch.cross(n, e1, dim=1))

        F = torch.stack([e1, e2, n], dim=2)  # columns
        return F
    def _L_tensor(self, x: torch.Tensor, L: float = None) -> torch.Tensor:
        L = self.L if L is None else L
        return torch.as_tensor(L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor, L: float = None) -> torch.Tensor:
        L_t = self._L_tensor(dx, L)
        return dx - L_t * torch.round(dx / L_t)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L_t = self._L_tensor(x, self.L)
        return x - L_t * torch.floor(x / L_t)

    def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        """
        x_sol: (B,3,3)
        Make solute whole relative to atom 0.
        """
        x0 = x_sol[:, 0:1, :]
        d = self.mic(x_sol - x0)
        return x0 + d

    def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        """
        x_w: (B,3,3) with OHH ordering
        Make water whole relative to oxygen.
        """
        O = x_w[:, 0:1, :]
        d = self.mic(x_w - O)
        return O + d

    # ------------------------------------------------------------------
    # SO(3) helpers (column-vector convention)
    # ------------------------------------------------------------------
    def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (B,3)
        returns R: (B,3,3), acting on column vectors
        """
        B = w.shape[0]
        theta = torch.linalg.norm(w, dim=1, keepdim=True)  # (B,1)

        R = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(B, 1, 1)

        small = theta[:, 0] < 1e-8
        big = ~small

        if big.any():
            th = theta[big]
            k = w[big] / th.clamp_min(self.eps)

            kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
            K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
            K[:, 0, 1] = -kz
            K[:, 0, 2] =  ky
            K[:, 1, 0] =  kz
            K[:, 1, 2] = -kx
            K[:, 2, 0] = -ky
            K[:, 2, 1] =  kx

            I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(big.sum(), 1, 1)
            ct = torch.cos(th).view(-1, 1, 1)
            st = torch.sin(th).view(-1, 1, 1)
            R[big] = I + st * K + (1.0 - ct) * (K @ K)

        if small.any():
            ws = w[small]
            K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
            K[:, 0, 1] = -ws[:, 2]
            K[:, 0, 2] =  ws[:, 1]
            K[:, 1, 0] =  ws[:, 2]
            K[:, 1, 2] = -ws[:, 0]
            K[:, 2, 0] = -ws[:, 1]
            K[:, 2, 1] =  ws[:, 0]
            I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(small.sum(), 1, 1)
            R[small] = I + K

        return R

    def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: (B,3,3), acting on column vectors
        returns w: (B,3)
        """
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        if small.any():
            A = 0.5 * (R[small] - R[small].transpose(1, 2))
            w[small, 0] = A[:, 2, 1]
            w[small, 1] = A[:, 0, 2]
            w[small, 2] = A[:, 1, 0]

        big = ~small
        if big.any():
            th = theta[big]
            denom = (2.0 * torch.sin(th)).clamp_min(self.eps)
            axis = torch.stack([
                (R[big, 2, 1] - R[big, 1, 2]) / denom,
                (R[big, 0, 2] - R[big, 2, 0]) / denom,
                (R[big, 1, 0] - R[big, 0, 1]) / denom,
            ], dim=1)
            w[big] = axis * th.unsqueeze(1)

        return w

    # ------------------------------------------------------------------
    # Kabsch: return column-vector rotation mapping reference -> target
    # ------------------------------------------------------------------
    @staticmethod
    def kabsch_ref_to_target_colvec(X_ref: torch.Tensor, Y_tgt: torch.Tensor) -> torch.Tensor:
        """
        X_ref, Y_tgt: (B,N,3), centered row-stacked point clouds.
        Returns R_col acting on column vectors so that ref -> target.
        """
        C = X_ref.transpose(1, 2) @ Y_tgt
        U, S, Vh = torch.linalg.svd(C)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)

        det = torch.det(V @ Ut)
        D = torch.eye(3, device=X_ref.device, dtype=X_ref.dtype).unsqueeze(0).repeat(X_ref.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

        R_row = V @ D @ Ut
        R_col = R_row.transpose(1, 2)
        return R_col

    def water_rotation_from_positions(self, w_rel: torch.Tensor) -> torch.Tensor:
        """
        w_rel: (B,3,3), water coordinates relative to oxygen
        returns R: (B,3,3), column-vector rotation mapping reference water -> current water
        """
        w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
        X_ref = self.ref_water_centered.to(w_rel.device, w_rel.dtype).unsqueeze(0).expand(w_rel.shape[0], 3, 3)
        return self.kabsch_ref_to_target_colvec(X_ref, w_cent)

    # ------------------------------------------------------------------
    # inverse: Cartesian -> internal
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N) flattened Cartesian

        Returns
        -------
        i: (B, 6 + 6*n_waters)
        logdet_xi: (B,) = log |det(di/dx)|
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Make solute whole, define atom 0 as anchor
        sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
        origin = sol[:, 0:1, :]                     # (B,1,3)

        # Canonical MIC-relative coordinates to atom 0
        x_rel = self.mic(x - origin)                # (B,N,3)

        # Solute internal coordinates
        v1 = x_rel[:, 1, :]
        v2 = x_rel[:, 2, :]
        pieces = [v1, v2]

        logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

        F_ref = self.ref_water_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k

            # Water coordinates relative to solute atom 0
            w_rel_to_sol = x_rel[:, s:s + 3, :]     # (B,3,3), OHH

            # Make water whole relative to its oxygen
            w = self.make_whole_water(w_rel_to_sol)

            # Oxygen position relative to solute atom 0
            O_rel = w[:, 0, :]                      # (B,3)

            # New: convert oxygen Cartesian displacement to latent oxygen variable
            z_O, logdet_O_inv = self.oxygen_cartesian_to_latent(O_rel)
            logdet = logdet + logdet_O_inv

            # Water orientation from deterministic labeled frame
            w_rel = w - w[:, 0:1, :]               # O at origin
            F_cur = self.water_frame_from_rel_positions(w_rel)  # (B,3,3)

            # Rotation mapping reference frame -> current frame
            R = torch.einsum("bij,bkj->bik", F_cur, F_ref)
            omega = self.rotmat_to_rotvec(R)

            pieces += [z_O, omega]

        i = torch.cat(pieces, dim=1)
        return i, logdet

    def forward(self, i: torch.Tensor):
        """
        i: (B, 6 + 6*n_waters)

        Returns
        -------
        x: (B, 3*N) flattened Cartesian in [0, L)
        logdet_ix: (B,) = log |det(dx/di)|
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # Solute internal coords
        v1 = i[:, 0:3]
        v2 = i[:, 3:6]

        center = torch.full((B, 3), 0.5 * self.L, device=i.device, dtype=i.dtype)

        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype)
        H2_ref = self.ref_H2.to(i.device, i.dtype)

        idx = 6
        for k in range(self.n_waters):
            # New: oxygen internal is latent z_O, not direct Cartesian O_rel
            z_O = i[:, idx:idx + 3]
            omega = i[:, idx + 3:idx + 6]
            idx += 6

            O_rel, logdet_O = self.oxygen_latent_to_cartesian(z_O)
            logdet = logdet + logdet_O

            R = self.rotvec_to_rotmat(omega)

            O_abs = center + O_rel
            H1 = torch.einsum("bij,j->bi", R, H1_ref)
            H2 = torch.einsum("bij,j->bi", R, H2_ref)

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1
            x[:, s + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet

# class PBCGlobal3PointSphericalTransform(_BaseFlow):
#     """
#     Fixed-box PBC-safe transform for:
#       - one flexible triatomic solute (atoms 0,1,2)
#       - rigid water solvents in OHH ordering

#     Internal coordinates:
#       solute:
#         v1 = MIC(x1 - x0)                          -> 3
#         v2 = MIC(x2 - x0)                          -> 3

#       each water:
#         O_rel   = MIC(O - x0)                      -> 3
#         omega   = rigid-water rotation vector      -> 3

#     Total internal dim:
#         6 + 6 * n_waters

#     Properties
#     ----------
#     - removes only global translation (atom 0 anchored in forward)
#     - keeps global rotation (important for fixed-box PBC)
#     - preserves water ordering (permutation symmetry can be handled by the flow)
#     - locally invertible almost everywhere
#     - not globally bijective because:
#         * rotvec has SO(3) branch cut at angle pi
#         * MIC has measure-zero half-box ambiguities
#     """

#     def __init__(self, L: float, system=None, transform_data=None, internal_dim=None, eps: float = 1e-8):
#         super().__init__()
#         self.L = float(L)
#         self.system = system
#         self.transform_data = transform_data
#         self.eps = eps

#         assert transform_data is not None and transform_data.shape[0] == 1
#         self.n_atoms = transform_data.shape[1] // 3
#         self.n_solute = 3
#         self.n_atoms_per_water = 3
#         self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_water
#         assert self.n_solute + 3 * self.n_waters == self.n_atoms

#         expected_dim = 6 + 6 * self.n_waters
#         self.internal_dim = expected_dim if internal_dim is None else internal_dim
#         assert self.internal_dim == expected_dim, (self.internal_dim, expected_dim)

#         # Build reference rigid-water geometry from transform_data
#         with torch.no_grad():
#             x0 = transform_data.reshape(1, self.n_atoms, 3).clone()
#             w0 = x0[:, self.n_solute:self.n_solute + 3, :][0]   # (3,3), OHH
#             O = w0[0:1]
#             w_rel = w0 - O

#             # Reference H vectors used in forward()
#             self.ref_H1 = w_rel[1].clone()   # (3,)
#             self.ref_H2 = w_rel[2].clone()   # (3,)

#             # Centered reference cloud for Kabsch in inverse()
#             ref_cent = w_rel.mean(dim=0, keepdim=True)
#             self.ref_water_centered = (w_rel - ref_cent).clone()  # (3,3)
        
#         # Reference frame for labeled-water orientation
#         ref_w_rel = torch.stack(
#             [
#                 torch.zeros_like(self.ref_H1),
#                 self.ref_H1,
#                 self.ref_H2,
#             ],
#             dim=0,
#         ).unsqueeze(0)  # (1,3,3)

#         self.ref_water_frame = self.water_frame_from_rel_positions(ref_w_rel)[0]  # (3,3)

#     # ------------------------------------------------------------------
#     # PBC helpers
#     # ------------------------------------------------------------------
#     def _normalize(self, v: torch.Tensor) -> torch.Tensor:
#         return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

#     def water_frame_from_rel_positions(self, w_rel: torch.Tensor) -> torch.Tensor:
#         """
#         Build a deterministic labeled body frame for water from relative coordinates.

#         Parameters
#         ----------
#         w_rel : (B,3,3)
#             Water coordinates relative to oxygen, in OHH ordering:
#             w_rel[:,0,:] = 0
#             w_rel[:,1,:] = H1 - O
#             w_rel[:,2,:] = H2 - O

#         Returns
#         -------
#         F : (B,3,3)
#             Frame matrix with columns [e1, e2, n], acting on column vectors.
#         """
#         u1 = w_rel[:, 1, :]   # H1 - O
#         u2 = w_rel[:, 2, :]   # H2 - O

#         e1 = self._normalize(u1)
#         n = self._normalize(torch.cross(u1, u2, dim=1))
#         e2 = self._normalize(torch.cross(n, e1, dim=1))

#         F = torch.stack([e1, e2, n], dim=2)  # columns
#         return F
#     def _L_tensor(self, x: torch.Tensor, L: float = None) -> torch.Tensor:
#         L = self.L if L is None else L
#         return torch.as_tensor(L, device=x.device, dtype=x.dtype)

#     def mic(self, dx: torch.Tensor, L: float = None) -> torch.Tensor:
#         L_t = self._L_tensor(dx, L)
#         return dx - L_t * torch.round(dx / L_t)

#     def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
#         L_t = self._L_tensor(x, self.L)
#         return x - L_t * torch.floor(x / L_t)

#     def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
#         """
#         x_sol: (B,3,3)
#         Make solute whole relative to atom 0.
#         """
#         x0 = x_sol[:, 0:1, :]
#         d = self.mic(x_sol - x0)
#         return x0 + d

#     def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
#         """
#         x_w: (B,3,3) with OHH ordering
#         Make water whole relative to oxygen.
#         """
#         O = x_w[:, 0:1, :]
#         d = self.mic(x_w - O)
#         return O + d

#     # ------------------------------------------------------------------
#     # SO(3) helpers (column-vector convention)
#     # ------------------------------------------------------------------
#     def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
#         """
#         w: (B,3)
#         returns R: (B,3,3), acting on column vectors
#         """
#         B = w.shape[0]
#         theta = torch.linalg.norm(w, dim=1, keepdim=True)  # (B,1)

#         R = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(B, 1, 1)

#         small = theta[:, 0] < 1e-8
#         big = ~small

#         if big.any():
#             th = theta[big]
#             k = w[big] / th.clamp_min(self.eps)

#             kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
#             K = torch.zeros((big.sum(), 3, 3), device=w.device, dtype=w.dtype)
#             K[:, 0, 1] = -kz
#             K[:, 0, 2] =  ky
#             K[:, 1, 0] =  kz
#             K[:, 1, 2] = -kx
#             K[:, 2, 0] = -ky
#             K[:, 2, 1] =  kx

#             I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(big.sum(), 1, 1)
#             ct = torch.cos(th).view(-1, 1, 1)
#             st = torch.sin(th).view(-1, 1, 1)
#             R[big] = I + st * K + (1.0 - ct) * (K @ K)

#         if small.any():
#             ws = w[small]
#             K = torch.zeros((small.sum(), 3, 3), device=w.device, dtype=w.dtype)
#             K[:, 0, 1] = -ws[:, 2]
#             K[:, 0, 2] =  ws[:, 1]
#             K[:, 1, 0] =  ws[:, 2]
#             K[:, 1, 2] = -ws[:, 0]
#             K[:, 2, 0] = -ws[:, 1]
#             K[:, 2, 1] =  ws[:, 0]
#             I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).repeat(small.sum(), 1, 1)
#             R[small] = I + K

#         return R

#     def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
#         """
#         R: (B,3,3), acting on column vectors
#         returns w: (B,3)
#         """
#         B = R.shape[0]
#         trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
#         cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
#         theta = torch.acos(cos_theta)

#         w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

#         small = theta < 1e-6
#         if small.any():
#             A = 0.5 * (R[small] - R[small].transpose(1, 2))
#             w[small, 0] = A[:, 2, 1]
#             w[small, 1] = A[:, 0, 2]
#             w[small, 2] = A[:, 1, 0]

#         big = ~small
#         if big.any():
#             th = theta[big]
#             denom = (2.0 * torch.sin(th)).clamp_min(self.eps)
#             axis = torch.stack([
#                 (R[big, 2, 1] - R[big, 1, 2]) / denom,
#                 (R[big, 0, 2] - R[big, 2, 0]) / denom,
#                 (R[big, 1, 0] - R[big, 0, 1]) / denom,
#             ], dim=1)
#             w[big] = axis * th.unsqueeze(1)

#         return w

#     # ------------------------------------------------------------------
#     # Kabsch: return column-vector rotation mapping reference -> target
#     # ------------------------------------------------------------------
#     @staticmethod
#     def kabsch_ref_to_target_colvec(X_ref: torch.Tensor, Y_tgt: torch.Tensor) -> torch.Tensor:
#         """
#         X_ref, Y_tgt: (B,N,3), centered row-stacked point clouds.
#         Returns R_col acting on column vectors so that ref -> target.
#         """
#         C = X_ref.transpose(1, 2) @ Y_tgt
#         U, S, Vh = torch.linalg.svd(C)
#         V = Vh.transpose(1, 2)
#         Ut = U.transpose(1, 2)

#         det = torch.det(V @ Ut)
#         D = torch.eye(3, device=X_ref.device, dtype=X_ref.dtype).unsqueeze(0).repeat(X_ref.shape[0], 1, 1)
#         D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

#         R_row = V @ D @ Ut
#         R_col = R_row.transpose(1, 2)
#         return R_col

#     def water_rotation_from_positions(self, w_rel: torch.Tensor) -> torch.Tensor:
#         """
#         w_rel: (B,3,3), water coordinates relative to oxygen
#         returns R: (B,3,3), column-vector rotation mapping reference water -> current water
#         """
#         w_cent = w_rel - w_rel.mean(dim=1, keepdim=True)
#         X_ref = self.ref_water_centered.to(w_rel.device, w_rel.dtype).unsqueeze(0).expand(w_rel.shape[0], 3, 3)
#         return self.kabsch_ref_to_target_colvec(X_ref, w_cent)

#     # ------------------------------------------------------------------
#     # inverse: Cartesian -> internal
#     # ------------------------------------------------------------------
#     def inverse(self, x: torch.Tensor):
#         """
#         x: (B, 3*N) flattened Cartesian

#         Returns
#         -------
#         i: (B, 6 + 6*n_waters)
#         logdet_xi: (B,)
#         """
#         B = x.shape[0]
#         x = x.view(B, self.n_atoms, 3)

#         # Make solute whole, define atom 0 as anchor
#         sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
#         origin = sol[:, 0:1, :]                     # (B,1,3)

#         # Canonical MIC-relative coordinates to atom 0
#         x_rel = self.mic(x - origin)                # (B,N,3)

#         # Solute internal coordinates
#         v1 = x_rel[:, 1, :]
#         v2 = x_rel[:, 2, :]
#         pieces = [v1, v2]

#         logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

#         # Reference water frame
#         F_ref = self.ref_water_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

#         # Waters
#         for k in range(self.n_waters):
#             s = self.n_solute + 3 * k

#             # Water coordinates relative to solute atom 0
#             w_rel_to_sol = x_rel[:, s:s + 3, :]     # (B,3,3), OHH

#             # Make water whole relative to its oxygen
#             w = self.make_whole_water(w_rel_to_sol)

#             # Oxygen position relative to solute atom 0
#             O_rel = w[:, 0, :]                      # (B,3)

#             # Water orientation from deterministic labeled frame
#             w_rel = w - w[:, 0:1, :]               # O at origin
#             F_cur = self.water_frame_from_rel_positions(w_rel)  # (B,3,3)

#             # Rotation mapping reference frame -> current frame
#             R = torch.einsum("bij,bkj->bik", F_cur, F_ref)
#             # equivalently: R = F_cur @ F_ref.transpose(1, 2)

#             omega = self.rotmat_to_rotvec(R)

#             pieces += [O_rel, omega]

#         i = torch.cat(pieces, dim=1)
#         return i, logdet

#     def forward(self, i: torch.Tensor):
#         """
#         i: (B, 6 + 6*n_waters)

#         Returns
#         -------
#         x: (B, 3*N) flattened Cartesian in [0, L)
#         logdet_ix: (B,)
#         """
#         B = i.shape[0]
#         assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

#         # Solute internal coords
#         v1 = i[:, 0:3]
#         v2 = i[:, 3:6]

#         center = torch.full((B, 3), 0.5 * self.L, device=i.device, dtype=i.dtype)

#         x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
#         x[:, 0, :] = center
#         x[:, 1, :] = center + v1
#         x[:, 2, :] = center + v2

#         logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

#         H1_ref = self.ref_H1.to(i.device, i.dtype)
#         H2_ref = self.ref_H2.to(i.device, i.dtype)

#         idx = 6
#         for k in range(self.n_waters):

#             O_rel = i[:, idx:idx + 3]
#             omega = i[:, idx + 3:idx + 6]
#             idx += 6

#             R = self.rotvec_to_rotmat(omega)

#             O_abs = center + O_rel
#             H1 = torch.einsum("bij,j->bi", R, H1_ref)
#             H2 = torch.einsum("bij,j->bi", R, H2_ref)

#             s = self.n_solute + 3 * k
#             x[:, s + 0, :] = O_abs
#             x[:, s + 1, :] = O_abs + H1
#             x[:, s + 2, :] = O_abs + H2

#         x = self.wrap_0L(x)
#         return x.view(B, -1), logdet

# class PBCGlobal3PointSphericalTransform(nf.flows.Flow):
#     """
#     PBC + rigid-water-friendly coordinate transform.

#     Internal coords i:
#       - solute: v1 = x1-x0, v2 = x2-x0  (each MIC wrapped)  -> 6 dims
#       - each water: O position relative to solute origin (MIC) -> 3 dims
#                   + rotation vector omega (axis-angle)        -> 3 dims
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
#         # self.internal_dim = 6 + 6 * self.n_waters
#         self.internal_dim = internal_dim
#         # print("internal_dim  in transform:", self.internal_dim )

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

#         # Optional: keep theta in [0, pi] already guaranteed by acos.
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

#     # ---------- nf API ----------
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
#         # x_rel = x - origin

#         # Solute internal: two MIC bond vectors from atom0
#         v1 = x_rel[:, 1, :]  # already mic-wrapped
#         v2 = x_rel[:, 2, :]

#         pieces = [v1, v2]

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

#         # Unpack solute
#         v1 = i[:, 0:3]
#         v2 = i[:, 3:6]

#         # Place solute atom0 at box center (nice gauge choice for OpenMM)
#         center = 0.5 * self.L
#         x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
#         x[:, 0, :] = center
#         x[:, 1, :] = center + v1
#         x[:, 2, :] = center + v2

#         # Waters
#         idx = 6
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




class PBCGlobal3PointSphericalTransform2(nf.flows.Flow):
    """
    PBC + rigid-water-friendly coordinate transform.

    Internal coords i:
      - solute: shape only -> (r1, r2, theta)                  -> 3 dims
      - waters: 
          each water: O position relative to solute origin     -> 3 dims
                    + rotation vector omega                    -> 3 dims
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
        self.internal_dim = internal_dim
        print("internal_dim  in transform:", self.internal_dim )

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

        pieces = [r1, r2, theta]

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

        # # Unpack solute
        # v1 = i[:, 0:3]
        # v2 = i[:, 3:6]

        # # Place solute atom0 at box center
        # center = 0.5 * self.L
        # x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
        # x[:, 0, :] = center
        # x[:, 1, :] = center + v1
        # x[:, 2, :] = center + v2

        # Unpack solute shape
        r1 = i[:, 0:1].clamp_min(1e-6)          # (B,1)
        r2 = i[:, 1:2].clamp_min(1e-6)          # (B,1)
        theta = i[:, 2:3].clamp(1e-3, math.pi - 1e-3)   # (B,1)

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # S at center
        x[:, 0, :] = center

        # O1 on +x
        x[:, 1, 0] = center + r1[:, 0]
        x[:, 1, 1] = center
        x[:, 1, 2] = center

        # O2 in xy-plane
        x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
        x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
        x[:, 2, 2] = center

        # Waters
        # idx = 6
        idx = 3
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


class PBCGlobal3PointSphericalTransform3(nf.flows.Flow):
    """
    PBC + rigid-water + translation/rotation-equivariant coordinate transform.

    Internal coordinates:
      - t_global : 3
      - v1       : 3
      - v2       : 3
      - for each rigid water:
          O_body : 3
          c_rel  : 3   (Cayley coordinates of water orientation relative to solute frame)

    Total dim = 9 + 6 * n_waters

    Properties:
      - translation equivariant
      - rotation equivariant (away from frame degeneracies / PBC discontinuities)
      - bijective almost everywhere on a canonical PBC chart
      - exact rigid-water reconstruction

    Important:
      - NOT globally smooth/invertible at:
          * MIC tie surfaces
          * solute frame degeneracy (v1 || v2)
          * relative rotations with angle pi (Cayley singularity)
      - log|det J| is NOT zero; this implementation leaves it unimplemented on purpose.
    """

    def __init__(self, L: float, transform_data: torch.Tensor, internal_dim=None):
        super().__init__()
        self.L = float(L)

        assert transform_data is not None and transform_data.shape[0] == 1

        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        expected_dim = 9 + 6 * self.n_waters
        if internal_dim is None:
            self.internal_dim = expected_dim
        else:
            assert internal_dim == expected_dim, (internal_dim, expected_dim)
            self.internal_dim = internal_dim

        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3)[0].clone()

            # Reference solute frame from reference configuration
            sol = x0[:self.n_solute]
            sol_rel = self.mic(sol.unsqueeze(0) - sol[0].view(1, 1, 3), self.L)[0]
            Rsol0 = self._build_solute_frame_single(sol_rel)

            # Reference water geometry from first water
            s = self.n_solute
            wref = x0[s:s + 3]  # [O, H1, H2]
            wref_rel = self.mic(wref.unsqueeze(0) - wref[0].view(1, 1, 3), self.L)[0]

            # Store rigid reference geometry relative to oxygen
            self.register_buffer("h1_ref", wref_rel[1].clone())
            self.register_buffer("h2_ref", wref_rel[2].clone())

            # Deterministic reference water frame in lab/body coords
            Rw0 = self._build_water_frame_single(wref_rel)

            # Relative reference orientation in solute frame
            Rrel0 = Rsol0.T @ Rw0
            self.register_buffer("Rrel_ref", Rrel0.clone())

    # ---------------------------
    # PBC helpers
    # ---------------------------
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

    # ---------------------------
    # Solute frame
    # ---------------------------
    @staticmethod
    def _build_solute_frame_single(sol_rel: torch.Tensor) -> torch.Tensor:
        """
        sol_rel: (3,3), atom0 at origin
        returns R: (3,3), columns = body axes in lab coordinates
        """
        v1 = sol_rel[1]
        v2 = sol_rel[2]

        e1 = v1 / torch.norm(v1).clamp_min(1e-12)
        tmp = v2 - torch.dot(v2, e1) * e1
        e2 = tmp / torch.norm(tmp).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)

    @staticmethod
    def _build_solute_frame_batch(sol_rel: torch.Tensor) -> torch.Tensor:
        """
        sol_rel: (B,3,3), atom0 at origin
        returns R: (B,3,3), columns = body axes in lab coordinates
        """
        v1 = sol_rel[:, 1, :]
        v2 = sol_rel[:, 2, :]

        e1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp_min(1e-12)
        tmp = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
        e2 = tmp / torch.linalg.norm(tmp, dim=-1, keepdim=True).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=-1)

        return torch.stack([e1, e2, e3], dim=2)

    # ---------------------------
    # Water frame (deterministic from rigid geometry)
    # ---------------------------
    @staticmethod
    def _build_water_frame_single(w_rel: torch.Tensor) -> torch.Tensor:
        """
        w_rel: (3,3), with O at origin
        returns Rw: (3,3), columns = water-frame axes in lab coordinates
        """
        h1 = w_rel[1]
        h2 = w_rel[2]

        e1 = h1 / torch.norm(h1).clamp_min(1e-12)
        tmp = h2 - torch.dot(h2, e1) * e1
        e2 = tmp / torch.norm(tmp).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)

    @staticmethod
    def _build_water_frame_batch(w_rel: torch.Tensor) -> torch.Tensor:
        """
        w_rel: (B,3,3), with O at origin
        returns Rw: (B,3,3), columns = water-frame axes in lab coordinates
        """
        h1 = w_rel[:, 1, :]
        h2 = w_rel[:, 2, :]

        e1 = h1 / torch.linalg.norm(h1, dim=-1, keepdim=True).clamp_min(1e-12)
        tmp = h2 - (e1 * h2).sum(dim=-1, keepdim=True) * e1
        e2 = tmp / torch.linalg.norm(tmp, dim=-1, keepdim=True).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=-1)

        return torch.stack([e1, e2, e3], dim=2)

    # ---------------------------
    # SO(3) <-> Cayley coordinates
    # ---------------------------
    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        """
        v: (B,3)
        returns [v]_x : (B,3,3)
        """
        B = v.shape[0]
        M = torch.zeros((B, 3, 3), device=v.device, dtype=v.dtype)
        M[:, 0, 1] = -v[:, 2]
        M[:, 0, 2] =  v[:, 1]
        M[:, 1, 0] =  v[:, 2]
        M[:, 1, 2] = -v[:, 0]
        M[:, 2, 0] = -v[:, 1]
        M[:, 2, 1] =  v[:, 0]
        return M

    def cayley_to_rotmat(self, c: torch.Tensor) -> torch.Tensor:
        """
        c in R^3 -> R in SO(3), valid for all c.
        R = (I + [c]_x)(I - [c]_x)^{-1}
        """
        B = c.shape[0]
        I = torch.eye(3, device=c.device, dtype=c.dtype).unsqueeze(0).expand(B, 3, 3)
        C = self._skew(c)
        return torch.linalg.solve(I - C, I + C)

    def rotmat_to_cayley(self, R: torch.Tensor) -> torch.Tensor:
        """
        R in SO(3), excluding angle pi.
        c is defined by [c]_x = (R - I)(R + I)^{-1}
        """
        B = R.shape[0]
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, 3, 3)

        # Singular when det(R + I) = 0, i.e. angle = pi
        A = torch.linalg.solve(R + I, R - I)  # equals [c]_x in exact arithmetic

        c = torch.stack([
            A[:, 2, 1],
            A[:, 0, 2],
            A[:, 1, 0],
        ], dim=1)
        return c

    # ---------------------------
    # nf API
    # ---------------------------
    def inverse(self, x: torch.Tensor):
        """
        Cartesian -> internal coordinates

        x: (B, 3*N)
        returns:
          i: (B, 9 + 6*n_waters)
          logdet_x_to_i: NOT IMPLEMENTED
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Canonicalize solute relative to anchor atom 0
        sol = self.make_whole_solute(x[:, :self.n_solute, :])
        origin = sol[:, 0:1, :]                      # (B,1,3)
        x_rel = self.mic(x - origin, self.L)         # all coords rel to anchor atom 0

        v1 = x_rel[:, 1, :]
        v2 = x_rel[:, 2, :]
        t_global = origin[:, 0, :]

        pieces = [t_global, v1, v2]

        # Solute frame
        sol_rel = torch.zeros((B, 3, 3), device=x.device, dtype=x.dtype)
        sol_rel[:, 1, :] = v1
        sol_rel[:, 2, :] = v2
        Rsol = self._build_solute_frame_batch(sol_rel)

        Rrel_ref = self.Rrel_ref.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s + 3, :]          # anchor-relative
            w = self.make_whole_water(w)      # make water whole relative to oxygen

            O_rel = w[:, 0, :]                # anchor-relative oxygen
            O_body = torch.einsum("bij,bj->bi", Rsol.transpose(1, 2), O_rel)

            w_rel_O = w - w[:, 0:1, :]        # relative to oxygen
            Rw = self._build_water_frame_batch(w_rel_O)

            # relative orientation of water w.r.t. solute
            Rrel = torch.einsum("bij,bjk->bik", Rsol.transpose(1, 2), Rw)

            # remove fixed reference orientation so c_rel=0 at reference geometry
            Rdelta = torch.einsum("bij,bjk->bik", Rrel, Rrel_ref.transpose(1, 2))
            c_rel = self.rotmat_to_cayley(Rdelta)

            pieces += [O_body, c_rel]

        i = torch.cat(pieces, dim=1)

        raise NotImplementedError(
            "This transform has a nontrivial Jacobian. "
            "Do not return logdet=0 for BG likelihood training. "
            "Derive or compute log|det J| separately."
        )

    def forward(self, i: torch.Tensor):
        """
        Internal coordinates -> Cartesian

        i: (B, 9 + 6*n_waters)
        returns:
          x: (B, 3*N)
          logdet_i_to_x: NOT IMPLEMENTED
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        t_global = i[:, 0:3]
        v1 = i[:, 3:6]
        v2 = i[:, 6:9]

        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # Reconstruct solute
        x[:, 0, :] = t_global
        x[:, 1, :] = t_global + v1
        x[:, 2, :] = t_global + v2

        sol_rel = torch.zeros((B, 3, 3), device=i.device, dtype=i.dtype)
        sol_rel[:, 1, :] = v1
        sol_rel[:, 2, :] = v2
        Rsol = self._build_solute_frame_batch(sol_rel)

        h1_ref = self.h1_ref.to(i.device, i.dtype).view(1, 3, 1)
        h2_ref = self.h2_ref.to(i.device, i.dtype).view(1, 3, 1)
        Rrel_ref = self.Rrel_ref.to(i.device, i.dtype).unsqueeze(0).expand(B, 3, 3)

        idx = 9
        for k in range(self.n_waters):
            O_body = i[:, idx:idx + 3]
            c_rel  = i[:, idx + 3:idx + 6]
            idx += 6

            O_lab = torch.einsum("bij,bj->bi", Rsol, O_body) + t_global

            Rdelta = self.cayley_to_rotmat(c_rel)
            Rrel = torch.einsum("bij,bjk->bik", Rdelta, Rrel_ref)
            Rw = torch.einsum("bij,bjk->bik", Rsol, Rrel)

            H1_lab = (Rw @ h1_ref.expand(B, 3, 1)).squeeze(-1) + O_lab
            H2_lab = (Rw @ h2_ref.expand(B, 3, 1)).squeeze(-1) + O_lab

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_lab
            x[:, s + 1, :] = H1_lab
            x[:, s + 2, :] = H2_lab

        x = self.wrap_0L(x)

        raise NotImplementedError(
            "This transform has a nontrivial Jacobian. "
            "Do not return logdet=0 for BG likelihood training. "
            "Derive or compute log|det J| separately."
        )
# class PBCGlobal3PointSphericalTransform2(nf.flows.Flow):
#     """
#     PBC + rigid-water-friendly + translation/rotation-reduced transform.

#     Internal coords i:
#       - solute shape only -> (r1, r2, theta)                  : 3 dims
#       - each water:
#           O position in instantaneous solute frame            : 3 dims
#           rigid-body rotation vector in solute frame          : 3 dims

#     Total dim = 3 + 6 * n_waters

#     Conventions:
#       - solute atoms are [0, 1, 2]
#       - water atoms are ordered [O, H1, H2]
#       - solute atom 0 is the anchor
#       - forward reconstructs the solute in a canonical frame:
#           atom0 at box center
#           atom1 on +x
#           atom2 in xy-plane
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

#         expected_dim = 3 + 6 * self.n_waters
#         if internal_dim is None:
#             self.internal_dim = expected_dim
#         else:
#             assert internal_dim == expected_dim, (internal_dim, expected_dim)
#             self.internal_dim = internal_dim

#         with torch.no_grad():
#             x0 = transform_data.reshape(1, self.n_atoms, 3).clone()[0]  # (N,3)

#             # --- Reference solute, made whole and expressed in its own canonical frame
#             sol = x0[:self.n_solute]  # (3,3)
#             sol_rel = self.mic(sol.unsqueeze(0) - sol[0].view(1, 1, 3), self.L)[0]  # (3,3)

#             R0 = self._build_solute_frame_single(sol_rel)  # (3,3), columns = basis in lab coords

#             # --- Reference rigid water, rotated into the reference solute frame
#             w0_start = self.n_solute
#             ref = x0[w0_start:w0_start + 3]  # (3,3) = [O,H1,H2]
#             ref_rel = self.mic(ref.unsqueeze(0) - ref[0].view(1, 1, 3), self.L)[0]  # O at origin, whole water
#             ref_rel_sol = (R0.T @ ref_rel.T).T  # water in canonical solute frame

#             ref_cent = ref_rel_sol.mean(dim=0, keepdim=True)

#             self.register_buffer("ref_H1", ref_rel_sol[1].clone())
#             self.register_buffer("ref_H2", ref_rel_sol[2].clone())
#             self.register_buffer("ref_water_kabsch", (ref_rel_sol - ref_cent).clone())

#     # ------------------------------------------------------------------
#     # PBC helpers
#     # ------------------------------------------------------------------
#     def _L_tensor(self, x: torch.Tensor, L: float) -> torch.Tensor:
#         return torch.as_tensor(L, device=x.device, dtype=x.dtype)

#     def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
#         L_t = self._L_tensor(dx, L)
#         return dx - L_t * torch.round(dx / L_t)

#     def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
#         L_t = self._L_tensor(x, self.L)
#         return x - L_t * torch.floor(x / L_t)

#     def make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
#         # x_sol: (B,3,3)
#         x0 = x_sol[:, 0:1, :]
#         d = self.mic(x_sol - x0, self.L)
#         return x0 + d

#     def make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
#         # x_w: (B,3,3) with ordering [O,H1,H2]
#         O = x_w[:, 0:1, :]
#         d = self.mic(x_w - O, self.L)
#         return O + d

#     # ------------------------------------------------------------------
#     # Solute frame helpers
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _build_solute_frame_single(sol_rel: torch.Tensor) -> torch.Tensor:
#         """
#         sol_rel: (3,3), whole solute relative to atom 0
#         returns R: (3,3), columns are canonical basis vectors in lab coordinates
#         """
#         v1 = sol_rel[1]
#         v2 = sol_rel[2]

#         e1 = v1 / torch.norm(v1).clamp_min(1e-12)
#         tmp = v2 - torch.dot(v2, e1) * e1
#         e2 = tmp / torch.norm(tmp).clamp_min(1e-12)
#         e3 = torch.cross(e1, e2, dim=-1)

#         # Optional sign convention for stability
#         if e3[2] < 0:
#             e2 = -e2
#             e3 = -e3

#         return torch.stack([e1, e2, e3], dim=1)

#     @staticmethod
#     def _build_solute_frame_batch(sol_rel: torch.Tensor) -> torch.Tensor:
#         """
#         sol_rel: (B,3,3), whole solute relative to atom 0
#         returns R: (B,3,3), columns are canonical basis vectors in lab coordinates
#         """
#         v1 = sol_rel[:, 1, :]
#         v2 = sol_rel[:, 2, :]

#         e1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp_min(1e-12)
#         tmp = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
#         e2 = tmp / torch.linalg.norm(tmp, dim=-1, keepdim=True).clamp_min(1e-12)
#         e3 = torch.cross(e1, e2, dim=-1)

#         # Optional sign convention for stability
#         flip = e3[:, 2] < 0
#         if flip.any():
#             e2 = e2.clone()
#             e3 = e3.clone()
#             e2[flip] = -e2[flip]
#             e3[flip] = -e3[flip]

#         return torch.stack([e1, e2, e3], dim=2)

#     @staticmethod
#     def angle_abc(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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

#     # ------------------------------------------------------------------
#     # SO(3) maps
#     # ------------------------------------------------------------------
#     def rotvec_to_rotmat(self, w: torch.Tensor) -> torch.Tensor:
#         """
#         w: (B,3) rotation vector (axis * angle)
#         returns R: (B,3,3)
#         """
#         B = w.shape[0]
#         theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
#         k = w / theta

#         kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
#         K = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
#         K[:, 0, 1] = -kz
#         K[:, 0, 2] = ky
#         K[:, 1, 0] = kz
#         K[:, 1, 2] = -kx
#         K[:, 2, 0] = -ky
#         K[:, 2, 1] = kx

#         I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
#         ct = torch.cos(theta).view(B, 1, 1)
#         st = torch.sin(theta).view(B, 1, 1)
#         return I + st * K + (1.0 - ct) * (K @ K)

#     def rotmat_to_rotvec(self, R: torch.Tensor) -> torch.Tensor:
#         """
#         R: (B,3,3)
#         returns w: (B,3)
#         """
#         B = R.shape[0]
#         trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
#         cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
#         theta = torch.acos(cos_theta)

#         w = torch.zeros((B, 3), device=R.device, dtype=R.dtype)

#         small = theta < 1e-6
#         if small.any():
#             Rt = R[small].transpose(1, 2)
#             A = 0.5 * (R[small] - Rt)
#             w[small, 0] = A[:, 2, 1]
#             w[small, 1] = A[:, 0, 2]
#             w[small, 2] = A[:, 1, 0]

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
#         theta = torch.linalg.norm(w, dim=1).clamp_min(1e-12)
#         half = 0.5 * theta
#         return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

#     # ------------------------------------------------------------------
#     # Kabsch
#     # ------------------------------------------------------------------
#     @staticmethod
#     def kabsch_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         source: (B,3,3) centered
#         target: (B,3,3) centered
#         returns R such that source @ R ≈ target
#         """
#         C = source.transpose(1, 2) @ target
#         U, _, Vh = torch.linalg.svd(C)
#         V = Vh.transpose(1, 2)
#         Ut = U.transpose(1, 2)

#         det = torch.det(V @ Ut)
#         D = torch.eye(3, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(source.shape[0], 1, 1)
#         D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

#         return V @ D @ Ut

#     # ------------------------------------------------------------------
#     # nf API
#     # ------------------------------------------------------------------
#     def inverse(self, x: torch.Tensor):
#         """
#         x: (B, 3*N) flattened Cartesian
#         returns:
#           i: (B, 3 + 6*n_waters)
#           logdet_xi: (B,)
#         """
#         B = x.shape[0]
#         x = x.view(B, self.n_atoms, 3)

#         # Make solute whole and center on atom 0
#         sol = self.make_whole_solute(x[:, :3, :])   # (B,3,3)
#         origin = sol[:, 0:1, :]
#         x_rel = self.mic(x - origin, self.L)        # all atoms relative to atom0 under MIC
        

#         # Solute internal shape only
#         v1 = x_rel[:, 1, :]  # atom0 -> atom1
#         v2 = x_rel[:, 2, :]  # atom0 -> atom2

#         r1 = torch.linalg.norm(v1, dim=-1, keepdim=True)
#         r2 = torch.linalg.norm(v2, dim=-1, keepdim=True)
#         theta = self.angle_abc(
#             x_rel[:, 1, :],
#             x_rel[:, 0, :],
#             x_rel[:, 2, :],
#         ).unsqueeze(1)

#         pieces = [r1, r2, theta]
#         logdet = torch.zeros((B,), device=x.device, dtype=x.dtype)

#         # Build instantaneous solute frame
#         Rsol = self._build_solute_frame_batch(x_rel[:, :3, :])  # (B,3,3)

#         # Reference water in canonical solute frame
#         Yref = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

#         for k in range(self.n_waters):
#             s = self.n_solute + 3 * k
#             w = x_rel[:, s:s + 3, :]          # (B,3,3) [O,H1,H2], still in lab frame rel to atom0
#             # w = self.make_whole_water(w)      # make H whole wrt O
#             O0 = w[:, 0:1, :]
#             w = O0 + self.mic(w - O0, self.L)

#             # O position in instantaneous solute frame
#             O_lab = w[:, 0, :]
#             O = torch.einsum("bij,bj->bi", Rsol.transpose(1, 2), O_lab)

#             # Water geometry relative to O, then rotate into solute frame
#             w_rel = w - w[:, 0:1, :]
#             w_rel_sol = torch.einsum("bij,bnj->bni", Rsol.transpose(1, 2), w_rel)

#             # Infer rigid orientation in solute frame
#             w_cent = w_rel_sol - w_rel_sol.mean(dim=1, keepdim=True)
#             Rw = self.kabsch_rotation(Yref, w_cent)
#             omega = self.rotmat_to_rotvec(Rw)

#             pieces += [O, omega]

#             # Optional Haar measure correction
#             # logdet = logdet + self.so3_logdet_exp(omega)

#         i = torch.cat(pieces, dim=1)
#         return i, logdet

#     def forward(self, i: torch.Tensor):
#         """
#         i: (B, 3 + 6*n_waters)
#         returns:
#           x: (B, 3*N) flattened Cartesian in [0, L)
#           logdet_ix: (B,)
#         """
#         B = i.shape[0]
#         assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

#         # Unpack solute shape
#         r1 = i[:, 0:1].clamp_min(1e-6)
#         r2 = i[:, 1:2].clamp_min(1e-6)
#         theta = i[:, 2:3].clamp(1e-3, math.pi - 1e-3)

#         center = 0.5 * self.L
#         x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

#         # Canonical solute:
#         # atom0 at center
#         # atom1 on +x
#         # atom2 in xy-plane
#         x[:, 0, 0] = center
#         x[:, 0, 1] = center
#         x[:, 0, 2] = center

#         x[:, 1, 0] = center + r1[:, 0]
#         x[:, 1, 1] = center
#         x[:, 1, 2] = center

#         x[:, 2, 0] = center + r2[:, 0] * torch.cos(theta[:, 0])
#         x[:, 2, 1] = center + r2[:, 0] * torch.sin(theta[:, 0])
#         x[:, 2, 2] = center

#         # Waters, already represented in this canonical solute frame
#         idx = 3
#         logdet = torch.zeros((B,), device=i.device, dtype=i.dtype)

#         H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
#         H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

#         solute0_abs = x[:, 0, :]

#         for k in range(self.n_waters):
#             O = i[:, idx:idx + 3]
#             omega = i[:, idx + 3:idx + 6]
#             idx += 6

#             R = self.rotvec_to_rotmat(omega)

#             O_abs = solute0_abs + O
#             x[:, self.n_solute + 3 * k + 0, :] = O_abs

#             H1 = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
#             H2 = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

#             x[:, self.n_solute + 3 * k + 1, :] = O_abs + H1
#             x[:, self.n_solute + 3 * k + 2, :] = O_abs + H2

#             # Optional Haar measure correction
#             # logdet = logdet + self.so3_logdet_exp(omega)

#         x = self.wrap_0L(x)
#         return x.view(B, -1), logdet

class PBCFixedSoluteTransform(nf.flows.Flow):
    """
    Fully fixed solute:
      - translation fixed (center of box)
      - rotation fixed (canonical)
      - bond lengths fixed
      - bond angle fixed

    Internal coords:
      - waters only: (O position 3 + omega 3) per water
    """

    def __init__(self, L: float, system=None, transform_data=None, internal_dim=None):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.internal_dim = internal_dim

        assert transform_data is not None and transform_data.shape[0] == 1

        self.n_atoms = transform_data.shape[1] // 3
        self.n_solute = 3
        self.n_atoms_per_mol = 3
        self.n_waters = (self.n_atoms - self.n_solute) // self.n_atoms_per_mol

        # -------------------------
        # FIXED SOLUTE GEOMETRY
        # -------------------------
        with torch.no_grad():
            x0 = transform_data.reshape(1, self.n_atoms, 3)[0]

            sol = x0[:3].clone()  # (3,3)

            # make solute whole
            sol_rel = self.mic(sol.unsqueeze(0) - sol[0].view(1,1,3))[0]
            v1 = sol_rel[1]
            v2 = sol_rel[2]

            # build canonical frame
            e1 = v1 / torch.norm(v1)
            tmp = v2 - torch.dot(v2, e1) * e1
            e2 = tmp / torch.norm(tmp)
            e3 = torch.cross(e1, e2, dim=-1)

            R = torch.stack([e1, e2, e3], dim=1)  # (3,3)

            # rotate into canonical frame
            sol_fixed = (R.T @ sol_rel.T).T

            # self.solute_fixed = sol_fixed.clone()  # store canonical geometry

            # -------------------------
            # WATER REFERENCE
            # -------------------------
            w0 = x0[self.n_solute:self.n_solute+3]
            w0_rel = self.mic(w0.unsqueeze(0) - w0[0].view(1,1,3))[0]
            w0_rel_canon = (R.T @ w0_rel.T).T

            # self.ref_H1 = w0_rel[1].clone()
            # self.ref_H2 = w0_rel[2].clone()
            self.register_buffer("solute_fixed", sol_fixed.clone())
            self.register_buffer("ref_H1", w0_rel_canon[1].clone())
            self.register_buffer("ref_H2", w0_rel_canon[2].clone())

            # ref_cent = w0_rel.mean(dim=0, keepdim=True)
            # self.ref_water_kabsch = (w0_rel - ref_cent).clone()
            ref_cent = w0_rel_canon.mean(dim=0, keepdim=True)
            self.register_buffer("ref_water_kabsch", (w0_rel_canon - ref_cent).clone())

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: (B,3,3) current points (centered)
        Y: (B,3,3) reference points (centered)
        Returns R such that source @ R ≈ target
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
    def mic(self, dx):
        L = self.L
        return dx - L * torch.round(dx / L)

    def wrap(self, x):
        L = self.L
        return x - L * torch.floor(x / L)

    def rotvec_to_rotmat(self, w):
        theta = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
        k = w / theta

        K = torch.zeros((w.shape[0], 3, 3), device=w.device, dtype=w.dtype)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] =  k[:, 1]
        K[:, 1, 0] =  k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] =  k[:, 0]

        I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0)
        ct = torch.cos(theta).view(-1,1,1)
        st = torch.sin(theta).view(-1,1,1)

        return I + st*K + (1-ct)*(K@K)

    def rotmat_to_rotvec(self, R):
        trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        theta = torch.acos(((trace-1)/2).clamp(-1,1))

        w = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

        small = theta < 1e-6
        if small.any():
            A = (R[small] - R[small].transpose(1,2))*0.5
            w[small,0] = A[:,2,1]
            w[small,1] = A[:,0,2]
            w[small,2] = A[:,1,0]

        big = ~small
        if big.any():
            th = theta[big]
            denom = (2*torch.sin(th)).clamp_min(1e-12)
            w[big,0] = (R[big,2,1]-R[big,1,2]) / denom
            w[big,1] = (R[big,0,2]-R[big,2,0]) / denom
            w[big,2] = (R[big,1,0]-R[big,0,1]) / denom
            w[big] *= th.unsqueeze(1)

        return w

    # -------------------------
    # inverse
    # -------------------------
    def inverse(self, x):
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # --- make solute whole relative to atom 0
        sol = x[:, :3]                                  # (B,3,3)
        sol0 = sol[:, 0:1]
        sol_rel = self.mic(sol - sol0)                  # whole solute, relative to atom 0

        v1 = sol_rel[:, 1]
        v2 = sol_rel[:, 2]

        e1 = v1 / torch.linalg.norm(v1, dim=1, keepdim=True).clamp_min(1e-12)
        tmp = v2 - (e1 * v2).sum(dim=1, keepdim=True) * e1
        e2 = tmp / torch.linalg.norm(tmp, dim=1, keepdim=True).clamp_min(1e-12)
        e3 = torch.cross(e1, e2, dim=-1)

        # columns are basis vectors in lab frame
        Rcur = torch.stack([e1, e2, e3], dim=2)         # (B,3,3)

        # all atoms relative to solute atom 0, minimum imaged
        x_rel = self.mic(x - sol0)                      # (B,N,3)

        pieces = []
        Yref = self.ref_water_kabsch.to(x.device).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = x_rel[:, s:s+3]                         # lab-frame, relative to sol0

            # oxygen position in canonical solute frame
            O_lab = w[:, 0]                             # (B,3)
            O = torch.einsum("bij,bj->bi", Rcur.transpose(1, 2), O_lab)

            # water internal orientation: first put water in solute frame
            w_rel = w - w[:, 0:1]                       # relative to O
            w_rel_canon = torch.einsum("bij,bnj->bni", Rcur.transpose(1, 2), w_rel)

            w_cent = w_rel_canon - w_rel_canon.mean(dim=1, keepdim=True)

            # reference -> current-in-canonical rotation
            Rw = self.kabsch_rotation(Yref, w_cent)
            omega = self.rotmat_to_rotvec(Rw)

            pieces += [O, omega]

        i = torch.cat(pieces, dim=1)
        return i, torch.zeros(B, device=x.device, dtype=x.dtype)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, i):
        B = i.shape[0]

        center = 0.5 * self.L

        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # place fixed solute
        sol = self.solute_fixed.to(i.device)
        x[:, :3] = center + sol
        solute0_abs = x[:, 0]          # fixed solute atom 0 position

        idx = 0

        H1_ref = self.ref_H1.to(i.device).view(1,3,1)
        H2_ref = self.ref_H2.to(i.device).view(1,3,1)

        for k in range(self.n_waters):
            O = i[:, idx:idx+3]
            omega = i[:, idx+3:idx+6]
            idx += 6

            R = self.rotvec_to_rotmat(omega)

            
            O_abs = solute0_abs + O
            x[:, self.n_solute + 3*k] = O_abs

            H1 = (R @ H1_ref.expand(B,3,1)).squeeze(-1)
            H2 = (R @ H2_ref.expand(B,3,1)).squeeze(-1)

            x[:, self.n_solute + 3*k + 1] = O_abs + H1
            x[:, self.n_solute + 3*k + 2] = O_abs + H2

        x = self.wrap(x)
        return x.view(B,-1), torch.zeros(B, device=i.device)


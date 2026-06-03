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

class Global3PointRadialRotvecTransform(_BaseFlow):
    """
    PBC coordinate transform for:
      - one flexible triatomic solute (atoms 0,1,2)
      - rigid water solvents in OHH ordering

    Internal coordinates:
      solute:
        v1 = MIC(x1 - x0)                          -> 3
        v2 = MIC(x2 - x0)                          -> 3

      each water:
        z_O  = radial latent of MIC(O - x0),        -> 3
               z_O = rho * (O_rel / ||O_rel||)
               where rho = (||O_rel|| - r_min) / r_scale
               requires ||O_rel|| >= r_min
        omega = rotation vector (axis * angle)      -> 3

    Total internal dim:
        6 + 6 * n_waters

    Properties
    ----------
    - removes only global translation (atom 0 anchored in forward)
    - keeps global rotation (lab-frame encoding, appropriate for fixed-box PBC)
    - preserves water ordering (permutation symmetry can be handled by the flow)
    - locally invertible almost everywhere
    - not globally bijective because:
        * rotvec has SO(3) branch cut at angle pi
        * MIC has measure-zero half-box ambiguities
        * oxygen positions with ||O_rel|| < r_min have no valid preimage

    logdet accounts for:
        * radial oxygen map (z_O <-> O_rel)
        * SO(3) exponential map (omega <-> R)
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

        Requires ||O_rel|| >= r_min.  Configurations that violate this have no
        valid preimage; a warning is emitted and rho is clamped to eps.
        """
        r = torch.linalg.norm(O_rel, dim=-1, keepdim=True).clamp_min(self.eps)   # (B,1)
        u = O_rel / r                                                             # (B,3)

        # invert r = r_min + r_scale * rho
        rho_raw = (r - self.oxygen_r_min) / self.oxygen_r_scale                  # (B,1)
        if (rho_raw < 0).any():
            import warnings
            n_bad = int((rho_raw < 0).sum().item())
            warnings.warn(
                f"oxygen_cartesian_to_latent: {n_bad} oxygen(s) closer than "
                f"r_min={self.oxygen_r_min} to solute atom 0. "
                "These configurations have no valid preimage; logdet will be incorrect.",
                RuntimeWarning,
                stacklevel=2,
            )
        rho = rho_raw.clamp_min(self.eps)                                        # (B,1)
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

    def so3_logdet_exp(self, omega: torch.Tensor) -> torch.Tensor:
        """
        log |det J| for the SO(3) exponential map omega -> R(omega).

        For the change-of-variables from rotation vector omega in R^3 to the
        orientation of a rigid body on SO(3), the Jacobian of the exponential
        map contributes:

            log|det J_exp(omega)| = 2 * log(sin(theta/2) / (theta/2))

        where theta = ||omega||.  This equals 0 at theta=0 and is negative for
        theta > 0, reflecting volume contraction of the exponential map.

        omega: (B,3)
        returns: (B,)
        """
        theta = torch.linalg.norm(omega, dim=-1).clamp_min(1e-12)  # (B,)
        half = 0.5 * theta
        return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

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

            # SO(3) log-map Jacobian: inverse of the exp-map logdet
            logdet = logdet - self.so3_logdet_exp(omega)

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

            # SO(3) exp-map Jacobian
            logdet = logdet + self.so3_logdet_exp(omega)

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


class LabFrameTorusTransform(_BaseFlow):
    """
    Lab-frame torus coordinate transform for:
      - one flexible triatomic solute  (atoms 0,1,2)
      - rigid OHH water solvents
      - cubic PBC box of edge length L

    Designed for small boxes (~2 solvation shells) where the solute orientation
    relative to the box is physically meaningful and must not be removed.
    The torus correctly represents the periodic topology of oxygen positions.

    Internal coordinates
    --------------------
    Layout: [v1(3), v2(3), tau_1(3), omega_1(3), ..., tau_W(3), omega_W(3)]
    Total:  6 + 6 * n_waters

      v1, v2   : MIC bond vectors in lab frame (no canonical rotation)   R^3 each
      tau_k    : torus angle of oxygen k,  tau = (2pi/L)*MIC(O_k - x0)  T^3
      omega_k  : rotation vector of water k orientation in lab frame     R^3

    The torus angles tau live in [-pi, pi)^3 and must be handled by a
    circular/torus-aware flow. omega lives in R^3 and is compatible with
    any standard flow.

    Change-of-variables logdet
    --------------------------
    inverse (Cartesian -> internal):
      +3*log(2pi/L)          per water  [torus scaling, constant]
      -so3_logdet_exp(omega) per water  [SO(3) log-map Jacobian]

    forward (internal -> Cartesian):
      +3*log(L/2pi)          per water  [torus scaling, constant]
      +so3_logdet_exp(omega) per water  [SO(3) exp-map Jacobian]
    """

    def __init__(
        self,
        L: float,
        transform_data: torch.Tensor,
        internal_dim: int = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.L   = float(L)
        self.eps = float(eps)

        assert transform_data is not None and transform_data.shape[0] == 1
        n_atoms = transform_data.shape[1] // 3
        self.n_atoms           = n_atoms
        self.n_solute          = 3
        self.n_atoms_per_water = 3
        self.n_waters = (n_atoms - self.n_solute) // self.n_atoms_per_water
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        expected_dim = 6 + 6 * self.n_waters
        self.internal_dim = expected_dim if internal_dim is None else internal_dim
        assert self.internal_dim == expected_dim, (self.internal_dim, expected_dim)

        # Build reference rigid-water geometry from first water in transform_data
        with torch.no_grad():
            x0  = transform_data.reshape(1, n_atoms, 3)[0].clone()
            w0  = x0[self.n_solute:self.n_solute + 3]   # (3,3) OHH
            O   = w0[0:1]
            w_rel = w0 - O                               # O at origin

            ref_H1 = w_rel[1].clone()                   # (3,)
            ref_H2 = w_rel[2].clone()                   # (3,)

            # Reference body frame with columns [e1, e2, n]
            ref_w_batch = torch.stack([
                torch.zeros_like(ref_H1),
                ref_H1,
                ref_H2,
            ], dim=0).unsqueeze(0)                      # (1,3,3)
            ref_frame = self._water_frame(ref_w_batch)  # (1,3,3)

        self.register_buffer("ref_H1",    ref_H1)        # (3,)
        self.register_buffer("ref_H2",    ref_H2)        # (3,)
        self.register_buffer("ref_frame", ref_frame[0])  # (3,3)

    # ------------------------------------------------------------------
    # PBC helpers
    # ------------------------------------------------------------------
    def _L(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L(dx)
        return dx - L * torch.round(dx / L)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L(x)
        return x - L * torch.floor(x / L)

    def _make_whole(self, x: torch.Tensor) -> torch.Tensor:
        """Make molecule whole relative to its first atom.  x: (B, M, 3)"""
        anchor = x[:, 0:1, :]
        return anchor + self.mic(x - anchor)

    # ------------------------------------------------------------------
    # Water body-frame helper
    # ------------------------------------------------------------------
    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

    def _water_frame(self, w_rel: torch.Tensor) -> torch.Tensor:
        """
        Build deterministic labeled body frame from relative water coords.

        w_rel : (B,3,3)  O at index 0 (origin), H1 at 1, H2 at 2
        Returns F : (B,3,3) with columns [e1, e2, n]

          e1 = normalize(H1)
          n  = normalize(H1 x H2)
          e2 = n x e1
        """
        u1 = w_rel[:, 1, :]
        u2 = w_rel[:, 2, :]
        e1 = self._normalize(u1)
        n  = self._normalize(torch.cross(u1, u2, dim=-1))
        e2 = self._normalize(torch.cross(n, e1, dim=-1))
        return torch.stack([e1, e2, n], dim=-1)  # (B,3,3)

    # ------------------------------------------------------------------
    # SO(3) exp-map Jacobian
    # ------------------------------------------------------------------
    def _so3_logdet_exp(self, omega: torch.Tensor) -> torch.Tensor:
        """
        log|det J| for the SO(3) exponential map  omega -> R.

        = 2 * log( sin(theta/2) / (theta/2) ),  theta = ||omega||

        Equals 0 at theta=0 and is negative for theta > 0 (volume
        contraction of the exp map).

        omega : (B,3)
        returns : (B,)
        """
        theta = torch.linalg.norm(omega, dim=-1).clamp_min(1e-12)
        half  = 0.5 * theta
        return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

    # ------------------------------------------------------------------
    # inverse:  Cartesian  ->  internal
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        Map flattened Cartesian coordinates to internal coordinates.

        x : (B, 3*N)
        Returns
        -------
        i       : (B, 6 + 6*n_waters)
        logdet  : (B,)   log|di/dx|
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Translation gauge: anchor solute atom 0, all coords relative to it
        sol    = self._make_whole(x[:, :3, :])   # (B,3,3)
        origin = sol[:, 0:1, :]                  # (B,1,3)
        x_rel  = self.mic(x - origin)            # (B,N,3)

        # Solute: bond vectors in lab frame (orientation is kept, not removed)
        v1 = x_rel[:, 1, :]   # (B,3)
        v2 = x_rel[:, 2, :]   # (B,3)
        pieces = [v1, v2]

        # Constant torus logdet per water: log|(2pi/L)^3|
        log_torus = 3.0 * math.log(2.0 * math.pi / self.L)
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)

        F_ref = self.ref_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            # Make water whole relative to its own oxygen
            w = self._make_whole(x_rel[:, s:s + 3, :])   # (B,3,3)

            # Oxygen torus angle in lab frame
            O_rel = w[:, 0, :]                            # (B,3)
            tau   = wrap_to_pi((2.0 * math.pi / self.L) * O_rel)  # (B,3)
            logdet = logdet + log_torus

            # Water orientation: frame -> rotation matrix -> rotation vector
            w_body = w - w[:, 0:1, :]                     # O at origin (B,3,3)
            F_cur  = self._water_frame(w_body)            # (B,3,3)
            R      = F_cur @ F_ref.transpose(-1, -2)      # R: F_cur = R @ F_ref
            R      = project_to_so3(R)
            omega  = so3_log(R)                           # (B,3)
            logdet = logdet - self._so3_logdet_exp(omega)

            pieces += [tau, omega]

        i = torch.cat(pieces, dim=1)   # (B, 6+6W)
        return i, logdet

    # ------------------------------------------------------------------
    # forward:  internal  ->  Cartesian
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        Map internal coordinates to flattened Cartesian coordinates.

        i : (B, 6 + 6*n_waters)
        Returns
        -------
        x       : (B, 3*N)  Cartesian positions in [0, L)
        logdet  : (B,)      log|dx/di|
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        v1 = i[:, 0:3]
        v2 = i[:, 3:6]

        center = torch.full((B, 3), 0.5 * self.L, device=i.device, dtype=i.dtype)
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # Solute atom 0 at box center, atoms 1 and 2 displaced by bond vectors
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        # Constant torus logdet per water: log|(L/2pi)^3|
        log_torus = 3.0 * math.log(self.L / (2.0 * math.pi))
        logdet = torch.zeros(B, device=i.device, dtype=i.dtype)

        H1_ref = self.ref_H1.to(i.device, i.dtype)
        H2_ref = self.ref_H2.to(i.device, i.dtype)

        idx = 6
        for k in range(self.n_waters):
            tau   = i[:, idx:idx + 3]       # (B,3) torus angles in [-pi, pi)
            omega = i[:, idx + 3:idx + 6]   # (B,3) rotation vector
            idx  += 6

            # tau -> oxygen displacement in lab frame
            O_rel = (self.L / (2.0 * math.pi)) * tau   # (B,3) in [-L/2, L/2)
            logdet = logdet + log_torus

            # omega -> rotation matrix -> H positions
            R      = so3_exp(omega)                     # (B,3,3)
            logdet = logdet + self._so3_logdet_exp(omega)

            O_abs = center + O_rel
            H1    = torch.einsum("bij,j->bi", R, H1_ref)
            H2    = torch.einsum("bij,j->bi", R, H2_ref)

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1
            x[:, s + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet


class SFICTorusTransform(_BaseFlow):
    """
    Solute-Frame Internal Coordinate transform with torus oxygen parameterisation.

    WARNING — ISOTROPIC BULK ONLY
    ==============================
    This transform removes 3 global rotation DOF by expressing all coordinates
    in a canonical solute frame.  This is physically valid ONLY when the
    Boltzmann distribution is approximately rotation-invariant, i.e.:
      * large cubic box (many solvation shells, bulk-like interior)
      * homogeneous isotropic solvent (no interface, membrane, or surface)
      * freely-rotating solute (no external orienting field)

    For small boxes (~2 solvation shells), systems near interfaces, or any
    case where the solute orientation relative to the box is energetically
    meaningful, use LabFrameTorusTransform instead.

    System requirements
    -------------------
    - One flexible triatomic solute  (atoms 0,1,2)
    - Rigid OHH water molecules
    - Cubic PBC box of edge length L
    - Simulation MUST use rigid water (rigid_water=True in OpenMM); flexible
      H fluctuations are silently discarded by inverse(), causing energy errors.

    Internal coordinates
    --------------------
    Layout: [z_r1(1), z_r2(1), z_theta(1), tau_1(3), omega_1(3), ..., tau_W(3), omega_W(3)]
    Total:  3 + 6 * n_waters

      z_r1    = log(|x1 - x0|)               unconstrained real
      z_r2    = log(|x2 - x0|)               unconstrained real
      z_theta = logit(angle(x1,x0,x2) / pi)  unconstrained real
      tau_k   = (2pi/L) * O_canon_k           torus angle T^3  (for circular flow)
      omega_k = rotation vector of water k    unconstrained R^3

    Oxygen positions are expressed in the canonical solute frame and stored as
    torus angles in [-pi, pi)^3.  This correctly captures the periodic topology
    of the PBC box and avoids the arctanh boundary blow-up of SFICTransform.
    omega lives on flat R^3 and is handled by any standard flow layer.

    Comparison with SFICTransform
    ------------------------------
    Identical except for the oxygen parameterisation:
      SFICTransform:    z_O  = arctanh(2*O_canon/L)   bounded, boundary diverges
      SFICTorusTransform: tau = (2pi/L)*O_canon        periodic, constant logdet

    Change-of-variables logdet
    --------------------------
    inverse (Cartesian -> internal):
      solute:    -(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
      per water: +3*log(2pi/L)           [torus, constant]
                 -so3_logdet_exp(omega)  [SO(3) log-map]

    forward (internal -> Cartesian):
      solute:    +(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
      per water: +3*log(L/2pi)           [torus, constant]
                 +so3_logdet_exp(omega)  [SO(3) exp-map]
    """

    def __init__(
        self,
        L: float,
        transform_data: torch.Tensor,
        internal_dim: int = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.L   = float(L)
        self.eps = float(eps)

        assert transform_data is not None and transform_data.shape[0] == 1
        n_atoms = transform_data.shape[1] // 3
        self.n_atoms          = n_atoms
        self.n_solute         = 3
        self.n_atoms_per_mol  = 3
        self.n_waters = (n_atoms - self.n_solute) // self.n_atoms_per_mol
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        expected_dim = 3 + 6 * self.n_waters
        self.internal_dim = expected_dim if internal_dim is None else internal_dim
        assert self.internal_dim == expected_dim, (self.internal_dim, expected_dim)

        # Reference water geometry in canonical solute frame from transform_data
        with torch.no_grad():
            x0 = transform_data.reshape(1, n_atoms, 3)[0].clone()

            # Canonical solute frame from reference configuration
            sol_ref = x0[:self.n_solute, :]                   # (3,3)
            v1_ref  = sol_ref[1] - sol_ref[0]
            v2_ref  = sol_ref[2] - sol_ref[0]
            e1_ref  = v1_ref / v1_ref.norm().clamp_min(1e-12)
            v2_perp = v2_ref - (v2_ref * e1_ref).sum() * e1_ref
            e2_ref  = v2_perp / v2_perp.norm().clamp_min(1e-12)
            e3_ref  = torch.cross(e1_ref, e2_ref, dim=0)
            # Rows are canonical basis vectors: R_sol_ref @ v_lab = v_canonical
            R_sol_ref = torch.stack([e1_ref, e2_ref, e3_ref], dim=0)  # (3,3)

            # Reference water in canonical solute frame
            s = self.n_solute
            ref_water_abs   = x0[s:s + 3, :]                  # (3,3) [O,H1,H2]
            ref_water_rel   = ref_water_abs - sol_ref[0:1, :]  # relative to atom 0
            ref_water_canon = (R_sol_ref @ ref_water_rel.T).T  # (3,3) canonical

            ref_O_canon    = ref_water_canon[0:1, :]
            ref_rel_canon  = ref_water_canon - ref_O_canon     # O at origin

            ref_H1 = ref_rel_canon[1].clone()                  # (3,)
            ref_H2 = ref_rel_canon[2].clone()                  # (3,)

            # Centred reference geometry for Kabsch alignment
            ref_cent          = ref_rel_canon.mean(dim=0, keepdim=True)
            ref_water_kabsch  = (ref_rel_canon - ref_cent).clone()  # (3,3)

        self.register_buffer("ref_H1",           ref_H1)
        self.register_buffer("ref_H2",           ref_H2)
        self.register_buffer("ref_water_kabsch", ref_water_kabsch)

    # ------------------------------------------------------------------
    # PBC helpers
    # ------------------------------------------------------------------
    def _L(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.L, device=x.device, dtype=x.dtype)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L(dx)
        return dx - L * torch.round(dx / L)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L(x)
        return x - L * torch.floor(x / L)

    def _make_whole_solute(self, x_sol: torch.Tensor) -> torch.Tensor:
        x0 = x_sol[:, 0:1, :]
        return x0 + self.mic(x_sol - x0)

    def _make_whole_water(self, x_w: torch.Tensor) -> torch.Tensor:
        O = x_w[:, 0:1, :]
        return O + self.mic(x_w - O)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _angle_abc(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Angle ABC in radians.  a,b,c: (B,3) -> (B,)"""
        ba = a - b
        bc = c - b
        ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.acos((ba * bc).sum(dim=-1).clamp(-1.0, 1.0))

    def _compute_R_sol(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Per-sample rotation matrix from lab frame to canonical solute frame.
        Canonical: v1 along +x, projection of v2 perpendicular to v1 along +y.
        v1, v2: (B,3)  ->  R_sol: (B,3,3),  R_sol @ v_lab = v_canonical
        """
        eps = 1e-12
        e1      = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(eps)
        v2_perp = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
        e2      = v2_perp / v2_perp.norm(dim=-1, keepdim=True).clamp_min(eps)
        e3      = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)   # (B,3,3) rows = canonical basis

    @staticmethod
    def _kabsch_rotation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Optimal rotation R minimising ||X R - Y||_F  (orthogonal Procrustes).
        X, Y: (B,3,3) centred point clouds (3 points x 3 coords per row).
        Returns R: (B,3,3) such that X @ R ≈ Y.

        Derivation: maximise Tr(R^T C) with C = X^T Y = U S Vh.
        The solution is R = U D Vh where D corrects for reflections.

        Note: the formula R = V D U^T (used in SFICTransform.kabsch_rotation)
        solves the TRANSPOSED problem (Y @ R ≈ X) and returns R_omega^T here.
        This implementation solves the correct problem (X @ R ≈ Y) and returns R_omega.
        """
        C   = X.transpose(1, 2) @ Y           # C = X^T Y
        U, _, Vh = torch.linalg.svd(C)        # C = U S Vh
        det = torch.det(U @ Vh)               # det(R) without correction
        D   = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return U @ D @ Vh                      # R = U D Vh  =>  X @ R ≈ Y

    # ------------------------------------------------------------------
    # SO(3) exp-map Jacobian
    # ------------------------------------------------------------------
    def _so3_logdet_exp(self, omega: torch.Tensor) -> torch.Tensor:
        """
        log|det J| for SO(3) exp-map  omega -> R.
        = 2*log(sin(theta/2)/(theta/2)),  theta = ||omega||.
        omega: (B,3) -> (B,)
        """
        theta = torch.linalg.norm(omega, dim=-1).clamp_min(1e-12)
        half  = 0.5 * theta
        return 2.0 * (torch.log(torch.sin(half).clamp_min(1e-12)) - torch.log(half))

    # ------------------------------------------------------------------
    # inverse:  Cartesian  ->  internal
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x : (B, 3*N) flattened Cartesian
        Returns i: (B, 3+6W), logdet: (B,)   [log|di/dx|]
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        # Anchor at solute atom 0
        sol    = self._make_whole_solute(x[:, :3, :])   # (B,3,3)
        origin = sol[:, 0:1, :]                          # (B,1,3)
        x_rel  = self.mic(x - origin)                   # (B,N,3)

        v1 = x_rel[:, 1, :]   # (B,3)
        v2 = x_rel[:, 2, :]   # (B,3)

        # Solute shape in polar-like coords (rotation-invariant)
        r1    = torch.linalg.norm(v1, dim=-1)
        r2    = torch.linalg.norm(v2, dim=-1)
        theta = self._angle_abc(x_rel[:, 1, :], x_rel[:, 0, :], x_rel[:, 2, :])

        eps  = 1e-7
        r1_c = r1.clamp_min(eps)
        r2_c = r2.clamp_min(eps)
        th_c = theta.clamp(eps, math.pi - eps)

        z_r1    = torch.log(r1_c)
        z_r2    = torch.log(r2_c)
        z_theta = torch.log(th_c / (math.pi - th_c))   # logit(theta/pi)

        pieces = [z_r1.unsqueeze(1), z_r2.unsqueeze(1), z_theta.unsqueeze(1)]

        # Solute logdet: spherical Jacobian + log-distance + logit-angle reparam
        logdet = -(
            3.0 * torch.log(r1_c) +
            3.0 * torch.log(r2_c) +
            torch.log(th_c.sin().clamp_min(eps)) +
            torch.log(th_c) +
            torch.log((math.pi - th_c).clamp_min(eps)) -
            math.log(math.pi)
        )

        # Rotation mapping lab frame -> canonical solute frame
        R_sol  = self._compute_R_sol(v1, v2)                          # (B,3,3)

        log_torus = 3.0 * math.log(2.0 * math.pi / self.L)            # constant
        Yref   = self.ref_water_kabsch.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = self._make_whole_water(x_rel[:, s:s + 3, :])          # (B,3,3)

            # Transform water to canonical solute frame
            w_canon = torch.bmm(R_sol, w.transpose(1, 2)).transpose(1, 2)  # (B,3,3)

            # Oxygen torus angle in canonical frame
            O_canon = w_canon[:, 0, :]                                 # (B,3)
            tau     = wrap_to_pi((2.0 * math.pi / self.L) * O_canon)  # (B,3)
            logdet  = logdet + log_torus

            # Water orientation via Kabsch in canonical frame
            w_rel_canon  = w_canon - w_canon[:, 0:1, :]
            w_cent_canon = w_rel_canon - w_rel_canon.mean(dim=1, keepdim=True)
            R_water      = self._kabsch_rotation(w_cent_canon, Yref)   # (B,3,3)
            omega        = so3_log(R_water)                            # (B,3)
            logdet       = logdet - self._so3_logdet_exp(omega)

            pieces += [tau, omega]

        i = torch.cat(pieces, dim=1)   # (B, 3+6W)
        return i, logdet

    # ------------------------------------------------------------------
    # forward:  internal  ->  Cartesian
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        i : (B, 3+6W)
        Returns x: (B, 3*N) Cartesian in [0,L), logdet: (B,)  [log|dx/di|]

        Note: generated samples always have the solute in canonical orientation
        (atom 1 on +x axis, atom 2 in the xy-plane).  For an isotropic bulk
        system all orientations are equivalent by assumption.
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        z_r1    = i[:, 0]
        z_r2    = i[:, 1]
        z_theta = i[:, 2]

        r1    = torch.exp(z_r1)                     # > 0
        r2    = torch.exp(z_r2)                     # > 0
        theta = math.pi * torch.sigmoid(z_theta)   # in (0, pi)

        center = 0.5 * self.L
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)

        # Solute in canonical orientation: atom 0 at center, atom 1 on +x, atom 2 in xy
        x[:, 0, :] = center
        x[:, 1, 0] = center + r1
        x[:, 1, 1] = center
        x[:, 1, 2] = center
        x[:, 2, 0] = center + r2 * torch.cos(theta)
        x[:, 2, 1] = center + r2 * torch.sin(theta)
        x[:, 2, 2] = center

        eps = 1e-7
        logdet = (
            3.0 * torch.log(r1) +
            3.0 * torch.log(r2) +
            torch.log(theta.sin().clamp_min(eps)) +
            torch.log(theta.clamp_min(eps)) +
            torch.log((math.pi - theta).clamp_min(eps)) -
            math.log(math.pi)
        )

        log_torus = 3.0 * math.log(self.L / (2.0 * math.pi))          # constant

        H1_ref = self.ref_H1.to(i.device, i.dtype).view(1, 3, 1)
        H2_ref = self.ref_H2.to(i.device, i.dtype).view(1, 3, 1)

        idx = 3
        for k in range(self.n_waters):
            tau   = i[:, idx:idx + 3]       # (B,3) torus angles in [-pi, pi)
            omega = i[:, idx + 3:idx + 6]   # (B,3) rotation vector
            idx  += 6

            # tau -> oxygen position in canonical frame (= lab frame here)
            O_canon = (self.L / (2.0 * math.pi)) * tau   # (B,3) in [-L/2, L/2)
            logdet  = logdet + log_torus

            # omega -> rotation matrix -> H positions
            R      = so3_exp(omega)                       # (B,3,3)
            logdet = logdet + self._so3_logdet_exp(omega)

            O_abs = center + O_canon
            H1    = (R @ H1_ref.expand(B, 3, 1)).squeeze(-1)
            H2    = (R @ H2_ref.expand(B, 3, 1)).squeeze(-1)

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1
            x[:, s + 2, :] = O_abs + H2

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


import torch
import math
from torch import nn

try:
    import normflows as nf
    _BaseFlow = nf.flows.Flow
except Exception:
    _BaseFlow = nn.Module

from fab.utils.manifold_utils import (
    wrap_to_pi,
    so3_exp, so3_log,
)

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
        canonical_sorting: bool = False,
    ):
        super().__init__()
        self.L                 = float(L)
        self.eps               = float(eps)
        self.canonical_sorting = canonical_sorting

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
        The solution is R = V D U^T = Vh^T D U^T where D corrects for reflections.
        """
        C   = X.transpose(1, 2) @ Y           # C = X^T Y
        U, _, Vh = torch.linalg.svd(C)        # C = U S Vh
        det = torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2))  # det(V U^T)
        D   = torch.eye(3, device=X.device, dtype=X.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        return Vh.transpose(1, 2) @ D @ U.transpose(1, 2)  # R = V D U^T  =>  X @ R ≈ Y

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

        if self.canonical_sorting:
            # Sort waters by O-to-solute distance in the lab frame.
            # Rotation to canonical solute frame preserves norms, so this is
            # equivalent to sorting by ||O_canon||.  Permutation has |det|=1.
            waters = x_rel[:, self.n_solute:, :].view(B, self.n_waters, 3, 3)
            dists  = torch.linalg.norm(waters[:, :, 0, :], dim=-1)  # (B,W)
            sort_idx = torch.argsort(dists, dim=-1)                  # (B,W)
            waters_sorted = waters[
                torch.arange(B, device=x.device).unsqueeze(-1), sort_idx
            ]
            x_rel = torch.cat(
                [x_rel[:, :self.n_solute, :], waters_sorted.view(B, 3 * self.n_waters, 3)],
                dim=1,
            )

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


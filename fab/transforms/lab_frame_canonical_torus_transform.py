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
    mrp_exp, mrp_log, mrp_logdet_exp,
    project_to_so3,
)

class LabFrameCanonicalTorusTransform(_BaseFlow):
    """
    Lab-frame transform with canonical-solute-frame water coordinates.

    This transform combines the strengths of LabFrameGeometricTorusTransform
    and SFICTorusTransform:

      LabFrameGeometricTorusTransform  -- keeps explicit p_sol (physical orientation),
                                          valid for small boxes; torus angles in lab frame
      SFICTorusTransform               -- torus angles in canonical solute frame,
                                          solvation structure expressed relative to solute;
                                          drops p_sol (assumes isotropic box)

    Here we keep p_sol AND express water coordinates in the canonical solute frame.
    This makes the solvation-shell structure approximately rotationally invariant
    within the latent space (the flow only needs to learn radial/angular structure
    relative to the solute), while still encoding the physical orientation so the
    transform is valid for anisotropic or small-box systems.

    System
    ------
    - One flexible triatomic solute (atoms 0, 1, 2)
    - Rigid OHH water molecules
    - Cubic PBC box of edge length L

    Internal coordinates
    --------------------
    Layout: [z_r1(1), z_r2(1), z_theta(1), p_sol(3),
             tau_1(3), p_1(3), ..., tau_W(3), p_W(3)]
    Total:  6 + 6 * n_waters

      z_r1    = log(|x1 - x0|)                     unconstrained R
      z_r2    = log(|x2 - x0|)                     unconstrained R
      z_theta = logit(angle(x1, x0, x2) / pi)      unconstrained R
      p_sol   = MRP of solute frame in lab          R^3  (physical orientation)
      tau_k   = (2pi/L) * MIC(O_k - x0) in canon   T^3  (requires circular flow)
      p_k     = MRP of water k orient. in canon     R^3

    R_sol convention: COLUMNS are the canonical basis vectors in the lab frame.
      - R_sol @ v_canon = v_lab   (canonical -> lab)
      - R_sol.T @ v_lab = v_canon (lab -> canonical)
    This matches _build_solute_frame and is consistent with mrp_exp/mrp_log.

    Jacobian approximation
    ----------------------
    The exact Jacobian has off-diagonal blocks d(tau_k, p_k)/d(v_sol) because
    R_sol depends on v_sol = (v1, v2).  These blocks are ignored, giving a
    block-diagonal approximation identical to that of SFICTorusTransform.
    LabFrameGeometricTorusTransform has an exactly block-diagonal Jacobian
    (its torus angles are in the lab frame, not the solute frame), so use that
    if an exact density is required.

    Change-of-variables logdet
    --------------------------
    inverse (Cartesian -> internal):
      solute: -(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
              - mrp_logdet_exp(p_sol)
      per water: +3*log(2pi/L)        [torus, constant]
                 - mrp_logdet_exp(p_k)

    forward (internal -> Cartesian):
      solute: +(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
              + mrp_logdet_exp(p_sol)
      per water: +3*log(L/2pi)
                 + mrp_logdet_exp(p_k)
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
        self.n_atoms           = n_atoms
        self.n_solute          = 3
        self.n_atoms_per_water = 3
        self.n_waters = (n_atoms - self.n_solute) // self.n_atoms_per_water
        assert self.n_solute + 3 * self.n_waters == self.n_atoms

        expected_dim = 6 + 6 * self.n_waters
        self.internal_dim = expected_dim if internal_dim is None else internal_dim
        assert self.internal_dim == expected_dim, (self.internal_dim, expected_dim)

        # Build reference buffers in canonical solute frame
        with torch.no_grad():
            x0 = transform_data.reshape(1, n_atoms, 3)[0].clone()   # (N, 3)

            # Canonical solute frame of the reference configuration.
            # R_sol_ref columns = canonical basis in lab:
            #   R_sol_ref.T @ v_lab = v_canon
            sol_ref   = x0[:self.n_solute, :]                        # (3, 3)
            v1_ref    = sol_ref[1] - sol_ref[0]
            v2_ref    = sol_ref[2] - sol_ref[0]
            e1_ref    = v1_ref / v1_ref.norm().clamp_min(1e-12)
            v2_perp   = v2_ref - (v2_ref * e1_ref).sum() * e1_ref
            e2_ref    = v2_perp / v2_perp.norm().clamp_min(1e-12)
            e3_ref    = torch.cross(e1_ref, e2_ref, dim=0)
            R_sol_ref = torch.stack([e1_ref, e2_ref, e3_ref], dim=-1)  # (3,3) columns

            # Reference water expressed in canonical frame
            s = self.n_solute
            ref_water_abs   = x0[s:s + 3, :]                        # (3, 3)
            ref_water_rel   = ref_water_abs - sol_ref[0:1, :]        # relative to atom 0
            # v_canon = R_sol.T @ v_lab
            ref_water_canon = (R_sol_ref.T @ ref_water_rel.T).T      # (3, 3)

            ref_O_canon   = ref_water_canon[0:1, :]
            ref_rel_canon = ref_water_canon - ref_O_canon             # O at origin

            ref_H1_canon = ref_rel_canon[1].clone()                  # (3,)
            ref_H2_canon = ref_rel_canon[2].clone()                  # (3,)

            # Water body frame in canonical frame (used as reference in inverse)
            ref_w_batch     = ref_rel_canon.unsqueeze(0)             # (1, 3, 3)
            ref_frame_canon = self._water_frame(ref_w_batch)         # (1, 3, 3)

        self.register_buffer("ref_H1_canon",    ref_H1_canon)        # (3,)
        self.register_buffer("ref_H2_canon",    ref_H2_canon)        # (3,)
        self.register_buffer("ref_frame_canon", ref_frame_canon[0])  # (3, 3)

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
    # Frame helpers
    # ------------------------------------------------------------------
    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

    def _water_frame(self, w_rel: torch.Tensor) -> torch.Tensor:
        """
        Deterministic labeled body frame for a water molecule.
        w_rel: (B, 3, 3) with O at index 0 (origin), H1 at 1, H2 at 2.
        Returns F: (B, 3, 3) with columns [e1, e2, n]
          e1 = normalize(H1),  n = normalize(H1 x H2),  e2 = n x e1
        """
        u1 = w_rel[:, 1, :]
        u2 = w_rel[:, 2, :]
        e1 = self._normalize(u1)
        n  = self._normalize(torch.cross(u1, u2, dim=-1))
        e2 = self._normalize(torch.cross(n, e1, dim=-1))
        return torch.stack([e1, e2, n], dim=-1)                       # (B, 3, 3)

    def _build_solute_frame(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Build R_sol with columns [e1, e2, e3] = canonical basis in lab frame.
          e1 = v1 / |v1|
          e2 = GramSchmidt(v2 perp e1) / |...|
          e3 = e1 x e2

        R_sol @ v_canon = v_lab   (canonical -> lab)
        R_sol.T @ v_lab = v_canon (lab -> canonical)

        v1, v2: (B, 3)  ->  R_sol: (B, 3, 3)
        """
        e1      = self._normalize(v1)
        v2_perp = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
        e2      = self._normalize(v2_perp)
        e3      = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1)                      # (B, 3, 3) columns

    # ------------------------------------------------------------------
    # inverse:  Cartesian  ->  internal
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N) -> i: (B, 6+6W), logdet: (B,)   [log|di/dx|]

        Approximation: ignores d(tau_k, p_k)/d(v_sol) coupling in the Jacobian.
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        origin = x[:, 0:1, :]
        x_rel  = self.mic(x - origin)                                 # (B, N, 3)

        # ---- Solute shape ----
        v1 = x_rel[:, 1, :]                                           # (B, 3)
        v2 = x_rel[:, 2, :]                                           # (B, 3)

        eps = 1e-7
        r1  = torch.linalg.norm(v1, dim=-1).clamp_min(eps)
        r2  = torch.linalg.norm(v2, dim=-1).clamp_min(eps)
        cos_th = (
            (v1 / r1.unsqueeze(-1)) * (v2 / r2.unsqueeze(-1))
        ).sum(-1).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(cos_th)

        z_r1    = torch.log(r1)
        z_r2    = torch.log(r2)
        z_theta = torch.log(theta / (math.pi - theta))                # logit(theta/pi)

        # ---- Solute orientation ----
        # R_sol columns = canonical basis in lab; mrp_log/exp uses column convention
        R_sol = project_to_so3(self._build_solute_frame(v1, v2))      # (B, 3, 3)
        p_sol = mrp_log(R_sol)                                        # (B, 3)

        # Solute logdet (block-diagonal block only)
        th_c   = theta.clamp(eps, math.pi - eps)
        logdet = -(
            3.0 * torch.log(r1) +
            3.0 * torch.log(r2) +
            torch.log(th_c.sin().clamp_min(eps)) +
            torch.log(th_c) +
            torch.log((math.pi - th_c).clamp_min(eps)) -
            math.log(math.pi)
        ) - mrp_logdet_exp(p_sol)

        pieces = [z_r1.unsqueeze(1), z_r2.unsqueeze(1), z_theta.unsqueeze(1), p_sol]

        # ---- Optional canonical sorting (by O-solute distance) ----
        if self.canonical_sorting:
            waters   = x_rel[:, self.n_solute:, :].view(B, self.n_waters, 3, 3)
            dists    = torch.linalg.norm(waters[:, :, 0, :], dim=-1)  # (B, W)
            sort_idx = torch.argsort(dists, dim=-1)
            waters_s = waters[torch.arange(B, device=x.device).unsqueeze(-1), sort_idx]
            x_rel = torch.cat(
                [x_rel[:, :self.n_solute, :], waters_s.view(B, 3 * self.n_waters, 3)],
                dim=1,
            )

        # R_sol.T maps lab -> canonical frame
        R_sol_T = R_sol.transpose(1, 2)                               # (B, 3, 3)

        log_torus   = 3.0 * math.log(2.0 * math.pi / self.L)
        F_ref_canon = self.ref_frame_canon.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = self._make_whole(x_rel[:, s:s + 3, :])               # (B, 3, 3) OHH

            # Rotate water coordinates to canonical solute frame
            # w[:, i, :] are row vectors; bmm(R_sol_T, w.T).T rotates each row
            w_canon = torch.bmm(R_sol_T, w.transpose(1, 2)).transpose(1, 2)  # (B, 3, 3)

            # Oxygen torus angle in canonical frame
            O_canon = w_canon[:, 0, :]                                # (B, 3)
            tau     = wrap_to_pi((2.0 * math.pi / self.L) * O_canon) # (B, 3)
            logdet  = logdet + log_torus

            # Water orientation in canonical frame
            w_body_canon = w_canon - w_canon[:, 0:1, :]              # O at origin
            F_cur_canon  = self._water_frame(w_body_canon)            # (B, 3, 3)
            R_water      = project_to_so3(
                F_cur_canon @ F_ref_canon.transpose(-1, -2)
            )                                                         # (B, 3, 3)
            p_k    = mrp_log(R_water)                                # (B, 3)
            logdet = logdet - mrp_logdet_exp(p_k)

            pieces += [tau, p_k]

        return torch.cat(pieces, dim=1), logdet                       # (B, 6+6W), (B,)

    # ------------------------------------------------------------------
    # forward:  internal  ->  Cartesian
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        i: (B, 6+6W) -> x: (B, 3*N) Cartesian in [0, L), logdet: (B,)  [log|dx/di|]

        Atom 0 is placed at the box centre.  Global translation gauge is fixed
        identically to LabFrameGeometricTorusTransform.
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # ---- Solute ----
        z_r1    = i[:, 0]
        z_r2    = i[:, 1]
        z_theta = i[:, 2]
        p_sol   = i[:, 3:6]

        r1    = torch.exp(z_r1)
        r2    = torch.exp(z_r2)
        theta = math.pi * torch.sigmoid(z_theta)

        R_sol = mrp_exp(p_sol)                                        # (B, 3, 3)
        # Solute atom positions in lab frame: v_lab = R_sol @ v_canon
        # In canonical frame: atom 1 = r1*e_x, atom 2 = r2*(cos theta*e_x + sin theta*e_y)
        e1 = R_sol[:, :, 0]                                           # (B, 3) first column
        e2 = R_sol[:, :, 1]                                           # (B, 3) second column
        v1 = r1.unsqueeze(-1) * e1
        v2 = r2.unsqueeze(-1) * (
            torch.cos(theta).unsqueeze(-1) * e1 +
            torch.sin(theta).unsqueeze(-1) * e2
        )

        center = torch.full((B, 3), 0.5 * self.L, device=i.device, dtype=i.dtype)
        x = torch.zeros((B, self.n_atoms, 3), device=i.device, dtype=i.dtype)
        x[:, 0, :] = center
        x[:, 1, :] = center + v1
        x[:, 2, :] = center + v2

        eps    = 1e-7
        th_c   = theta.clamp(eps, math.pi - eps)
        logdet = (
            3.0 * z_r1 +
            3.0 * z_r2 +
            torch.log(th_c.sin().clamp_min(eps)) +
            torch.log(th_c) +
            torch.log((math.pi - th_c).clamp_min(eps)) -
            math.log(math.pi)
        ) + mrp_logdet_exp(p_sol)

        # ---- Waters ----
        log_torus = 3.0 * math.log(self.L / (2.0 * math.pi))
        H1_ref_c  = self.ref_H1_canon.to(i.device, i.dtype)          # (3,)
        H2_ref_c  = self.ref_H2_canon.to(i.device, i.dtype)          # (3,)

        idx = 6
        for k in range(self.n_waters):
            tau = i[:, idx:idx + 3]                                   # (B, 3) torus angles
            p_k = i[:, idx + 3:idx + 6]                               # (B, 3) MRP
            idx += 6

            # Oxygen position in canonical frame
            O_canon = (self.L / (2.0 * math.pi)) * tau               # (B, 3) in [-L/2, L/2)
            logdet  = logdet + log_torus

            # Water orientation in canonical frame -> H positions in canonical frame
            R_water  = mrp_exp(p_k)                                   # (B, 3, 3)
            logdet   = logdet + mrp_logdet_exp(p_k)
            H1_canon = torch.einsum("bij,j->bi", R_water, H1_ref_c)  # (B, 3)
            H2_canon = torch.einsum("bij,j->bi", R_water, H2_ref_c)  # (B, 3)

            # Rotate canonical -> lab frame:  v_lab = R_sol @ v_canon
            O_lab  = torch.bmm(R_sol, O_canon.unsqueeze(-1)).squeeze(-1)   # (B, 3)
            H1_lab = torch.bmm(R_sol, H1_canon.unsqueeze(-1)).squeeze(-1)  # (B, 3)
            H2_lab = torch.bmm(R_sol, H2_canon.unsqueeze(-1)).squeeze(-1)  # (B, 3)

            O_abs = center + O_lab
            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1_lab
            x[:, s + 2, :] = O_abs + H2_lab

        x = self.wrap_0L(x)

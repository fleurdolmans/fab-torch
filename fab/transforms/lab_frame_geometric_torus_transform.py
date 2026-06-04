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

class LabFrameGeometricTorusTransform(_BaseFlow):
    """
    Lab-frame coordinate transform with geometric solute decomposition and
    torus oxygen positions.

    Motivation
    ----------
    LabFrameTorusTransform encodes the solute as raw bond vectors (v1, v2) in
    R^6.  This mixes bond geometry (r1, r2, theta) and orientation (SO(3)) at
    very different scales, making the flow's task harder.  This transform
    separates them explicitly:

      Geometry:    z_r1, z_r2, z_theta  -- log-bond-lengths + logit-angle,
                                           approximately Gaussian, uncorrelated
      Orientation: p_sol in R^3         -- MRP of solute frame, bounded for
                                           all physical configs (singularity
                                           only at theta=2pi, unreachable)

    Water coordinates are identical to LabFrameTorusTransform but use MRP
    instead of axis-angle for orientations.

    Design
    ------
    - For small boxes (~2 solvation shells) where the solute orientation
      relative to the box is physically meaningful.
    - Keeps global translation gauge (atom 0 anchored at box centre in
      forward).  Global rotation is NOT removed -- the 3 orientation DOF
      of p_sol represent the real, physically meaningful orientation.
    - Use canonical_sorting=True to make inverse permutation-invariant over
      water molecules.

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

      z_r1    = log(|x1 - x0|)                unconstrained R
      z_r2    = log(|x2 - x0|)                unconstrained R
      z_theta = logit(angle(x1,x0,x2) / pi)   unconstrained R
      p_sol   = MRP of solute frame in lab     R^3
      tau_k   = (2pi/L)*MIC(O_k - x0)         T^3  (requires circular flow)
      p_k     = MRP of water k orientation     R^3

    Change-of-variables logdet
    --------------------------
    inverse (Cartesian -> internal):
      solute: -(3*log(r1) + 3*log(r2) + log(sin(theta)*theta*(pi-theta)/pi))
              - mrp_logdet_exp(p_sol)
      per water: +3*log(2pi/L)           [torus, constant]
                 - mrp_logdet_exp(p_k)   [MRP log-map Jacobian]

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

        # Reference water body frame built from first water in transform_data
        with torch.no_grad():
            x0    = transform_data.reshape(1, n_atoms, 3)[0].clone()
            w0    = x0[self.n_solute:self.n_solute + 3]   # (3,3) OHH
            w_rel = w0 - w0[0:1]                          # O at origin

            ref_H1 = w_rel[1].clone()                     # (3,)
            ref_H2 = w_rel[2].clone()                     # (3,)

            ref_w_batch = torch.stack(
                [torch.zeros_like(ref_H1), ref_H1, ref_H2], dim=0
            ).unsqueeze(0)                                # (1,3,3)
            ref_frame = self._water_frame(ref_w_batch)    # (1,3,3)

        self.register_buffer("ref_H1",    ref_H1)         # (3,)
        self.register_buffer("ref_H2",    ref_H2)         # (3,)
        self.register_buffer("ref_frame", ref_frame[0])   # (3,3)

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
        """Make molecule whole relative to its first atom.  x: (B,M,3)"""
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
        w_rel: (B,3,3) with O at index 0 (origin), H1 at 1, H2 at 2.
        Returns F: (B,3,3) with columns [e1, e2, n]
          e1 = normalize(H1),  n = normalize(H1 x H2),  e2 = n x e1
        """
        u1 = w_rel[:, 1, :]
        u2 = w_rel[:, 2, :]
        e1 = self._normalize(u1)
        n  = self._normalize(torch.cross(u1, u2, dim=-1))
        e2 = self._normalize(torch.cross(n, e1, dim=-1))
        return torch.stack([e1, e2, n], dim=-1)           # (B,3,3)

    def _build_solute_frame(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Build rotation matrix R_sol with columns [e1, e2, e3] in the lab frame.

          e1 = v1 / |v1|
          e2 = GramSchmidt(v2 perp to e1) / |...|
          e3 = e1 x e2   (right-handed, det = +1)

        In the forward, v1 = r1 * R_sol[:,0] and
        v2 = r2 * (cos(theta)*R_sol[:,0] + sin(theta)*R_sol[:,1]).

        v1, v2: (B,3)  ->  R_sol: (B,3,3)
        """
        e1      = self._normalize(v1)
        v2_perp = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
        e2      = self._normalize(v2_perp)
        e3      = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1)          # (B,3,3) columns

    # ------------------------------------------------------------------
    # inverse:  Cartesian  ->  internal
    # ------------------------------------------------------------------
    def inverse(self, x: torch.Tensor):
        """
        x: (B, 3*N) -> i: (B, 6+6W), logdet: (B,)  [log|di/dx|]
        """
        B = x.shape[0]
        x = x.view(B, self.n_atoms, 3)

        origin = x[:, 0:1, :]                             # (B,1,3)
        x_rel  = self.mic(x - origin)                     # (B,N,3)

        # ---- Solute internal coordinates ----
        v1 = x_rel[:, 1, :]                               # (B,3)
        v2 = x_rel[:, 2, :]                               # (B,3)

        eps = 1e-7
        r1  = torch.linalg.norm(v1, dim=-1).clamp_min(eps)   # (B,)
        r2  = torch.linalg.norm(v2, dim=-1).clamp_min(eps)   # (B,)
        cos_th = ((v1 / r1.unsqueeze(-1)) * (v2 / r2.unsqueeze(-1))).sum(-1).clamp(-1 + eps, 1 - eps)
        theta  = torch.acos(cos_th)                       # (B,) in (0, pi)

        z_r1    = torch.log(r1)                           # (B,)
        z_r2    = torch.log(r2)                           # (B,)
        z_theta = torch.log(theta / (math.pi - theta))   # logit(theta/pi), (B,)

        R_sol = project_to_so3(self._build_solute_frame(v1, v2))  # (B,3,3)
        p_sol = mrp_log(R_sol)                            # (B,3)

        # Solute logdet: Jacobian of (v1,v2) -> (z_r1, z_r2, z_theta, p_sol)
        # = -(3*log r1 + 3*log r2 + log(sin theta * theta * (pi-theta)/pi))
        #   - mrp_logdet_exp(p_sol)
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

        # ---- Waters ----
        if self.canonical_sorting:
            waters   = x_rel[:, self.n_solute:, :].view(B, self.n_waters, 3, 3)
            dists    = torch.linalg.norm(waters[:, :, 0, :], dim=-1)  # (B,W)
            sort_idx = torch.argsort(dists, dim=-1)
            waters_s = waters[torch.arange(B, device=x.device).unsqueeze(-1), sort_idx]
            x_rel    = torch.cat(
                [x_rel[:, :self.n_solute, :], waters_s.view(B, 3 * self.n_waters, 3)],
                dim=1,
            )

        log_torus = 3.0 * math.log(2.0 * math.pi / self.L)
        F_ref = self.ref_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            w = self._make_whole(x_rel[:, s:s + 3, :])   # (B,3,3) OHH

            O_rel  = w[:, 0, :]
            tau    = wrap_to_pi((2.0 * math.pi / self.L) * O_rel)  # (B,3)
            logdet = logdet + log_torus

            w_body = w - w[:, 0:1, :]
            F_cur  = self._water_frame(w_body)
            R      = project_to_so3(F_cur @ F_ref.transpose(-1, -2))
            p_k    = mrp_log(R)                           # (B,3) MRP
            logdet = logdet - mrp_logdet_exp(p_k)

            pieces += [tau, p_k]

        return torch.cat(pieces, dim=1), logdet           # (B, 6+6W), (B,)

    # ------------------------------------------------------------------
    # forward:  internal  ->  Cartesian
    # ------------------------------------------------------------------
    def forward(self, i: torch.Tensor):
        """
        i: (B, 6+6W) -> x: (B, 3*N) Cartesian in [0,L), logdet: (B,)  [log|dx/di|]
        """
        B = i.shape[0]
        assert i.shape[1] == self.internal_dim, (i.shape, self.internal_dim)

        # ---- Solute ----
        z_r1    = i[:, 0]                                 # (B,)
        z_r2    = i[:, 1]                                 # (B,)
        z_theta = i[:, 2]                                 # (B,)
        p_sol   = i[:, 3:6]                               # (B,3)

        r1    = torch.exp(z_r1)                           # (B,)
        r2    = torch.exp(z_r2)                           # (B,)
        theta = math.pi * torch.sigmoid(z_theta)          # (B,) in (0, pi)

        R_sol = mrp_exp(p_sol)                            # (B,3,3)
        # v1 = r1 * e1,  v2 = r2 * (cos(theta)*e1 + sin(theta)*e2)
        e1 = R_sol[:, :, 0]                               # (B,3) first column
        e2 = R_sol[:, :, 1]                               # (B,3) second column
        v1 = r1.unsqueeze(-1) * e1                        # (B,3)
        v2 = r2.unsqueeze(-1) * (
            torch.cos(theta).unsqueeze(-1) * e1 +
            torch.sin(theta).unsqueeze(-1) * e2
        )                                                 # (B,3)

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
        H1_ref = self.ref_H1.to(i.device, i.dtype)
        H2_ref = self.ref_H2.to(i.device, i.dtype)

        idx = 6
        for k in range(self.n_waters):
            tau = i[:, idx:idx + 3]                       # (B,3) torus angles
            p_k = i[:, idx + 3:idx + 6]                   # (B,3) MRP
            idx += 6

            O_rel  = (self.L / (2.0 * math.pi)) * tau    # (B,3) in [-L/2, L/2)
            logdet = logdet + log_torus

            R      = mrp_exp(p_k)                         # (B,3,3)
            logdet = logdet + mrp_logdet_exp(p_k)

            O_abs = center + O_rel
            H1    = torch.einsum("bij,j->bi", R, H1_ref)
            H2    = torch.einsum("bij,j->bi", R, H2_ref)

            s = self.n_solute + 3 * k
            x[:, s + 0, :] = O_abs
            x[:, s + 1, :] = O_abs + H1
            x[:, s + 2, :] = O_abs + H2

        x = self.wrap_0L(x)
        return x.view(B, -1), logdet


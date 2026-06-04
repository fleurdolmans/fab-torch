import torch
import math
from torch import nn

try:
    import normflows as nf
    _BaseFlow = nf.flows.Flow
except Exception:
    _BaseFlow = nn.Module

from fab.utils.manifold_utils import (
    wrap_to_pi, torus_add, torus_sub,
    so3_exp, so3_log,
    mrp_exp, mrp_log, mrp_logdet_exp,
    project_to_so3,
)

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
        canonical_sorting: bool = False,
        use_mrp: bool = False,
    ):
        super().__init__()
        self.L                  = float(L)
        self.eps                = float(eps)
        self.canonical_sorting  = canonical_sorting
        self.use_mrp            = use_mrp

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

        # Translation gauge: anchor solute atom 0, all coords relative to it.
        # _make_whole with anchor=atom0 leaves atom0 unchanged, so origin is
        # simply x[:,0:1,:].  No need to call _make_whole for the solute here.
        origin = x[:, 0:1, :]                   # (B,1,3)
        x_rel  = self.mic(x - origin)            # (B,N,3)

        # Solute: bond vectors in lab frame (orientation is kept, not removed)
        v1 = x_rel[:, 1, :]   # (B,3)
        v2 = x_rel[:, 2, :]   # (B,3)
        pieces = [v1, v2]

        # Constant torus logdet per water: log|(2pi/L)^3|
        log_torus = 3.0 * math.log(2.0 * math.pi / self.L)
        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.canonical_sorting:
            # Sort water molecules by O-to-solute distance so that inverse is
            # invariant to water permutation.  A permutation matrix has |det|=1
            # so the logdet is unaffected.
            waters = x_rel[:, self.n_solute:, :].view(B, self.n_waters, 3, 3)
            dists  = torch.linalg.norm(waters[:, :, 0, :], dim=-1)  # (B,W) O distances
            sort_idx = torch.argsort(dists, dim=-1)                  # (B,W)
            waters_sorted = waters[
                torch.arange(B, device=x.device).unsqueeze(-1), sort_idx
            ]
            x_rel = torch.cat(
                [x_rel[:, :self.n_solute, :], waters_sorted.view(B, 3 * self.n_waters, 3)],
                dim=1,
            )

        F_ref = self.ref_frame.to(x.device, x.dtype).unsqueeze(0).expand(B, 3, 3)

        for k in range(self.n_waters):
            s = self.n_solute + 3 * k
            # Make water whole relative to its own oxygen
            w = self._make_whole(x_rel[:, s:s + 3, :])   # (B,3,3)

            # Oxygen torus angle in lab frame
            O_rel = w[:, 0, :]                            # (B,3)
            tau   = wrap_to_pi((2.0 * math.pi / self.L) * O_rel)  # (B,3)
            logdet = logdet + log_torus

            # Water orientation: frame -> rotation matrix -> rotation/MRP vector
            w_body = w - w[:, 0:1, :]                     # O at origin (B,3,3)
            F_cur  = self._water_frame(w_body)            # (B,3,3)
            R      = F_cur @ F_ref.transpose(-1, -2)      # R: F_cur = R @ F_ref
            R      = project_to_so3(R)
            if self.use_mrp:
                omega  = mrp_log(R)                       # (B,3) MRP vector
                logdet = logdet - mrp_logdet_exp(omega)
            else:
                omega  = so3_log(R)                       # (B,3) axis-angle
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
            if self.use_mrp:
                R      = mrp_exp(omega)                 # (B,3,3) from MRP vector
                logdet = logdet + mrp_logdet_exp(omega)
            else:
                R      = so3_exp(omega)                 # (B,3,3) from axis-angle
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


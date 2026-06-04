import torch
import math
from torch import nn

try:
    import normflows as nf
    _BaseFlow = nf.flows.Flow
except Exception:
    _BaseFlow = nn.Module

from fab.utils.manifold_utils import wrap_to_pi, ManifoldState

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
        canonical_sorting: bool = False,
    ):
        super().__init__()
        self.L = float(L)
        self.system = system
        self.transform_data = transform_data
        self.eps = eps
        self.canonical_sorting = canonical_sorting

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

        if self.canonical_sorting:
            # Sort water molecules by O-to-solute distance.  Permutation has
            # |det|=1, so logdet is unaffected.
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



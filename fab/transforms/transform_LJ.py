import math
import torch
from torch import nn
import normflows as nf

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)



class SolventOnlyTransform(nn.Module):
    """
    Internal coordinates = flattened solvent Cartesian coordinates only.

    forward(internal):
        solvent-only flat coords -> full flat Cartesian coords
        by inserting fixed solute coordinates in front

    inverse(full):
        full flat Cartesian coords -> solvent-only flat coords
        by removing the fixed solute coordinates

    Assumes atom order:
        [solute atoms..., solvent atoms...]
    """

    def __init__(self, solute_positions_nm, n_solute: int, n_solvent: int):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")

        if solute_positions_nm.shape[0] != self.n_solute:
            raise ValueError(
                f"solute_positions_nm has {solute_positions_nm.shape[0]} entries, expected {self.n_solute}"
            )

        self.register_buffer(
            "solute_flat",
            solute_positions_nm.reshape(1, 3 * self.n_solute),
        )

        self.internal_dim = 3 * self.n_solvent
        self.cartesian_dim = 3 * (self.n_solute + self.n_solvent)

    def forward(self, z: torch.Tensor):
        """
        z: (B, 3*n_solvent)
        returns:
            x_full: (B, 3*(n_solute+n_solvent))
            logdet: (B,) = 0
        """
        if z.ndim != 2 or z.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected z shape (B, {self.internal_dim}), got {tuple(z.shape)}"
            )

        B = z.shape[0]
        sol = self.solute_flat.to(device=z.device, dtype=z.dtype).expand(B, -1)
        x_full = torch.cat([sol, z], dim=-1)
        logdet = torch.zeros(B, device=z.device, dtype=z.dtype)
        return x_full, logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 3*(n_solute+n_solvent))
        returns:
            z: (B, 3*n_solvent)
            logdet: (B,) = 0
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        z = x_full[:, 3 * self.n_solute :]
        logdet = torch.zeros(x_full.shape[0], device=x_full.device, dtype=x_full.dtype)
        return z, logdet

class SoluteCenteredSolventTransform(nn.Module):
    """
    Translation-reduced transform for fixed-solute LJ systems.

    Internal coordinates:
        solvent positions relative to a fixed solute reference point,
        flattened as (B, 3 * n_solvent)

    forward(internal):
        relative solvent coords -> full Cartesian coords
        by reinserting fixed solute coordinates and adding the reference origin

    inverse(full):
        full Cartesian coords -> relative solvent coords
        by subtracting the fixed reference point and dropping solute coords

    Notes:
      - assumes solute coordinates are fixed
      - uses minimum-image relative coordinates in [-L/2, L/2)
      - exact inverse as long as you commit to the principal MIC branch
    """

    def __init__(
        self,
        solute_positions_nm,
        n_solute: int,
        n_solvent: int,
        box_length_nm: float,
        reference: str = "solute_centroid",   # or "first_solute"
    ):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)
        self.tail_bound = 0.5 * float(box_length_nm)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")
        if solute_positions_nm.shape[0] != self.n_solute:
            raise ValueError(
                f"solute_positions_nm has {solute_positions_nm.shape[0]} rows, expected {self.n_solute}"
            )

        if reference == "solute_centroid":
            ref = solute_positions_nm.mean(dim=0)
        elif reference == "first_solute":
            ref = solute_positions_nm[0]
        else:
            raise ValueError(f"Unknown reference={reference}")

        self.register_buffer("solute_positions", solute_positions_nm)
        self.register_buffer("reference_point", ref.reshape(1, 3))

        self.internal_dim = 3 * self.n_solvent
        self.cartesian_dim = 3 * (self.n_solute + self.n_solvent)

    def _L(self, x: torch.Tensor):
        return torch.as_tensor(self.box_length_nm, device=x.device, dtype=x.dtype)

    def wrap_0L(self, x: torch.Tensor) -> torch.Tensor:
        L = self._L(x)
        return x - L * torch.floor(x / L)

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = self._L(dx)
        return dx - L * torch.round(dx / L)

    def forward(self, z: torch.Tensor):
        """
        z: (B, 3*n_solvent), solvent rel coords in principal MIC branch
        returns:
            x_full: (B, 3*(n_solute+n_solvent))
            logdet: zeros
        """
        if z.ndim != 2 or z.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected z shape (B, {self.internal_dim}), got {tuple(z.shape)}"
            )

        B = z.shape[0]
        z_rel = z.view(B, self.n_solvent, 3)

        ref = self.reference_point.to(device=z.device, dtype=z.dtype)         # (1,3)
        sol = self.solute_positions.to(device=z.device, dtype=z.dtype)        # (n_solute,3)

        sol = sol.unsqueeze(0).expand(B, -1, -1)                              # (B,n_solute,3)
        solv = ref.unsqueeze(1) + z_rel                                        # (B,n_solvent,3)

        x = torch.cat([sol, solv], dim=1)
        x = self.wrap_0L(x)

        logdet = torch.zeros(B, device=z.device, dtype=z.dtype)
        return x.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 3*(n_solute+n_solvent))
        returns:
            z: (B, 3*n_solvent)
            logdet: zeros
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 3)
        solvent = x[:, self.n_solute:, :]

        ref = self.reference_point.to(device=x.device, dtype=x.dtype)         # (1,3)
        z_rel = self.mic(solvent - ref.unsqueeze(1))

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return z_rel.reshape(B, -1), logdet


class TorusCartesianTransform(nn.Module):
    def __init__(self, solute_positions_nm, n_solute, n_solvent, box_length_nm):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.shape != (self.n_solute, 3):
            raise ValueError(
                f"solute_positions_nm must have shape ({self.n_solute}, 3), "
                f"got {tuple(solute_positions_nm.shape)}"
            )

        self.register_buffer("solute_positions", solute_positions_nm)

        self.internal_dim = 3 * self.n_solvent
        self.cartesian_dim = 3 * (self.n_solute + self.n_solvent)

    def wrap_unit(self, u: torch.Tensor) -> torch.Tensor:
        return u - torch.floor(u)

    def wrap_box(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length_nm, device=x.device, dtype=x.dtype)
        return x - L * torch.floor(x / L)

    def forward(self, u_flat: torch.Tensor):
        """
        u_flat: (B, 3*n_solvent) with torus coords in [0,1)
        returns:
            x_full: (B, 3*(n_solute+n_solvent))
            logdet: (B,)
        """
        if u_flat.ndim != 2 or u_flat.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected u_flat shape (B, {self.internal_dim}), got {tuple(u_flat.shape)}"
            )

        B = u_flat.shape[0]
        u = self.wrap_unit(u_flat.view(B, self.n_solvent, 3))
        solvent = self.box_length_nm * u

        sol = self.solute_positions.to(device=u.device, dtype=u.dtype)
        sol = sol.unsqueeze(0).expand(B, -1, -1)

        x_full = torch.cat([sol, solvent], dim=1)   # (B, n_solute+n_solvent, 3)
        x_full = self.wrap_box(x_full)

        logdet = torch.zeros(B, device=u.device, dtype=u.dtype)
        return x_full.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 3*(n_solute+n_solvent))
        returns:
            u_flat: (B, 3*n_solvent)
            logdet: (B,)
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 3)
        solvent = x[:, self.n_solute:, :]  # drop fixed solute
        u = self.wrap_unit(solvent / self.box_length_nm)

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return u.reshape(B, -1), logdet



# ============================================================
# Utilities
# ============================================================

def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    """Wrap coordinates into [0, 1)."""
    return x - torch.floor(x)


def wrap_centered_unit(x: torch.Tensor) -> torch.Tensor:
    """Wrap coordinates into [-0.5, 0.5)."""
    return x - torch.round(x)


def mic_unit(du: torch.Tensor) -> torch.Tensor:
    """
    Minimum-image displacement in unit-box coordinates.
    Input and output are in box fractions, not nm.
    """
    return du - torch.round(du)


def unit_to_box(u: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=u.device, dtype=u.dtype)
    return L * u


def box_to_unit(x: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    return x / L


# ============================================================
# Transform: fixed-solute + torus solvent coordinates
# ============================================================

class FixedSoluteUnitTorusTransform(nn.Module):
    """
    Internal coordinates:
        solvent positions only, in unit-torus coordinates [0,1),
        flattened as (B, 3 * n_solvent)

    Physical Cartesian coordinates:
        full coordinates [solute..., solvent...] in nm, in [0, L)

    This is the cleaner periodic version for a fixed-solute LJ system.

    forward(u_flat):
        solvent torus coords -> full Cartesian coords

    inverse(x_full):
        full Cartesian coords -> solvent torus coords
    """

    def __init__(
        self,
        solute_positions_nm,
        n_solute: int,
        n_solvent: int,
        box_length_nm: float,
    ):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=torch.float64)
        if solute_positions_nm.shape != (self.n_solute, 3):
            raise ValueError(
                f"solute_positions_nm must have shape ({self.n_solute}, 3), "
                f"got {tuple(solute_positions_nm.shape)}"
            )

        self.register_buffer("solute_positions_nm", solute_positions_nm)

        self.internal_dim = 3 * self.n_solvent
        self.cartesian_dim = 3 * (self.n_solute + self.n_solvent)

    def forward(self, u_flat: torch.Tensor):
        """
        u_flat: (B, 3*n_solvent), solvent coords in [0,1)

        returns:
            x_full: (B, 3*(n_solute+n_solvent)) in nm
            logdet: zeros
        """
        if u_flat.ndim != 2 or u_flat.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected u_flat shape (B, {self.internal_dim}), got {tuple(u_flat.shape)}"
            )

        B = u_flat.shape[0]
        u = wrap_unit(u_flat.view(B, self.n_solvent, 3))
        solvent_nm = unit_to_box(u, self.box_length_nm)

        solute_nm = self.solute_positions_nm.to(device=u.device, dtype=u.dtype)
        solute_nm = solute_nm.unsqueeze(0).expand(B, -1, -1)

        x_full = torch.cat([solute_nm, solvent_nm], dim=1)
        logdet = torch.zeros(B, device=u.device, dtype=u.dtype)
        return x_full.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 3*(n_solute+n_solvent)) in nm

        returns:
            u_flat: (B, 3*n_solvent) in [0,1)
            logdet: zeros
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 3)
        solvent_nm = x[:, self.n_solute:, :]
        u = wrap_unit(box_to_unit(solvent_nm, self.box_length_nm))

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return u.reshape(B, -1), logdet

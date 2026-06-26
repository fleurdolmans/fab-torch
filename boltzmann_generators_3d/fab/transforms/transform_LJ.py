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
            logdet: (B,) = 3*n_solvent*log(L)
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
        log_scale = 3 * self.n_solvent * math.log(self.box_length_nm)
        logdet = torch.full((B,), log_scale, device=u.device, dtype=u.dtype)
        return x_full.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 3*(n_solute+n_solvent)) in nm

        returns:
            u_flat: (B, 3*n_solvent) in [0,1)
            logdet: (B,) = -3*n_solvent*log(L)
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 3)
        solvent_nm = x[:, self.n_solute:, :]
        u = wrap_unit(box_to_unit(solvent_nm, self.box_length_nm))

        log_scale = -3 * self.n_solvent * math.log(self.box_length_nm)
        logdet = torch.full((B,), log_scale, device=x.device, dtype=x.dtype)
        return u.reshape(B, -1), logdet

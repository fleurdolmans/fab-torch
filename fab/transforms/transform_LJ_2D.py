import math
import torch
from torch import nn


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


def mic_unit(du: torch.Tensor) -> torch.Tensor:
    return du - torch.round(du)


def box_to_unit(x: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=x.device, dtype=x.dtype)
    return x / L


def unit_to_box(u: torch.Tensor, box_length: float) -> torch.Tensor:
    L = torch.as_tensor(box_length, device=u.device, dtype=u.dtype)
    return L * u


class FixedSoluteUnitTorusTransform2D(nn.Module):
    """
    2D version of FixedSoluteUnitTorusTransform.

    Internal coordinates:
        solvent positions only, in unit-torus coordinates [0,1),
        flattened as (B, 2 * n_solvent)

    Physical Cartesian coordinates:
        full coordinates [solute..., solvent...] in 2D box coordinates [0, L)

    Assumes particle order:
        particle 0 = solute
        particles 1: = solvent
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        n_solute: int = 1,
    ):
        super().__init__()
        self.n_solute = int(n_solute)
        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)

        self.internal_dim = 2 * self.n_solvent
        self.cartesian_dim = 2 * (self.n_solute + self.n_solvent)

    def forward(self, u_flat: torch.Tensor):
        """
        u_flat: (B, 2*n_solvent), solvent coords in [0,1)

        returns:
            x_full: (B, 2*(n_solute+n_solvent)) in nm
            logdet: zeros
        """
        if u_flat.ndim != 2 or u_flat.shape[1] != self.internal_dim:
            raise ValueError(
                f"Expected u_flat shape (B, {self.internal_dim}), got {tuple(u_flat.shape)}"
            )

        B = u_flat.shape[0]
        u = wrap_unit(u_flat.view(B, self.n_solvent, 2))
        solvent_nm = unit_to_box(u, self.box_length_nm)

        solute_nm = torch.zeros(
            (B, self.n_solute, 2),
            device=u.device,
            dtype=u.dtype,
        )

        x_full = torch.cat([solute_nm, solvent_nm], dim=1)
        logdet = torch.zeros(B, device=u.device, dtype=u.dtype)
        return x_full.reshape(B, -1), logdet

    def inverse(self, x_full: torch.Tensor):
        """
        x_full: (B, 2*(n_solute+n_solvent)) in nm

        returns:
            u_flat: (B, 2*n_solvent) in [0,1)
            logdet: zeros
        """
        if x_full.ndim != 2 or x_full.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"Expected x_full shape (B, {self.cartesian_dim}), got {tuple(x_full.shape)}"
            )

        B = x_full.shape[0]
        x = x_full.view(B, self.n_solute + self.n_solvent, 2)
        solvent_nm = x[:, self.n_solute:, :]
        u = wrap_unit(box_to_unit(solvent_nm, self.box_length_nm))

        logdet = torch.zeros(B, device=x.device, dtype=x.dtype)
        return u.reshape(B, -1), logdet

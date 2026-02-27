import torch
import torch
from torch import nn, Tensor
import numpy as np

class PBCPreprocessTransform(torch.nn.Module):
    def __init__(self, L, n_solute, n_waters, anchor_idx=0, do_center=False):
        super().__init__()
        self.L = float(L)
        self.n_solute = int(n_solute)
        self.n_waters = int(n_waters)
        self.anchor_idx = int(anchor_idx)
        self.do_center = bool(do_center)

        if self.n_solute != 3:
            raise ValueError("Expected triatomic solute (n_solute==3)")

    def _L_tensor(self, L: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(L, device=ref.device, dtype=ref.dtype)

    def wrap(self, x: torch.Tensor, L: float) -> torch.Tensor:
        L = self._L_tensor(L, x)
        return torch.remainder(x, L)

    def mic(self, dx: torch.Tensor, L: float) -> torch.Tensor:
        L = self._L_tensor(L, dx)
        return dx - L * torch.round(dx / L)

    def make_triatomic_solute_whole(self, X: torch.Tensor, L: float, anchor_idx: int = 0) -> torch.Tensor:
        X = self.wrap(X, L).clone()
        a0 = X[:, anchor_idx, :]
        X[:, 1, :] = a0 + self.mic(X[:, 1, :] - a0, L)
        X[:, 2, :] = a0 + self.mic(X[:, 2, :] - a0, L)
        return self.wrap(X, L)

    def make_waters_whole(self, X: torch.Tensor, L: float, n_solute: int, n_waters: int) -> torch.Tensor:
        X = self.wrap(X, L).clone()
        start = n_solute
        for w in range(n_waters):
            i = start + 3 * w
            O  = X[:, i + 0, :]
            H1 = X[:, i + 1, :]
            H2 = X[:, i + 2, :]
            X[:, i + 1, :] = O + self.mic(H1 - O, L)
            X[:, i + 2, :] = O + self.mic(H2 - O, L)
        return self.wrap(X, L)

    def canonicalize_translation(self, X: torch.Tensor, L: float, anchor_idx: int = 0, to_center: bool = True) -> torch.Tensor:
        X = self.wrap(X, L)
        anchor = X[:, anchor_idx, :]
        target = (0.5 * float(L)) if to_center else 0.0
        shift = (target - anchor)[:, None, :]
        return self.wrap(X + shift, L)

    def forward(self, z: torch.Tensor):
        # z: (B, 3N)
        if z.ndim != 2 or (z.shape[1] % 3 != 0):
            raise ValueError(f"Expected z (B,3N), got {tuple(z.shape)}")

        B, D = z.shape
        N = D // 3
        X = z.view(B, N, 3)

        X = self.wrap(X, self.L)
        X = self.make_triatomic_solute_whole(X, self.L, anchor_idx=self.anchor_idx)
        X = self.make_waters_whole(X, self.L, n_solute=self.n_solute, n_waters=self.n_waters)

        if self.do_center:
            # I would keep this False for the energy path initially.
            X = self.canonicalize_translation(X, self.L, anchor_idx=self.anchor_idx, to_center=True)
            # if you center, re-whole once to be safe
            X = self.make_triatomic_solute_whole(X, self.L, anchor_idx=self.anchor_idx)
            X = self.make_waters_whole(X, self.L, n_solute=self.n_solute, n_waters=self.n_waters)

        x = X.view(B, 3 * N)
        log_det = torch.zeros(B, device=z.device, dtype=z.dtype)
        return x, log_det
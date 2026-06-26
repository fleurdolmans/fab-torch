import numpy as np
import torch
import torch.nn as nn
import normflows as nf


class MLP(nn.Module):
    """Simple MLP with ReLU activations and optional dropout."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def water_block_permutation(dim: int, n_prefix: int, block_size: int, seed: int) -> torch.Tensor:
    """
    Keep the first n_prefix dims fixed, permute the remaining dims in whole blocks.
    """
    assert (dim - n_prefix) % block_size == 0
    n_blocks = (dim - n_prefix) // block_size

    rng = np.random.RandomState(seed)
    bperm = rng.permutation(n_blocks)

    perm = list(range(n_prefix))
    for b in bperm:
        start = n_prefix + b * block_size
        perm.extend(range(start, start + block_size))

    return torch.tensor(perm, dtype=torch.long)


class PermuteFixed(nf.flows.Flow):
    """A fixed permutation flow that permutes groups of dims (whole water blocks)."""

    def __init__(self, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("perm", perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        self.register_buffer("inv_perm", inv)

    def forward(self, z):
        z = z[:, self.perm]
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, log_det

    def inverse(self, z):
        z = z[:, self.inv_perm]
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, log_det

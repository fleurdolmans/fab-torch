# pbc_water_preprocess.py
import torch
import torch
from torch import nn, Tensor
import numpy as np


def _stat(name, x):
    x = x.detach()
    print(f"[{name}] shape={tuple(x.shape)} "
          f"min={x.min().item():.6g} max={x.max().item():.6g} "
          f"mean={x.mean().item():.6g} std={x.std().item():.6g}", flush=True)

def check_in_box(X, L, name="X"):
    # X: (B,N,3)
    inside = (X >= 0).all() and (X < L).all()
    print(f"[{name}] inside [0,L): {inside}", flush=True)
    if not inside:
        # show worst violations
        below = (X < 0).sum().item()
        above = (X >= L).sum().item()
        print(f"[{name}] below0={below} aboveL={above}", flush=True)

def water_oh_stats_mic(
    X, L, n_solute, n_waters, name="",
    max_frames=256,
    max_bonds=200000,
):
    """
    X: (B,N,3)
    Computes MIC O-H distances but subsamples to avoid huge tensors.
    """
    B = X.shape[0]
    if B > max_frames:
        idx = torch.randperm(B, device=X.device)[:max_frames]
        X = X[idx]
        B = X.shape[0]

    start = n_solute
    d_all = []
    for w in range(n_waters):
        i = start + 3*w
        O  = X[:, i+0, :]
        H1 = X[:, i+1, :]
        H2 = X[:, i+2, :]
        d1 = torch.norm(mic(H1 - O, L), dim=-1)
        d2 = torch.norm(mic(H2 - O, L), dim=-1)
        d_all.append(d1)
        d_all.append(d2)

    d = torch.cat(d_all, dim=0)  # (2*B*n_waters,)

    # Subsample bonds for quantiles if too many
    if d.numel() > max_bonds:
        j = torch.randperm(d.numel(), device=d.device)[:max_bonds]
        d = d[j]

    d_cpu = d.detach().cpu()
    # Use numpy for quantiles (fast, avoids torch quantile issues)
    arr = d_cpu.numpy()
    p1, p99 = np.quantile(arr, [0.01, 0.99])

    print(
        f"[OH_MIC {name}] n={arr.size} mean={arr.mean():.6f} std={arr.std():.6f} "
        f"p1={p1:.6f} p99={p99:.6f} min={arr.min():.6f} max={arr.max():.6f}",
        flush=True
    )

def water_split_fraction_raw(X, L, n_solute, n_waters, thresh=0.2, name=""):
    """
    Fraction of O-H bonds whose *raw* displacement has any component > thresh*L
    (indicates straddling boundary in raw coords). Not a bug—just diagnostic.
    """
    start = n_solute
    bad = 0
    tot = 0
    for w in range(n_waters):
        i = start + 3*w
        O  = X[:, i+0, :]
        H1 = X[:, i+1, :]
        H2 = X[:, i+2, :]
        for H in (H1, H2):
            dx = H - O
            # raw displacement components
            is_split = (dx.abs() > (thresh * L)).any(dim=-1)  # (B,)
            bad += is_split.sum().item()
            tot += is_split.numel()
    frac = bad / max(1, tot)
    print(f"[SPLIT_RAW {name}] frac={frac:.6f} (thresh={thresh}*L)", flush=True)


def _L_tensor(L: float, ref: Tensor) -> Tensor:
    return torch.as_tensor(L, device=ref.device, dtype=ref.dtype)

def wrap(x: Tensor, L: float) -> Tensor:
    L = _L_tensor(L, x)
    return torch.remainder(x, L)

def mic(dx: Tensor, L: float) -> Tensor:
    L = _L_tensor(L, dx)
    return dx - L * torch.round(dx / L)

def make_waters_whole(X: Tensor, L: float, n_solute: int, n_waters: int) -> Tensor:
    """
    X: (B, N, 3) wrapped or unwrapped; returns wrapped (B, N, 3)
    Assumes water atom order is O H H repeating after the solute atoms.
    """
    X = wrap(X, L).clone()  # clone to avoid modifying views

    start = n_solute
    for w in range(n_waters):
        i = start + 3 * w
        O  = X[:, i+0, :]
        H1 = X[:, i+1, :]
        H2 = X[:, i+2, :]
        X[:, i+1, :] = O + mic(H1 - O, L)
        X[:, i+2, :] = O + mic(H2 - O, L)

    return wrap(X, L)

def canonicalize_translation(X: Tensor, L: float, anchor_idx: int = 0, to_center: bool = True) -> Tensor:
    """
    Shift entire configuration so anchor atom is at box center (or origin).
    X: (B, N, 3)
    """
    X = wrap(X, L)
    anchor = X[:, anchor_idx, :]  # (B,3)
    target = (0.5 * float(L)) if to_center else 0.0
    shift = (target - anchor)[:, None, :]  # (B,1,3)
    return wrap(X + shift, L)

def preprocess_frame_batch(X_flat: Tensor, L: float, n_solute: int, n_waters: int, anchor_idx: int = 0, debug=False) -> Tensor:
    """
    X_flat: (B, 3N) -> returns (B, 3N), wrapped and canonicalized for PBC.
    """
    if X_flat.ndim != 2:
        raise ValueError(f"X_flat must be (B,3N), got {tuple(X_flat.shape)}")
    B, D = X_flat.shape
    if D % 3 != 0:
        raise ValueError(f"Expected D multiple of 3, got D={D}")
    N = D // 3
    X = X_flat.view(B, N, 3)

    if debug:
        _stat("raw X", X)

    X = wrap(X, L)
    if debug:
        check_in_box(X, L, "after wrap")
        water_oh_stats_mic(X, L, n_solute, n_waters, name="after wrap")
        water_split_fraction_raw(X, L, n_solute, n_waters, name="after wrap")

    X = make_waters_whole(X, L, n_solute=n_solute, n_waters=n_waters)
    if debug:
        water_oh_stats_mic(X, L, n_solute, n_waters, name="after whole1")
        water_split_fraction_raw(X, L, n_solute, n_waters, name="after whole1")

    X = canonicalize_translation(X, L, anchor_idx=anchor_idx, to_center=True)
    if debug:
        check_in_box(X, L, "after canonicalize")
        water_oh_stats_mic(X, L, n_solute, n_waters, name="after canonicalize")
        water_split_fraction_raw(X, L, n_solute, n_waters, name="after canonicalize")

    X = make_waters_whole(X, L, n_solute=n_solute, n_waters=n_waters)
    if debug:
        water_oh_stats_mic(X, L, n_solute, n_waters, name="after whole2")
        water_split_fraction_raw(X, L, n_solute, n_waters, name="after whole2")

    X = wrap(X, L)
    if debug:
        check_in_box(X, L, "final")
        water_oh_stats_mic(X, L, n_solute, n_waters, name="final")
        water_split_fraction_raw(X, L, n_solute, n_waters, name="final")

    return X.view(B, 3 * N)

def max_mic_abs_diff(X1: Tensor, X0: Tensor, L: float) -> float:
    if X1.ndim == 2:
        X1 = X1.view(X1.shape[0], -1, 3)
    if X0.ndim == 2:
        X0 = X0.view(X0.shape[0], -1, 3)
    d = mic(X1 - X0, L)
    return float(d.abs().max().item())


class IdentityTransform(nn.Module):
    """Transform that maps z->x with zero logdet."""
    def forward(self, z: Tensor):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, log_det

    __call__ = forward
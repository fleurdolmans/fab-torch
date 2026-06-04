"""
Basic manifold helpers for SO(3), torus T^n, and MRP parameterisations.

Used by the PBC coordinate transforms in fab.transforms.
"""

import math
import torch
from dataclasses import dataclass


# ============================================================
# Torus helpers
# ============================================================

def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def torus_add(theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """T^n group operation in angle coordinates."""
    return wrap_to_pi(theta + delta)


def torus_sub(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Difference on T^n in principal branch."""
    return wrap_to_pi(theta_a - theta_b)


# ============================================================
# SO(3) helpers
# ============================================================

def hat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (..., 3)
    returns skew matrix (..., 3, 3)
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([O,   -wz,  wy], dim=-1),
        torch.stack([wz,   O,  -wx], dim=-1),
        torch.stack([-wy, wx,   O], dim=-1),
    ], dim=-2)
    return K


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Exponential map so(3) -> SO(3)
    w: (..., 3)
    R: (..., 3, 3)
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (..., 1)
    K = hat(w)

    I = torch.eye(3, device=w.device, dtype=w.dtype)
    I = I.expand(w.shape[:-1] + (3, 3))

    theta2 = theta ** 2
    A = torch.where(
        theta > eps,
        torch.sin(theta) / theta,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
    )
    B = torch.where(
        theta > eps,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
    )

    A = A.unsqueeze(-1)
    B = B.unsqueeze(-1)
    return I + A * K + B * (K @ K)


def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Principal logarithm SO(3) -> so(3) ~ R^3
    R: (..., 3, 3)
    w: (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    A = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack([
        A[..., 2, 1],
        A[..., 0, 2],
        A[..., 1, 0],
    ], dim=-1)

    scale = torch.where(
        theta < 1e-6,
        torch.ones_like(theta),
        theta / torch.sin(theta).clamp_min(eps),
    )
    return vee * scale.unsqueeze(-1)


# ============================================================
# Modified Rodrigues Parameters (MRP)
# ============================================================

def mrp_exp(p: torch.Tensor) -> torch.Tensor:
    """
    Modified Rodrigues Parameters (MRP) exponential map: R^3 -> SO(3)

    p = tan(theta/4) * n_hat  ->  R

    Uses the direct rational formula (no sqrt, no trig):
      R = [(1 - 6s + s^2)*I + 8*p*p^T + 4*(1-s)*hat(p)] / (1+s)^2
    where s = ||p||^2.

    The Jacobian log-det is: log(64) - 3*log(1 + ||p||^2).
    Singularity at s -> inf (theta -> 2*pi), outside the principal
    branch [0, pi] and unreachable for near-equilibrium MD configurations.

    p: (..., 3)
    R: (..., 3, 3)
    """
    s    = (p * p).sum(dim=-1)                                         # (...,) = ||p||^2
    denom = (1.0 + s) ** 2                                             # (...,)
    a = (1.0 - 6.0 * s + s ** 2) / denom                              # (...,)
    b = 8.0 / denom                                                    # (...,)
    c = 4.0 * (1.0 - s) / denom                                       # (...,)

    I   = torch.eye(3, device=p.device, dtype=p.dtype).expand(p.shape[:-1] + (3, 3))
    ppT = p.unsqueeze(-1) * p.unsqueeze(-2)                            # (..., 3, 3)
    K   = hat(p)                                                       # (..., 3, 3)

    sh = p.shape[:-1] + (1, 1)
    return a.view(sh) * I + b.view(sh) * ppT + c.view(sh) * K


def mrp_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    MRP log map: SO(3) -> R^3

    R -> p = tan(theta/4) * n_hat

    Formula:  p = vee(R - R^T) / (2 * cos(theta/2) * (1 + cos(theta/2)))

    Well-conditioned for theta in [0, pi).

    R: (..., 3, 3)
    p: (..., 3)
    """
    trace     = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)

    A   = 0.5 * (R - R.mT)
    vee = torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], dim=-1)

    cos_half = torch.sqrt(((1.0 + cos_theta) / 2.0).clamp_min(eps))   # (...,)
    denom    = (2.0 * cos_half * (1.0 + cos_half)).clamp_min(eps)      # (...,)
    return vee / denom.unsqueeze(-1)


def mrp_logdet_exp(p: torch.Tensor) -> torch.Tensor:
    """
    log|det J| for the MRP exponential map p -> R.

    = log(64) - 3*log(1 + ||p||^2)

    Always finite for p in R^3 (singularity at ||p|| -> inf i.e. theta -> 2*pi).
    Equals log(64) at p=0 (identity) and log(8) at ||p||=1 (theta=pi).

    p: (..., 3)
    returns: (...,)
    """
    return math.log(64.0) - 3.0 * torch.log1p((p * p).sum(dim=-1))


# ============================================================
# SO(3) projection
# ============================================================

def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """
    Differentiable polar projection to nearest rotation matrix.
    M: (..., 3, 3)

    Uses R = U @ diag([1, 1, det(U)*det(Vh)]) @ Vh, which enforces det(R)=+1
    without conditional branching or in-place operations.
    """
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U) * torch.linalg.det(Vh)          # (...,)
    ones = torch.ones(M.shape[:-2] + (2,), device=M.device, dtype=M.dtype)
    diag = torch.cat([ones, d.unsqueeze(-1)], dim=-1)        # (..., 3)
    return U @ torch.diag_embed(diag) @ Vh


# ============================================================
# Structured state container
# ============================================================

@dataclass
class ManifoldState:
    # solute bond vectors anchored at atom 0
    solute: torch.Tensor   # (B, 6), [v1(3), v2(3)]
    # torus angles for oxygen positions relative to atom 0
    tau: torch.Tensor      # (B, W, 3), each in [-pi, pi)
    # water orientations as rotation matrices
    R: torch.Tensor        # (B, W, 3, 3)

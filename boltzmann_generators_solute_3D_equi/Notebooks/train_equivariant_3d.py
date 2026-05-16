"""
train_equivariant_3d.py
=======================
Train an E(n)-equivariant Boltzmann generator for the 3D PBC solute-in-LJ-bath
system. Logs training metrics and evaluation plots to Weights & Biases.

Usage
-----
    python train_equivariant_3d.py [options]

Typical Snellius invocation (single GPU node):
    python train_equivariant_3d.py \\
        --n_mc 200000 \\
        --n_epochs_ml 300 --n_epochs_kl 100 \\
        --wandb_project my_project --wandb_entity my_entity \\
        --save_dir /scratch/$USER/equivariant_3d \\
        --mc_cache /scratch/$USER/equivariant_3d/xtraj.npy

The MC trajectory is cached (--mc_cache), so re-running with the same cache
path skips the expensive sampling step.
"""
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json
import argparse
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve().parent          # Notebooks/
_ROOT = _THIS.parent                             # boltzmann_generators_3d_equivariant/
_LIB  = _ROOT / "Library"
_REPO = _ROOT.parent                             # fab-torch/

for _p in [str(_LIB), str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project imports  (after path setup)
# ---------------------------------------------------------------------------
from potentials_pbc import SoluteSimulationPBC3D          # noqa: E402
from build_system   import build_solute_coords_3d_rsa     # noqa: E402
from sampling       import MetropolisSampler               # noqa: E402
from generator      import build_egnn_solute_flow_3d       # noqa: E402


# ---------------------------------------------------------------------------
# g(r) helper
# ---------------------------------------------------------------------------

def compute_gr(traj, l_box, n_bins=30, r_max=None):
    """
    Solute-solvent g(r).

    Parameters
    ----------
    traj  : (N_frames, N, 3)  full system, solute at index 0
    l_box : float             half-box; full box side L = 2*l_box
    """
    L = 2.0 * l_box
    if r_max is None:
        r_max = l_box
    bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(n_bins)
    n_frames, N, _ = traj.shape
    rho = (N - 1) / L**3                               # bulk solvent density
    for frame in traj[::10]:
        delta = frame[1:] - frame[0]                   # solvent - solute
        delta = delta - L * np.round(delta / L)        # minimum-image
        r = np.linalg.norm(delta, axis=-1)
        h, _ = np.histogram(r, bins=bins)
        hist += h
    n_used = len(traj[::10])
    shell_vols = (4 / 3) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    gr = hist / (n_used * rho * shell_vols)             # g(r) = <n(r)> / (ρ V_shell)
    return r_centers, gr


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loader(array: np.ndarray, batch_size: int, shuffle: bool = True):
    tensor = torch.from_numpy(array.astype(np.float32))
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)

def compute_grad_norm(parameters):
    """
    Total L2 norm of gradients before clipping.
    """
    total_sq = 0.0

    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item()
            total_sq += param_norm ** 2

    return total_sq ** 0.5


@torch.no_grad()
def estimate_logq_logp_ess(flow, system, n_samples, device, energy_cap=None):
    """
    Estimate diagnostics from generated samples:

      mean_log_q_total = E_q[log q(x)]
      mean_log_p       = E_q[log p(x)] = E_q[-U(x)] up to a constant
      ESS              = effective sample size of importance weights p/q
      ESS fraction     = ESS / n_samples

    For EGNNEquivariantFlow:
      x, logdet_zx = flow.generator(z)
      log q(x) = log p_z(z) - logdet_zx
    """
    was_training = flow.training
    flow.eval()

    z = flow.prior.sample(n_samples).to(device)
    x, logdet_zx = flow.generator(z)

    log_pz = flow.prior.log_prob(z)
    if log_pz.ndim > 1:
        log_pz = log_pz.view(log_pz.shape[0], -1).sum(dim=1)

    if logdet_zx.ndim > 1:
        logdet_zx = logdet_zx.view(logdet_zx.shape[0], -1).sum(dim=1)

    # This matches your EGNNEquivariantFlow.loss_KL convention.
    log_q_total = log_pz - logdet_zx

    if getattr(flow, "fixed_solute", False):
        x_energy = flow._add_fixed_solute(x)
    else:
        x_energy = x

    u_x = system.get_energy_batch(x_energy)

    if energy_cap is not None:
        u_x = torch.where(
            u_x < energy_cap,
            u_x,
            energy_cap + torch.log1p(u_x - energy_cap),
        )

    log_p = -u_x

    log_w = log_p - log_q_total
    finite = torch.isfinite(log_w)

    if finite.sum() == 0:
        metrics = {
            "mean_log_q_total": float("nan"),
            "mean_log_p": float("nan"),
            "ess": 0.0,
            "ess_fraction": 0.0,
        }
    else:
        log_w_f = log_w[finite]
        log_w_f = log_w_f - torch.max(log_w_f)

        w = torch.exp(log_w_f)
        ess = (w.sum() ** 2) / torch.sum(w ** 2)
        ess_fraction = ess / n_samples

        metrics = {
            "mean_log_q_total": float(log_q_total[finite].mean().item()),
            "mean_log_p": float(log_p[finite].mean().item()),
            "ess": float(ess.item()),
            "ess_fraction": float(ess_fraction.item()),
        }

    if was_training:
        flow.train()

    return metrics

def append_metrics_csv(path: Path, row: dict):
    """
    Append a row of metrics to CSV.

    If new metric columns appear later, rewrite the CSV with the expanded header.
    This is useful when expensive metrics are logged every N epochs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames or []

    new_fields = list(old_fields)
    for key in row.keys():
        if key not in new_fields:
            new_fields.append(key)

    old_rows.append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(old_rows)


def save_args_json(args, path: Path):
    """
    Save command-line args for reproducibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(vars(args), f, indent=2)


def save_checkpoint(flow, optimizer, path: Path, epoch: int, loss: float, args):
    """
    Save full training checkpoint, not only model weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
            "args": vars(args),
        },
        path,
    )


def load_model_weights(flow, checkpoint_path: str, device):
    """
    Load either a full checkpoint dict or a plain state_dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        flow.load_state_dict(ckpt["model_state_dict"])
    else:
        flow.load_state_dict(ckpt)

    print(f"Loaded checkpoint from {checkpoint_path}")

def wrap_pbc_positions(positions, l_box):
    """
    Wrap coordinates into [-l_box, l_box).
    positions: (..., 3)
    """
    L = 2.0 * l_box
    return ((positions + l_box) % L) - l_box


def make_box_edges(l_box):
    """
    Return line segments for a cubic box [-l_box, l_box]^3.
    """
    lo, hi = -l_box, l_box

    corners = np.array([
        [lo, lo, lo],
        [hi, lo, lo],
        [hi, hi, lo],
        [lo, hi, lo],
        [lo, lo, hi],
        [hi, lo, hi],
        [hi, hi, hi],
        [lo, hi, hi],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    x, y, z = [], [], []
    for i, j in edges:
        x += [corners[i, 0], corners[j, 0], None]
        y += [corners[i, 1], corners[j, 1], None]
        z += [corners[i, 2], corners[j, 2], None]

    return x, y, z


def plot_3d_pbc_interactive(
    positions,
    l_box,
    sigma=1.1,
    title="3D PBC configuration",
    wrap=True,
    show=False,
):
    """
    positions : (N, 3), particle 0 is solute.
    l_box     : half box width, so box is [-l_box, l_box]^3.
    sigma     : particle diameter scale for marker size.
    wrap      : if True, wrap coordinates into the central PBC image.
    """
    pos = np.asarray(positions, dtype=float)

    if wrap:
        pos = wrap_pbc_positions(pos, l_box)

    solute = pos[:1]
    solvent = pos[1:]

    # Plotly marker size is in pixels, not data units.
    marker_size = max(4, 18 * sigma / l_box)

    bx, by, bz = make_box_edges(l_box)

    traces = [
        go.Scatter3d(
            x=solvent[:, 0],
            y=solvent[:, 1],
            z=solvent[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color="steelblue",
                opacity=0.75,
            ),
            name="Solvent",
        ),
        go.Scatter3d(
            x=solute[:, 0],
            y=solute[:, 1],
            z=solute[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size * 1.4,
                color="firebrick",
                opacity=1.0,
            ),
            name="Solute",
        ),
        go.Scatter3d(
            x=bx,
            y=by,
            z=bz,
            mode="lines",
            line=dict(color="black", width=4),
            name="PBC box",
        ),
    ]

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        width=750,
        height=750,
        scene=dict(
            xaxis=dict(title="x", range=[-l_box, l_box]),
            yaxis=dict(title="y", range=[-l_box, l_box]),
            zaxis=dict(title="z", range=[-l_box, l_box]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if show:
        fig.show()

    return fig


def train_epoch(flow, loader_x, loader_z, optimizer,
                w_ml, w_kl, energy_cap, device, clip_grad):
    """
    One full training epoch.

    Returns:
        mean_loss, mean_ml_loss, mean_kl_loss, mean_grad_norm
    """
    flow.train()
    losses, ml_losses, kl_losses, grad_norms = [], [], [], []

    if loader_x is not None and loader_z is not None:
        n = max(len(loader_x), len(loader_z))
        pairs = zip(cycle(loader_x) if len(loader_x) < n else loader_x,
                    cycle(loader_z) if len(loader_z) < n else loader_z)
    elif loader_x is not None:
        pairs = ((bx, None) for bx in loader_x)
    else:
        pairs = ((None, bz) for bz in loader_z)

    for batch_x, batch_z in pairs:
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        ml_val = kl_val = 0.0

        if w_ml > 0 and batch_x is not None:
            ml = flow.loss_ML(batch_x[0].to(device))
            loss = loss + w_ml * ml
            ml_val = ml.item()

        if w_kl > 0 and batch_z is not None:
            kl = flow.loss_KL(batch_z[0].to(device), energy_cap=energy_cap)
            loss = loss + w_kl * kl
            kl_val = kl.item()

        loss.backward()

        grad_norm = compute_grad_norm(flow.parameters())

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(flow.parameters(), clip_grad)

        optimizer.step()

        losses.append(loss.item())
        ml_losses.append(ml_val)
        kl_losses.append(kl_val)
        grad_norms.append(grad_norm)

    return (
        float(np.mean(losses)),
        float(np.mean(ml_losses)),
        float(np.mean(kl_losses)),
        float(np.mean(grad_norms)),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train E(n)-equivariant Boltzmann generator (3D PBC solute system)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # System
    g = p.add_argument_group("System")
    g.add_argument("--n_particles", type=int,   default=37)
    g.add_argument("--n_solvent",   type=int,   default=36)
    g.add_argument("--l_box",       type=float, default=2.0,  help="Box half-width; full box = [-l,l]^3")
    g.add_argument("--sigma",       type=float, default=1.1)
    g.add_argument("--epsilon",     type=float, default=1.0)
    g.add_argument("--k_center",    type=float, default=20.0, help="Harmonic centering spring constant")
    g.add_argument("--temperature", type=float, default=1.0)

    # Monte Carlo
    g = p.add_argument_group("Monte Carlo")
    g.add_argument("--n_eq",     type=int,   default=50_000,  help="Equilibration steps")
    g.add_argument("--n_mc",     type=int,   default=200_000, help="Production steps")
    g.add_argument("--mc_sigma", type=float, default=0.04,    help="MC displacement std")
    g.add_argument("--mc_stride",type=int,   default=1,       help="Save every n steps")
    g.add_argument("--mc_seed",  type=int,   default=42)
    g.add_argument("--mc_cache", type=str,   default=None,
                   help="Path to cache MC trajectory (.npy). If exists, skip MC sampling.")

    # Flow architecture
    g = p.add_argument_group("Flow architecture")
    g.add_argument("--n_blocks",    type=int,   default=8)
    g.add_argument("--egnn_hidden", type=int,   default=64)
    g.add_argument("--egnn_layers", type=int,   default=3)
    g.add_argument("--num_bins",    type=int,   default=8,   help="RQ spline bins")
    g.add_argument("--tail_bound",  type=float, default=4.0)
    g.add_argument("--att_heads",   type=int,   default=4)

    # Stage 1 — ML
    g = p.add_argument_group("Stage 1 (ML / forward KL)")
    g.add_argument("--n_epochs_ml",   type=int,   default=300)
    g.add_argument("--lr_ml",         type=float, default=3e-4)
    g.add_argument("--batch_size_ml", type=int,   default=256)
    g.add_argument("--patience_ml",   type=int,   default=40)

    # Stage 2 — KL
    g = p.add_argument_group("Stage 2 (KL / reverse KL fine-tuning)")
    g.add_argument("--n_epochs_kl",   type=int,   default=50)
    g.add_argument("--lr_kl",         type=float, default=5e-6)
    g.add_argument("--batch_size_kl", type=int,   default=256)
    g.add_argument("--w_ml",          type=float, default=0.8, help="ML weight in stage 2")
    g.add_argument("--w_kl",          type=float, default=0.2, help="KL weight in stage 2")
    g.add_argument("--energy_cap",    type=float, default=1000.0)
    g.add_argument("--n_kl_samples",  type=int,   default=50_000)
    g.add_argument("--patience_kl",   type=int,   default=30)

    # Stage 3 — pure KL
    g = p.add_argument_group("Stage 3 (pure KL)")
    g.add_argument("--n_epochs_stage3", type=int, default=100)
    g.add_argument("--lr_stage3", type=float, default=1e-5)
    g.add_argument("--batch_size_stage3", type=int, default=256)
    g.add_argument("--patience_stage3", type=int, default=30)

    # Misc
    g = p.add_argument_group("Misc")
    g.add_argument("--clip_grad",       type=float, default=1.0, help="Gradient clipping (0=off)")
    g.add_argument("--data_stride",     type=int,   default=5,   help="Thinning of MC trajectory")
    g.add_argument("--seed",            type=int,   default=0)
    g.add_argument("--save_dir",        type=str,   default=None)
    g.add_argument("--checkpoint_every",type=int,   default=20,  help="Save checkpoint every N epochs")
    g.add_argument("--eval_samples",    type=int,   default=2000)
    g.add_argument("--metric_samples", type=int, default=2048,
               help="Number of generated samples for logq/logp/ESS estimates")
    g.add_argument("--metric_every", type=int, default=1,
               help="Compute logq/logp/ESS every N epochs")
    g.add_argument("--run_md", action="store_true",
               help="Run MC/MD production only and save trajectory.")
    g.add_argument("--run_stage1", action="store_true",
                help="Run Stage 1: ML training.")
    g.add_argument("--run_stage2", action="store_true",
                help="Run Stage 2: mixed ML + KL training.")
    g.add_argument("--run_stage3", action="store_true",
                help="Run Stage 3: pure KL training.")
    g.add_argument("--stage1_checkpoint", type=str, default=None,
                help="Checkpoint to initialize Stage 2. Defaults to save_dir/stage1_ml/final.pt.")
    g.add_argument("--stage2_checkpoint", type=str, default=None,
                help="Checkpoint to initialize Stage 3. Defaults to save_dir/stage2_ml_kl/final.pt.")
    g.add_argument("--plot_every", type=int, default=20,
               help="Generate RDF, energy, and 3D plots every N epochs during training. 0 disables.")

    # WandB
    g = p.add_argument_group("Weights & Biases")
    g.add_argument("--wandb_project", type=str, default="egnn-boltzmann-3d")
    g.add_argument("--wandb_entity",  type=str, default=None)
    g.add_argument("--wandb_name",    type=str, default=None)
    g.add_argument("--wandb_group",   type=str, default=None)
    g.add_argument("--no_wandb",      action="store_true", help="Disable WandB logging")



    return p.parse_args()


def mic(dx: torch.Tensor, L: float) -> torch.Tensor:
    L_t = torch.as_tensor(float(L), device=dx.device, dtype=dx.dtype)
    return dx - L_t * torch.round(dx / L_t)

def pairwise_min_distances_same_group(X: torch.Tensor, L: float):
    B, N, _ = X.shape
    if N < 2:
        return np.full(B, np.inf)

    dmin = torch.full((B,), float("inf"), device=X.device, dtype=X.dtype)
    for i in range(N):
        for j in range(i + 1, N):
            dij = torch.linalg.norm(mic(X[:, i, :] - X[:, j, :], L), dim=-1)
            dmin = torch.minimum(dmin, dij)
    return dmin.detach().cpu().numpy()

def pairwise_min_distances_cross_group(A: torch.Tensor, B: torch.Tensor, L: float):
    d = mic(A[:, :, None, :] - B[:, None, :, :], L)
    d = torch.linalg.norm(d, dim=-1)
    return d.amin(dim=(1, 2)).detach().cpu().numpy()

# Draw cubic PBC box [-L_BOX, L_BOX]^3
def draw_box_3d(ax, L_BOX, color="gray", linestyle="--", linewidth=1):
    corners = np.array([
        [-L_BOX, -L_BOX, -L_BOX],
        [ L_BOX, -L_BOX, -L_BOX],
        [ L_BOX,  L_BOX, -L_BOX],
        [-L_BOX,  L_BOX, -L_BOX],
        [-L_BOX, -L_BOX,  L_BOX],
        [ L_BOX, -L_BOX,  L_BOX],
        [ L_BOX,  L_BOX,  L_BOX],
        [-L_BOX,  L_BOX,  L_BOX],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    return ax

def position_density(xtraj, L_BOX, title, stride=20):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xtraj[::stride, 1:, 0].ravel(),
        xtraj[::stride, 1:, 1].ravel(),
        xtraj[::stride, 1:, 2].ravel(),
        s=0.3,
        c="steelblue",
        alpha=0.02,
        rasterized=True,
        label="Solvent",
    )

    ax.scatter(
        [0], [0], [0],
        s=60,
        c="orangered",
        depthshade=False,
        zorder=10,
        label="Solute",
    )

    draw_box_3d(ax, L_BOX)

    ax.set_xlim(-L_BOX, L_BOX)
    ax.set_ylim(-L_BOX, L_BOX)
    ax.set_zlim(-L_BOX, L_BOX)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect((1, 1, 1))
    ax.legend(markerscale=3)
    ax.set_title(title)

    plt.tight_layout()
    return fig

@torch.no_grad()
def evaluate_and_log(
    flow,
    system,
    x_samples,
    r_ref,
    gr_ref,
    args,
    device,
    eval_dir,
    use_wandb=False,
    wandb=None,
    prefix="eval",
    make_3d=True,
    wandb_step=None,
):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    was_training = flow.training
    flow.eval()

    z_eval = flow.prior.sample(args.eval_samples).to(device)
    x_gen_flat, _ = flow.generator(z_eval)

    x_gen = x_gen_flat.detach().cpu().numpy().reshape(-1, args.n_solvent, 3)
    x_gen_full = np.concatenate(
        [np.zeros((len(x_gen), 1, 3)), x_gen],
        axis=1,
    )

    r_gen, gr_gen = compute_gr(x_gen_full, args.l_box)

    gr_ref_i = np.interp(r_gen, r_ref, gr_ref)
    gr_l2 = float(np.sqrt(np.mean((gr_gen - gr_ref_i) ** 2)))

    n_eval = min(args.eval_samples, len(x_samples), len(x_gen_full))

    x_ref_full = np.concatenate(
        [
            np.zeros((n_eval, 1, 3)),
            x_samples[:n_eval].reshape(n_eval, args.n_solvent, 3),
        ],
        axis=1,
    )

    x_gen_full_eval = x_gen_full[:n_eval]

    e_ref = system.get_energy_batch(
        torch.tensor(
            x_ref_full.reshape(n_eval, -1),
            dtype=torch.float32,
            device=device,
        )
    ).detach().cpu().numpy()

    e_gen = system.get_energy_batch(
        torch.tensor(
            x_gen_full_eval.reshape(n_eval, -1),
            dtype=torch.float32,
            device=device,
        )
    ).detach().cpu().numpy()

    # fin_ref = e_ref[np.isfinite(e_ref) & (e_ref < 1e4)]
    # fin_gen = e_gen[np.isfinite(e_gen) & (e_gen < 1e4)]
    fin_ref = e_ref[np.isfinite(e_ref)]
    fin_gen = e_gen[np.isfinite(e_gen)]

    finite_frac = len(fin_gen) / n_eval
    overlap_frac = float((e_gen < -1e3).mean()) if len(e_gen) > 0 else float("nan")

    L = 2.0 * args.l_box

    md_full_t = torch.tensor(x_ref_full, dtype=torch.float32, device=device)
    flow_full_t = torch.tensor(x_gen_full_eval, dtype=torch.float32, device=device)

    md_solute = md_full_t[:, :1, :]
    md_solvent = md_full_t[:, 1:, :]

    flow_solute = flow_full_t[:, :1, :]
    flow_solvent = flow_full_t[:, 1:, :]

    md_min_ss = pairwise_min_distances_same_group(md_solvent, L)
    flow_min_ss = pairwise_min_distances_same_group(flow_solvent, L)

    md_min_solv_solute = pairwise_min_distances_cross_group(md_solvent, md_solute, L)
    flow_min_solv_solute = pairwise_min_distances_cross_group(flow_solvent, flow_solute, L)

    # Plot minimum distance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(md_min_ss, bins=40, alpha=0.45, label="MC")
    axes[0].hist(flow_min_ss, bins=40, alpha=0.45, label="Flow")
    axes[0].axvline(args.sigma, color="gray", ls="--", lw=0.8, label=f"σ={args.sigma}")
    axes[0].set_xlabel("min solvent-solvent distance")
    axes[0].set_ylabel("count")
    axes[0].set_title("Closest solvent-solvent distance")
    axes[0].legend()

    axes[1].hist(md_min_solv_solute, bins=40, alpha=0.45, label="MC")
    axes[1].hist(flow_min_solv_solute, bins=40, alpha=0.45, label="Flow")
    axes[1].axvline(args.sigma, color="gray", ls="--", lw=0.8, label=f"σ={args.sigma}")
    axes[1].set_xlabel("min solvent-solute distance")
    axes[1].set_ylabel("count")
    axes[1].set_title("Closest solvent-solute distance")
    axes[1].legend()

    plt.tight_layout()

    dist_path = eval_dir / f"{prefix}_minimum_distance_plot.png"
    fig.savefig(dist_path, dpi=200)

    if use_wandb:
        wandb.log({"Media/minimum_distance": wandb.Image(fig)}, step=wandb_step)

    plt.close(fig)

    # Position distribution plot

    fig = position_density(
        wrap_pbc_positions(x_gen_full_eval, args.l_box),
        args.l_box,
        "3D solvent position distribution"
    )
    gr_path = eval_dir / f"{prefix}_position_dist_plot.png"
    fig.savefig(gr_path, dpi=200)

    if use_wandb:
        wandb.log({"Media/position_distribution": wandb.Image(fig)}, step=wandb_step)

    plt.close(fig)


    # RDF plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(r_ref, gr_ref, label="MC reference")
    ax.plot(r_gen, gr_gen, "--", label="Equivariant flow")
    ax.axvline(args.sigma, color="gray", ls="--", lw=0.8, label=f"σ={args.sigma}")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Solute–solvent g(r)")
    ax.legend()
    plt.tight_layout()

    gr_path = eval_dir / f"{prefix}_gr_plot.png"
    fig.savefig(gr_path, dpi=200)

    if use_wandb:
        wandb.log({"Media/rdf": wandb.Image(fig)}, step=wandb_step)

    plt.close(fig)

    # Energy plot, robust to huge generated-energy outliers

    fig, ax = plt.subplots(figsize=(7, 4))

    if len(fin_ref) > 0 and len(fin_gen) > 0:
        combined_energy = np.concatenate([fin_ref, fin_gen])
        energy_plot_low = float(np.percentile(combined_energy, 1))
        energy_plot_high = float(np.percentile(combined_energy, 99))

        if (
            not np.isfinite(energy_plot_low)
            or not np.isfinite(energy_plot_high)
            or energy_plot_high <= energy_plot_low
        ):
            energy_plot_low = float(np.min(combined_energy))
            energy_plot_high = float(np.max(combined_energy))

        fin_ref_plot = fin_ref[
            (fin_ref >= energy_plot_low) & (fin_ref <= energy_plot_high)
        ]
        fin_gen_plot = fin_gen[
            (fin_gen >= energy_plot_low) & (fin_gen <= energy_plot_high)
        ]

        ref_energy_plot_outlier_frac = 1.0 - len(fin_ref_plot) / max(len(fin_ref), 1)
        gen_energy_plot_outlier_frac = 1.0 - len(fin_gen_plot) / max(len(fin_gen), 1)

    elif len(fin_ref) > 0:
        energy_plot_low = float(np.percentile(fin_ref, 1))
        energy_plot_high = float(np.percentile(fin_ref, 99))
        fin_ref_plot = fin_ref
        fin_gen_plot = fin_gen
        ref_energy_plot_outlier_frac = 0.0
        gen_energy_plot_outlier_frac = float("nan")

    elif len(fin_gen) > 0:
        energy_plot_low = float(np.percentile(fin_gen, 1))
        energy_plot_high = float(np.percentile(fin_gen, 99))
        fin_ref_plot = fin_ref
        fin_gen_plot = fin_gen
        ref_energy_plot_outlier_frac = float("nan")
        gen_energy_plot_outlier_frac = 0.0

    else:
        energy_plot_low = float("nan")
        energy_plot_high = float("nan")
        fin_ref_plot = fin_ref
        fin_gen_plot = fin_gen
        ref_energy_plot_outlier_frac = float("nan")
        gen_energy_plot_outlier_frac = float("nan")

    if len(fin_ref_plot) > 0:
        ax.hist(fin_ref_plot, bins=50, density=True, alpha=0.5, label="MC")

    if len(fin_gen_plot) > 0:
        ax.hist(fin_gen_plot, bins=50, density=True, alpha=0.5, label="Flow")

    if np.isfinite(energy_plot_low) and np.isfinite(energy_plot_high):
        ax.set_xlim(energy_plot_low, energy_plot_high)

    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title(
        "Energy distribution "
        f"(1–99% window; flow outliers={gen_energy_plot_outlier_frac:.1%})"
    )

    if len(fin_ref_plot) > 0 or len(fin_gen_plot) > 0:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No finite energies to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    energy_path = eval_dir / f"{prefix}_energy_plot.png"
    fig.savefig(energy_path, dpi=200)

    if use_wandb:
        wandb.log({"Media/energy_distribution": wandb.Image(fig)}, step=wandb_step)

    plt.close(fig)

    # 3D HTML plots
    if make_3d:
        fig_mc_3d = plot_3d_pbc_interactive(
            x_ref_full[0],
            l_box=args.l_box,
            sigma=args.sigma,
            title=f"{prefix}: MC reference configuration",
            wrap=True,
            show=False,
        )
        fig_mc_3d.write_html(eval_dir / f"{prefix}_mc_reference_pbc_3d.html")

        fig_gen_3d = plot_3d_pbc_interactive(
            x_gen_full_eval[0],
            l_box=args.l_box,
            sigma=args.sigma,
            title=f"{prefix}: Generated flow configuration",
            wrap=True,
            show=False,
        )
        fig_gen_3d.write_html(eval_dir / f"{prefix}_generated_pbc_3d.html")

        if use_wandb:
            wandb.log({
                "Media/mc_reference_pbc_3d": wandb.Html(
                    fig_mc_3d.to_html(include_plotlyjs="cdn")
                ),
                "Media/generated_pbc_3d": wandb.Html(
                    fig_gen_3d.to_html(include_plotlyjs="cdn")
                ),
            })
    
    metrics = {
        "mean_energy_mc": float(fin_ref.mean()) if len(fin_ref) > 0 else float("nan"),
        "std_energy_mc": float(fin_ref.std()) if len(fin_ref) > 0 else float("nan"),
        "mean_energy_flow": float(fin_gen.mean()) if len(fin_gen) > 0 else float("nan"),
        "std_energy_flow": float(fin_gen.std()) if len(fin_gen) > 0 else float("nan"),
        "finite_fraction": float(finite_frac),
        "gr_l2_error": float(gr_l2),
        "overlap_frac": float(overlap_frac),
        "energy_plot_low": float(energy_plot_low),
        "energy_plot_high": float(energy_plot_high),
        "ref_energy_plot_outlier_frac": float(ref_energy_plot_outlier_frac),
        "gen_energy_plot_outlier_frac": float(gen_energy_plot_outlier_frac),
        "md_min_ss_mean": float(np.mean(md_min_ss)),
        "md_min_ss_median": float(np.median(md_min_ss)),
        "md_min_ss_min": float(np.min(md_min_ss)),

        "flow_min_ss_mean": float(np.mean(flow_min_ss)),
        "flow_min_ss_median": float(np.median(flow_min_ss)),
        "flow_min_ss_min": float(np.min(flow_min_ss)),

        "md_min_solv_solute_mean": float(np.mean(md_min_solv_solute)),
        "md_min_solv_solute_median": float(np.median(md_min_solv_solute)),
        "md_min_solv_solute_min": float(np.min(md_min_solv_solute)),

        "flow_min_solv_solute_mean": float(np.mean(flow_min_solv_solute)),
        "flow_min_solv_solute_median": float(np.median(flow_min_solv_solute)),
        "flow_min_solv_solute_min": float(np.min(flow_min_solv_solute)),
    }

    append_metrics_csv(eval_dir / "eval_metrics.csv", metrics)

    if use_wandb:
        wandb.log({
                "Statistics/gr_l2_error": metrics["gr_l2_error"],
                "Statistics/finite_fraction": metrics["finite_fraction"],
                "Statistics/overlap_frac": metrics["overlap_frac"],
                "Statistics/energy_plot_low": metrics["energy_plot_low"],
                "Statistics/energy_plot_high": metrics["energy_plot_high"],
                "Statistics/ref_energy_plot_outlier_frac": metrics["ref_energy_plot_outlier_frac"],
                "Statistics/gen_energy_plot_outlier_frac": metrics["gen_energy_plot_outlier_frac"],
                "Statistics/md_min_ss_mean": metrics["md_min_ss_mean"],
                "Statistics/flow_min_ss_mean": metrics["flow_min_ss_mean"],
                "Statistics/md_min_solv_solute_mean": metrics["md_min_solv_solute_mean"],
                "Statistics/flow_min_solv_solute_mean": metrics["flow_min_solv_solute_mean"],
            }, step=wandb_step)

    if was_training:
        flow.train()

    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not any([args.run_md, args.run_stage1, args.run_stage2, args.run_stage3]):
        raise ValueError(
            "No run stage selected. Use at least one of: "
            "--run_md, --run_stage1, --run_stage2, --run_stage3"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Save directory
    if args.save_dir is None:
        args.save_dir = str(_ROOT / "Notebooks" / "Trained_models" / "Solute_PBC_Equivariant")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    md_dir = save_dir / "md"
    stage1_dir = save_dir / "stage1_ml"
    stage2_dir = save_dir / "stage2_ml_kl"
    stage3_dir = save_dir / "stage3_kl"

    for d in [md_dir, stage1_dir, stage2_dir, stage3_dir]:

        d.mkdir(parents=True, exist_ok=True)

    save_args_json(args, save_dir / "run_args.json")

    # ------------------------------------------------------------------ WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            config=vars(args),
        )
        print(f"WandB run: {run.url}")

    global_step = 0   # continuous step counter for wandb x-axis

    # -------------------------------------------------------------- 1. System
    print("\n=== 1. System setup ===")
    V = (2 * args.l_box) ** 3
    rho_star = args.n_solvent * args.sigma ** 3 / V
    print(f"Reduced density ρ* = {rho_star:.3f}")

    system = SoluteSimulationPBC3D(
        n_particles=args.n_particles,
        epsilon=args.epsilon,
        sigma=args.sigma,
        l_box=args.l_box,
        center_solute=True,
        k_center=args.k_center,
    )

    # --------------------------------------------------------------- 2. MC / MD data
    print("\n=== 2. MC/MD production or loading ===")

    if args.mc_cache is None:
        args.mc_cache = str(md_dir / "xtraj.npy")

    mc_cache_path = Path(args.mc_cache)

    if args.run_md:
        print("Running MC/MD production and saving trajectory.")

        coords0 = build_solute_coords_3d_rsa(
            N=args.n_particles,
            l_box=args.l_box,
            sigma=args.sigma,
            seed=args.mc_seed,
        )
        print(f"RSA initial config — energy: {system.get_energy(coords0):.2f}")

        # Equilibration
        sim_eq = MetropolisSampler(
            system,
            temp=args.temperature,
            sigma=args.mc_sigma,
            stride=100,
        )
        sim_eq.run(coords0, nsteps=args.n_eq)

        # Production
        sim = MetropolisSampler(
            system,
            temp=args.temperature,
            sigma=args.mc_sigma,
            stride=args.mc_stride,
        )
        sim.run(sim_eq.xtraj[-1], nsteps=args.n_mc)

        xtraj = np.array(sim.xtraj)
        etraj = np.array(sim.etraj)

        print(
            f"Production — {xtraj.shape[0]} frames, "
            f"energy {etraj.mean():.2f} ± {etraj.std():.2f}"
        )

        mc_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mc_cache_path, xtraj)
        np.save(md_dir / "etraj.npy", etraj)

        append_metrics_csv(
            md_dir / "metrics.csv",
            {
                "n_frames": xtraj.shape[0],
                "mean_energy": float(etraj.mean()),
                "std_energy": float(etraj.std()),
                "temperature": args.temperature,
                "mc_sigma": args.mc_sigma,
                "n_eq": args.n_eq,
                "n_mc": args.n_mc,
                "mc_stride": args.mc_stride,
            },
        )

        if use_wandb:
            wandb.log({
                "md/mean_energy": float(etraj.mean()),
                "md/std_energy": float(etraj.std()),
                "md/n_frames": xtraj.shape[0],
            })

    else:
        if not mc_cache_path.exists():
            raise FileNotFoundError(
                f"MC/MD cache not found: {mc_cache_path}\n"
                f"Run first with --run_md --mc_cache {mc_cache_path}"
            )

        print(f"Loading MC/MD trajectory from {mc_cache_path}")
        xtraj = np.load(mc_cache_path)
        print(f"Loaded: {xtraj.shape}")
    
    if args.run_md and not any([args.run_stage1, args.run_stage2, args.run_stage3]):
        if use_wandb:
            wandb.finish()
        print("\nMD/MC production complete.")
        return


    # -------------------------------------------------------- 3. Training data
    need_training_data = args.run_stage1 or args.run_stage2 or args.run_stage3
    need_reference_data = args.run_stage1 or args.run_stage2 or args.run_stage3

    if need_reference_data:
        r_ref, gr_ref = compute_gr(xtraj, args.l_box)

    if need_training_data:
        print("\n=== 3. Training data ===")
        x_solvent = xtraj[::args.data_stride, 1:, :]
        x_samples = x_solvent.reshape(len(x_solvent), -1).astype(np.float32)
        np.random.shuffle(x_samples)

        r_all = np.linalg.norm(x_solvent.reshape(-1, 3), axis=-1)
        print(
            f"Training samples: {x_samples.shape}  "
            f"r ∈ [{r_all.min():.2f}, {r_all.max():.2f}]"
        )

        loader_x_ml = make_loader(x_samples, args.batch_size_ml)
        loader_x_kl = make_loader(x_samples, args.batch_size_kl)
    else:
        x_samples = None
        loader_x_ml = None
        loader_x_kl = None

    # ----------------------------------------------------------- 4. Build flow
    print("\n=== 4. Building flow ===")
    flow = build_egnn_solute_flow_3d(
        system=system,
        n_particles=args.n_solvent,
        n_blocks=args.n_blocks,
        egnn_hidden=args.egnn_hidden,
        egnn_layers=args.egnn_layers,
        num_bins=args.num_bins,
        tail_bound=args.tail_bound,
        l_box=args.l_box,
        att_heads=args.att_heads,
    ).to(device)

    n_params = sum(p.numel() for p in flow.parameters())
    print(f"Parameters: {n_params:,}  |  Coupling layers: {len(flow.coupling_layers)}")
    if use_wandb:
        wandb.summary["flow/n_params"] = n_params

    # --------------------------------------------------- 5. Stage 1 — ML
    if args.run_stage1:
        print("\n=== 5. Stage 1 — ML training ===")

        optimizer_ml = torch.optim.Adam(flow.parameters(), lr=args.lr_ml)

        best_ml = float("inf")
        patience_ctr = 0

        save_args_json(args, stage1_dir / "args.json")

        for epoch in tqdm(range(args.n_epochs_ml), desc="Stage 1 ML"):
            loss, ml_loss, _, grad_norm = train_epoch(
                flow,
                loader_x_ml,
                None,
                optimizer_ml,
                w_ml=1.0,
                w_kl=0.0,
                energy_cap=None,
                device=device,
                clip_grad=args.clip_grad,
            )
            global_step += 1

            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": loss,
                "ml_loss": ml_loss,
                "kl_loss": 0.0,
                "grad_norm": grad_norm,
                "learning_rate": optimizer_ml.param_groups[0]["lr"],
            }

            if (epoch + 1) % args.metric_every == 0:
                metrics = estimate_logq_logp_ess(
                    flow=flow,
                    system=system,
                    n_samples=args.metric_samples,
                    device=device,
                    energy_cap=args.energy_cap,
                )
                row.update(metrics)

            append_metrics_csv(stage1_dir / "metrics.csv", row)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "Statistics/loss": row["loss"],
                    "Statistics/ml_loss": row["ml_loss"],
                    "Statistics/grad_norm": row["grad_norm"],
                    "Statistics/learning_rate": row["learning_rate"],
                    **({
                        "Statistics/mean_log_q_total": row["mean_log_q_total"],
                        "Statistics/mean_log_p": row["mean_log_p"],
                        "Statistics/ess": row["ess"],
                        "Statistics/ess_fraction": row["ess_fraction"],
                    } if "ess" in row else {}),
                }, step=global_step)

            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    flow,
                    optimizer_ml,
                    stage1_dir / f"ckpt_ep{epoch+1:04d}.pt",
                    epoch + 1,
                    loss,
                    args,
                )

            if loss < best_ml - 1e-3:
                best_ml = loss
                patience_ctr = 0
                save_checkpoint(
                    flow,
                    optimizer_ml,
                    stage1_dir / "best.pt",
                    epoch + 1,
                    loss,
                    args,
                )
            else:
                patience_ctr += 1

            if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
                evaluate_and_log(
                    flow=flow,
                    system=system,
                    x_samples=x_samples,
                    r_ref=r_ref,
                    gr_ref=gr_ref,
                    args=args,
                    device=device,
                    eval_dir=stage1_dir / "intermediate_eval",
                    use_wandb=use_wandb,
                    wandb=wandb if use_wandb else None,
                    prefix=f"stage1_epoch_{epoch+1:04d}",
                    make_3d=(epoch + 1) % (args.plot_every) == 0,
                    wandb_step=global_step,
                )

            if patience_ctr >= args.patience_ml:
                tqdm.write(f"\nStage 1 early stop at epoch {epoch + 1}")
                break
        

        save_checkpoint(
            flow,
            optimizer_ml,
            stage1_dir / "final.pt",
            epoch + 1,
            loss,
            args,
        )
        torch.save(flow.state_dict(), stage1_dir / "final_state_dict.pt")

        print(f"Stage 1 done — best loss: {best_ml:.4f}")

    # --------------------------------------------------- 6. Stage 2 — ML + KL
    if args.run_stage2:
        print("\n=== 6. Stage 2 — ML + KL fine-tuning ===")

        if not args.run_stage1:
            stage1_ckpt = args.stage1_checkpoint
            if stage1_ckpt is None:
                stage1_ckpt = str(stage1_dir / "final.pt")

            if not Path(stage1_ckpt).exists():
                raise FileNotFoundError(
                    f"Stage 1 checkpoint not found: {stage1_ckpt}\n"
                    "Provide --stage1_checkpoint or run --run_stage1 first."
                )

            load_model_weights(flow, stage1_ckpt, device)

        with torch.no_grad():
            z_kl_t = flow.prior.sample(args.n_kl_samples)

        z_kl = z_kl_t.cpu().numpy().astype(np.float32)
        loader_z_kl = make_loader(z_kl, args.batch_size_kl)

        optimizer_kl = torch.optim.Adam(flow.parameters(), lr=args.lr_kl)

        best_kl = float("inf")
        patience_ctr = 0

        save_args_json(args, stage2_dir / "args.json")

        for epoch in tqdm(range(args.n_epochs_kl), desc="Stage 2 ML+KL"):
            loss, ml_loss, kl_loss, grad_norm = train_epoch(
                flow,
                loader_x_kl,
                loader_z_kl,
                optimizer_kl,
                w_ml=args.w_ml,
                w_kl=args.w_kl,
                energy_cap=args.energy_cap,
                device=device,
                clip_grad=args.clip_grad,
            )
            global_step += 1

            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": loss,
                "ml_loss": ml_loss,
                "kl_loss": kl_loss,
                "grad_norm": grad_norm,
                "learning_rate": optimizer_kl.param_groups[0]["lr"],
                "w_ml": args.w_ml,
                "w_kl": args.w_kl,
            }

            if (epoch + 1) % args.metric_every == 0:
                metrics = estimate_logq_logp_ess(
                    flow=flow,
                    system=system,
                    n_samples=args.metric_samples,
                    device=device,
                    energy_cap=args.energy_cap,
                )
                row.update(metrics)

            append_metrics_csv(stage2_dir / "metrics.csv", row)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "Statistics/loss": row["loss"],
                    "Statistics/ml_loss": row["ml_loss"],
                    "Statistics/kl_loss": row["kl_loss"],
                    "Statistics/grad_norm": row["grad_norm"],
                    "Statistics/learning_rate": row["learning_rate"],
                    "Statistics/w_ml": row["w_ml"],
                    "Statistics/w_kl": row["w_kl"],
                    **({
                        "Statistics/mean_log_q_total": row["mean_log_q_total"],
                        "Statistics/mean_log_p": row["mean_log_p"],
                        "Statistics/ess": row["ess"],
                        "Statistics/ess_fraction": row["ess_fraction"],
                    } if "ess" in row else {}),
                }, step=global_step)

            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    flow,
                    optimizer_kl,
                    stage2_dir / f"ckpt_ep{epoch+1:04d}.pt",
                    epoch + 1,
                    loss,
                    args,
                )

            if loss < best_kl - 1e-3:
                best_kl = loss
                patience_ctr = 0
                save_checkpoint(
                    flow,
                    optimizer_kl,
                    stage2_dir / "best.pt",
                    epoch + 1,
                    loss,
                    args,
                )
            else:
                patience_ctr += 1

            if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
                evaluate_and_log(
                    flow=flow,
                    system=system,
                    x_samples=x_samples,
                    r_ref=r_ref,
                    gr_ref=gr_ref,
                    args=args,
                    device=device,
                    eval_dir=stage2_dir / "intermediate_eval",
                    use_wandb=use_wandb,
                    wandb=wandb if use_wandb else None,
                    prefix=f"stage2_epoch_{epoch+1:04d}",
                    make_3d=(epoch + 1) % (args.plot_every) == 0,
                    wandb_step=global_step,
                )

            if patience_ctr >= args.patience_kl:
                tqdm.write(f"\nStage 2 early stop at epoch {epoch + 1}")
                break

        save_checkpoint(
            flow,
            optimizer_kl,
            stage2_dir / "final.pt",
            epoch + 1,
            loss,
            args,
        )
        torch.save(flow.state_dict(), stage2_dir / "final_state_dict.pt")

        print(f"Stage 2 done — best loss: {best_kl:.4f}")

    # --------------------------------------------------- 7. Stage 3 — pure KL
    if args.run_stage3:
        print("\n=== 7. Stage 3 — pure KL fine-tuning ===")

        if not args.run_stage2:
            stage2_ckpt = args.stage2_checkpoint
            if stage2_ckpt is None:
                stage2_ckpt = str(stage2_dir / "final.pt")

            if not Path(stage2_ckpt).exists():
                raise FileNotFoundError(
                    f"Stage 2 checkpoint not found: {stage2_ckpt}\n"
                    "Provide --stage2_checkpoint or run --run_stage2 first."
                )

            load_model_weights(flow, stage2_ckpt, device)

        with torch.no_grad():
            z_stage3_t = flow.prior.sample(args.n_kl_samples)

        z_stage3 = z_stage3_t.cpu().numpy().astype(np.float32)

        batch_size_stage3 = getattr(args, "batch_size_stage3", args.batch_size_kl)
        loader_z_stage3 = make_loader(z_stage3, batch_size_stage3)

        optimizer_stage3 = torch.optim.Adam(flow.parameters(), lr=args.lr_stage3)

        best_stage3 = float("inf")
        patience_ctr = 0

        save_args_json(args, stage3_dir / "args.json")

        for epoch in tqdm(range(args.n_epochs_stage3), desc="Stage 3 KL"):
            loss, ml_loss, kl_loss, grad_norm = train_epoch(
                flow,
                None,
                loader_z_stage3,
                optimizer_stage3,
                w_ml=0.0,
                w_kl=1.0,
                energy_cap=args.energy_cap,
                device=device,
                clip_grad=args.clip_grad,
            )
            global_step += 1

            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": loss,
                "ml_loss": ml_loss,
                "kl_loss": kl_loss,
                "grad_norm": grad_norm,
                "learning_rate": optimizer_stage3.param_groups[0]["lr"],
                "w_ml": 0.0,
                "w_kl": 1.0,
            }

            if (epoch + 1) % args.metric_every == 0:
                metrics = estimate_logq_logp_ess(
                    flow=flow,
                    system=system,
                    n_samples=args.metric_samples,
                    device=device,
                    energy_cap=args.energy_cap,
                )
                row.update(metrics)

            append_metrics_csv(stage3_dir / "metrics.csv", row)

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "Statistics/loss": row["loss"],
                    "Statistics/kl_loss": row["kl_loss"],
                    "Statistics/grad_norm": row["grad_norm"],
                    "Statistics/learning_rate": row["learning_rate"],
                    "Statistics/w_ml": row["w_ml"],
                    "Statistics/w_kl": row["w_kl"],
                    **({
                        "Statistics/mean_log_q_total": row["mean_log_q_total"],
                        "Statistics/mean_log_p": row["mean_log_p"],
                        "Statistics/ess": row["ess"],
                        "Statistics/ess_fraction": row["ess_fraction"],
                    } if "ess" in row else {}),
                }, step=global_step)

            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    flow,
                    optimizer_stage3,
                    stage3_dir / f"ckpt_ep{epoch+1:04d}.pt",
                    epoch + 1,
                    loss,
                    args,
                )

            if loss < best_stage3 - 1e-3:
                best_stage3 = loss
                patience_ctr = 0
                save_checkpoint(
                    flow,
                    optimizer_stage3,
                    stage3_dir / "best.pt",
                    epoch + 1,
                    loss,
                    args,
                )
            else:
                patience_ctr += 1

            if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
                evaluate_and_log(
                    flow=flow,
                    system=system,
                    x_samples=x_samples,
                    r_ref=r_ref,
                    gr_ref=gr_ref,
                    args=args,
                    device=device,
                    eval_dir=stage3_dir / "intermediate_eval",
                    use_wandb=use_wandb,
                    wandb=wandb if use_wandb else None,
                    prefix=f"stage3_epoch_{epoch+1:04d}",
                    make_3d=(epoch + 1) % (5 * args.plot_every) == 0,
                    wandb_step=global_step,
                )

            if patience_ctr >= args.patience_stage3:
                tqdm.write(f"\nStage 3 early stop at epoch {epoch + 1}")
                break

        save_checkpoint(
            flow,
            optimizer_stage3,
            stage3_dir / "final.pt",
            epoch + 1,
            loss,
            args,
        )
        torch.save(flow.state_dict(), stage3_dir / "final_state_dict.pt")

        print(f"Stage 3 done — best loss: {best_stage3:.4f}")

    # --------------------------------------------------- 8. Final evaluation
    print("\n=== 8. Evaluation ===")

    if args.run_stage3:
        eval_dir = stage3_dir
    elif args.run_stage2:
        eval_dir = stage2_dir
    elif args.run_stage1:
        eval_dir = stage1_dir
    else:
        eval_dir = md_dir

    metrics = evaluate_and_log(
        flow=flow,
        system=system,
        x_samples=x_samples,
        r_ref=r_ref,
        gr_ref=gr_ref,
        args=args,
        device=device,
        eval_dir=eval_dir,
        use_wandb=use_wandb,
        wandb=wandb if use_wandb else None,
        prefix="eval_final",
        make_3d=True,
        wandb_step=global_step,
    )

    print(f"g(r) L2 error: {metrics['gr_l2_error']:.4f}")
    print(
        f"MC   energy : {metrics['mean_energy_mc']:.2f} "
        f"± {metrics['std_energy_mc']:.2f}"
    )
    print(
        f"Flow energy : {metrics['mean_energy_flow']:.2f} "
        f"± {metrics['std_energy_flow']:.2f}"
    )
    print(f"Finite frac : {metrics['finite_fraction']:.3f}")

    if use_wandb:
        wandb.finish()

    final_path = eval_dir / "flow_final.pt"
    torch.save(flow.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

if __name__ == "__main__":
    main()

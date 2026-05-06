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

def compute_gr(traj, l_box, n_bins=100, r_max=None):
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


def train_epoch(flow, loader_x, loader_z, optimizer,
                w_ml, w_kl, energy_cap, device, clip_grad):
    """
    One full training epoch.  Returns (mean_loss, mean_ml_loss, mean_kl_loss).
    For a pure ML epoch pass loader_z=None, w_kl=0.
    For a mixed epoch, the shorter loader is cycled to match the longer one.
    """
    flow.train()
    losses, ml_losses, kl_losses = [], [], []

    if loader_x is not None and loader_z is not None:
        # cycle the shorter loader
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
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(flow.parameters(), clip_grad)
        optimizer.step()

        losses.append(loss.item())
        ml_losses.append(ml_val)
        kl_losses.append(kl_val)

    return np.mean(losses), np.mean(ml_losses), np.mean(kl_losses)


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
    g.add_argument("--n_epochs_kl",   type=int,   default=100)
    g.add_argument("--lr_kl",         type=float, default=3e-5)
    g.add_argument("--batch_size_kl", type=int,   default=256)
    g.add_argument("--w_ml",          type=float, default=0.1, help="ML weight in stage 2")
    g.add_argument("--w_kl",          type=float, default=0.9, help="KL weight in stage 2")
    g.add_argument("--energy_cap",    type=float, default=500.0)
    g.add_argument("--n_kl_samples",  type=int,   default=50_000)
    g.add_argument("--patience_kl",   type=int,   default=30)

    # Misc
    g = p.add_argument_group("Misc")
    g.add_argument("--clip_grad",       type=float, default=1.0, help="Gradient clipping (0=off)")
    g.add_argument("--data_stride",     type=int,   default=5,   help="Thinning of MC trajectory")
    g.add_argument("--seed",            type=int,   default=0)
    g.add_argument("--save_dir",        type=str,   default=None)
    g.add_argument("--checkpoint_every",type=int,   default=50,  help="Save checkpoint every N epochs")
    g.add_argument("--eval_samples",    type=int,   default=2000)

    # WandB
    g = p.add_argument_group("Weights & Biases")
    g.add_argument("--wandb_project", type=str, default="egnn-boltzmann-3d")
    g.add_argument("--wandb_entity",  type=str, default=None)
    g.add_argument("--wandb_name",    type=str, default=None)
    g.add_argument("--wandb_group",   type=str, default=None)
    g.add_argument("--no_wandb",      action="store_true", help="Disable WandB logging")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Save directory
    if args.save_dir is None:
        args.save_dir = str(_ROOT / "Notebooks" / "Trained_models" / "Solute_PBC_Equivariant")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # --------------------------------------------------------------- 2. MC data
    print("\n=== 2. Monte Carlo sampling ===")
    if args.mc_cache and Path(args.mc_cache).exists():
        print(f"Loading MC trajectory from {args.mc_cache}")
        xtraj = np.load(args.mc_cache)
        print(f"Loaded: {xtraj.shape}")
    else:
        coords0 = build_solute_coords_3d_rsa(
            N=args.n_particles, l_box=args.l_box,
            sigma=args.sigma, seed=args.mc_seed,
        )
        print(f"RSA initial config — energy: {system.get_energy(coords0):.2f}")

        # Equilibration
        sim_eq = MetropolisSampler(system, temp=args.temperature,
                                   sigma=args.mc_sigma, stride=100)
        sim_eq.run(coords0, nsteps=args.n_eq)
        print(f"Equilibration done — acceptance: {sim_eq.accept_rate:.3f}, "
              f"energy: {system.get_energy(sim_eq.xtraj[-1]):.2f}")

        # Production
        sim = MetropolisSampler(system, temp=args.temperature,
                                sigma=args.mc_sigma, stride=args.mc_stride)
        sim.run(sim_eq.xtraj[-1], nsteps=args.n_mc)
        xtraj = np.array(sim.xtraj)
        etraj  = np.array(sim.etraj)
        print(f"Production — {xtraj.shape[0]} frames, "
              f"acceptance {sim.accept_rate:.3f}, "
              f"energy {etraj.mean():.2f} ± {etraj.std():.2f}")

        if use_wandb:
            wandb.log({
                "mc/accept_rate": sim.accept_rate,
                "mc/mean_energy": float(etraj.mean()),
                "mc/n_frames":    xtraj.shape[0],
            })

        if args.mc_cache:
            Path(args.mc_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.mc_cache, xtraj)
            print(f"Saved trajectory to {args.mc_cache}")

    # Reference g(r) (computed once; used in evaluation plot)
    r_ref, gr_ref = compute_gr(xtraj, args.l_box)

    # -------------------------------------------------------- 3. Training data
    print("\n=== 3. Training data ===")
    x_solvent = xtraj[::args.data_stride, 1:, :]          # (n, N_solvent, 3)
    x_samples = x_solvent.reshape(len(x_solvent), -1).astype(np.float32)
    np.random.shuffle(x_samples)
    r_all = np.linalg.norm(x_solvent.reshape(-1, 3), axis=-1)
    print(f"Training samples: {x_samples.shape}  "
          f"r ∈ [{r_all.min():.2f}, {r_all.max():.2f}]")

    loader_x_ml = make_loader(x_samples, args.batch_size_ml)
    loader_x_kl = make_loader(x_samples, args.batch_size_kl)

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
    print("\n=== 5. Stage 1 — ML training ===")
    optimizer_ml = torch.optim.Adam(flow.parameters(), lr=args.lr_ml)

    best_ml, patience_ctr = float("inf"), 0

    for epoch in tqdm(range(args.n_epochs_ml), desc="ML"):
        loss, ml_loss, _ = train_epoch(
            flow, loader_x_ml, None, optimizer_ml,
            w_ml=1.0, w_kl=0.0, energy_cap=None,
            device=device, clip_grad=args.clip_grad,
        )
        global_step += 1

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "step": global_step,
                       "stage1/loss": loss, "stage1/ml_loss": ml_loss})

        # Checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(flow.state_dict(), save_dir / f"ckpt_ml_ep{epoch+1:04d}.pt")

        # Early stopping
        if loss < best_ml - 1e-3:
            best_ml, patience_ctr = loss, 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience_ml:
                tqdm.write(f"\nML early stop at epoch {epoch + 1}")
                break

    torch.save(flow.state_dict(), save_dir / "flow_stage1_ml.pt")
    print(f"Stage 1 done — best loss: {best_ml:.4f}")

    # --------------------------------------------------- 6. Stage 2 — KL
    print("\n=== 6. Stage 2 — KL fine-tuning ===")

    with torch.no_grad():
        z_kl_t = flow.prior.sample(args.n_kl_samples)
    z_kl = z_kl_t.cpu().numpy().astype(np.float32)
    loader_z_kl = make_loader(z_kl, args.batch_size_kl)

    optimizer_kl = torch.optim.Adam(flow.parameters(), lr=args.lr_kl)

    best_kl, patience_ctr = float("inf"), 0

    for epoch in tqdm(range(args.n_epochs_kl), desc="KL"):
        loss, ml_loss, kl_loss = train_epoch(
            flow, loader_x_kl, loader_z_kl, optimizer_kl,
            w_ml=args.w_ml, w_kl=args.w_kl,
            energy_cap=args.energy_cap,
            device=device, clip_grad=args.clip_grad,
        )
        global_step += 1

        if use_wandb:
            wandb.log({
                "epoch": args.n_epochs_ml + epoch + 1,
                "step": global_step,
                "stage2/loss":    loss,
                "stage2/ml_loss": ml_loss,
                "stage2/kl_loss": kl_loss,
            })

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(flow.state_dict(), save_dir / f"ckpt_kl_ep{epoch+1:04d}.pt")

        if loss < best_kl - 1e-3:
            best_kl, patience_ctr = loss, 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience_kl:
                tqdm.write(f"\nKL early stop at epoch {epoch + 1}")
                break

    torch.save(flow.state_dict(), save_dir / "flow_stage2_kl.pt")
    print(f"Stage 2 done — best loss: {best_kl:.4f}")

    # --------------------------------------------------------- 7. Evaluation
    print("\n=== 7. Evaluation ===")
    flow.eval()

    with torch.no_grad():
        z_eval = flow.prior.sample(args.eval_samples).to(device)
        x_gen_flat, _ = flow.generator(z_eval)

    x_gen = x_gen_flat.cpu().numpy().reshape(-1, args.n_solvent, 3)
    # Prepend fixed solute at origin for energy / g(r) evaluation
    x_gen_full = np.concatenate([np.zeros((len(x_gen), 1, 3)), x_gen], axis=1)

    # g(r)
    r_gen, gr_gen = compute_gr(x_gen_full, args.l_box)

    # g(r) L2 error (interpolate reference onto generated grid)
    gr_ref_i = np.interp(r_gen, r_ref, gr_ref)
    gr_l2    = float(np.sqrt(np.mean((gr_gen - gr_ref_i) ** 2)))
    print(f"g(r) L2 error: {gr_l2:.4f}")

    # Energy distribution
    n_eval = args.eval_samples
    x_ref_full = np.concatenate(
        [np.zeros((n_eval, 1, 3)),
         x_samples[:n_eval].reshape(n_eval, args.n_solvent, 3)],
        axis=1,
    )
    with torch.no_grad():
        e_ref = system.get_energy_batch(
            torch.tensor(x_ref_full.reshape(n_eval, -1))
        ).cpu().numpy()
        e_gen = system.get_energy_batch(
            torch.tensor(x_gen_full.reshape(len(x_gen_full), -1))
        ).cpu().numpy()

    fin_ref = e_ref[np.isfinite(e_ref) & (e_ref < 1e4)]
    fin_gen = e_gen[np.isfinite(e_gen) & (e_gen < 1e4)]
    finite_frac = len(fin_gen) / args.eval_samples

    print(f"MC   energy : {fin_ref.mean():.2f} ± {fin_ref.std():.2f}")
    print(f"Flow energy : {fin_gen.mean():.2f} ± {fin_gen.std():.2f}")
    print(f"Finite frac : {finite_frac:.3f}")

    # Overlap check
    overlap_frac = float((e_gen < -1e3).mean()) if len(e_gen) > 0 else float("nan")

    if use_wandb:
        import matplotlib.pyplot as plt

        # g(r) figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(r_ref, gr_ref, label="MC reference")
        ax.plot(r_gen, gr_gen, "--", label="Equivariant flow")
        ax.axvline(args.sigma, color="gray", ls="--", lw=0.8, label=f"σ={args.sigma}")
        ax.set_xlabel("r"); ax.set_ylabel("g(r)"); ax.set_title("Solute–solvent g(r)")
        ax.legend(); plt.tight_layout()
        wandb.log({"eval/gr_plot": wandb.Image(fig)})
        plt.close(fig)

        # Energy distribution figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(fin_ref, bins=50, density=True, alpha=0.5, label="MC")
        ax.hist(fin_gen, bins=50, density=True, alpha=0.5, label="Flow")
        ax.set_xlabel("Energy"); ax.set_ylabel("Density")
        ax.set_title("Energy distribution"); ax.legend(); plt.tight_layout()
        wandb.log({"eval/energy_plot": wandb.Image(fig)})
        plt.close(fig)

        wandb.log({
            "eval/mean_energy_mc":     float(fin_ref.mean()),
            "eval/std_energy_mc":      float(fin_ref.std()),
            "eval/mean_energy_flow":   float(fin_gen.mean()),
            "eval/std_energy_flow":    float(fin_gen.std()),
            "eval/finite_fraction":    finite_frac,
            "eval/gr_l2_error":        gr_l2,
        })
        wandb.finish()

    # Final model
    final_path = save_dir / "flow_final.pt"
    torch.save(flow.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()

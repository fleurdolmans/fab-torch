"""
FAB training for the 3D solute-in-LJ-bath system using the CartesianSplineFlow.

Uses:
  - SoluteSimulation3D    (potentials.py)
  - build_solute_spline_flow_3d  (spline_flow.py)  — 108D (36 solvent × 3D)
  - SoluteTarget3D        (target.py)              — FAB target
  - SoluteFlowFAB3D       (flow_adapter.py)        — FAB wrapper
  - FABModel + Metropolis + PrioritisedReplayBuffer + PrioritisedBufferTrainer

Workflow:
  1. Pretrain with ML + KL in the notebook  →  save checkpoint
  2. Set PRETRAINED_PATH below
  3. python boltzmann_generators_3d/fab/train_fab_spline_3d.py

Run from the repo root:
    python boltzmann_generators_3d/fab/train_fab_spline_3d.py
"""

import os
import sys
import pathlib

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent          # .../fab/
_REPO = _HERE.parent.parent                              # fab-torch/
_LIB  = _HERE.parent / "Library"                         # boltzmann_generators_3d/Library

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_LIB))

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from potentials  import SoluteSimulation3D
from spline_flow import build_solute_spline_flow_3d

from fab                                     import FABModel, Metropolis, PrioritisedBufferTrainer
from fab.utils.prioritised_replay_buffer     import PrioritisedReplayBuffer
from fab.utils.logging                       import ListLogger

from target       import SoluteTarget3D
from flow_adapter import SoluteFlowFAB3D


# ===========================================================================
# Configuration
# ===========================================================================

# --- System ---
N_PARTICLES   = 37          # 1 solute + 36 solvent
N_SOLVENT     = 36
DIM           = 3
L_BOX         = 2.3
SIGMA         = 1.1
EPSILON       = 1.0
K_BOX         = 100.0
CENTER_SOLUTE = True
K_CENTER      = 20.0
TEMPERATURE   = 1.0

# --- Spline flow architecture ---
N_BLOCKS   = 8
N_NODES    = 256
N_HIDDEN   = 3
NUM_BINS   = 8
TAIL_BOUND = 7.0

# --- FAB / AIS ---
N_INTERMEDIATE      = 16
N_METROPOLIS_STEPS  = 15
METROPOLIS_MAX_STEP = 0.2
METROPOLIS_MIN_STEP = 0.02
FAB_ALPHA           = 2.0

# --- Prioritised replay buffer ---
BUFFER_MAX_LENGTH  = 10_000
BUFFER_MIN_LENGTH  = 2_000
N_BATCHES_SAMPLING = 2
W_ADJUST_MAX_CLIP  = 10.0

# --- Training ---
BATCH_SIZE    = 512
LR            = 3e-4
MAX_GRAD_NORM = 5.0
SEED          = 42

# --- Temperature curriculum ---
# Each entry: (temperature, n_iterations).
# The buffer is rebuilt at each stage transition (log_w depends on T).
# Set to a single stage [(1.0, N)] to disable curriculum.
T_SCHEDULE = [
    (5.0,  30),
    (3.0,  40),
    (2.0,  50),
    (1.0,  80),
]

# --- I/O ---
SAVE_DIR = str(_HERE / "runs" / "fab_spline_3d")
# Path to a pre-trained 3D spline flow checkpoint, or None to train from scratch.
PRETRAINED_PATH = None   # e.g. str(_HERE.parent / "Notebooks" / "Trained_models" / "model_KL")

N_EVAL        = 10
N_CHECKPOINTS = 5

# --- MC data for buffer seeding ---
# Path to MC trajectory .npz with key "xtraj" of shape (N_frames, 37, 3).
# Set to None to use AIS-only buffer initialisation.
MC_DATA_PATH = None   # e.g. str(_HERE.parent / "Notebooks" / "MC_data" / "Solute3D_prod.npz")


# ===========================================================================
# Setup helpers
# ===========================================================================

def build_system():
    return SoluteSimulation3D(
        n_particles=N_PARTICLES,
        epsilon=EPSILON,
        sigma=SIGMA,
        l_box=L_BOX,
        k_box=K_BOX,
        center_solute=CENTER_SOLUTE,
        k_center=K_CENTER,
    )


def build_flow(system):
    flow = build_solute_spline_flow_3d(
        system=system,
        n_particles=N_SOLVENT,
        n_blocks=N_BLOCKS,
        n_nodes=N_NODES,
        n_hidden=N_HIDDEN,
        num_bins=NUM_BINS,
        tail_bound=TAIL_BOUND,
    )
    if PRETRAINED_PATH is not None and os.path.isfile(PRETRAINED_PATH):
        state = torch.load(PRETRAINED_PATH, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        flow.load_state_dict(state)
        print(f"Loaded pre-trained flow from {PRETRAINED_PATH}")
    else:
        print("Starting flow from random initialisation.")
    return flow


def build_target():
    return SoluteTarget3D(
        n_solvent=N_SOLVENT,
        epsilon=EPSILON,
        sigma=SIGMA,
        l_box=L_BOX,
        k_box=K_BOX,
        center_solute=False,   # solute fixed at origin; no centering penalty needed
        k_center=K_CENTER,
        temperature=TEMPERATURE,
    )


def build_fab_model(flow_adapter: SoluteFlowFAB3D, target: SoluteTarget3D) -> FABModel:
    dim = N_SOLVENT * DIM  # 108

    transition_operator = Metropolis(
        n_ais_intermediate_distributions=N_INTERMEDIATE,
        dim=dim,
        base_log_prob=flow_adapter.log_prob,
        target_log_prob=target.log_prob,
        p_target=False,
        alpha=FAB_ALPHA,
        n_updates=N_METROPOLIS_STEPS,
        adjust_step_size=True,
        target_p_accept=0.65,
        max_step_size=METROPOLIS_MAX_STEP,
        min_step_size=METROPOLIS_MIN_STEP,
    )

    fab_model = FABModel(
        flow=flow_adapter,
        target_distribution=target,
        n_intermediate_distributions=N_INTERMEDIATE,
        transition_operator=transition_operator,
        alpha=FAB_ALPHA,
        loss_type=None,
    )
    return fab_model


def load_mc_data():
    """Load MC trajectory, extract solvent, apply Hungarian ordering → (N, 108) tensor."""
    if MC_DATA_PATH is None or not os.path.isfile(MC_DATA_PATH):
        print("MC_DATA_PATH not set or not found — skipping MC buffer seeding.")
        return None

    d = np.load(MC_DATA_PATH)
    xtraj = d["xtraj"]                   # (N_frames, 37, 3)

    solvent = xtraj[:, 1:, :]            # (N_frames, 36, 3)  drop fixed solute

    # Hungarian ordering — align particle labels to the last frame as reference
    ref = solvent[-1]
    ordered = solvent.copy()
    for i in range(len(solvent)):
        cost = np.linalg.norm(ref[:, None, :] - solvent[i][None, :, :], axis=-1)
        _, ci = linear_sum_assignment(cost)
        ordered[i] = solvent[i][ci]

    x_mc = ordered.reshape(len(ordered), -1).astype(np.float32)  # (N_frames, 108)
    tensor = torch.tensor(x_mc)
    print(f"Loaded MC data: {tensor.shape}  (Hungarian-ordered solvent, 108D)")
    return tensor


def build_buffer(fab_model, target, mc_samples_tensor, device):
    def initial_sampler():
        point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
            BATCH_SIZE, logging=False
        )
        return point.x.detach(), log_w.detach(), point.log_q.detach()

    buffer = PrioritisedReplayBuffer(
        dim=N_SOLVENT * DIM,
        max_length=BUFFER_MAX_LENGTH,
        min_sample_length=BUFFER_MIN_LENGTH,
        initial_sampler=initial_sampler,
        fill_buffer_during_init=False,
        device=device,
    )

    # Seed with MC samples if available
    if mc_samples_tensor is not None:
        mc = mc_samples_tensor.to(device)
        with torch.no_grad():
            log_q_old = fab_model.flow.log_prob(mc)
            log_p     = target.log_prob(mc)
            log_w_mc  = FAB_ALPHA * (log_p - log_q_old)
        finite = torch.isfinite(log_w_mc)
        buffer.add(mc[finite], log_w_mc[finite], log_q_old[finite])
        print(f"Added {finite.sum().item()} / {len(finite)} MC samples to buffer")

    while not buffer.can_sample:
        x, log_w, log_q = initial_sampler()
        buffer.add(x, log_w, log_q)

    return buffer


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Build components ---
    system       = build_system()
    flow         = build_flow(system)
    target       = build_target()
    flow_adapter = SoluteFlowFAB3D(flow)

    if device == "cuda":
        flow_adapter.cuda()

    fab_model = build_fab_model(flow_adapter, target)

    if device == "cuda":
        fab_model.transition_operator.cuda()

    n_params = sum(p.numel() for p in flow.parameters())
    print(f"Flow parameters: {n_params:,}")

    # --- Quick sanity check ---
    with torch.no_grad():
        x_test, lq_test = flow_adapter.sample_and_log_prob((4,))
        lp_test = target.log_prob(x_test)
        print(f"Sanity check — log q: {lq_test.mean():.2f}  log p: {lp_test.mean():.2f}")
        print(f"Fraction finite target log probs: {torch.isfinite(lp_test).float().mean():.2f}")

    # --- Buffer (build at first curriculum temperature) ---
    target.temperature = T_SCHEDULE[0][0]
    mc_samples_tensor  = load_mc_data()
    print(f"Filling replay buffer at T={T_SCHEDULE[0][0]:.1f} …")
    buffer = build_buffer(fab_model, target, mc_samples_tensor, device)
    print(f"Buffer ready — {buffer.get_buffer_size()} samples")

    # --- Optimiser / logger / trainer ---
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    logger    = ListLogger()
    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    trainer = PrioritisedBufferTrainer(
        model=fab_model,
        optimizer=optimizer,
        buffer=buffer,
        alpha=FAB_ALPHA,
        n_batches_buffer_sampling=N_BATCHES_SAMPLING,
        max_gradient_norm=MAX_GRAD_NORM,
        w_adjust_max_clip=W_ADJUST_MAX_CLIP,
        logger=logger,
        save_path=SAVE_DIR,
        print_eval=True,
    )

    # --- Curriculum training loop ---
    for stage_idx, (temp, n_iters) in enumerate(T_SCHEDULE):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx+1}/{len(T_SCHEDULE)}: T={temp:.1f},  {n_iters} iterations")
        print(f"{'='*60}")

        target.temperature = temp
        if stage_idx > 0:
            print(f"Rebuilding buffer for T={temp:.1f} …")
            buffer = build_buffer(fab_model, target, mc_samples_tensor, device)
            trainer.buffer = buffer
            print(f"Buffer ready — {buffer.get_buffer_size()} samples")

        trainer.run(
            n_iterations=n_iters,
            batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            n_eval=max(1, n_iters // 10),
            n_checkpoints=1,
            save=True,
        )

        stage_path = os.path.join(SAVE_DIR, f"model_stage{stage_idx+1}_T{temp:.1f}.pt")
        torch.save(flow.state_dict(), stage_path)
        print(f"Saved stage {stage_idx+1} checkpoint: {stage_path}")

    # --- Final save ---
    final_path = os.path.join(SAVE_DIR, "model_final.pt")
    torch.save(flow.state_dict(), final_path)
    print(f"Saved final flow to {final_path}")

    return logger


if __name__ == "__main__":
    main()

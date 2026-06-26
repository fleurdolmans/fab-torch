"""
FAB training for the 2D solute-in-LJ-bath system using the coupling spline flow.

Uses:
  - SoluteSplineFlow         (build_solute_spline_flow from Library/generator.py)
  - SoluteTarget2D           (target.py) — 72D target (36 solvent, solute fixed at origin)
  - SoluteFlowFAB            (flow_adapter.py) — wraps the spline flow for FAB
  - FABModel + Metropolis + PrioritisedReplayBuffer + PrioritisedBufferTrainer

The spline flow models the 36 solvent particles in 72D Cartesian space (Hungarian
preprocessing is applied during ML/KL pretraining in the notebook, and the resulting
checkpoint is loaded here before FAB fine-tuning).

Curriculum learning (temperature annealing)
-------------------------------------------
Training proceeds in stages defined by T_SCHEDULE: [(temperature, n_iterations), ...].
Starting at a high temperature makes the target distribution smoother and much easier
for AIS to bridge.  The buffer is rebuilt at each stage transition because stored
importance weights log_w = α*(log p_T - log q) depend on T and become stale.

Workflow:
  1. Pretrain with ML+KL in the notebook  →  save checkpoint
  2. Set PRETRAINED_PATH below
  3. python boltzmann_generators_solute/fab/train_fab_spline.py

Run from the repo root:
    python boltzmann_generators_solute/fab/train_fab_spline.py
"""

import os
import sys
import pathlib

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE      = pathlib.Path(__file__).resolve().parent          # .../fab/
_ROOT      = _HERE.parent                                     # boltzmann_generators_2d/
_SLIB      = _ROOT / "Library"                                # boltzmann_generators_2d/Library
_FAB_TORCH = _ROOT.parent / "fab-torch"                       # ../fab-torch/ (FAB library)

sys.path.insert(0, str(_ROOT))       # so "from Library.x import y" works
sys.path.insert(0, str(_SLIB))       # so "from potentials import ..." works (fab-local modules)
sys.path.insert(0, str(_FAB_TORCH))  # must be first so "from fab import FABModel" finds fab-torch, not local fab/

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from potentials import SoluteSimulation2D
from Library.boltzmann import BoltzmannGenerator2D


from boltzmann_generators_3d.fab                                        import FABModel, Metropolis, PrioritisedBufferTrainer
from boltzmann_generators_3d.fab.utils.prioritised_replay_buffer        import PrioritisedReplayBuffer
from boltzmann_generators_3d.fab.utils.logging                          import ListLogger

from target       import SoluteTarget2D
from flow_adapter import SoluteFlowFAB


# ===========================================================================
# Configuration
# ===========================================================================

# --- System ---
N_PARTICLES    = 33          # 1 solute + 32 solvent (needed for energy calc in system)
N_SOLVENT      = 32
DIM            = 2
L_BOX          = 4.5
SIGMA          = 1.1
EPSILON        = 1.0
K_BOX          = 100.0
CENTER_SOLUTE  = True
K_CENTER       = 20.0
TEMPERATURE    = 1.0

# --- Spline flow architecture (must match pre-trained model if loading one) ---
N_BLOCKS       = 8           # coupling-block pairs (total layers = 2 * N_BLOCKS)
N_NODES        = 256         # MLP hidden width in each coupling conditioner
N_HIDDEN       = 3           # number of hidden layers in each coupling MLP
NUM_BINS       = 8          # spline bins
TAIL_BOUND     = 5.5         # spline tail bound (default from build_solute_spline_flow)

# --- FAB / AIS ---
N_INTERMEDIATE      = 16
N_METROPOLIS_STEPS  = 15
METROPOLIS_MAX_STEP = 0.2
METROPOLIS_MIN_STEP = 0.02
FAB_ALPHA           = 2.0

# --- Prioritised replay buffer ---
BUFFER_MAX_LENGTH   = 25_600
BUFFER_MIN_LENGTH   = 2560
N_BATCHES_SAMPLING  = 2
W_ADJUST_MAX_CLIP   = 5.0
ENERGY_CUTOFF       = 1e3   # used to filter out extreme outliers in log_w and log_p when filling the buffer

# --- Training ---
BATCH_SIZE     = 128
LR             = 5e-5
MAX_GRAD_NORM  = 5.0
SEED           = 42

# --- Temperature curriculum ---
# Each entry: (temperature, n_iterations).
# Training runs stage-by-stage, annealing from a soft target to the true one.
# The buffer is rebuilt at each stage transition (log_w values depend on T).
# Set to a single stage [(1.0, N)] to disable curriculum (standard FAB).

T_SCHEDULE = [
    (5.0,  30),   # stage 1: very diffuse target, AIS can bridge easily
    (3.0,  40),   # stage 2: moderate difficulty
    (2.0,  50),   # stage 3: near-physical
    (1.0,  600),   # stage 4: true physical target
]

# --- I/O ---
SAVE_DIR        = str(_HERE.parent / "Trained_models" /"FAB" / 
                      "buffer_passing" / "fab")  # where to save FAB checkpoints
# Set to the path of a pretrained spline flow .pt file, or None to train from scratch.

# PRETRAINED_PATH = None
PRETRAINED_PATH = str(_HERE.parent / "Trained_models" /
                      "spline" / "finetuning" /
                      "model_v1_spline_s3_w_loss-0-1-30_N33_L4p5_E1000")

N_EVAL         = 10
N_CHECKPOINTS  = 5

# --- MD data for buffer seeding ---
# Points to the MC trajectory .npz saved by the notebook.
# Set to None to skip MD seeding (buffer filled by AIS only).
MD_DATA_PATH = str(_HERE.parent / "Notebooks" / "MC_data" / "Solute2D_prod_L4.5_33_T1.0.npz")

# ===========================================================================
# Setup helpers
# ===========================================================================

def build_system():
    return SoluteSimulation2D(
        n_particles=N_PARTICLES,
        epsilon=EPSILON,
        sigma=SIGMA,
        l_box=L_BOX,
        k_box=K_BOX,
        center_solute=CENTER_SOLUTE,
        k_center=K_CENTER,
    )


def build_flow(system, BG):
    """Build or load the coupling spline flow for 36 solvent particles (72D)."""
    flow = BG.build(system)

    if PRETRAINED_PATH is not None and os.path.isfile(PRETRAINED_PATH):
        state = torch.load(PRETRAINED_PATH, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        flow.load_state_dict(state)
        print(f"Loaded pre-trained spline flow from {PRETRAINED_PATH}")
    else:
        print("Starting spline flow from random initialisation.")

    return flow


def build_target():
    """72D target: solute fixed at origin, 36 solvent particles free."""
    return SoluteTarget2D(
        n_solvent=N_SOLVENT,
        epsilon=EPSILON,
        sigma=SIGMA,
        l_box=L_BOX,
        k_box=K_BOX,
        center_solute=False,   # solute is fixed; no centering penalty needed
        k_center=K_CENTER,
        temperature=TEMPERATURE,
    )


def build_fab_model(flow_adapter: SoluteFlowFAB, target: SoluteTarget2D) -> FABModel:
    dim = N_SOLVENT * DIM  # 72

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


def load_reference_frame():
    """Return the canonical reference frame used for Hungarian alignment.

    This is the last MC frame (solvent only), which is the same reference
    used when pre-training the flow in the notebook.  Shape: (N_solvent, 2).
    Returns None if MD_DATA_PATH is not set or does not exist.
    """
    if MD_DATA_PATH is None or not os.path.isfile(MD_DATA_PATH):
        print("MD_DATA_PATH not set or file not found — Hungarian alignment disabled.")
        return None
    d = np.load(MD_DATA_PATH)
    xtraj = d["xtraj"]                    # (N_frames, 33, 2)
    ref = xtraj[-1, 1:, :].astype(np.float32)   # (N_solvent, 2), solvent only
    return torch.from_numpy(ref)


def load_md_data():
    """Load MC trajectory, extract solvent coords, apply Hungarian ordering, return (N, 72) tensor."""
    if MD_DATA_PATH is None or not os.path.isfile(MD_DATA_PATH):
        print("MD_DATA_PATH not set or file not found — skipping MD buffer seeding.")
        return None

    d = np.load(MD_DATA_PATH)
    xtraj = d["xtraj"]                  # (N_frames, 37, 2): solute at [0], solvent at [1:]

    # Extract solvent only (drop solute at index 0)
    solvent = xtraj[:, 1:, :]           # (N_frames, 36, 2)

    # Apply Hungarian ordering so particle labels match the flow's training data.
    # Use the last frame as reference (same convention as the pretraining notebook).
    ref = solvent[-1]                   # (36, 2)
    ordered = solvent.copy()
    for i in range(len(solvent)):
        cost = np.linalg.norm(ref[:, None, :] - solvent[i][None, :, :], axis=-1)  # (36, 36)
        _, col_idx = linear_sum_assignment(cost)
        ordered[i] = solvent[i][col_idx]

    x_md = ordered.reshape(len(ordered), -1).astype(np.float32)  # (N_frames, 72)
    tensor = torch.tensor(x_md)
    print(f"Loaded MD data: {tensor.shape}  (Hungarian-ordered solvent, 72D)")
    return tensor


def reweight_buffer(buffer, target, device, chunk_size: int = 512):
    """Recompute importance weights in-place after a temperature change.

    Instead of discarding all buffered samples, we recover U(x) from the
    stored (log_w, log_q_old) and compute new weights at the current
    target.temperature.  Samples with non-finite new weights are killed
    (set to -inf) so the prioritised sampler ignores them.
    """
    n = buffer.get_buffer_size()
    x       = buffer.buffer.x[:n]
    log_q   = buffer.buffer.log_q_old[:n]

    # Re-evaluate log p at the new temperature in chunks to avoid OOM.
    log_p_chunks = []
    for start in range(0, n, chunk_size):
        with torch.no_grad():
            lp = target.log_prob(x[start:start + chunk_size].to(device))
        log_p_chunks.append(lp)
    log_p = torch.cat(log_p_chunks)                       # (n,)

    log_w_new = FAB_ALPHA * (log_p - log_q)

    valid = torch.isfinite(log_w_new)
    log_w_new[~valid] = -float("inf")

    buffer.buffer.log_w[:n] = log_w_new
    print(f"Buffer reweighted: {valid.sum().item()}/{n} finite weights "
          f"at T={target.temperature:.1f}")


def build_buffer(fab_model, target, md_samples_tensor, device):
    # md_samples_tensor: shape [N, 72], already on CPU, from MD trajectory

    def initial_sampler():
        point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
            BATCH_SIZE, logging=False
        )
        x    = point.x.detach()
        log_w = log_w.detach()
        log_q = point.log_q.detach()

        # Filter out samples with non-finite weights (overlaps, numerical blow-ups)
        with torch.no_grad():
            log_p = target.log_prob(x)
        valid = (
            torch.isfinite(log_w)
            & torch.isfinite(log_q)
            & torch.isfinite(log_p)
            & (log_p > -ENERGY_CUTOFF)   # catches large-but-finite overlaps
        )

        return x[valid], log_w[valid], log_q[valid]


    # def initial_sampler():
    #     point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
    #         BATCH_SIZE, logging=False
    #     )
    #     return point.x.detach(), log_w.detach(), point.log_q.detach()

    buffer = PrioritisedReplayBuffer(
        dim=N_SOLVENT * DIM,
        max_length=BUFFER_MAX_LENGTH,
        min_sample_length=BUFFER_MIN_LENGTH,
        initial_sampler=initial_sampler,
        fill_buffer_during_init=False,   # manual init below
        device=device,
    )

    # Seed with MD samples (if available)
    if md_samples_tensor is not None:
        md = md_samples_tensor.to(device)
        with torch.no_grad():
            log_q_old = fab_model.flow.log_prob(md)
            log_p = target.log_prob(md)
            log_w_md = FAB_ALPHA * (log_p - log_q_old)
        # Drop any frames where the energy is non-finite (e.g. hard overlaps in MC data)
        finite = torch.isfinite(log_w_md)
        buffer.add(md[finite], log_w_md[finite], log_q_old[finite])
        print(f"Added {finite.sum().item()} / {len(finite)} MD samples to buffer")

    # Fill remaining capacity with AIS
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

    # --- BoltzmannGenerator2D wrapper (build / save / load in ML/KL-compatible format) ---
    BG = BoltzmannGenerator2D({
        'flow_type': 'fab',
        'reshape':     (N_SOLVENT, DIM),
        'n_blocks':    N_BLOCKS,
        'n_nodes':     N_NODES,
        'n_hidden':    N_HIDDEN,
        'num_bins':    NUM_BINS,
        'tail_bound':  TAIL_BOUND,
        'lr':          LR,
        'batch_size':  BATCH_SIZE,
    })

    # --- Build components ---
    system       = build_system()
    flow         = build_flow(system, BG)
    target       = build_target()
    reference    = load_reference_frame()
    flow_adapter = SoluteFlowFAB(flow, reference=reference)
    if reference is not None:
        print(f"Hungarian alignment enabled — reference frame shape: {reference.shape}")

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
        finite_frac = torch.isfinite(lp_test).float().mean().item()
        print(f"Fraction of finite target log probs: {finite_frac:.2f}")

    # --- Buffer ---
    # Set initial temperature (first curriculum stage) before filling the buffer,
    # because log_w = α*(log p_T - log q) depends on T.
    target.temperature = T_SCHEDULE[0][0]
    md_samples_tensor = load_md_data()
    print(f"Filling replay buffer at T={T_SCHEDULE[0][0]:.1f} …")
    buffer = build_buffer(fab_model, target, md_samples_tensor, device)
    print(f"Buffer ready — {buffer.get_buffer_size()} samples")

    # --- Optimiser ---
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

    # --- Logger ---
    logger = ListLogger()

    # --- Trainer ---
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

        # Update target temperature and reweight existing buffer samples in-place.
        # log_w = α*(log p_T - log q) depends on T, so weights must be updated,
        # but the samples x and log_q values remain valid and are kept.
        target.temperature = temp
        if stage_idx > 0:
            print(f"Reweighting buffer for T={temp:.1f} …")
            reweight_buffer(buffer, target, device)
            trainer.buffer = buffer

        n_eval_stage = max(1, n_iters // 10)
        trainer.run(
            n_iterations=n_iters,
            batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            n_eval=n_eval_stage,
            n_checkpoints=1,
            save=True,
        )

        stage_path = os.path.join(SAVE_DIR, f"model_stage{stage_idx+1}_T{temp:.1f}")
        BG.loss_iteration = logger.history.get('loss', [])
        BG.save(flow, stage_path)
        print(f"Saved stage {stage_idx+1} checkpoint: {stage_path}")

    # --- Final save ---
    final_path = os.path.join(SAVE_DIR, "model_final")
    BG.loss_iteration = logger.history.get('loss', [])
    BG.save(flow, final_path)
    print(f"Saved final flow to {final_path}")

    return logger


if __name__ == "__main__":
    main()

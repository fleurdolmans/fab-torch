"""
FAB training for the 2D solute-in-LJ-bath system.

Uses:
  - RealNVP           (from boltzmann_generators_solute/Library/generator.py)
  - BoltzmannGenerator (from boltzmann_generators_solute/Library/training.py)
  - SoluteFullTarget2D (target.py)  — 74D target (all 37 particles)
  - RealNVPFlowFAB    (flow_adapter.py)  — wraps the flow for FAB
  - FABModel + Metropolis + PrioritisedReplayBuffer + PrioritisedBufferTrainer

The RealNVP models all 37 particles (74D); the centering penalty in the energy
function keeps the solute near the origin. Pretrain with Hungarian preprocessing
in the notebook before running FAB.

Run from the repo root:
    python boltzmann_generators_solute/fab/train_fab.py
"""

import os
import sys
import pathlib

# ---------------------------------------------------------------------------
# Path setup — import from both the solute library and the fab library
# ---------------------------------------------------------------------------
_HERE   = pathlib.Path(__file__).resolve().parent          # .../fab/
_REPO   = _HERE.parent.parent                              # fab-torch/
_SLIB   = _HERE.parent / "Library"                         # boltzmann_generators_solute/Library

sys.path.insert(0, str(_REPO))   # makes `fab` importable
sys.path.insert(0, str(_SLIB))   # makes `generator`, `potentials` importable

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import torch
import numpy as np

from potentials import SoluteSimulation2D
from training   import BoltzmannGenerator

from fab                                        import FABModel, Metropolis, PrioritisedBufferTrainer
from fab.utils.prioritised_replay_buffer        import PrioritisedReplayBuffer
from fab.utils.logging                          import ListLogger

from target       import SoluteTarget2D
from flow_adapter import RealNVPFlowFAB


# ===========================================================================
# Configuration  (edit these values; no config-file system needed)
# ===========================================================================

# --- System ---
N_PARTICLES    = 37          # 1 solute + 36 solvent
N_SOLVENT      = 36
DIM            = 2
L_BOX          = 5.0        # soft wall at ±5 nm
SIGMA          = 1.1
EPSILON        = 1.0
K_BOX          = 100.0
CENTER_SOLUTE  = True
K_CENTER       = 20.0
TEMPERATURE    = 1.0

# --- Flow architecture (must match the pre-trained model if loading one) ---
N_BLOCKS       = 8           # coupling-block pairs  (total layers = 2 * N_BLOCKS)
N_NODES        = 256         # hidden layer width in each coupling MLP
N_LAYERS       = 3           # number of hidden layers in each coupling MLP
PRIOR_SIGMA    = 1.0         # isotropic Gaussian prior std
S_SCALE        = 0.5         # tanh clamp on s-net output (keeps scale bounded)


# --- FAB / AIS ---
N_INTERMEDIATE = 16      # number of AIS intermediate distributions; was 8       
N_METROPOLIS_STEPS = 15       # MCMC steps per AIS transition, was 5
METROPOLIS_MAX_STEP = 0.2    # initial max step size for Metropolis; was 0.5
METROPOLIS_MIN_STEP = 0.02      # minimum max step size after adjustment; was 0.05
FAB_ALPHA      = 2.0         # α for the α-divergence

# --- Prioritised replay buffer ---
BUFFER_MAX_LENGTH   = 50_000
BUFFER_MIN_LENGTH   = 2_000   # wait until this many samples before training
N_BATCHES_SAMPLING  = 2       # gradient steps per AIS batch
W_ADJUST_MAX_CLIP   = 10.0

# --- Training ---
N_ITERATIONS   = 2_00
BATCH_SIZE     = 512
LR             = 3e-4
MAX_GRAD_NORM  = 5.0
SEED           = 42

# --- I/O ---
SAVE_DIR        = str(_HERE / "runs" / "fab_realnvp")
# Set to None to train from scratch; set to a path to load a pre-trained RealNVP.
PRETRAINED_PATH = str(_HERE.parent / "Notebooks" / "Trained_models" /
                      "Solute" / "model_KL_center_v3")

N_EVAL         = 10          # how many times to log evaluation metrics
N_CHECKPOINTS  = 5          # how many checkpoints to save


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


def build_flow(system):
    """Build or load the RealNVP flow for the 36 solvent particles (72D).

    The solute is fixed at the origin; loss_KL prepends it via fixed_solute=True
    before computing energy, mirroring the SoluteSplineFlow approach.
    """
    bg = BoltzmannGenerator({
        'n_blocks':     N_BLOCKS,
        'dimension':    N_SOLVENT * 2,    # 72 — solvent only
        'reshape':      (N_SOLVENT, 2),   # (36, 2)
        'n_nodes':      N_NODES,
        'n_layers':     N_LAYERS,
        'prior_sigma':  PRIOR_SIGMA,
        's_scale':      S_SCALE,
        'fixed_solute': True,             # prepend solute at origin for energy eval
    })
    flow = bg.build(system)

    if PRETRAINED_PATH is not None and os.path.isfile(PRETRAINED_PATH):
        state = torch.load(PRETRAINED_PATH, map_location="cpu")
        # BoltzmannGenerator saves {"model": state_dict, "loss": ...} or raw state_dict.
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        try:
            flow.load_state_dict(state)
            print(f"Loaded pre-trained RealNVP from {PRETRAINED_PATH}")
        except RuntimeError as e:
            print(f"Could not load checkpoint (likely old 74D model): {e}")
            print("Starting RealNVP from random initialisation.")
    else:
        print("Starting RealNVP from random initialisation.")

    return flow


def build_target():
    return SoluteTarget2D(
        n_solvent=N_SOLVENT,
        epsilon=EPSILON,
        sigma=SIGMA,
        l_box=L_BOX,
        k_box=K_BOX,
        center_solute=CENTER_SOLUTE,
        k_center=K_CENTER,
        temperature=TEMPERATURE,
    )


def build_fab_model(flow_adapter: RealNVPFlowFAB, target: SoluteTarget2D) -> FABModel:
    dim = N_SOLVENT * DIM  # 72

    transition_operator = Metropolis(
        n_ais_intermediate_distributions=N_INTERMEDIATE,
        dim=dim,
        base_log_prob=flow_adapter.log_prob,
        target_log_prob=target.log_prob,
        p_target=False,            # target is p^α / q^(α-1)
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
        loss_type=None,            # PrioritisedBufferTrainer manages its own loss
    )

    return fab_model

def load_md_data(device):
    x_md = np.load(MD_DATA_PATH).astype("float32")

    # Strip solute (particle 0) to get 72D solvent-only data.
    # Handles both (n_frames, 37, 2) and flat (n_frames, 74) inputs.
    if x_md.ndim == 3 and x_md.shape[1] == N_PARTICLES:
        x_md = x_md[:, 1:, :].reshape(len(x_md), -1)   # (n, 36, 2) → (n, 72)
    elif x_md.ndim == 2 and x_md.shape[1] == N_PARTICLES * DIM:
        x_md = x_md.reshape(-1, N_PARTICLES, DIM)[:, 1:, :].reshape(-1, N_SOLVENT * DIM)

    x_md = torch.tensor(x_md, device=device, dtype=torch.float32)
    print(f"Loaded MD data: {x_md.shape}")
    return x_md


def sample_md_batch(x_md, batch_size):
    idx = torch.randint(
        0,
        x_md.shape[0],
        (batch_size,),
        device=x_md.device,
    )
    return x_md[idx]

P_DATA_INIT = 0.8

def build_buffer(fab_model: FABModel, device: str, x_md=None) -> PrioritisedReplayBuffer:
    dim = N_SOLVENT * DIM  # 72

    def initial_sampler():
        if x_md is None:
            point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                BATCH_SIZE, logging=False, purpose="buffer init"
            )
            return point.x.detach(), log_w.detach(), point.log_q.detach()

        n_data = int(P_DATA_INIT * BATCH_SIZE)
        n_ais = BATCH_SIZE - n_data

        xs, log_ws, log_qs = [], [], []

        if n_data > 0:
            x_data = sample_md_batch(x_md, n_data)
            log_q_data = fab_model.flow.log_prob(x_data)
            log_p_data = fab_model.target_distribution.log_prob(x_data)
            log_w_data = log_p_data - log_q_data

            xs.append(x_data)
            log_ws.append(log_w_data)
            log_qs.append(log_q_data)

        if n_ais > 0:
            point, log_w_ais = fab_model.annealed_importance_sampler.sample_and_log_weights(
                n_ais, logging=False, purpose="buffer init"
            )

            xs.append(point.x)
            log_ws.append(log_w_ais)
            log_qs.append(point.log_q)

        x = torch.cat(xs, dim=0)
        log_w = torch.cat(log_ws, dim=0)
        log_q = torch.cat(log_qs, dim=0)

        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm].detach(), log_w[perm].detach(), log_q[perm].detach()

    return PrioritisedReplayBuffer(
        dim=dim,
        max_length=BUFFER_MAX_LENGTH,
        min_sample_length=BUFFER_MIN_LENGTH,
        initial_sampler=initial_sampler,
        fill_buffer_during_init=True,
        device=device,
    )

# def build_buffer(fab_model: FABModel, device: str) -> PrioritisedReplayBuffer:
#     dim = N_PARTICLES * DIM  # 74

#     def initial_sampler():
#         point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
#             BATCH_SIZE, logging=False, purpose="buffer init"
#         )
#         return point.x.detach(), log_w.detach(), point.log_q.detach()

#     return PrioritisedReplayBuffer(
#         dim=dim,
#         max_length=BUFFER_MAX_LENGTH,
#         min_sample_length=BUFFER_MIN_LENGTH,
#         initial_sampler=initial_sampler,
#         fill_buffer_during_init=True,
#         device=device,
#     )


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
    flow_adapter = RealNVPFlowFAB(flow)

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
    print("Filling replay buffer via AIS …")
    buffer = build_buffer(fab_model, device)
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

    eval_batch_size = BATCH_SIZE

    trainer.run(
        n_iterations=N_ITERATIONS,
        batch_size=BATCH_SIZE,
        eval_batch_size=eval_batch_size,
        n_eval=N_EVAL,
        n_checkpoints=N_CHECKPOINTS,
        save=True,
    )

    # --- Final save ---
    final_path = os.path.join(SAVE_DIR, "model_final.pt")
    torch.save(flow.state_dict(), final_path)
    print(f"Saved final flow to {final_path}")

    return logger


if __name__ == "__main__":
    main()

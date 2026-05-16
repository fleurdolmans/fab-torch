"""
EGNN: E(n)-Equivariant Graph Neural Network for particle systems with PBC.

Based on Satorras et al. 2021 "E(n) Equivariant Normalizing Flows".
Uses torch_scatter (part of the PyG ecosystem) for efficient aggregation.

All operations are batched: input shape (B, N, 3) for positions, (B, N, d) for features.
Internally the batch dimension is folded into the node dimension (B*N, ...) and
a complete-graph edge_index is constructed once and reused.

Key classes
-----------
EGNNLayer   : single message-passing layer
EGNN        : stacked layers producing invariant features + equivariant vectors
"""

import math
import torch
import torch.nn as nn


def _scatter_add(src: torch.Tensor, index: torch.Tensor,
                 dim_size: int) -> torch.Tensor:
    """
    Scatter-add: aggregates src values by index along dim=0.
    Equivalent to torch_scatter.scatter(src, index, dim=0, reduce='add').

    src   : (E, d)
    index : (E,)  destination node indices
    returns: (dim_size, d)
    """
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _min_image(diff: torch.Tensor, l_box: float) -> torch.Tensor:
    """Minimum-image convention for PBC: diff → diff - L * round(diff/L)."""
    return diff - l_box * torch.round(diff / l_box)


def _make_complete_edge_index(n: int, batch_size: int, device) -> torch.Tensor:
    """
    Complete-graph edge_index (no self-loops) for B graphs of N nodes each.

    Returns
    -------
    edge_index : (2, B * N * (N-1)) long tensor on `device`
        Row 0: source indices, row 1: destination indices.
        Nodes in graph b occupy positions [b*N, (b+1)*N).
    """
    idx = torch.arange(n, device=device)
    row, col = torch.meshgrid(idx, idx, indexing='ij')   # (N, N) each
    mask = row != col                                      # exclude diagonal
    row, col = row[mask], col[mask]                       # (N*(N-1),) each

    offsets = torch.arange(batch_size, device=device) * n  # (B,)
    row = (row[None] + offsets[:, None]).reshape(-1)       # (B * N*(N-1),)
    col = (col[None] + offsets[:, None]).reshape(-1)

    return torch.stack([row, col], dim=0)                  # (2, B*N*(N-1))


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2,
         activation=nn.SiLU, zero_init_last: bool = False) -> nn.Sequential:
    """Build a simple MLP."""
    layers = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    final = nn.Linear(hidden_dim, out_dim)
    if zero_init_last:
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Single EGNN message-passing layer
# ---------------------------------------------------------------------------

class EGNNLayer(nn.Module):
    """
    One E(n)-equivariant message-passing layer (Satorras et al. 2021).

    Computes
    --------
    h_i' = φ_h( h_i,  Σ_{j≠i} m_{ij} )                  (invariant)
    v_i  = Σ_{j≠i} φ_x( m_{ij} ) * (x_j - x_i)_MIC     (equivariant, optional)

    where  m_{ij} = φ_e( h_i, h_j, ||x_i - x_j||²_MIC ).

    Parameters
    ----------
    in_features     : input node-feature dimension
    hidden_features : width of edge & node MLPs
    out_features    : output node-feature dimension
    update_coords   : whether to also output the equivariant vector v_i
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 update_coords: bool = False):
        super().__init__()
        # φ_e : (h_i, h_j, r_ij²) → m_ij
        self.phi_e = _mlp(2 * in_features + 1, hidden_features, hidden_features,
                          n_hidden=2, zero_init_last=False)
        # φ_h : (h_i, agg_m) → h_i'
        self.phi_h = _mlp(in_features + hidden_features, hidden_features, out_features,
                          n_hidden=1, zero_init_last=(in_features == out_features))
        # φ_x : m_ij → scalar weight (equivariant coordinate update)
        self.update_coords = update_coords
        if update_coords:
            self.phi_x = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, 1),
                nn.Tanh(),   # bound to avoid instability
            )

    def forward(self, h: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, l_box: float | None = None):
        """
        Parameters
        ----------
        h          : (B*N, in_features)
        pos        : (B*N, 3)
        edge_index : (2, E)  E = B * N * (N-1)
        l_box      : float or None — PBC box size

        Returns
        -------
        h_out : (B*N, out_features)
        v_out : (B*N, 3) or None
        """
        src, dst = edge_index                             # (E,) each

        # --- Edge geometry ---
        diff = pos[src] - pos[dst]                        # (E, 3): x_src - x_dst
        if l_box is not None:
            diff = _min_image(diff, l_box)
        dist2 = (diff ** 2).sum(-1, keepdim=True)         # (E, 1)

        # --- Edge messages: φ_e(h_i, h_j, r_ij²) ---
        m = self.phi_e(torch.cat([h[src], h[dst], dist2], dim=-1))  # (E, hidden)

        # --- Aggregate to destination nodes ---
        n_nodes = pos.shape[0]
        m_agg = _scatter_add(m, dst, dim_size=n_nodes)     # (B*N, hidden)

        # --- Node update: φ_h(h_i, Σ m_ji) ---
        h_out = self.phi_h(torch.cat([h, m_agg], dim=-1))  # (B*N, out)

        # --- Equivariant coordinate update: v_dst = Σ_src φ_x(m) * diff ---
        v_out = None
        if self.update_coords:
            w = self.phi_x(m)                              # (E, 1)
            # diff = x_src - x_dst → weighted sum pointing from dst to src neighbors
            v_out = _scatter_add(w * diff, dst, dim_size=n_nodes)   # (B*N, 3)

        return h_out, v_out


# ---------------------------------------------------------------------------
# Full EGNN (stacked layers)
# ---------------------------------------------------------------------------

class EGNN(nn.Module):
    """
    E(n)-equivariant GNN: stacked EGNNLayer with PBC-aware complete graph.

    Takes a BATCH of particle configurations (B, N, 3) and produces:
      - invariant per-particle features  h_out : (B, N, out_features)
      - equivariant per-particle vectors v_out : (B, N, 3)  (last layer only)

    Parameters
    ----------
    n_particles     : number of particles in the graph
    hidden_features : width of all MLPs
    out_features    : output feature dimension per particle
    n_layers        : number of stacked EGNNLayer
    update_coords   : whether the last layer outputs equivariant vectors
    l_box           : float or None — PBC box size (passed to every layer)
    """

    def __init__(self, n_particles: int, hidden_features: int, out_features: int,
                 n_layers: int = 3, update_coords: bool = False,
                 l_box: float | None = None):
        super().__init__()
        self.n_particles = n_particles
        self.hidden_features = hidden_features
        self.l_box = l_box
        self.update_coords = update_coords

        # Initial node embedding: start from zeros (position-only input)
        # h^(0)_i ≡ 0 ∈ R^hidden  — positions enter only via distances in messages
        self.h0_dim = hidden_features

        layers = []
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            in_f = hidden_features
            out_f = out_features if is_last else hidden_features
            layers.append(EGNNLayer(
                in_features=in_f,
                hidden_features=hidden_features,
                out_features=out_f,
                update_coords=(update_coords and is_last),
            ))
        self.layers = nn.ModuleList(layers)

        # Cache edge_index for repeated calls with same (N, B, device)
        self._cached_edge_index: dict = {}

    def _get_edge_index(self, batch_size: int, device) -> torch.Tensor:
        key = (batch_size, str(device))
        if key not in self._cached_edge_index:
            self._cached_edge_index[key] = _make_complete_edge_index(
                self.n_particles, batch_size, device
            )
        return self._cached_edge_index[key]

    def forward(self, pos: torch.Tensor):
        """
        Parameters
        ----------
        pos : (B, N, 3) — particle positions (Cartesian, possibly PBC-wrapped)

        Returns
        -------
        h_out : (B, N, out_features) — invariant per-particle features
        v_out : (B, N, 3) or None    — equivariant vectors (if update_coords)
        """
        B, N, _ = pos.shape
        assert N == self.n_particles, f"Expected {self.n_particles} particles, got {N}"

        # Build complete-graph edge_index (cached)
        edge_index = self._get_edge_index(B, pos.device)  # (2, B*N*(N-1))

        # Fold batch into node dim: (B, N, 3) → (B*N, 3)
        pos_flat = pos.reshape(B * N, 3)

        # Initial node features: zeros
        h = torch.zeros(B * N, self.h0_dim, device=pos.device, dtype=pos.dtype)

        # Forward through layers
        v_flat = None
        for layer in self.layers:
            h, v_flat = layer(h, pos_flat, edge_index, l_box=self.l_box)

        # Unfold back: (B*N, d) → (B, N, d)
        h_out = h.reshape(B, N, -1)
        v_out = v_flat.reshape(B, N, 3) if v_flat is not None else None

        return h_out, v_out


# ---------------------------------------------------------------------------
# Cross-group aggregation  (frozen group A → active group B)
# ---------------------------------------------------------------------------

class CrossGroupConditioner(nn.Module):
    """
    Aggregates EGNN features from frozen group A to produce per-particle
    context vectors for active group B.

    Uses distance-based attention: for each particle b in B, the context is
    a weighted sum of h_a features (A particles), with weights from a learned
    function of the A→B distances.

    Parameters
    ----------
    feature_dim : dimension of h_a features (= EGNN out_features)
    n_heads     : number of attention heads
    l_box       : float or None — PBC box size
    """

    def __init__(self, feature_dim: int, n_heads: int = 4,
                 l_box: float | None = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.l_box = l_box

        # Attention logit: from (h_a, r_ab²) to scalar logit
        self.att_net = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, n_heads),
        )
        # Project multi-head to feature_dim
        self.out_proj = nn.Linear(n_heads * feature_dim, feature_dim)

    def forward(self, h_a: torch.Tensor, pos_a: torch.Tensor,
                pos_b: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_a   : (B, N_A, feature_dim) — frozen group features from EGNN
        pos_a : (B, N_A, 3)           — frozen group positions
        pos_b : (B, N_B, 3)           — active group positions

        Returns
        -------
        context_b : (B, N_B, feature_dim) — context for each active particle
        """
        B, N_A, _ = h_a.shape
        N_B = pos_b.shape[1]

        # Pairwise displacement b → a: (B, N_B, N_A, 3)
        diff_ba = pos_b.unsqueeze(2) - pos_a.unsqueeze(1)  # b→a direction
        if self.l_box is not None:
            diff_ba = _min_image(diff_ba, self.l_box)
        r2_ba = (diff_ba ** 2).sum(-1, keepdim=True)        # (B, N_B, N_A, 1)

        # Attention logits from (h_a, r_ba²): (B, N_B, N_A, n_heads)
        h_a_exp = h_a.unsqueeze(1).expand(B, N_B, N_A, -1)  # (B, N_B, N_A, d)
        att_inp = torch.cat([h_a_exp, r2_ba], dim=-1)        # (B, N_B, N_A, d+1)
        logits = self.att_net(att_inp)                        # (B, N_B, N_A, heads)
        weights = torch.softmax(logits, dim=2)                # (B, N_B, N_A, heads)

        # Weighted sum of h_a: (B, N_B, heads, d)
        # weights: (B, N_B, N_A, heads), h_a_exp: (B, N_B, N_A, d)
        # want: (B, N_B, heads, d)
        ctx = torch.einsum('biah,biad->bihd', weights, h_a_exp)  # (B, N_B, heads, d)
        ctx = ctx.reshape(B, N_B, self.n_heads * self.feature_dim)
        context_b = self.out_proj(ctx)                        # (B, N_B, d)

        return context_b


class CrossGroupEquivariant(nn.Module):
    """
    Computes equivariant rotation parameters for active group B from frozen group A.

    To ensure invertibility of the AngularCouplingLayer, the rotation parameters
    must NOT depend on the active group positions pos_b (which change during the
    forward pass). Instead:

    - Equivariant axis (shared across all active particles):
        v = Σ_a φ_x(h_a, ||x_a||²) * x_a
      This is E(3)-equivariant: v → R @ v under rotation.

    - Invariant per-active-particle angles:
        theta_b = π * σ( MLP(mean(h_a)) )_b
      Each active particle b gets a different angle via a linear projection
      of the frozen group's mean features. Invariant under rotation.

    Parameters
    ----------
    feature_dim : dimension of h_a features
    hidden_dim  : hidden width of MLPs
    n_active    : number of active particles (N_B)
    l_box       : unused (kept for interface compatibility)
    """

    def __init__(self, feature_dim: int, hidden_dim: int,
                 n_active: int = 1,
                 l_box: float | None = None):
        super().__init__()
        self.n_active = n_active
        self.phi_x = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        # Outputs n_active rotation angles (one per active particle slot)
        self.phi_theta = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_active),
            nn.Sigmoid(),   # maps to (0, 1), scaled to (0, π) outside
        )

    def forward(self, h_a: torch.Tensor,
                pos_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_a   : (B, N_A, feature_dim) — frozen group features from EGNN
        pos_a : (B, N_A, 3)           — frozen group positions

        Returns
        -------
        v     : (B, 3)        — shared equivariant rotation axis
        theta_b : (B, N_active, 1) — per-active-particle rotation angles in (0, π)
        """
        B, N_A, _ = h_a.shape

        # Equivariant axis: weighted sum of frozen positions
        # Weight depends only on h_a and ||x_a||² (invariant under rotation)
        r2_a = (pos_a ** 2).sum(-1, keepdim=True)       # (B, N_A, 1)
        inp = torch.cat([h_a, r2_a], dim=-1)             # (B, N_A, d+1)
        w = self.phi_x(inp)                               # (B, N_A, 1)
        v = (w * pos_a).sum(dim=1)                       # (B, 3)

        # Per-active-particle invariant angles from mean frozen features
        h_agg = h_a.mean(dim=1)                          # (B, d)
        theta_b = math.pi * self.phi_theta(h_agg)        # (B, n_active)
        theta_b = theta_b.unsqueeze(-1)                   # (B, n_active, 1)

        return v, theta_b

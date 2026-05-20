import copy
import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# ---------------------------------------------------------------------------
# RealNVP  (CPU version)
# ---------------------------------------------------------------------------

class RealNVP(nn.Module):
    """
    RealNVP normalising flow for N-particle systems in arbitrary dimension.

    The flat configuration vector has length ``dimension = N * d`` where ``d``
    is the spatial dimension (e.g. d=3 for 3D).  The network operates on this
    flat vector; ``sys_dim`` is the tuple used to reshape it back to (N, d)
    when computing the energy.

    Parameters
    ----------
    s_net, t_net : callable
        Factory functions that return a new nn.Module when called (one per
        coupling block).
    mask : torch.Tensor, shape (2 * n_blocks, dimension)
        Binary mask for the affine coupling layers.
    prior : torch.distributions object
        Base distribution in the latent space (usually N(0, I)).
    system : potentials object
        Must expose ``get_energy_batch(batch)`` → (B,) tensor.
    sys_dim : tuple
        Shape used to reshape a flat sample, e.g. (N, 3).
    """

    def __init__(self, s_net, t_net, mask, prior, system, sys_dim):
        super().__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nn.ModuleList([t_net() for _ in range(len(mask))])
        self.s = nn.ModuleList([s_net() for _ in range(len(mask))])
        self.system = system
        self.sys_dim = sys_dim

    # ------------------------------------------------------------------
    # Flow transforms
    # ------------------------------------------------------------------

    def inverse_generator(self, x):
        """x (configuration) → z (latent),  returns (z, log|det J_xz|)."""
        z = x
        log_R_xz = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z * torch.exp(s) + t)
            log_R_xz += torch.sum(s, dim=-1)
        return z, log_R_xz

    def generator(self, z):
        """z (latent) → x (configuration),  returns (x, log|det J_zx|)."""
        x = z
        log_R_zx = z.new_zeros(z.shape[0])
        for i in range(len(self.t)):
            x_ = self.mask[i] * x
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x - t) * torch.exp(-s)
            log_R_zx -= torch.sum(s, dim=-1)
        return x, log_R_zx

    # ------------------------------------------------------------------
    # Energy (vectorised over batch)
    # ------------------------------------------------------------------

    def calculate_energy(self, batch, space):
        """
        Batch energy computation — no per-sample Python loop.

        Parameters
        ----------
        batch : torch.Tensor, shape (B, dimension)
        space : str
            'configuration' uses ``system.get_energy_batch``;
            'latent' uses isotropic Gaussian energy 0.5 * ||z||^2.

        Returns
        -------
        energy : torch.Tensor, shape (B,)
        """
        if space == 'configuration':
            energy = self.system.get_energy_batch(batch)
        elif space == 'latent':
            energy = 0.5 * (batch ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown space '{space}'. Use 'configuration' or 'latent'.")
        return energy

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def loss_ML(self, batch_x, weighted=False):
        """Forward KL  J_ML = E_data[ u_z(F^{-1}(x)) - log|det J_xz| ]."""
        z, log_R_xz = self.inverse_generator(batch_x)
        u_z = self.calculate_energy(z, space='latent')
        if not weighted:
            return (u_z - log_R_xz).mean()
        else:
            u_x = self.calculate_energy(batch_x, space='configuration')
            w = torch.exp(-u_x)
            return (w * (u_z - log_R_xz)).sum() / w.sum()

    def loss_KL(self, batch_z, weighted=False, energy_cap=None):
        """Reverse KL  J_KL = E_z[ u_x(F(z)) - log|det J_zx| ].

        energy_cap : float or None
            If set, energies above this value are soft-capped via
            cap + log1p(u_x - cap) so gradients still flow but the
            loss cannot explode from overlapping configurations.
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration')
        if energy_cap is not None:
            u_x = torch.where(
                u_x < energy_cap,
                u_x,
                energy_cap + torch.log1p(u_x - energy_cap),
            )
        if not weighted:
            return (u_x - log_R_zx).mean()
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            w = torch.exp(-u_z)
            return (w * (u_x - log_R_zx)).sum() / w.sum()

    def expectation(self, observable, weights=None):
        if weights is None:
            return observable.mean()
        return (observable * weights).sum() / weights.sum()


# ---------------------------------------------------------------------------
# E(n)-equivariant flow factory
# ---------------------------------------------------------------------------

def build_egnn_solute_flow_3d(
    system,
    n_particles: int = 36,
    n_blocks: int = 8,
    egnn_hidden: int = 64,
    egnn_layers: int = 3,
    num_bins: int = 8,
    tail_bound: float = 4.0,
    l_box: float | None = None,
    att_heads: int = 4,
    r_min_factor: float = 0.5,
    prior_sigma_r: float = 1.0,
):
    """
    Build an E(n)-equivariant normalizing flow for the 3D PBC solute system.

    Each block alternates between:
      - RadialCouplingLayer  (EGNN-conditioned spline on log r_i)
      - AngularCouplingLayer (EGNN equivariant Rodrigues rotation of u_i)

    The flow is a drop-in replacement for CartesianSplineFlow in training.py.

    Parameters
    ----------
    system        : SoluteSimulationPBC3D
    n_particles   : int    number of SOLVENT particles (36)
    n_blocks      : int    A-B coupling cycles; total layers = n_blocks * 4
    egnn_hidden   : int    hidden width of EGNN MLPs
    egnn_layers   : int    number of EGNN message-passing layers per conditioner
    num_bins      : int    RQ spline bins for the radial transformation
    tail_bound    : float  spline covers [-tail_bound, tail_bound] on log(r)
    l_box         : float or None  PBC box half-width (None = no PBC)
    att_heads     : int    cross-group attention heads in CrossGroupConditioner
    r_min_factor  : float  each RadialCouplingLayer clamps generated r to at
                           least r_min_factor * system.sigma.  Set to 0 to
                           disable.  Default 0.5 gives r_min = 0.5*sigma.
    prior_sigma_r : float  std of log(r) in the SphericalPrior.  Default 1.0.

    Returns
    -------
    EGNNEquivariantFlow
    """
    import sys
    from pathlib import Path
    _LIB = Path(__file__).resolve().parent
    if str(_LIB) not in sys.path:
        sys.path.insert(0, str(_LIB))

    import torch
    import torch.nn as nn
    from egnn import EGNN, CrossGroupConditioner, CrossGroupEquivariant
    from egnn_flow import (RadialCouplingLayer, AngularCouplingLayer,
                           EGNNEquivariantFlow)
    from spline_flow import SphericalPrior

    # ---- Particle groups: even / odd index split ----
    all_idx = torch.arange(n_particles)
    group_A = all_idx[all_idx % 2 == 0]   # 0,2,4,...,34  (18 particles)
    group_B = all_idx[all_idx % 2 == 1]   # 1,3,5,...,35  (18 particles)
    N_A = len(group_A)
    N_B = len(group_B)

    ppc = 3 * num_bins - 1               # RQ spline params per scalar
    r_min = r_min_factor * system.sigma if r_min_factor > 0.0 else 0.0

    def _make_radial_layer(frozen_idx, active_idx, n_frozen, n_active):
        egnn_f = EGNN(
            n_particles=n_frozen,
            hidden_features=egnn_hidden,
            out_features=egnn_hidden,
            n_layers=egnn_layers,
            update_coords=False,
            l_box=l_box,
        )
        cross = CrossGroupConditioner(
            feature_dim=egnn_hidden,
            n_heads=att_heads,
            l_box=l_box,
        )
        param_head = nn.Sequential(
            nn.Linear(egnn_hidden, egnn_hidden),
            nn.SiLU(),
            nn.Linear(egnn_hidden, ppc),
        )
        nn.init.zeros_(param_head[-1].weight)
        nn.init.zeros_(param_head[-1].bias)

        return RadialCouplingLayer(
            egnn_frozen=egnn_f,
            cross_cond=cross,
            param_head=param_head,
            frozen_idx=frozen_idx,
            active_idx=active_idx,
            num_bins=num_bins,
            tail_bound=tail_bound,
            l_box=l_box,
            r_min=r_min,
        )

    def _make_angular_layer(frozen_idx, active_idx, n_frozen, n_active):
        egnn_e = EGNN(
            n_particles=n_frozen,
            hidden_features=egnn_hidden,
            out_features=egnn_hidden,
            n_layers=egnn_layers,
            update_coords=False,
            l_box=l_box,
        )
        cross_eq = CrossGroupEquivariant(
            feature_dim=egnn_hidden,
            hidden_dim=egnn_hidden,
            n_active=n_active,
            l_box=l_box,
        )
        return AngularCouplingLayer(
            egnn_equivariant=egnn_e,
            cross_equi=cross_eq,
            frozen_idx=frozen_idx,
            active_idx=active_idx,
            l_box=l_box,
        )

    layers = []
    for _ in range(n_blocks):
        layers.append(_make_radial_layer(group_A, group_B, N_A, N_B))
        layers.append(_make_angular_layer(group_A, group_B, N_A, N_B))
        layers.append(_make_radial_layer(group_B, group_A, N_B, N_A))
        layers.append(_make_angular_layer(group_B, group_A, N_B, N_A))

    prior = SphericalPrior(n_particles=n_particles, sigma_r=prior_sigma_r)

    return EGNNEquivariantFlow(
        layers=layers,
        prior=prior,
        system=system,
        n_particles=n_particles,
        fixed_solute=True,
    )


# ---------------------------------------------------------------------------
# RealNVP  (CUDA / GPU version — identical logic, tensors on device)
# ---------------------------------------------------------------------------

class RealNVPCUDA(nn.Module):
    """GPU-enabled RealNVP.  Same interface as RealNVP."""

    def __init__(self, s_net, t_net, mask, prior, system, sys_dim):
        super().__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False).to(device)
        self.t = nn.ModuleList([t_net() for _ in range(len(mask))]).to(device)
        self.s = nn.ModuleList([s_net() for _ in range(len(mask))]).to(device)
        self.system = system
        self.sys_dim = sys_dim

    def inverse_generator(self, x):
        z = x
        log_R_xz = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z * torch.exp(s) + t)
            log_R_xz += torch.sum(s, dim=-1)
        return z, log_R_xz

    def generator(self, z):
        x = z
        log_R_zx = z.new_zeros(z.shape[0])
        for i in range(len(self.t)):
            x_ = self.mask[i] * x
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x - t) * torch.exp(-s)
            log_R_zx -= torch.sum(s, dim=-1)
        return x, log_R_zx

    def calculate_energy(self, batch, space):
        if space == 'configuration':
            return self.system.get_energy_batch(batch)
        elif space == 'latent':
            return 0.5 * (batch ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown space '{space}'.")

    def loss_ML(self, batch_x, weighted=False):
        z, log_R_xz = self.inverse_generator(batch_x)
        u_z = self.calculate_energy(z, space='latent')
        if not weighted:
            return (u_z - log_R_xz).mean()
        else:
            u_x = self.calculate_energy(batch_x, space='configuration')
            w = torch.exp(-u_x)
            return (w * (u_z - log_R_xz)).sum() / w.sum()

    def loss_KL(self, batch_z, weighted=False, energy_cap=None):
        """Reverse KL  J_KL = E_z[ u_x(F(z)) - log|det J_zx| ].

        energy_cap : float or None
            If set, energies above this value are soft-capped via
            cap + log1p(u_x - cap) so gradients still flow but the
            loss cannot explode from overlapping configurations.
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration')
        if energy_cap is not None:
            u_x = torch.where(
                u_x < energy_cap,
                u_x,
                energy_cap + torch.log1p(u_x - energy_cap),
            )
        if not weighted:
            return (u_x - log_R_zx).mean()
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            w = torch.exp(-u_z)
            return (w * (u_x - log_R_zx)).sum() / w.sum()

    def expectation(self, observable, weights=None):
        if weights is None:
            return observable.mean()
        return (observable * weights).sum() / weights.sum()

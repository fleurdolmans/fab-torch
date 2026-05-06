import math
import torch
import torch.nn as nn

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


# ============================================================
# Prior
# ============================================================

class GaussianPrior:
    """Isotropic Gaussian prior over flattened 2D solvent coordinates."""

    def __init__(self, dim, sigma=1.0):
        self.dim = int(dim)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * self.dim * (
            math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma)
        )

    def sample(self, shape, device=None, dtype=torch.float32):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.randn(*shape, self.dim, device=device, dtype=dtype) * self.sigma

    def log_prob(self, z):
        return -0.5 * (z / self.sigma).pow(2).sum(dim=-1) + self._log_norm


# ============================================================
# Small utilities
# ============================================================

def rotate_vectors(x, angle):
    """
    Rotate each 2D vector x_i by its own scalar angle_i.

    x     : (B, N, 2)
    angle : (B, N)
    """
    c = torch.cos(angle)
    s = torch.sin(angle)

    x0 = x[..., 0]
    x1 = x[..., 1]

    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1

    return torch.stack([y0, y1], dim=-1)


def safe_norm(x, eps=1e-8):
    return torch.sqrt((x * x).sum(dim=-1) + eps)


# ============================================================
# Permutation-equivariant radial spline conditioner
# ============================================================

class DirectionalRadialConditioner(nn.Module):
    """
    Conditioner for radial spline layers.

    Input:
        u : (B, N, 2), unit direction vectors x_i / ||x_i||

    Output:
        spline parameters per particle:
        params : (B, N, 3 * num_bins - 1)

    The conditioner uses pairwise dot products u_i · u_j.
    These are rotation-invariant and permutation-equivariant.
    """

    def __init__(self, num_bins, hidden=128):
        super().__init__()

        self.num_bins = int(num_bins)
        self.params_per_particle = 3 * self.num_bins - 1

        self.edge_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.node_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.params_per_particle),
        )

        # Near-identity initialisation.
        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    def forward(self, u):
        # cos_ij: (B, N, N, 1)
        cos_ij = torch.einsum("bid,bjd->bij", u, u).unsqueeze(-1)

        messages = self.edge_net(cos_ij)       # (B, N, N, H)
        context = messages.sum(dim=2)          # (B, N, H)

        params = self.node_net(context)        # (B, N, 3*num_bins - 1)
        return params


# ============================================================
# Permutation-equivariant angular conditioner
# ============================================================

class RadiusAngularConditioner(nn.Module):
    """
    Conditioner for angular rotation layers.

    Input:
        log_r : (B, N)

    Output:
        delta_theta : (B, N)

    This is permutation-equivariant. Since it depends only on radii,
    the angular layer is exactly invertible: radii are unchanged.
    """

    def __init__(self, hidden=128, max_angle=0.5):
        super().__init__()

        self.max_angle = float(max_angle)

        self.edge_net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.node_net = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # Near-identity initialisation.
        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    def forward(self, log_r):
        """
        log_r : (B, N)
        """
        ri = log_r.unsqueeze(2)                 # (B, N, 1)
        rj = log_r.unsqueeze(1)                 # (B, 1, N)

        diff = ri - rj                          # (B, N, N)
        absdiff = diff.abs()

        edge_features = torch.stack(
            [
                ri.expand_as(diff),
                rj.expand_as(diff),
                absdiff,
            ],
            dim=-1,
        )                                       # (B, N, N, 3)

        messages = self.edge_net(edge_features) # (B, N, N, H)
        context = messages.sum(dim=2)           # (B, N, H)

        node_input = torch.cat(
            [log_r.unsqueeze(-1), context],
            dim=-1,
        )                                       # (B, N, H+1)

        raw_delta = self.node_net(node_input).squeeze(-1)  # (B, N)

        # Bounded angle update for training stability.
        delta = self.max_angle * torch.tanh(raw_delta)

        return delta


# ============================================================
# Radial spline layer
# ============================================================

class SphericalRadialSplineLayer(nn.Module):
    """
    Rotation-equivariant, permutation-equivariant radial spline layer.

    Decompose each particle as:

        x_i = r_i u_i

    Then transform:

        log r_i -> spline_i(log r_i)
        u_i unchanged

    Because the conditioner depends only on directions u_i, which are unchanged
    by this layer, forward and inverse use the same conditioner values.
    """

    def __init__(
        self,
        num_bins=8,
        hidden=128,
        tail_bound=5.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        eps=1e-8,
    ):
        super().__init__()

        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.eps = eps

        self.conditioner = DirectionalRadialConditioner(
            num_bins=num_bins,
            hidden=hidden,
        )

    def _split_params(self, params):
        nb = self.num_bins
        widths = params[..., :nb]
        heights = params[..., nb:2 * nb]
        derivatives = params[..., 2 * nb:]
        return widths, heights, derivatives

    def forward(self, x):
        """
        x : (B, N, 2)

        Returns
        -------
        y      : (B, N, 2)
        logdet : (B,)
        """
        r = safe_norm(x, self.eps)                  # (B, N)
        u = x / r.unsqueeze(-1)                     # (B, N, 2)
        log_r = torch.log(r)                        # (B, N)

        params = self.conditioner(u)
        widths, heights, derivatives = self._split_params(params)

        log_r_new, ld_spline = unconstrained_rational_quadratic_spline(
            inputs=log_r,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        r_new = torch.exp(log_r_new)
        y = r_new.unsqueeze(-1) * u

        # In 2D:
        # Cartesian -> (log r, theta): -2 log r
        # spline on log r: ld_spline
        # (log r_new, theta) -> Cartesian: +2 log r_new
        logdet = (ld_spline + 2.0 * (log_r_new - log_r)).sum(dim=1)

        return y, logdet

    def inverse(self, y):
        r_new = safe_norm(y, self.eps)
        u = y / r_new.unsqueeze(-1)
        log_r_new = torch.log(r_new)

        params = self.conditioner(u)
        widths, heights, derivatives = self._split_params(params)

        log_r, ld_spline_inv = unconstrained_rational_quadratic_spline(
            inputs=log_r_new,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        r = torch.exp(log_r)
        x = r.unsqueeze(-1) * u

        logdet = (ld_spline_inv + 2.0 * (log_r - log_r_new)).sum(dim=1)

        return x, logdet


# ============================================================
# Angular equivariant rotation layer
# ============================================================

class SphericalAngularLayer(nn.Module):
    """
    Rotation-equivariant, permutation-equivariant angular layer.

    Transform:

        x_i -> R(delta_i) x_i

    where delta_i is a scalar predicted from all particle radii.

    Radii are unchanged, so inverse is exact:

        x_i -> R(-delta_i) x_i

    Log determinant is zero because this is a rotation per particle.
    """

    def __init__(self, hidden=128, max_angle=0.5, eps=1e-8):
        super().__init__()
        self.conditioner = RadiusAngularConditioner(
            hidden=hidden,
            max_angle=max_angle,
        )
        self.eps = eps

    def forward(self, x):
        r = safe_norm(x, self.eps)
        log_r = torch.log(r)

        delta = self.conditioner(log_r)
        y = rotate_vectors(x, delta)

        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return y, logdet

    def inverse(self, y):
        r = safe_norm(y, self.eps)
        log_r = torch.log(r)

        delta = self.conditioner(log_r)
        x = rotate_vectors(y, -delta)

        logdet = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        return x, logdet


# ============================================================
# Full spherical equivariant Boltzmann generator
# ============================================================

class SphericalEquivariantBoltzmannGenerator(nn.Module):
    """
    Spherical equivariant Boltzmann generator for a 2D solute-solvent system.

    Models only solvent particles. The solute is assumed fixed at the origin.

    Symmetries:
      - permutation equivariant over solvent particles
      - SO(2) rotation equivariant
      - solute-centered translation handling
      - bijective except at r=0, which is avoided numerically by eps

    API:
      - generator(z)
      - inverse_generator(x)
      - loss_ML(batch_x)
      - loss_KL(batch_z)
      - sample(n)
    """

    def __init__(
        self,
        system,
        n_particles=36,
        n_blocks=8,
        num_bins=8,
        hidden=128,
        tail_bound=5.0,
        max_angle=0.5,
        energy_cap=1e4,
        eps=1e-8,
    ):
        super().__init__()

        self.system = system
        self.n_particles = int(n_particles)
        self.sys_dim = (self.n_particles, 2)
        self.dim = 2 * self.n_particles
        self.energy_cap = float(energy_cap)
        self.eps = eps

        layers = []
        for _ in range(n_blocks):
            layers.append(
                SphericalRadialSplineLayer(
                    num_bins=num_bins,
                    hidden=hidden,
                    tail_bound=tail_bound,
                    eps=eps,
                )
            )
            layers.append(
                SphericalAngularLayer(
                    hidden=hidden,
                    max_angle=max_angle,
                    eps=eps,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.prior = GaussianPrior(dim=self.dim)

    def _to_particle(self, flat):
        return flat.view(flat.shape[0], self.n_particles, 2)

    def _to_flat(self, x):
        return x.reshape(x.shape[0], -1)

    def _add_fixed_solute(self, solvent_flat):
        B = solvent_flat.shape[0]

        solvent = solvent_flat.view(B, self.n_particles, 2)

        solute = torch.zeros(
            B,
            1,
            2,
            device=solvent.device,
            dtype=solvent.dtype,
        )

        full = torch.cat([solute, solvent], dim=1)
        return full.reshape(B, -1)

    def forward_map(self, x):
        """
        Configuration x -> latent z.

        x : (B, N, 2)
        """
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld

        return x, logdet

    def inverse_map(self, z):
        """
        Latent z -> configuration x.

        z : (B, N, 2)
        """
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld

        return z, logdet

    def inverse_generator(self, x_flat):
        """
        Configuration -> latent.

        x_flat : (B, N*2)

        Returns
        -------
        z_flat     : (B, N*2)
        logdet_xz  : (B,)
        """
        x = self._to_particle(x_flat)
        z, logdet = self.forward_map(x)
        return self._to_flat(z), logdet

    def generator(self, z_flat):
        """
        Latent -> configuration.

        z_flat : (B, N*2)

        Returns
        -------
        x_flat     : (B, N*2)
        logdet_zx  : (B,)
        """
        z = self._to_particle(z_flat)
        x, logdet = self.inverse_map(z)
        return self._to_flat(x), logdet

    def calculate_energy(self, solvent_flat):
        """
        solvent_flat : (B, N*2)

        Prepends the fixed solute at the origin, then evaluates system energy.
        """
        full_flat = self._add_fixed_solute(solvent_flat)

        energies = []
        for i in range(full_flat.shape[0]):
            e = self.system.get_energy(
                full_flat[i].reshape(self.n_particles + 1, 2)
            )

            if not isinstance(e, torch.Tensor):
                e = torch.tensor(
                    e,
                    device=full_flat.device,
                    dtype=full_flat.dtype,
                )
            else:
                e = e.to(device=full_flat.device, dtype=full_flat.dtype)

            energies.append(e)

        energies = torch.stack(energies)

        cap = self.energy_cap
        energies = torch.where(
            energies < cap,
            energies,
            cap + torch.log1p(energies - cap),
        )

        return energies

    def loss_ML(self, batch_x):
        """
        Maximum-likelihood loss on samples from the target distribution.

        batch_x : (B, N*2)
        """
        z, logdet_xz = self.inverse_generator(batch_x)
        log_pz = self.prior.log_prob(z)

        return -(log_pz + logdet_xz).mean()

    def loss_KL(self, batch_z):
        """
        Reverse-KL / energy training loss.

        J_KL = E_z[ u_x(G(z)) - log |det dG/dz| ]

        batch_z : (B, N*2)
        """
        x, logdet_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x)

        return (u_x - logdet_zx).mean()

    def sample(self, n_samples, device=None, dtype=torch.float32):
        z = self.prior.sample(
            n_samples,
            device=device,
            dtype=dtype,
        )
        return self.generator(z)


def build_spherical_equivariant_flow(
    system,
    n_particles=36,
    n_blocks=8,
    num_bins=8,
    hidden=128,
    tail_bound=5.0,
    max_angle=0.5,
):
    return SphericalEquivariantBoltzmannGenerator(
        system=system,
        n_particles=n_particles,
        n_blocks=n_blocks,
        num_bins=num_bins,
        hidden=hidden,
        tail_bound=tail_bound,
        max_angle=max_angle,
    )


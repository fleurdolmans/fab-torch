import math
import torch
import torch.nn as nn

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


class GaussianPrior:
    def __init__(self, dim, sigma=1.0):
        self.dim = int(dim)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * self.dim * (
            math.log(2 * math.pi) + 2 * math.log(self.sigma)
        )

    def sample(self, shape, device=None, dtype=torch.float32):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.randn(*shape, self.dim, device=device, dtype=dtype) * self.sigma

    def log_prob(self, z):
        return -0.5 * (z / self.sigma).pow(2).sum(-1) + self._log_norm


class DirectionEquivariantConditioner(nn.Module):
    """
    Permutation-equivariant, rotation-invariant conditioner.

    Input:
        u: (B, N, 2) unit vectors from solute to solvent particles

    Output:
        spline parameters per particle: (B, N, 3*num_bins - 1)

    Uses only angular information via dot products u_i · u_j.
    This keeps the radial transform exactly invertible because directions
    are unchanged by the radial layer.
    """

    def __init__(self, num_bins, hidden=128):
        super().__init__()
        self.num_bins = num_bins
        self.params_per_particle = 3 * num_bins - 1

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

        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    def forward(self, u):
        cos_ij = torch.einsum("bid,bjd->bij", u, u).unsqueeze(-1)
        messages = self.edge_net(cos_ij)
        h = messages.sum(dim=2)
        return self.node_net(h)


class RadiusEquivariantConditioner(nn.Module):
    """
    Permutation-equivariant, rotation-invariant conditioner using radii as features.

    Input:
        log_r: (B, N) log-radii

    Output:
        spline parameters per particle: (B, N, 3*num_bins - 1)

    Used by the angular spline layer.  Radii are unchanged in that layer, so
    conditioning on log_r_j (including j=i) is valid — no circular dependency.
    """

    def __init__(self, num_bins, hidden=128):
        super().__init__()
        self.num_bins = num_bins
        self.params_per_particle = 3 * num_bins - 1

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

        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    def forward(self, log_r):
        N = log_r.shape[1]
        # Edge feature: log r_j for each pair (i, j)
        log_r_j = log_r.unsqueeze(1).expand(-1, N, -1).unsqueeze(-1)  # (B, N, N, 1)
        messages = self.edge_net(log_r_j)                              # (B, N, N, hidden)
        h = messages.sum(dim=2)                                        # (B, N, hidden)
        return self.node_net(h)                                        # (B, N, params)


class RadialSplineEquivariantLayer(nn.Module):
    """
    Bijective radial spline layer.

    x_i = r_i u_i
    log r_i -> spline(log r_i)
    u_i unchanged

    Symmetries:
      - permutation equivariant
      - rotation equivariant
      - translation handled by using solute-centered coordinates
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
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.eps = eps

        self.conditioner = DirectionEquivariantConditioner(
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
        r = x.norm(dim=-1).clamp_min(self.eps)
        u = x / r.unsqueeze(-1)
        log_r = torch.log(r)

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

        # Cartesian logdet for 2D radial transform:
        # d(x,y) -> d(log r, theta): -2 log r
        # spline: ld_spline
        # d(log r_new, theta) -> d(x_new,y_new): +2 log r_new
        logdet = (ld_spline + 2.0 * (log_r_new - log_r)).sum(dim=1)

        return y, logdet

    def inverse(self, y):
        r_new = y.norm(dim=-1).clamp_min(self.eps)
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


class AngularSplineEquivariantLayer(nn.Module):
    """
    Bijective angular spline layer.

    x_i = r_i u_i
    theta_i = atan2(u_i[1], u_i[0]) -> spline(theta_i)
    r_i unchanged

    Conditioner uses log r (which is unchanged by this layer -> valid, exact inverse).

    Logdet derivation:
        d(x,y) = r d(r, theta)  [polar area element]
        Since r is unchanged: d(x_new, y_new) = r d(r, theta_new)
        Cartesian logdet = log |d theta_new / d theta| = ld_spline

    Branch-cut handling:
        tail_bound = pi covers the full domain (-pi, pi).
        Linear tails never activate for real angle values.

    Symmetries:
      - permutation equivariant (shared conditioner architecture, sum aggregation)
      - rotation equivariant (conditioning on radii is rotation-invariant)
    """

    def __init__(
        self,
        num_bins=8,
        hidden=128,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        eps=1e-8,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.eps = eps

        self.conditioner = RadiusEquivariantConditioner(
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
        r = x.norm(dim=-1).clamp_min(self.eps)
        u = x / r.unsqueeze(-1)
        theta = torch.atan2(u[..., 1], u[..., 0])   # (B, N)
        log_r = torch.log(r)

        params = self.conditioner(log_r)
        widths, heights, derivatives = self._split_params(params)

        theta_new, ld = unconstrained_rational_quadratic_spline(
            inputs=theta,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=math.pi,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        u_new = torch.stack([theta_new.cos(), theta_new.sin()], dim=-1)
        y = r.unsqueeze(-1) * u_new
        logdet = ld.sum(dim=1)

        return y, logdet

    def inverse(self, y):
        r = y.norm(dim=-1).clamp_min(self.eps)
        u = y / r.unsqueeze(-1)
        theta_new = torch.atan2(u[..., 1], u[..., 0])
        log_r = torch.log(r)   # r unchanged → identical conditioning as forward

        params = self.conditioner(log_r)
        widths, heights, derivatives = self._split_params(params)

        theta, ld = unconstrained_rational_quadratic_spline(
            inputs=theta_new,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=math.pi,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        u_old = torch.stack([theta.cos(), theta.sin()], dim=-1)
        x = r.unsqueeze(-1) * u_old
        logdet = ld.sum(dim=1)

        return x, logdet


class EquivariantBoltzmannGenerator(nn.Module):
    def __init__(
        self,
        system,
        n_particles=36,
        n_layers=8,
        num_bins=8,
        hidden=128,
        tail_bound=5.0,
        energy_cap=1e4,
        layer_type='radial',
    ):
        super().__init__()

        self.system = system
        self.n_particles = int(n_particles)
        self.sys_dim = (self.n_particles, 2)
        self.energy_cap = float(energy_cap)

        layers = []
        for i in range(n_layers):
            if layer_type == 'alternating' and i % 2 == 1:
                layers.append(AngularSplineEquivariantLayer(
                    num_bins=num_bins,
                    hidden=hidden,
                ))
            else:
                layers.append(RadialSplineEquivariantLayer(
                    num_bins=num_bins,
                    hidden=hidden,
                    tail_bound=tail_bound,
                ))
        self.layers = nn.ModuleList(layers)

        self.prior = GaussianPrior(dim=2 * self.n_particles)

    def _to_particle(self, flat):
        return flat.view(flat.shape[0], self.n_particles, 2)

    def _to_flat(self, x):
        return x.reshape(x.shape[0], -1)

    def _add_fixed_solute(self, solvent_flat):
        B = solvent_flat.shape[0]
        solvent = solvent_flat.view(B, self.n_particles, 2)

        solute = torch.zeros(
            B, 1, 2,
            device=solvent.device,
            dtype=solvent.dtype,
        )

        full = torch.cat([solute, solvent], dim=1)
        return full.reshape(B, -1)

    def forward_map(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld

        return x, logdet

    def inverse_map(self, z):
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld

        return z, logdet

    def inverse_generator(self, x_flat):
        x = self._to_particle(x_flat)
        z, logdet = self.forward_map(x)
        return self._to_flat(z), logdet

    def generator(self, z_flat):
        z = self._to_particle(z_flat)
        x, logdet = self.inverse_map(z)
        return self._to_flat(x), logdet

    def loss_ML(self, batch_x):
        z, logdet = self.inverse_generator(batch_x)
        log_pz = self.prior.log_prob(z)
        return -(log_pz + logdet).mean()

    def calculate_energy(self, x_flat):
        x_full = self._add_fixed_solute(x_flat)

        energies = torch.stack([
            self.system.get_energy(x_full[i].reshape(self.n_particles + 1, 2))
            for i in range(x_full.shape[0])
        ])

        cap = self.energy_cap
        return torch.where(
            energies < cap,
            energies,
            cap + torch.log1p(energies - cap),
        )

    def loss_KL(self, batch_z):
        x, logdet = self.generator(batch_z)
        u_x = self.calculate_energy(x)
        return (u_x - logdet).mean()


def build_equivariant_boltzmann_generator(
    system,
    n_particles=36,
    n_layers=8,
    num_bins=8,
    hidden=128,
    tail_bound=5.0,
    layer_type='radial',
):
    """
    Build an equivariant Boltzmann generator.

    Args:
        layer_type: 'radial' (default) — only radial spline layers, identical to the
                    original architecture.
                    'alternating' — interleave radial and angular spline layers
                    [R, A, R, A, ...] for n_layers total.  Angular layers transform
                    theta_i conditioned on radii r (valid: r unchanged in angular layers).
    """
    return EquivariantBoltzmannGenerator(
        system=system,
        n_particles=n_particles,
        n_layers=n_layers,
        num_bins=num_bins,
        hidden=hidden,
        tail_bound=tail_bound,
        layer_type=layer_type,
    )
import math
import torch
import torch.nn as nn


# ============================================================
# Prior
# ============================================================

class GaussianPrior:
    def __init__(self, dim, sigma=1.0):
        self.dim = int(dim)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * self.dim * (
            math.log(2.0 * math.pi) + 2.0 * math.log(self.sigma)
        )

    def sample(self, shape, device=None, dtype=torch.float32):
        if isinstance(shape, int):
            shape = (shape,)

        return torch.randn(
            (*shape, self.dim),
            device=device,
            dtype=dtype,
        ) * self.sigma

    def log_prob(self, z):
        return -0.5 * (z / self.sigma).pow(2).sum(dim=-1) + self._log_norm


# ============================================================
# Box transform: unconstrained R -> finite box [-L, L]
# ============================================================

class BoxTanhTransform(nn.Module):
    """
    Elementwise bijection:

        x = L * tanh(u)

    maps unconstrained coordinates u in R to physical coordinates x in (-L, L).

    This makes generated samples stay inside the square box.
    """

    def __init__(self, L_BOX, eps=1e-6):
        super().__init__()
        self.L_BOX = float(L_BOX)
        self.eps = float(eps)

    def forward(self, u):
        """
        u -> x_box

        u: (B, D)

        returns:
            x: (B, D)
            logdet: log |dx/du|, shape (B,)
        """
        y = torch.tanh(u)
        x = self.L_BOX * y

        # dx/du = L * (1 - tanh(u)^2)
        logdet = (
            math.log(self.L_BOX)
            + torch.log1p(-y.pow(2) + self.eps)
        ).sum(dim=-1)

        return x, logdet

    def inverse(self, x):
        """
        x_box -> u

        x: (B, D)

        returns:
            u: (B, D)
            logdet: log |du/dx|, shape (B,)
        """
        y = x / self.L_BOX
        y = y.clamp(-1.0 + self.eps, 1.0 - self.eps)

        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        # du/dx = 1 / [L * (1 - y^2)]
        logdet = -(
            math.log(self.L_BOX)
            + torch.log1p(-y.pow(2) + self.eps)
        ).sum(dim=-1)

        return u, logdet


# ============================================================
# Permutation-equivariant conditioner
# ============================================================

class PermEquivariantConditioner(nn.Module):
    """
    Maps per-particle frozen coordinates to per-particle scale/shift.

    Input:
        frozen: (B, N, 1)

    Output:
        s, t: each (B, N, 1)

    This is permutation equivariant:
        conditioner(Px) = P conditioner(x)
    """

    def __init__(self, hidden=128):
        super().__init__()

        self.edge_net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.node_net = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

        # Near-identity init: s=0, t=0.
        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    def forward(self, frozen):
        """
        frozen: (B, N, 1)
        """
        xi = frozen.unsqueeze(2)                  # (B, N, 1, 1)
        xj = frozen.unsqueeze(1)                  # (B, 1, N, 1)

        diff = xi - xj                            # (B, N, N, 1)
        absdiff = diff.abs()                      # (B, N, N, 1)

        edge_features = torch.cat(
            [
                xi.expand_as(diff),
                xj.expand_as(diff),
                absdiff,
            ],
            dim=-1,
        )                                         # (B, N, N, 3)

        messages = self.edge_net(edge_features)   # (B, N, N, H)
        context = messages.sum(dim=2)             # (B, N, H)

        node_input = torch.cat([frozen, context], dim=-1)
        params = self.node_net(node_input)        # (B, N, 2)

        s = params[..., 0:1]
        t = params[..., 1:2]

        return s, t


# ============================================================
# Cartesian coordinate coupling
# ============================================================

class CoordPermEquivariantCoupling(nn.Module):
    """
    Coupling layer over x/y coordinates, not particle indices.

    If transform_coord == "y":
        x-coordinate is frozen, y-coordinate is transformed.

    If transform_coord == "x":
        y-coordinate is frozen, x-coordinate is transformed.

    This preserves full solvent-particle permutation equivariance.
    """

    def __init__(self, transform_coord="y", hidden=128, s_max=2.0):
        super().__init__()

        assert transform_coord in ["x", "y"]

        self.transform_coord = transform_coord
        self.s_max = float(s_max)
        self.conditioner = PermEquivariantConditioner(hidden=hidden)

    def forward(self, x):
        """
        Forward map: u -> z convention.

        x: (B, N, 2)
        """
        y = x.clone()

        if self.transform_coord == "y":
            frozen = x[..., 0:1]
            active = x[..., 1:2]
            active_index = 1
        else:
            frozen = x[..., 1:2]
            active = x[..., 0:1]
            active_index = 0

        s, t = self.conditioner(frozen)
        s = self.s_max * torch.tanh(s)

        active_new = active * torch.exp(s) + t
        y[..., active_index:active_index + 1] = active_new

        logdet = s.sum(dim=(1, 2))

        return y, logdet

    def inverse(self, y):
        """
        Inverse map: z -> u convention.

        y: (B, N, 2)
        """
        x = y.clone()

        if self.transform_coord == "y":
            frozen = y[..., 0:1]
            active = y[..., 1:2]
            active_index = 1
        else:
            frozen = y[..., 1:2]
            active = y[..., 0:1]
            active_index = 0

        s, t = self.conditioner(frozen)
        s = self.s_max * torch.tanh(s)

        active_old = (active - t) * torch.exp(-s)
        x[..., active_index:active_index + 1] = active_old

        logdet = -s.sum(dim=(1, 2))

        return x, logdet


# ============================================================
# Penalties
# ============================================================

def overlap_penalty(solvent_flat, sigma, n_particles=36, factor=0.9):
    """
    Penalizes solute-solvent and solvent-solvent overlaps.

    solvent_flat: (B, 72)
    """
    B = solvent_flat.shape[0]
    solvent = solvent_flat.view(B, n_particles, 2)

    solute = torch.zeros(
        B, 1, 2,
        device=solvent.device,
        dtype=solvent.dtype,
    )

    x = torch.cat([solute, solvent], dim=1)  # (B, 37, 2)

    diff = x.unsqueeze(2) - x.unsqueeze(1)
    r2 = (diff * diff).sum(dim=-1)

    N = n_particles + 1
    idx = torch.triu_indices(N, N, offset=1, device=x.device)
    r2_pairs = r2[:, idx[0], idx[1]]

    cutoff2 = (factor * sigma) ** 2
    return torch.relu(cutoff2 - r2_pairs).pow(2).sum(dim=1).mean()


def box_penalty(solvent_flat, L_BOX, n_particles=36):
    """
    Usually not needed if using BoxTanhTransform, but useful as diagnostic.
    """
    x = solvent_flat.view(-1, n_particles, 2)
    excess = torch.relu(x.abs() - L_BOX)
    return excess.pow(2).sum(dim=(1, 2)).mean()


# ============================================================
# Optional differentiable Lennard-Jones energy
# ============================================================

def torch_lj_energy_solute(
    solvent_flat,
    sigma,
    epsilon=1.0,
    n_particles=36,
    r_min=1e-4,
):
    """
    Differentiable LJ energy for fixed solute at origin + solvent particles.

    U = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]

    Includes solute-solvent and solvent-solvent pairs.

    solvent_flat: (B, 72)
    returns: (B,)
    """
    B = solvent_flat.shape[0]
    solvent = solvent_flat.view(B, n_particles, 2)

    solute = torch.zeros(
        B, 1, 2,
        device=solvent.device,
        dtype=solvent.dtype,
    )

    x = torch.cat([solute, solvent], dim=1)  # (B, 37, 2)

    diff = x.unsqueeze(2) - x.unsqueeze(1)
    r2 = (diff * diff).sum(dim=-1).clamp_min(r_min ** 2)

    N = n_particles + 1
    idx = torch.triu_indices(N, N, offset=1, device=x.device)

    r2_pairs = r2[:, idx[0], idx[1]]

    inv_r2 = (sigma ** 2) / r2_pairs
    inv_r6 = inv_r2.pow(3)
    inv_r12 = inv_r6.pow(2)

    pair_energy = 4.0 * epsilon * (inv_r12 - inv_r6)

    return pair_energy.sum(dim=1)


# ============================================================
# Full model
# ============================================================

class PermEquivariantBoxBoltzmannGenerator(nn.Module):
    """
    Permutation-equivariant, box-bounded Boltzmann generator.

    Symmetries:
      - solvent permutation equivariant
      - solute-centered
      - not rotation equivariant, because square box breaks rotation symmetry

    API:
      - generator(z)
      - inverse_generator(x)
      - loss_ML(batch_x)
      - loss_KL(batch_z)
    """

    def __init__(
        self,
        system=None,
        n_particles=36,
        n_layers=12,
        hidden=128,
        L_BOX=5.0,
        sigma=1.0,
        epsilon=1.0,
        energy_cap=1e4,
        use_torch_lj=True,
        s_max=2.0,
    ):
        super().__init__()

        self.system = system
        self.n_particles = int(n_particles)
        self.dim = 2 * self.n_particles
        self.L_BOX = float(L_BOX)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.energy_cap = float(energy_cap)
        self.use_torch_lj = bool(use_torch_lj)

        self.prior = GaussianPrior(dim=self.dim)
        self.box_transform = BoxTanhTransform(L_BOX=self.L_BOX)

        layers = []
        for i in range(n_layers):
            if i % 2 == 0:
                layers.append(
                    CoordPermEquivariantCoupling(
                        transform_coord="y",
                        hidden=hidden,
                        s_max=s_max,
                    )
                )
            else:
                layers.append(
                    CoordPermEquivariantCoupling(
                        transform_coord="x",
                        hidden=hidden,
                        s_max=s_max,
                    )
                )

        self.layers = nn.ModuleList(layers)

    def _to_particle(self, flat):
        return flat.view(flat.shape[0], self.n_particles, 2)

    def _to_flat(self, x):
        return x.reshape(x.shape[0], -1)

    def forward_map(self, u):
        """
        Unconstrained coordinates u -> latent z.

        u: (B, N, 2)
        """
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)

        x = u
        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld

        return x, logdet

    def inverse_map(self, z):
        """
        Latent z -> unconstrained coordinates u.

        z: (B, N, 2)
        """
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld

        return x, logdet

    def generator(self, z_flat):
        """
        Latent -> physical solvent coordinates in box.

        returns:
            x_box: (B, 72)
            logdet_zx: log |dx/dz|
        """
        z = self._to_particle(z_flat)

        u_particle, logdet_flow = self.inverse_map(z)
        u_flat = self._to_flat(u_particle)

        x_box, logdet_box = self.box_transform.forward(u_flat)

        return x_box, logdet_flow + logdet_box

    def inverse_generator(self, x_box):
        """
        Physical solvent coordinates in box -> latent.

        returns:
            z: (B, 72)
            logdet_xz: log |dz/dx|
        """
        u_flat, logdet_box_inv = self.box_transform.inverse(x_box)
        u_particle = self._to_particle(u_flat)

        z_particle, logdet_flow = self.forward_map(u_particle)

        return self._to_flat(z_particle), logdet_box_inv + logdet_flow

    def regularize_energy(self, energies):
        cap = self.energy_cap
        return torch.where(
            energies < cap,
            energies,
            cap + torch.log1p(energies - cap),
        )

    def calculate_energy(self, solvent_flat):
        """
        Differentiable energy.

        Preferred: use built-in torch LJ.
        Avoid non-differentiable system.get_energy during KL training.
        """
        if self.use_torch_lj:
            energies = torch_lj_energy_solute(
                solvent_flat,
                sigma=self.sigma,
                epsilon=self.epsilon,
                n_particles=self.n_particles,
            )
            return self.regularize_energy(energies)

        # Fallback: useful for logging, but may not be differentiable.
        B = solvent_flat.shape[0]
        solvent = solvent_flat.view(B, self.n_particles, 2)
        solute = torch.zeros(B, 1, 2, device=solvent.device, dtype=solvent.dtype)
        full = torch.cat([solute, solvent], dim=1)

        energies = []
        for i in range(B):
            e = self.system.get_energy(full[i])
            if not isinstance(e, torch.Tensor):
                e = torch.tensor(e, device=solvent.device, dtype=solvent.dtype)
            energies.append(e)

        energies = torch.stack(energies)
        return self.regularize_energy(energies)

    def loss_ML(self, batch_x):
        """
        ML loss on MC samples.

        batch_x: (B, 72), solvent-only coordinates inside [-L_BOX, L_BOX].
        """
        z, logdet_xz = self.inverse_generator(batch_x)
        log_pz = self.prior.log_prob(z)
        return -(log_pz + logdet_xz).mean()

    def loss_KL(
        self,
        batch_z,
        w_overlap=0.0,
        overlap_factor=0.9,
    ):
        """
        Reverse KL / energy training.

        batch_z: (B, 72)
        """
        x, logdet_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x)

        loss = (u_x - logdet_zx).mean()

        if w_overlap > 0.0:
            loss = loss + w_overlap * overlap_penalty(
                x,
                sigma=self.sigma,
                n_particles=self.n_particles,
                factor=overlap_factor,
            )

        return loss

    def sample(self, n_samples, device=None, dtype=torch.float32):
        z = self.prior.sample(n_samples, device=device, dtype=dtype)
        return self.generator(z)


def build_perm_equivariant_box_flow(
    system=None,
    n_particles=36,
    n_layers=12,
    hidden=128,
    L_BOX=5.0,
    sigma=1.0,
    epsilon=1.0,
    energy_cap=1e4,
    use_torch_lj=True,
    s_max=2.0,
):
    return PermEquivariantBoxBoltzmannGenerator(
        system=system,
        n_particles=n_particles,
        n_layers=n_layers,
        hidden=hidden,
        L_BOX=L_BOX,
        sigma=sigma,
        epsilon=epsilon,
        energy_cap=energy_cap,
        use_torch_lj=use_torch_lj,
        s_max=s_max,
    )
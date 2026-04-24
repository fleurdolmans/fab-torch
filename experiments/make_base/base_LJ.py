import math
import torch
from torch import nn
from typing import Optional, Sequence


# ============================================================
# Utilities
# ============================================================

def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)

def mic_unit(du: torch.Tensor) -> torch.Tensor:
    return du - torch.round(du)


def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return (m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)


def safe_log(x: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

def fibonacci_sphere(n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Deterministic approximately uniform directions on the unit sphere.
    Returns shape (n, 3).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    i = torch.arange(n, device=device, dtype=dtype)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = 2.0 * math.pi * i / phi
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1)
    return dirs


# ============================================================
# 1D wrapped normal on [0,1)
# ============================================================

class WrappedNormal1D:
    """
    Finite-image wrapped normal on the unit circle [0,1).
    """

    def __init__(self, sigma: float, image_range: int = 1):
        self.sigma = float(sigma)
        self.image_range = int(image_range)

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        x, mu broadcastable
        returns log p(x | mu)
        """
        sigma = torch.as_tensor(self.sigma, device=x.device, dtype=x.dtype)
        offsets = torch.arange(
            -self.image_range,
            self.image_range + 1,
            device=x.device,
            dtype=x.dtype,
        )

        norm_const = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma)
        z = x.unsqueeze(-1) - mu.unsqueeze(-1) + offsets
        log_terms = norm_const - 0.5 * (z / sigma) ** 2
        return logsumexp(log_terms, dim=-1)

    def sample(self, mu: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(self.sigma, device=mu.device, dtype=mu.dtype)
        eps = sigma * torch.randn_like(mu)
        return wrap_unit(mu + eps)


# ============================================================
# Exact Gaussian-unit-torus base for v4
# ============================================================

class GaussianUnitTorusBase(nn.Module):
    """
    Exact factorized wrapped-Gaussian base on [0,1)^(dim).

    This is a simple exact base for v4.
    Each coordinate is independent and distributed as a wrapped 1D Gaussian
    with common mean and std in unit coordinates.

    q(u) = prod_j WrappedNormal(u_j | mean_j, sigma_unit)

    Parameters
    ----------
    dim : int
        Total event dimension, e.g. 3 * n_solvent.
    sigma_unit : float
        Wrapped Gaussian std in unit coordinates.
    mean_unit : float or tensor-like
        Mean in [0,1). Can be scalar or shape (dim,).
    image_range : int
        Number of periodic image copies on each side for log_prob.
    """

    def __init__(
        self,
        dim: int,
        sigma_unit: float = 0.20,
        mean_unit=0.5,
        image_range: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = int(dim)
        self.shape = (self.dim,)
        self.sigma_unit = float(sigma_unit)
        self.image_range = int(image_range)

        mean_unit = torch.as_tensor(mean_unit, dtype=dtype, device=device)
        if mean_unit.ndim == 0:
            mean_unit = mean_unit.repeat(self.dim)
        if mean_unit.shape != (self.dim,):
            raise ValueError(f"mean_unit must be scalar or shape ({self.dim},)")

        self.register_buffer("mean_unit", wrap_unit(mean_unit))
        self.register_buffer("_dummy", torch.zeros(1, dtype=dtype, device=device))

        self._wn = WrappedNormal1D(sigma=self.sigma_unit, image_range=self.image_range)

    def sample(self, n: int) -> torch.Tensor:
        mean = self.mean_unit.to(device=self._dummy.device, dtype=self._dummy.dtype)
        mean = mean.view(1, self.dim).expand(n, -1)
        return self._wn.sample(mean)

    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        u = wrap_unit(u)
        mean = self.mean_unit.to(device=u.device, dtype=u.dtype).view(1, self.dim)
        lp = self._wn.log_prob(u, mean.expand_as(u))
        return lp.sum(dim=-1)

    def forward(self, n: int):
        u = self.sample(n)
        log_q = self.log_prob(u)
        return u, log_q

    __call__ = forward


# ============================================================
# Better exact shell base for v4
# ============================================================

class ShellTorusSoluteBase(nn.Module):
    """
    Exact shell-based base on solvent unit-torus coordinates for v4.

    Each solvent particle is sampled independently from a mixture:
      - shell components around a chosen solute reference center
      - optional uniform background component

    Construction of each shell component:
      1. choose a shell center c_k on the torus
      2. sample radius r from a positive 1D Gaussian-like density
      3. sample direction uniformly on the sphere
      4. set x = c_k + (r/L) * direction on the unit torus

    The radial density used here is a truncated-normal-like density on r>0:
      q_r(r) ∝ exp(-0.5 * ((r - r0)/sigma_r)^2),  r > 0

    The full 3D shell density around center c is:
      q_shell(u) = q_r(r) / (4π r^2)    with r = ||MIC(L*(u-c))||

    This is exact for the implemented density and sampler.

    Notes
    -----
    - This is much better geometrically than a site-centered Gaussian mixture.
    - It is still independent across solvent particles, so it does not encode
      solvent-solvent exclusion.
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        solute_positions_nm,
        shell_radius_nm: float = 0.33,
        shell_sigma_nm: float = 0.035,
        n_reference_sites: Optional[int] = None,
        add_uniform_component: bool = True,
        uniform_weight: float = 0.25,
        site_weights=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.shape = (self.dim,)
        self.box_length_nm = float(box_length_nm)
        self.shell_radius_nm = float(shell_radius_nm)
        self.shell_sigma_nm = float(shell_sigma_nm)
        self.add_uniform_component = bool(add_uniform_component)
        self.uniform_weight = float(uniform_weight)

        self.register_buffer("_dummy", torch.zeros(1, dtype=dtype, device=device))

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=dtype, device=device)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")

        # Use provided solute sites directly as shell centers.
        # Optionally restrict to first n_reference_sites.
        if n_reference_sites is not None:
            solute_positions_nm = solute_positions_nm[: int(n_reference_sites)]

        if solute_positions_nm.shape[0] == 0:
            raise ValueError("Need at least one solute position for ShellTorusSoluteBase.")

        centers_u = wrap_unit(solute_positions_nm / self.box_length_nm)
        self.register_buffer("centers_u", centers_u)  # (K,3)

        K = centers_u.shape[0]
        if site_weights is None:
            site_weights = torch.ones(K, dtype=dtype, device=device)
        else:
            site_weights = torch.as_tensor(site_weights, dtype=dtype, device=device)
            if site_weights.shape != (K,):
                raise ValueError(f"site_weights must have shape ({K},)")

        if self.add_uniform_component:
            mix_weights = torch.cat(
                [site_weights, torch.tensor([self.uniform_weight], dtype=dtype, device=device)],
                dim=0,
            )
        else:
            mix_weights = site_weights

        mix_probs = mix_weights / mix_weights.sum()
        self.register_buffer("mix_probs", mix_probs)

        # positive-radius normalizer on r>0
        # Z = ∫_0^∞ exp(-0.5((r-r0)/σ)^2) dr
        #   = σ * sqrt(pi/2) * (1 + erf(r0/(sqrt(2)σ)))
        r0 = torch.as_tensor(self.shell_radius_nm, dtype=dtype, device=device)
        sr = torch.as_tensor(self.shell_sigma_nm, dtype=dtype, device=device)
        sqrt2 = torch.as_tensor(math.sqrt(2.0), dtype=dtype, device=device)
        radial_Z = sr * math.sqrt(math.pi / 2.0) * (1.0 + torch.erf(r0 / (sqrt2 * sr)))
        self.register_buffer("radial_Z", radial_Z)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def mic_unit(self, du: torch.Tensor) -> torch.Tensor:
        return du - torch.round(du)

    def _sample_unit_sphere(self, n: int) -> torch.Tensor:
        v = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
        return v

    def _sample_positive_radius(self, n: int) -> torch.Tensor:
        """
        Rejection from N(r0, sigma^2), truncated to r > 0.
        This is exact for the implemented positive-radius normal density.
        """
        r0 = self.shell_radius_nm
        sr = self.shell_sigma_nm

        out = torch.empty(n, device=self.device, dtype=self.dtype)
        filled = 0
        while filled < n:
            m = max(2 * (n - filled), 16)
            cand = r0 + sr * torch.randn(m, device=self.device, dtype=self.dtype)
            cand = cand[cand > 0.0]
            take = min(cand.numel(), n - filled)
            if take > 0:
                out[filled:filled + take] = cand[:take]
                filled += take
        return out

    def _sample_single_particle(self, n: int):
        """
        Returns
        -------
        u : (n,3)
        log_q : (n,)
        """
        K = self.mix_probs.shape[0]
        comp_idx = torch.multinomial(self.mix_probs, num_samples=n, replacement=True)  # (n,)

        is_uniform = self.add_uniform_component and (comp_idx == K - 1)

        u = torch.empty(n, 3, device=self.device, dtype=self.dtype)

        # Shell component samples
        n_shell = int((~is_uniform).sum().item()) if self.add_uniform_component else n
        if n_shell > 0:
            idx_shell = torch.where(~is_uniform)[0] if self.add_uniform_component else torch.arange(n, device=self.device)
            centers = self.centers_u[comp_idx[idx_shell]]  # (n_shell,3)

            dirs = self._sample_unit_sphere(n_shell)                    # (n_shell,3)
            radii_nm = self._sample_positive_radius(n_shell)            # (n_shell,)
            radii_unit = radii_nm / self.box_length_nm                  # (n_shell,)

            u_shell = wrap_unit(centers + radii_unit[:, None] * dirs)
            u[idx_shell] = u_shell

        # Uniform component samples
        if self.add_uniform_component:
            idx_uni = torch.where(is_uniform)[0]
            if idx_uni.numel() > 0:
                u[idx_uni] = torch.rand(idx_uni.numel(), 3, device=self.device, dtype=self.dtype)

        log_q = self._log_prob_single(u)
        return u, log_q

    def _shell_log_prob_component(self, u: torch.Tensor, centers_u: torch.Tensor) -> torch.Tensor:
        """
        u:         (B,3)
        centers_u: (K,3)

        returns:
            log density under each shell component: (B,K)
        """
        B = u.shape[0]
        K = centers_u.shape[0]

        du = self.mic_unit(u[:, None, :] - centers_u[None, :, :])     # (B,K,3)
        dx = self.box_length_nm * du                                   # nm
        r = torch.linalg.norm(dx, dim=-1).clamp_min(1e-12)            # (B,K)

        r0 = torch.as_tensor(self.shell_radius_nm, device=u.device, dtype=u.dtype)
        sr = torch.as_tensor(self.shell_sigma_nm, device=u.device, dtype=u.dtype)
        radial_Z = self.radial_Z.to(device=u.device, dtype=u.dtype)

        # log q_r(r)
        log_qr = -0.5 * ((r - r0) / sr) ** 2 - torch.log(radial_Z)

        # shell surface factor
        log_surface = math.log(4.0 * math.pi) + 2.0 * torch.log(r)

        # density in 3D around center
        return log_qr - log_surface

    def _log_prob_single(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B,3)
        returns log q_single(u): (B,)
        """
        u = wrap_unit(u)
        B = u.shape[0]

        mix_probs = self.mix_probs.to(device=u.device, dtype=u.dtype)
        centers_u = self.centers_u.to(device=u.device, dtype=u.dtype)

        log_terms = []

        # Shell components
        lp_shell = self._shell_log_prob_component(u, centers_u)  # (B,Kshell)
        lp_shell = lp_shell + torch.log(mix_probs[: centers_u.shape[0]])[None, :]
        log_terms.append(lp_shell)

        # Uniform component on [0,1)^3 has log density 0
        if self.add_uniform_component:
            lp_uni = torch.full(
                (B, 1),
                fill_value=torch.log(mix_probs[-1]).item(),
                device=u.device,
                dtype=u.dtype,
            )
            log_terms.append(lp_uni)

        all_lp = torch.cat(log_terms, dim=1)
        return logsumexp(all_lp, dim=1)

    def sample(self, n: int) -> torch.Tensor:
        u, _ = self._sample_single_particle(n * self.n_solvent)
        u = u.view(n, self.n_solvent, 3)
        return u.reshape(n, self.dim)

    def log_prob(self, u_flat: torch.Tensor) -> torch.Tensor:
        u_flat = wrap_unit(u_flat)
        B = u_flat.shape[0]
        u = u_flat.view(B, self.n_solvent, 3)

        lp = self._log_prob_single(u.reshape(B * self.n_solvent, 3))
        lp = lp.view(B, self.n_solvent).sum(dim=1)
        return lp

    def forward(self, n: int):
        u = self.sample(n)
        log_q = self.log_prob(u)
        return u, log_q

    __call__ = forward

class UnitTorusPeriodicBase(nn.Module):
    """
    Adapter that turns a base defined on the centered periodic chart
        x_c in [-L/2, L/2)
    into a base on unit-torus coordinates
        u in [0, 1)

    Assumes:
      - wrapped_base.sample / wrapped_base(n) returns x_c in [-L/2, L/2)
      - wrapped_base.log_prob(x_c) is defined on that same chart

    Coordinate conversion:
      x_c = wrap_centered(L * u)

    Returned log probs are with respect to unit-torus coordinates u.
    """

    def __init__(self, wrapped_base: nn.Module, dim: int, box_length: float):
        super().__init__()
        self.wrapped_base = wrapped_base
        self.dim = int(dim)
        self.box_length = float(box_length)
        self.shape = (self.dim,)

    def wrap_unit(self, u: torch.Tensor) -> torch.Tensor:
        return u - torch.floor(u)

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length, device=x.device, dtype=x.dtype)
        return torch.remainder(x + 0.5 * L, L) - 0.5 * L

    def unit_to_centered(self, u: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length, device=u.device, dtype=u.dtype)
        x = L * self.wrap_unit(u)
        return self.wrap_centered(x)

    def centered_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length, device=x.device, dtype=x.dtype)
        x = self.wrap_centered(x)
        return self.wrap_unit(x / L)

    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        u = self.wrap_unit(u)
        x_c = self.unit_to_centered(u)
        log_q_x = self.wrapped_base.log_prob(x_c)

        # x_c = L * u locally, so |det dx/du| = L^dim
        log_abs_det = self.dim * math.log(self.box_length)
        return log_q_x + log_abs_det

    def forward(self, n: int):
        x_c, log_q_x = self.wrapped_base(n)
        u = self.centered_to_unit(x_c)

        log_abs_det = self.dim * math.log(self.box_length)
        log_q_u = log_q_x + log_abs_det
        return u, log_q_u


class HardCoreRandomPlacementBase(nn.Module):
    """
    Hard-core solvent base in centered box coordinates [-L/2, L/2).

    Sequentially places solvent particles with rejection sampling using:
      - solvent-solvent exclusion radius r_min_ss
      - solvent-solute exclusion radius r_min_su

    Output:
      flattened solvent coordinates in centered chart,
      shape (B, 3 * n_solvent)

    Important:
      log_prob() is only a surrogate uniform-in-box density,
      not the exact rejection-sampling density.
    """

    def __init__(
        self,
        n_solvent: int,
        box_length: float,
        solute_positions_centered=None,
        r_min_ss: float = 0.24,
        r_min_su: float = 0.20,
        max_attempts_per_particle: int = 500,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.box_length = float(box_length)
        self.tail_bound = 0.5 * float(box_length)

        self.r_min_ss = float(r_min_ss)
        self.r_min_su = float(r_min_su)
        self.max_attempts_per_particle = int(max_attempts_per_particle)

        if solute_positions_centered is None:
            solute_positions_centered = torch.zeros((0, 3), dtype=dtype, device=device)

        solute_positions_centered = torch.as_tensor(
            solute_positions_centered, dtype=dtype, device=device
        )
        if solute_positions_centered.ndim != 2 or solute_positions_centered.shape[1] != 3:
            raise ValueError("solute_positions_centered must have shape (n_solute, 3)")

        self.register_buffer("solute_positions_centered", solute_positions_centered)
        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))

        self.shape = (self.dim,)
        self.last_sampling_stats = None

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def mic(self, dx: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length, device=dx.device, dtype=dx.dtype)
        return dx - L * torch.round(dx / L)

    def wrap_centered(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.box_length, device=x.device, dtype=x.dtype)
        return torch.remainder(x + 0.5 * L, L) - 0.5 * L

    def _valid_wrt_solute(self, cand: torch.Tensor) -> bool:
        if self.solute_positions_centered.numel() == 0:
            return True

        d = torch.linalg.norm(
            self.mic(self.solute_positions_centered - cand.unsqueeze(0)),
            dim=-1,
        )
        return bool(torch.all(d > self.r_min_su))

    def _valid_wrt_solvent(self, cand: torch.Tensor, placed: list[torch.Tensor]) -> bool:
        if len(placed) == 0:
            return True

        prev = torch.stack(placed, dim=0)
        d = torch.linalg.norm(self.mic(prev - cand.unsqueeze(0)), dim=-1)
        return bool(torch.all(d > self.r_min_ss))

    def _sample_one(self):
        placed = []

        total_attempts = 0
        attempts_per_particle = []
        n_last_resort = 0

        L = self.box_length

        for _ in range(self.n_solvent):
            accepted = False
            particle_attempts = 0

            for _ in range(self.max_attempts_per_particle):
                particle_attempts += 1
                total_attempts += 1

                cand = (torch.rand(3, device=self.device, dtype=self.dtype) - 0.5) * L
                cand = self.wrap_centered(cand)

                if not self._valid_wrt_solute(cand):
                    continue
                if not self._valid_wrt_solvent(cand, placed):
                    continue

                placed.append(cand)
                accepted = True
                break

            if not accepted:
                n_last_resort += 1
                best_cand = None
                best_score = -float("inf")

                for _ in range(self.max_attempts_per_particle):
                    particle_attempts += 1
                    total_attempts += 1

                    cand = (torch.rand(3, device=self.device, dtype=self.dtype) - 0.5) * L
                    cand = self.wrap_centered(cand)

                    if not self._valid_wrt_solute(cand):
                        continue

                    if len(placed) == 0:
                        best_cand = cand
                        break

                    prev = torch.stack(placed, dim=0)
                    d = torch.linalg.norm(self.mic(prev - cand.unsqueeze(0)), dim=-1)
                    score = float(d.min().item())

                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand is None:
                    best_cand = (torch.rand(3, device=self.device, dtype=self.dtype) - 0.5) * L
                    best_cand = self.wrap_centered(best_cand)

                placed.append(best_cand)

            attempts_per_particle.append(particle_attempts)

        x = torch.stack(placed, dim=0)
        stats = {
            "total_attempts": total_attempts,
            "mean_attempts_per_particle": total_attempts / max(self.n_solvent, 1),
            "max_attempts_for_particle": max(attempts_per_particle),
            "n_last_resort": n_last_resort,
        }
        return x, stats

    def sample(self, shape):
        n = int(shape[0])

        xs = []
        stats_list = []
        for _ in range(n):
            x_one, stats_one = self._sample_one()
            xs.append(x_one.reshape(-1))
            stats_list.append(stats_one)

        self.last_sampling_stats = {
            "mean_total_attempts": sum(s["total_attempts"] for s in stats_list) / n,
            "mean_attempts_per_particle": sum(s["mean_attempts_per_particle"] for s in stats_list) / n,
            "max_attempts_for_particle": max(s["max_attempts_for_particle"] for s in stats_list),
            "mean_last_resort": sum(s["n_last_resort"] for s in stats_list) / n,
        }

        return torch.stack(xs, dim=0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Surrogate only: uniform on centered-box support, -inf outside.
        Not the exact normalized log-density of the rejection sampler.
        """
        x = x.view(x.shape[0], self.n_solvent, 3)

        inside = (x >= -self.tail_bound) & (x < self.tail_bound)
        inside = inside.all(dim=(-1, -2))

        vol = self.box_length ** (3 * self.n_solvent)
        logp = torch.full(
            (x.shape[0],),
            fill_value=-math.log(vol),
            device=x.device,
            dtype=x.dtype,
        )
        logp = torch.where(inside, logp, torch.full_like(logp, float("-inf")))
        return logp

    def forward(self, n: int):
        x = self.sample((n,))
        logp = self.log_prob(x)
        return x, logp



class ExactTorusSoluteMixtureBase(nn.Module):
    """
    Exact tractable base on solvent unit-torus coordinates u in [0,1)^(3N).

    Each solvent particle is sampled independently from the same mixture of
    wrapped Gaussians centered near solute sites.

    This is a *correct* normalizing-flow base:
      - sample() and log_prob() match exactly for the implemented distribution

    Distribution:
      q(u_1,...,u_N) = prod_i q_single(u_i)

    where q_single is a mixture of factorized wrapped 1D Gaussians on [0,1).

    Parameters
    ----------
    n_solvent : int
        Number of solvent particles.
    box_length_nm : float
        Cubic box length.
    solute_positions_nm : tensor-like, shape (n_solute, 3)
        Fixed solute positions in nm.
    sigma_unit : float
        Standard deviation in unit-torus coordinates.
        Example: sigma_nm / box_length_nm.
    image_range : int
        Number of periodic images on each side used in wrapped-normal sum.
        image_range=1 means sum over {-1,0,1}.
    add_uniform_component : bool
        Whether to include one uniform component in the mixture.
    uniform_weight : float
        Unnormalized weight of the uniform component before normalization.
    site_weights : tensor-like or None
        Optional per-solute-site unnormalized weights.
    device, dtype
        Standard torch placement args.
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        solute_positions_nm,
        sigma_unit: float = 0.08,
        image_range: int = 1,
        add_uniform_component: bool = True,
        uniform_weight: float = 1.0,
        site_weights=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.box_length_nm = float(box_length_nm)
        self.sigma_unit = float(sigma_unit)
        self.image_range = int(image_range)
        self.add_uniform_component = bool(add_uniform_component)
        self.uniform_weight = float(uniform_weight)

        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))
        self.shape = (self.dim,)

        solute_positions_nm = torch.as_tensor(solute_positions_nm, dtype=dtype, device=device)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")

        # Convert solute positions to unit torus coordinates
        solute_u = wrap_unit(solute_positions_nm / self.box_length_nm)
        self.register_buffer("solute_u", solute_u)  # (K_solute, 3)

        n_sites = solute_u.shape[0]
        if n_sites == 0:
            raise ValueError("Need at least one solute site for this base.")

        if site_weights is None:
            site_weights = torch.ones(n_sites, dtype=dtype, device=device)
        else:
            site_weights = torch.as_tensor(site_weights, dtype=dtype, device=device)
            if site_weights.shape != (n_sites,):
                raise ValueError(f"site_weights must have shape ({n_sites},)")

        if self.add_uniform_component:
            mix_weights = torch.cat(
                [site_weights, torch.tensor([self.uniform_weight], dtype=dtype, device=device)],
                dim=0,
            )
        else:
            mix_weights = site_weights

        mix_probs = mix_weights / mix_weights.sum()
        self.register_buffer("mix_probs", mix_probs)

        # Component centers: solute sites + optional dummy center for uniform comp
        if self.add_uniform_component:
            dummy_center = torch.zeros((1, 3), dtype=dtype, device=device)
            centers = torch.cat([solute_u, dummy_center], dim=0)
        else:
            centers = solute_u
        self.register_buffer("component_centers", centers)  # (K,3)

        # Precompute image offsets for wrapped Gaussian
        offsets = torch.arange(-self.image_range, self.image_range + 1, device=device, dtype=dtype)
        self.register_buffer("image_offsets", offsets)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def _sample_single_particle(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        u : (n, 3)
        log_q : (n,)
        """
        K = self.mix_probs.shape[0]
        comp_idx = torch.multinomial(self.mix_probs, num_samples=n, replacement=True)  # (n,)

        is_uniform = self.add_uniform_component and (comp_idx == K - 1)

        centers = self.component_centers[comp_idx]  # (n,3)

        # Sample
        eps = self.sigma_unit * torch.randn(n, 3, device=self.device, dtype=self.dtype)
        u = wrap_unit(centers + eps)

        if self.add_uniform_component:
            # overwrite uniform-component samples with exact uniform torus samples
            n_uni = int(is_uniform.sum().item())
            if n_uni > 0:
                u[is_uniform] = torch.rand(n_uni, 3, device=self.device, dtype=self.dtype)

        # Exact log_prob under implemented mixture
        log_q = self._log_prob_single(u)
        return u, log_q

    def _wrapped_normal_1d_logpdf(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        x:  (...,)
        mu: (...,)
        returns log pdf of wrapped 1D normal on [0,1), using finite image sum
        """
        sigma = torch.as_tensor(self.sigma_unit, device=x.device, dtype=x.dtype)
        offsets = self.image_offsets.to(device=x.device, dtype=x.dtype)  # (M,)
        norm_const = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma)

        # shape broadcast: (..., M)
        z = x.unsqueeze(-1) - mu.unsqueeze(-1) + offsets
        log_terms = norm_const - 0.5 * (z / sigma) ** 2
        return logsumexp(log_terms, dim=-1)

    def _log_prob_single(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B,3)
        returns log q_single(u): (B,)
        """
        u = wrap_unit(u)
        B = u.shape[0]
        K = self.mix_probs.shape[0]

        centers = self.component_centers.to(device=u.device, dtype=u.dtype)  # (K,3)
        mix_probs = self.mix_probs.to(device=u.device, dtype=u.dtype)        # (K,)

        # Wrapped-Gaussian components for solute-site centers
        n_gauss = K - 1 if self.add_uniform_component else K
        log_comp = []

        if n_gauss > 0:
            mu = centers[:n_gauss]  # (Kg,3)

            # Compute per-component log density
            # result shape: (B, Kg)
            lp_x = self._wrapped_normal_1d_logpdf(
                u[:, None, 0].expand(B, n_gauss),
                mu[None, :, 0].expand(B, n_gauss),
            )
            lp_y = self._wrapped_normal_1d_logpdf(
                u[:, None, 1].expand(B, n_gauss),
                mu[None, :, 1].expand(B, n_gauss),
            )
            lp_z = self._wrapped_normal_1d_logpdf(
                u[:, None, 2].expand(B, n_gauss),
                mu[None, :, 2].expand(B, n_gauss),
            )

            lp_gauss = lp_x + lp_y + lp_z + torch.log(mix_probs[:n_gauss])[None, :]
            log_comp.append(lp_gauss)

        if self.add_uniform_component:
            # Uniform on [0,1)^3 has log density 0
            lp_uni = torch.full((B, 1), torch.log(mix_probs[-1]).item(), device=u.device, dtype=u.dtype)
            log_comp.append(lp_uni)

        all_lp = torch.cat(log_comp, dim=1)  # (B,K)
        return logsumexp(all_lp, dim=1)

    def sample(self, shape):
        n = int(shape[0])
        u, _ = self._sample_single_particle(n * self.n_solvent)
        u = u.view(n, self.n_solvent, 3)
        return u.reshape(n, self.dim)

    def log_prob(self, u_flat: torch.Tensor):
        u_flat = wrap_unit(u_flat)
        B = u_flat.shape[0]
        u = u_flat.view(B, self.n_solvent, 3)

        # Independent product across solvent particles
        lp = self._log_prob_single(u.reshape(B * self.n_solvent, 3))
        lp = lp.view(B, self.n_solvent).sum(dim=1)
        return lp

    def forward(self, n: int):
        u = self.sample((n,))
        log_q = self.log_prob(u)
        return u, log_q



# ============================================================
# Exact autoregressive shell base for v4
# ============================================================
class MultiShellAutoregressiveTorusBase(nn.Module):
    """
    Exact autoregressive multi-shell base on unit-torus solvent coordinates for v4.

    Factorization
    -------------
        q(u_1, ..., u_N) = prod_i q(u_i | u_<i)

    Each conditional is an exact tractable mixture:
        q(u_i | u_<i) =
            sum_k pi_k(u_<i) q_k(u_i)
            [+ pi_uniform(u_<i) * Uniform(u_i)]

    where:
      - q_k are wrapped-Gaussian components
      - component centers are placed on multiple shells around the solute(s)
      - mixture weights pi_k depend autoregressively on previously placed solvent
        through a soft crowding penalty

    This is an exact NF base because:
      - sample() draws from exactly these conditionals
      - log_prob() evaluates exactly these same conditionals

    Important
    ---------
    - No hard rejection / resampling is used.
    - Soft exclusion is introduced only through the mixture logits, so exactness is preserved.
    - This is NOT permutation invariant because of sequential ordering.
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        solute_positions_nm,
        shell_radii_nm: Sequence[float] = (0.34, 0.42, 0.52),
        shell_weights: Optional[Sequence[float]] = None,
        component_sigma_unit: float = 0.018,
        n_directions: int = 48,
        image_range: int = 1,
        add_uniform_component: bool = False,
        uniform_weight: float = 1e-3,
        crowding_strength: float = 20.0,
        crowding_sigma_nm: float = 0.08,
        hard_core_nm: float = 0.24,
        hard_core_strength: float = 10.0,
        site_weights=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.shape = (self.dim,)

        self.box_length_nm = float(box_length_nm)
        self.component_sigma_unit = float(component_sigma_unit)
        self.n_directions = int(n_directions)
        self.image_range = int(image_range)

        self.add_uniform_component = bool(add_uniform_component)
        self.uniform_weight = float(uniform_weight)

        self.crowding_strength = float(crowding_strength)
        self.crowding_sigma_nm = float(crowding_sigma_nm)
        self.hard_core_nm = float(hard_core_nm)
        self.hard_core_strength = float(hard_core_strength)

        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))

        solute_positions_nm = torch.as_tensor(solute_positions_nm, device=device, dtype=dtype)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")
        if solute_positions_nm.shape[0] == 0:
            raise ValueError("Need at least one solute site.")

        self.register_buffer("solute_u", wrap_unit(solute_positions_nm / self.box_length_nm))

        n_sites = self.solute_u.shape[0]
        if site_weights is None:
            site_weights = torch.ones(n_sites, device=device, dtype=dtype)
        else:
            site_weights = torch.as_tensor(site_weights, device=device, dtype=dtype)
            if site_weights.shape != (n_sites,):
                raise ValueError(f"site_weights must have shape ({n_sites},)")
        self.register_buffer("site_weights", site_weights)

        shell_radii_nm = torch.as_tensor(shell_radii_nm, device=device, dtype=dtype)
        if shell_radii_nm.ndim != 1 or shell_radii_nm.numel() == 0:
            raise ValueError("shell_radii_nm must be a non-empty 1D sequence")

        if shell_weights is None:
            shell_weights = torch.ones_like(shell_radii_nm)
        else:
            shell_weights = torch.as_tensor(shell_weights, device=device, dtype=dtype)
            if shell_weights.shape != shell_radii_nm.shape:
                raise ValueError("shell_weights must match shell_radii_nm shape")
        shell_weights = shell_weights / shell_weights.sum()

        self.register_buffer("shell_radii_nm", shell_radii_nm)
        self.register_buffer("shell_weights", shell_weights)

        dirs = fibonacci_sphere(self.n_directions, device=device, dtype=dtype)
        self.register_buffer("shell_dirs", dirs)

        centers = []
        comp_base_weights = []

        for s in range(n_sites):
            c = self.solute_u[s]  # (3,)
            site_w = site_weights[s]

            for r_nm, r_w in zip(shell_radii_nm, shell_weights):
                r_unit = r_nm / self.box_length_nm
                shell_centers = wrap_unit(c.view(1, 3) + r_unit * dirs)  # (D,3)
                centers.append(shell_centers)

                w = torch.full(
                    (self.n_directions,),
                    fill_value=(site_w * r_w / self.n_directions).item(),
                    device=device,
                    dtype=dtype,
                )
                comp_base_weights.append(w)

        centers = torch.cat(centers, dim=0)               # (K,3)
        comp_base_weights = torch.cat(comp_base_weights)  # (K,)
        comp_base_weights = comp_base_weights / comp_base_weights.sum()

        self.register_buffer("component_centers", centers)
        self.register_buffer("component_base_weights", comp_base_weights)
        self.n_components = centers.shape[0]

        self._wn = WrappedNormal1D(
            sigma=self.component_sigma_unit,
            image_range=self.image_range,
        )

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def _pair_dist_nm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A, B broadcastable with final dim 3
        returns Euclidean MIC distances in nm
        """
        du = mic_unit(A - B)
        dx = self.box_length_nm * du
        return torch.linalg.norm(dx, dim=-1)

    def _component_log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B,3)
        returns:
            log q_k(u) for each component k, shape (B,K)
        """
        u = wrap_unit(u)
        centers = self.component_centers.to(device=u.device, dtype=u.dtype)
        B = u.shape[0]
        K = centers.shape[0]

        lp_x = self._wn.log_prob(
            u[:, None, 0].expand(B, K),
            centers[None, :, 0].expand(B, K),
        )
        lp_y = self._wn.log_prob(
            u[:, None, 1].expand(B, K),
            centers[None, :, 1].expand(B, K),
        )
        lp_z = self._wn.log_prob(
            u[:, None, 2].expand(B, K),
            centers[None, :, 2].expand(B, K),
        )
        return lp_x + lp_y + lp_z

    def _crowding_penalty(self, prev_u: torch.Tensor) -> torch.Tensor:
        """
        prev_u: (M,3), already placed solvent particles
        returns:
            penalty for each component center, shape (K,)
        """
        K = self.n_components
        if prev_u.numel() == 0:
            return torch.zeros(K, device=self.device, dtype=self.dtype)

        centers = self.component_centers.to(device=prev_u.device, dtype=prev_u.dtype)  # (K,3)

        # distances from each center to each previously placed solvent
        d = self._pair_dist_nm(centers[:, None, :], prev_u[None, :, :])  # (K,M)

        sigma = torch.as_tensor(self.crowding_sigma_nm, device=d.device, dtype=d.dtype)
        hard_core = torch.as_tensor(self.hard_core_nm, device=d.device, dtype=d.dtype)

        # smooth crowding term
        soft_pen = torch.exp(-0.5 * (d / sigma) ** 2).sum(dim=1)  # (K,)

        # additional exact-in-logits hard-core-like discouragement
        # still exact because it only changes mixture weights, not the sampling rule
        hard_pen = (d < hard_core).sum(dim=1).to(d.dtype)  # (K,)

        return soft_pen + self.hard_core_strength * hard_pen

    def _conditional_mixture_probs(self, prev_u: torch.Tensor) -> torch.Tensor:
        """
        prev_u: (M,3)
        returns:
            probs over K components [+ optional uniform], shape (K,) or (K+1,)
        """
        base_w = self.component_base_weights.to(device=self.device, dtype=self.dtype)
        penalty = self._crowding_penalty(prev_u)

        logits = torch.log(base_w.clamp_min(1e-30)) - self.crowding_strength * penalty

        if self.add_uniform_component:
            logits = torch.cat(
                [
                    logits,
                    torch.tensor([math.log(self.uniform_weight)], device=self.device, dtype=self.dtype),
                ],
                dim=0,
            )

        return torch.softmax(logits, dim=0)

    def _conditional_log_prob_single(self, u_i: torch.Tensor, prev_u: torch.Tensor) -> torch.Tensor:
        """
        u_i:   (B,3)
        prev_u: (M,3)

        returns:
            log q(u_i | prev_u), shape (B,)
        """
        probs = self._conditional_mixture_probs(prev_u).to(device=u_i.device, dtype=u_i.dtype)

        lp_comp = self._component_log_prob(u_i)  # (B,K)
        lp_terms = lp_comp + torch.log(probs[: self.n_components].clamp_min(1e-30))[None, :]

        if self.add_uniform_component:
            lp_uni = torch.full(
                (u_i.shape[0], 1),
                fill_value=torch.log(probs[-1].clamp_min(torch.as_tensor(1e-30, device=u_i.device, dtype=u_i.dtype))).item(),
                device=u_i.device,
                dtype=u_i.dtype,
            )
            lp_terms = torch.cat([lp_terms, lp_uni], dim=1)

        return logsumexp(lp_terms, dim=1)

    def _sample_conditional_single(self, prev_u: torch.Tensor) -> torch.Tensor:
        """
        prev_u: (M,3)
        returns:
            one sample u_i, shape (3,)
        """
        probs = self._conditional_mixture_probs(prev_u)
        idx = torch.multinomial(probs, num_samples=1, replacement=True).item()

        if self.add_uniform_component and idx == self.n_components:
            return torch.rand(3, device=self.device, dtype=self.dtype)

        center = self.component_centers[idx].to(device=self.device, dtype=self.dtype)
        return self._wn.sample(center)

    def sample(self, n: int) -> torch.Tensor:
        """
        returns:
            u_flat: (n, 3*n_solvent)
        """
        out = []
        for _ in range(n):
            placed = []
            for _i in range(self.n_solvent):
                if len(placed) == 0:
                    prev_u = torch.zeros((0, 3), device=self.device, dtype=self.dtype)
                else:
                    prev_u = torch.stack(placed, dim=0)

                u_i = self._sample_conditional_single(prev_u)
                placed.append(u_i)

            u = torch.stack(placed, dim=0)   # (N,3)
            out.append(u.reshape(-1))

        return torch.stack(out, dim=0)

    def log_prob(self, u_flat: torch.Tensor) -> torch.Tensor:
        """
        returns:
            log q(u), shape (B,)
        """
        u_flat = wrap_unit(u_flat)
        B = u_flat.shape[0]
        u = u_flat.view(B, self.n_solvent, 3)

        log_q = torch.zeros(B, device=u.device, dtype=u.dtype)

        for i in range(self.n_solvent):
            u_i = u[:, i, :]  # (B,3)

            if i == 0:
                prev_u = torch.zeros((0, 3), device=u.device, dtype=u.dtype)
                log_q = log_q + self._conditional_log_prob_single(u_i, prev_u)
            else:
                vals = []
                for b in range(B):
                    prev_u_b = u[b, :i, :]
                    vals.append(self._conditional_log_prob_single(u_i[b:b+1], prev_u_b))
                vals = torch.cat(vals, dim=0)
                log_q = log_q + vals

        return log_q

    def forward(self, n: int):
        u = self.sample(n)
        log_q = self.log_prob(u)
        return u, log_q


class MultiShellIIDTorusBase(nn.Module):
    """
    Permutation-equivariant multi-shell IID base on unit-torus solvent coords for v4.

    Each solvent particle is independently sampled from the *same* mixture of
    wrapped-Gaussian components placed on multiple shells around the solute(s):

        q(u_1, ..., u_N) = prod_i q(u_i)     [IID — permutation equivariant]

        q(u_i) = sum_k w_k * WrappedNormal3D(u_i; c_k, sigma_unit)
                 [+ w_uniform * Uniform(u_i)]

    Component centers c_k are on Fibonacci-sphere directions at each shell radius
    around each solute site.  Weights w_k ∝ site_weight * shell_weight / n_directions.

    Unlike MultiShellAutoregressiveTorusBase:
    - No crowding penalty → permutation equivariant, exact IID factorisation
    - No sequential per-particle loop → fully batched O(B*N*K) log_prob / sample
    - Base samples CAN be overlapping; the flow + overlap penalty corrects this
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        solute_positions_nm,
        shell_radii_nm: Sequence[float] = (0.34, 0.42, 0.52),
        shell_weights: Optional[Sequence[float]] = None,
        component_sigma_unit: float = 0.018,
        n_directions: int = 48,
        image_range: int = 1,
        add_uniform_component: bool = False,
        uniform_weight: float = 0.10,
        site_weights=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_solvent = int(n_solvent)
        self.dim = 3 * self.n_solvent
        self.shape = (self.dim,)
        self.box_length_nm = float(box_length_nm)
        self.component_sigma_unit = float(component_sigma_unit)
        self.n_directions = int(n_directions)
        self.image_range = int(image_range)
        self.add_uniform_component = bool(add_uniform_component)
        self.uniform_weight = float(uniform_weight)

        self.register_buffer("_dummy", torch.zeros(1, device=device, dtype=dtype))

        solute_positions_nm = torch.as_tensor(solute_positions_nm, device=device, dtype=dtype)
        if solute_positions_nm.ndim != 2 or solute_positions_nm.shape[1] != 3:
            raise ValueError("solute_positions_nm must have shape (n_solute, 3)")
        if solute_positions_nm.shape[0] == 0:
            raise ValueError("Need at least one solute site.")
        self.register_buffer("solute_u", wrap_unit(solute_positions_nm / self.box_length_nm))

        n_sites = self.solute_u.shape[0]
        if site_weights is None:
            site_weights = torch.ones(n_sites, device=device, dtype=dtype)
        else:
            site_weights = torch.as_tensor(site_weights, device=device, dtype=dtype)
            if site_weights.shape != (n_sites,):
                raise ValueError(f"site_weights must have shape ({n_sites},)")

        shell_radii_nm_t = torch.as_tensor(shell_radii_nm, device=device, dtype=dtype)
        if shell_radii_nm_t.ndim != 1 or shell_radii_nm_t.numel() == 0:
            raise ValueError("shell_radii_nm must be a non-empty 1D sequence")

        if shell_weights is None:
            shell_weights_t = torch.ones_like(shell_radii_nm_t)
        else:
            shell_weights_t = torch.as_tensor(shell_weights, device=device, dtype=dtype)
            if shell_weights_t.shape != shell_radii_nm_t.shape:
                raise ValueError("shell_weights must match shell_radii_nm shape")
        shell_weights_t = shell_weights_t / shell_weights_t.sum()

        dirs = fibonacci_sphere(self.n_directions, device=device, dtype=dtype)
        self.register_buffer("shell_dirs", dirs)

        # Build component centers and base weights (same construction as autoregressive version)
        centers = []
        comp_base_weights = []
        for s in range(n_sites):
            c = self.solute_u[s]
            site_w = site_weights[s]
            for r_nm, r_w in zip(shell_radii_nm_t, shell_weights_t):
                r_unit = r_nm / self.box_length_nm
                shell_centers = wrap_unit(c.view(1, 3) + r_unit * dirs)  # (D, 3)
                centers.append(shell_centers)
                w = torch.full(
                    (self.n_directions,),
                    fill_value=(site_w * r_w / self.n_directions).item(),
                    device=device,
                    dtype=dtype,
                )
                comp_base_weights.append(w)

        centers_t = torch.cat(centers, dim=0)              # (K, 3)
        comp_weights_t = torch.cat(comp_base_weights)       # (K,)
        comp_weights_t = comp_weights_t / comp_weights_t.sum()

        self.register_buffer("component_centers", centers_t)
        self.n_components = centers_t.shape[0]

        # Final mixture probabilities (components + optional uniform)
        if self.add_uniform_component:
            all_w = torch.cat([comp_weights_t, torch.tensor([self.uniform_weight], device=device, dtype=dtype)])
        else:
            all_w = comp_weights_t
        self.register_buffer("mix_probs", all_w / all_w.sum())

        self._wn = WrappedNormal1D(sigma=self.component_sigma_unit, image_range=self.image_range)

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def _component_log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, 3)  →  log q_k(u) for each component k,  shape (B, K)
        Identical to MultiShellAutoregressiveTorusBase._component_log_prob.
        """
        u = wrap_unit(u)
        centers = self.component_centers.to(device=u.device, dtype=u.dtype)
        B, K = u.shape[0], centers.shape[0]
        lp_x = self._wn.log_prob(u[:, None, 0].expand(B, K), centers[None, :, 0].expand(B, K))
        lp_y = self._wn.log_prob(u[:, None, 1].expand(B, K), centers[None, :, 1].expand(B, K))
        lp_z = self._wn.log_prob(u[:, None, 2].expand(B, K), centers[None, :, 2].expand(B, K))
        return lp_x + lp_y + lp_z  # (B, K)

    def _log_prob_single(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, 3)  →  log q(u): (B,)  using fixed (non-crowded) weights
        """
        mix_probs = self.mix_probs.to(device=u.device, dtype=u.dtype)
        lp_comp = self._component_log_prob(u)                                  # (B, K)
        lp_terms = lp_comp + safe_log(mix_probs[:self.n_components])[None, :]  # (B, K)

        if self.add_uniform_component:
            lp_uni = torch.full(
                (u.shape[0], 1),
                fill_value=safe_log(mix_probs[-1]).item(),
                device=u.device,
                dtype=u.dtype,
            )
            lp_terms = torch.cat([lp_terms, lp_uni], dim=1)

        return logsumexp(lp_terms, dim=1)  # (B,)

    def log_prob(self, u_flat: torch.Tensor) -> torch.Tensor:
        """u_flat: (B, 3*N)  →  log q: (B,)"""
        u_flat = wrap_unit(u_flat)
        B = u_flat.shape[0]
        u = u_flat.view(B, self.n_solvent, 3)
        lp = self._log_prob_single(u.reshape(B * self.n_solvent, 3))  # (B*N,)
        return lp.view(B, self.n_solvent).sum(dim=1)                   # (B,)

    def sample(self, n: int) -> torch.Tensor:
        """Returns (n, 3*n_solvent). All particles sampled IID in one batched call."""
        M = n * self.n_solvent
        mix_probs = self.mix_probs.to(device=self.device, dtype=self.dtype)
        idx = torch.multinomial(mix_probs, num_samples=M, replacement=True)  # (M,)

        u = torch.empty(M, 3, device=self.device, dtype=self.dtype)

        if self.add_uniform_component:
            is_uniform = idx == self.n_components
            idx_shell = torch.where(~is_uniform)[0]
            idx_uni = torch.where(is_uniform)[0]
        else:
            idx_shell = torch.arange(M, device=self.device)
            idx_uni = idx.new_empty(0)

        if idx_shell.numel() > 0:
            centers_sel = self.component_centers[idx[idx_shell]]  # (n_shell, 3)
            noise = self.component_sigma_unit * torch.randn(
                idx_shell.numel(), 3, device=self.device, dtype=self.dtype
            )
            u[idx_shell] = wrap_unit(centers_sel + noise)

        if idx_uni.numel() > 0:
            u[idx_uni] = torch.rand(idx_uni.numel(), 3, device=self.device, dtype=self.dtype)

        return u.view(n, self.dim)

    def forward(self, n: int):
        u = self.sample(n)
        log_q = self.log_prob(u)
        return u, log_q

    __call__ = forward

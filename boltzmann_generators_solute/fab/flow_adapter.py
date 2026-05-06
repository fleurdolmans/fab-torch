"""
SoluteFlowFAB — FAB TrainableDistribution wrapper around SoluteSplineFlow.

The SoluteSplineFlow models 36 solvent particles. Its internal representation
is either Cartesian (use_polar=False) or log-polar (use_polar=True), but the
external interface of generator() / inverse_generator() is always Cartesian.

This adapter exposes the three methods that FAB requires:
  - log_prob(x_cart)              → log q(x) under the flow
  - sample_and_log_prob(shape)    → (x_cart, log q(x))
  - sample(shape)                 → x_cart

The "sample space" for FAB is therefore the 72-dimensional Cartesian coordinates
of the 36 solvent particles.  This matches the input expected by SoluteTarget2D.

Change-of-variables accounting
-------------------------------
SoluteSplineFlow conventions (verified from source):
  forward_map(x_repr)  → (z, logdet_fwd)   where logdet_fwd = log|det ∂z/∂x_repr|
  inverse_map(z)       → (x_repr, logdet_inv) where logdet_inv = log|det ∂x_repr/∂z|

  generator(z)         → (x_cart, gen_logdet)
      gen_logdet = log|det ∂x_cart/∂z|  (accounts for the polar Jacobian when use_polar=True)

  inverse_generator(x_cart) → (z, inv_gen_logdet)
      inv_gen_logdet = log|det ∂z/∂x_cart|

Therefore:
  log p_flow(x_cart) = log p_prior(z) + inv_gen_logdet   [from inverse_generator]
  log p_flow(x_cart) = log p_prior(z) - gen_logdet       [from generator]
"""

from typing import Tuple

import torch
import torch.nn as nn

from fab.trainable_distributions.base import TrainableDistribution


class SoluteFlowFAB(TrainableDistribution, nn.Module):
    """
    FAB-compatible wrapper for SoluteSplineFlow.

    Parameters
    ----------
    flow : SoluteSplineFlow
        Already-built flow (via build_solute_spline_flow).
        Can be pre-trained or freshly initialised.
    """

    def __init__(self, flow):
        nn.Module.__init__(self)
        TrainableDistribution.__init__(self)
        self._flow = flow
        # 36 solvent particles × 2 Cartesian dimensions
        self._event_shape = (flow.n_particles * 2,)

    # ------------------------------------------------------------------
    # TrainableDistribution / Distribution interface
    # ------------------------------------------------------------------

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def log_prob(self, x_cart: torch.Tensor) -> torch.Tensor:
        """
        Log probability of Cartesian samples under the flow.

        Parameters
        ----------
        x_cart : (B, 72)  Cartesian coordinates of the 36 solvent particles.

        Returns
        -------
        log_q : (B,)
        """
        z_flat, inv_gen_logdet = self._flow.inverse_generator(x_cart)
        log_q_z = self._flow.prior.log_prob(z_flat)
        # log p(x_cart) = log p(z) + log|det ∂z/∂x_cart|
        return log_q_z + inv_gen_logdet

    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw samples from the flow and return their log probabilities.

        Parameters
        ----------
        shape : (B,)  — tuple with a single integer (batch size)

        Returns
        -------
        x_cart : (B, 72)   Cartesian solvent coordinates
        log_q  : (B,)      log p_flow(x_cart)
        """
        n = shape[0]
        z_flat  = self._flow.prior.sample(n)                    # (B, 72)
        log_q_z = self._flow.prior.log_prob(z_flat)             # (B,)
        x_cart, gen_logdet = self._flow.generator(z_flat)       # (B,72), (B,)
        # gen_logdet = log|det ∂x_cart/∂z|
        # log p(x_cart) = log p(z) - log|det ∂x_cart/∂z|
        return x_cart, log_q_z - gen_logdet

    def sample(self, shape: Tuple) -> torch.Tensor:
        x, _ = self.sample_and_log_prob(shape)
        return x

    # ------------------------------------------------------------------
    # Pass-through helpers so FABModel.parameters() works correctly
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True):
        return self._flow.parameters(recurse=recurse)

    def state_dict(self, **kwargs):
        return self._flow.state_dict(**kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._flow.load_state_dict(state_dict, strict=strict)


class RealNVPFlowFAB(TrainableDistribution, nn.Module):
    """
    FAB-compatible wrapper for the RealNVP flow (74D, all 37 particles).

    RealNVP models the full 37-particle system (solute + 36 solvent) in 74D
    Cartesian space. The prior is torch.distributions.MultivariateNormal,
    whose .sample() takes a shape tuple rather than an int.

    Parameters
    ----------
    flow : RealNVP
        Built via BoltzmannGenerator.build(system).
    """

    def __init__(self, flow):
        nn.Module.__init__(self)
        TrainableDistribution.__init__(self)
        self._flow = flow
        self._event_shape = (flow.prior.event_shape[0],)  # (74,)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability of samples under the flow.

        Parameters
        ----------
        x : (B, 74)  Cartesian coordinates of all 37 particles.

        Returns
        -------
        log_q : (B,)
        """
        z, inv_logdet = self._flow.inverse_generator(x)
        log_q_z = self._flow.prior.log_prob(z)   # MultivariateNormal → (B,)
        return log_q_z + inv_logdet

    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw samples and return their log probabilities.

        Parameters
        ----------
        shape : (B,)

        Returns
        -------
        x     : (B, 74)
        log_q : (B,)
        """
        n = shape[0]
        z = self._flow.prior.sample((n,))          # (B, 74)
        log_q_z = self._flow.prior.log_prob(z)     # (B,)
        x, gen_logdet = self._flow.generator(z)    # (B, 74), (B,)
        return x, log_q_z - gen_logdet

    def sample(self, shape: Tuple) -> torch.Tensor:
        x, _ = self.sample_and_log_prob(shape)
        return x

    def parameters(self, recurse: bool = True):
        return self._flow.parameters(recurse=recurse)

    def state_dict(self, **kwargs):
        return self._flow.state_dict(**kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._flow.load_state_dict(state_dict, strict=strict)

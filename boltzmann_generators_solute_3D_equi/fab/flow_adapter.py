"""
SoluteFlowFAB3D — FAB TrainableDistribution wrapper for the 3D solute spline flow.

The CartesianSplineFlow (built with build_solute_spline_flow_3d) models 36
solvent particles in 3D Cartesian space (108D). The solute is fixed at the origin.

This adapter exposes the three methods FAB requires:
  - log_prob(x_cart)              → log q(x) under the flow
  - sample_and_log_prob(shape)    → (x_cart, log q(x))
  - sample(shape)                 → x_cart

The sample space for FAB is the 108-dimensional Cartesian coordinates of the
36 solvent particles. This matches the input expected by SoluteTarget3D.
"""

from typing import Tuple

import torch
import torch.nn as nn

from fab.trainable_distributions.base import TrainableDistribution


class SoluteFlowFAB3D(TrainableDistribution, nn.Module):
    """
    FAB-compatible wrapper for the 3D CartesianSplineFlow (solute system).

    Parameters
    ----------
    flow : CartesianSplineFlow
        Built via build_solute_spline_flow_3d, with fixed_solute=True.
        Can be pre-trained or freshly initialised.
    """

    def __init__(self, flow):
        nn.Module.__init__(self)
        TrainableDistribution.__init__(self)
        self._flow = flow
        # 36 solvent particles × 3 Cartesian dimensions
        self._event_shape = (flow.n_particles * 3,)

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
        x_cart : (B, 108)  Cartesian coordinates of the 36 solvent particles.

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
        x_cart : (B, 108)   Cartesian solvent coordinates
        log_q  : (B,)       log p_flow(x_cart)
        """
        n = shape[0]
        z_flat  = self._flow.prior.sample(n)                     # (B, 108)
        log_q_z = self._flow.prior.log_prob(z_flat)              # (B,)
        x_cart, gen_logdet = self._flow.generator(z_flat)        # (B,108), (B,)
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

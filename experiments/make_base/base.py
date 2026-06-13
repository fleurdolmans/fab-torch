import math

import torch
from torch import nn
import normflows as nf
from nflows.distributions.base import Distribution


class TrainableDiagonalNormal(Distribution):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, trainable: bool = True):
        super().__init__()

        if trainable:
            self.loc = nn.Parameter(mean.clone())
            self.log_scale = nn.Parameter(log_std.clone())
        else:
            self.register_buffer("loc", mean.clone())
            self.register_buffer("log_scale", log_std.clone())

        self._shape = torch.Size([mean.numel()])

    def _log_prob(self, inputs, context):
        log_2pi = math.log(2.0 * math.pi)
        z = (inputs - self.loc) * torch.exp(-self.log_scale)
        log_prob = -0.5 * (z**2 + 2.0 * self.log_scale + log_2pi)
        return log_prob.sum(dim=-1)

    def _sample(self, num_samples, context):
        eps = torch.randn(
            num_samples,
            *self._shape,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return self.loc.unsqueeze(0) + eps * torch.exp(self.log_scale).unsqueeze(0)


def make_nflows_diag_gaussian_from_target(target, trainable: bool = False, eps: float = 1e-3):
    if getattr(target, "train_data_i", None) is not None:
        data_i = target.train_data_i
    elif getattr(target, "val_data_i", None) is not None:
        data_i = target.val_data_i
    else:
        with torch.no_grad():
            ref_x = target.transform_data.reshape(1, -1).to(target.device)
            data_i, _ = target.coordinate_transform.inverse(ref_x)

    data_i = data_i.detach()
    mean = data_i.mean(dim=0)
    std = data_i.std(dim=0, unbiased=False).clamp_min(eps)

    return TrainableDiagonalNormal(
        mean=mean,
        log_std=torch.log(std),
        trainable=trainable,
    )


def make_structured_diag_gaussian_from_target(target, learn_mean_var: bool = True, eps: float = 1e-3):
    """
    Create a diagonal Gaussian base initialized from target internal-coordinate statistics.
    """
    import normflows as nf
    import torch

    dim = target.internal_dim

    # Prefer train data, then val, then transform_data mapped to i-space
    if getattr(target, "train_data_i", None) is not None:
        data_i = target.train_data_i
    elif getattr(target, "val_data_i", None) is not None:
        data_i = target.val_data_i
    else:
        # fallback: use single reference transformed point
        with torch.no_grad():
            ref_x = target.transform_data.reshape(1, -1).to(target.device)
            data_i, _ = target.coordinate_transform.inverse(ref_x)

    data_i = data_i.detach()
    mean = data_i.mean(dim=0)
    std = data_i.std(dim=0, unbiased=False).clamp_min(eps)
    std = std.clamp(min=0.05)

    base = nf.distributions.DiagGaussian(dim, trainable=learn_mean_var)

    # normflows stores loc/log_scale as parameters in many versions
    with torch.no_grad():
        if hasattr(base, "loc"):
            base.loc.copy_(mean)
        if hasattr(base, "log_scale"):
            base.log_scale.copy_(torch.log(std))

    return base

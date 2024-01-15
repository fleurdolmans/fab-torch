from typing import Tuple
from decimal import Decimal

import torch
from normflows import NormalizingFlow

from fab.trainable_distributions import TrainableDistribution


class WrappedNormFlowModel(TrainableDistribution):
    """Wraps the distribution from normflows library
    (https://github.com/VincentStimper/normalizing-flows) to work in this fab library."""

    def __init__(self, normalising_flow: NormalizingFlow):
        super(WrappedNormFlowModel, self).__init__()
        self._nf_model = normalising_flow

    def sample_and_log_prob(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) == 1
        return self._nf_model.sample(shape[0])

    def sample(self, shape: Tuple) -> torch.Tensor:
        return self.sample_and_log_prob(shape)[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._nf_model.log_prob(x)

        # # Explicit code from Normflows package copied here.
        # log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        # z = x
        # # print("I[0]", x[0])
        # for i in range(len(self._nf_model.flows) - 1, -1, -1):
        #     z, log_det = self._nf_model.flows[i].inverse(z)
        #     print("Layer {:>2}, Jac: {}".format(
        #         i, ', '.join([f'{Decimal(val):.2e}' for val in log_det.cpu().detach().numpy()])
        #     ))
        #     log_q += log_det
        # base_log_prob = self._nf_model.q0.log_prob(z)
        # # print(f"Base log Q: {base_log_prob.mean():.4f}, Log det Jac: {log_q.mean():.4f}")
        # # print("Z[0]", z[0])
        # # print("Base Q", base_log_prob)
        # # print("J_IZ", log_q)
        # return log_q + base_log_prob

    @property
    def event_shape(self) -> Tuple[int, ...]:
        try:
            return self._nf_model.q0.shape
        except:
            return self._nf_model.sample()[0].shape[1:]

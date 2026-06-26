import torch.nn as nn
from boltzmann_generators_3d.fab.types_ import Distribution


class TrainableDistribution(Distribution, nn.Module):
    """Base class for trainable distributions."""

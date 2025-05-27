"""
Mock model, to be used in algorithm test initialization.
"""

import torch as pt
import torch.nn as nn


class MockModel(nn.Module):
    """Mock PyTorch module that returns the input images it receives."""

    def __init__(self):
        super(MockModel, self).__init__()
        self.param = nn.Parameter(pt.ones(1, 1))

    def forward(self, x):
        """Return the input images."""
        return self.param * x


class MockSerializableModel3D(MockModel):
    """Mock serializable model that returns the input volumes it receives."""

    def __init__(self):
        super().__init__()
        self.init_params = dict(n_dims=3)

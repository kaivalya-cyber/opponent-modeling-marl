"""
MLP value network (critic).
Input: obs_dim (29)
Output: scalar value estimate shape (batch, 1)

Architecture: same as policy_net but output dim = 1.
"""

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """MLP value network producing a scalar value estimate."""

    def __init__(self, input_dim: int = 29) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim)

        Returns:
            value: (batch, 1)
        """
        value = self.net(x)
        assert value.shape[-1] == 1, f"Value output shape mismatch: {value.shape}"
        return value

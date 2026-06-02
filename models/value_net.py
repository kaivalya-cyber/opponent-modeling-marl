"""
MLP value network (critic).
Input: obs_dim (29)
Output: scalar value estimate shape (batch, 1)

Architecture: same as policy_net but output dim = 1.
Optional LayerNorm after each hidden layer (config.VALUE_USE_LAYERNORM).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ValueNet(nn.Module):
    """MLP value network producing a scalar value estimate."""

    def __init__(self, input_dim: int = 29) -> None:
        super().__init__()

        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        ]
        if config.VALUE_USE_LAYERNORM:
            layers.append(nn.LayerNorm(128))

        layers.extend([
            nn.Linear(128, 128),
            nn.ReLU(),
        ])
        if config.VALUE_USE_LAYERNORM:
            layers.append(nn.LayerNorm(128))

        layers.append(nn.Linear(128, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)
        assert value.shape[-1] == 1, f"Value output shape mismatch: {value.shape}"
        return value

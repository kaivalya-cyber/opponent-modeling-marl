"""
MLP policy network shared by all agents.
Input: obs_dim (29) + optional opponent_model_output_dim (if OM agent)
Output: logits over action_dim (5)

Architecture:
  Linear(input_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, action_dim)

Optional LayerNorm after each hidden layer (config.POLICY_USE_LAYERNORM).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PolicyNet(nn.Module):
    """MLP policy network producing action logits."""

    def __init__(self, input_dim: int = 29, action_dim: int = 5) -> None:
        super().__init__()
        self.action_dim = action_dim

        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        ]
        if config.POLICY_USE_LAYERNORM:
            layers.append(nn.LayerNorm(128))

        layers.extend([
            nn.Linear(128, 128),
            nn.ReLU(),
        ])
        if config.POLICY_USE_LAYERNORM:
            layers.append(nn.LayerNorm(128))

        layers.append(nn.Linear(128, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        temp = temperature if temperature is not None else config.SOFTMAX_TEMPERATURE
        logits = self.net(x)
        if temp != 1.0:
            logits = logits / temp
        return logits

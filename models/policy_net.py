"""
MLP policy network shared by all agents.
Input: obs_dim (29) + optional opponent_model_output_dim (if OM agent)
Output: logits over action_dim (5)

Architecture:
  Linear(input_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, action_dim)

Returns raw logits (NOT softmax). Distributions handle probabilities.
"""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """MLP policy network producing action logits."""

    def __init__(self, input_dim: int = 29, action_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, action_dim) — raw logits, no softmax.
        """
        logits = self.net(x)
        assert logits.shape[-1] == self.net[-1].out_features, (
            f"Logits last dim mismatch: {logits.shape}"
        )
        return logits

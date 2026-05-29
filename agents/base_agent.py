"""Abstract base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from training.rollout_buffer import RolloutBuffer


class BaseAgent(ABC):
    """Base class every agent must implement."""

    @abstractmethod
    def select_action(
        self, obs: np.ndarray, **kwargs
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Choose an action.

        Returns:
            action: int in [0, action_dim)
            log_prob: scalar tensor
            value: tensor of shape (1,)
        """

    def select_action_deterministic(self, obs: np.ndarray, **kwargs) -> int:
        """Choose an action greedily (argmax). Default: calls select_action then
        returns a deterministic choice via the policy logits.

        Override for agents with augmented observations (e.g., OM agents).

        Returns:
            action: int in [0, action_dim)
        """
        # Default implementation — subclasses with extra args should override
        action, _, _ = self.select_action(obs, **{k: v for k, v in kwargs.items() if k != 'last_opp_action'})
        return action

    @abstractmethod
    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        """Run PPO update on collected rollout data.

        Returns:
            dict with at least: policy_loss, value_loss, entropy
        """

    @abstractmethod
    def reset_episode(self) -> None:
        """Called at episode start — reset any recurrent state."""

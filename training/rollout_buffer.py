"""Rollout buffer for PPO — stores transitions and computes GAE returns."""

from __future__ import annotations

import numpy as np
import torch

import config


class RolloutBuffer:
    """Fixed-capacity buffer storing one PPO rollout for a single agent."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.opp_actions = np.zeros(capacity, dtype=np.int64)

        # --- NEW: Store next_obs for N-step returns ---
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

        self._ptr = 0

    @property
    def size(self) -> int:
        return self._ptr

    @property
    def full(self) -> bool:
        return self._ptr >= self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
        opp_action: int,
        **kwargs,
    ) -> None:
        assert self._ptr < self.capacity, "Buffer overflow"
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value.detach().cpu().item()
        self.log_probs[self._ptr] = log_prob.detach().cpu().item()
        self.dones[self._ptr] = done
        self.opp_actions[self._ptr] = opp_action

        # Store next_obs for N-step returns (if provided)
        if "next_obs" in kwargs:
            self.next_obs[self._ptr] = kwargs["next_obs"]
        self._ptr += 1

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        """Compute GAE advantages and discounted returns."""
        n = self._ptr
        self.advantages = np.zeros(n, dtype=np.float32)
        last_val = last_value.detach().cpu().item()

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_val
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values[:n]
        assert self.advantages.shape == (n,)
        assert self.returns.shape == (n,)

    def get_batches(self, batch_size: int):
        """Yield shuffled minibatches as dicts of tensors on config.DEVICE."""
        n = self._ptr
        assert self.advantages is not None, "Call compute_returns first"
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "obs": torch.tensor(self.obs[idx], dtype=torch.float32).to(config.DEVICE),
                "actions": torch.tensor(self.actions[idx], dtype=torch.long).to(config.DEVICE),
                "log_probs": torch.tensor(self.log_probs[idx], dtype=torch.float32).to(config.DEVICE),
                "advantages": torch.tensor(self.advantages[idx], dtype=torch.float32).to(config.DEVICE),
                "returns": torch.tensor(self.returns[idx], dtype=torch.float32).to(config.DEVICE),
                "values": torch.tensor(self.values[idx], dtype=torch.float32).to(config.DEVICE),
                "opp_actions": torch.tensor(self.opp_actions[idx], dtype=torch.long).to(config.DEVICE),
                "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32).to(config.DEVICE),
            }

    def reset(self) -> None:
        self._ptr = 0
        self.advantages = None
        self.returns = None

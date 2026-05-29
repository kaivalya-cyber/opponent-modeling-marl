"""Standard PPO agent — baseline without opponent modeling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from agents.base_agent import BaseAgent
from models.policy_net import PolicyNet
from models.value_net import ValueNet
from training.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""

    def __init__(self, obs_dim: int = 29, action_dim: int = 5) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy = PolicyNet(input_dim=obs_dim, action_dim=action_dim).to(config.DEVICE)
        self.value_net = ValueNet(input_dim=obs_dim).to(config.DEVICE)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=config.LEARNING_RATE,
        )

    def select_action(
        self, obs: np.ndarray, **kwargs
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select action from current policy.

        Returns:
            action: int
            log_prob: scalar tensor
            value: tensor shape (1,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self.policy(obs_t)
            value = self.value_net(obs_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def select_action_deterministic(self, obs: np.ndarray, **kwargs) -> int:
        """Select action greedily (argmax) — used for evaluation.

        Returns:
            action: int in [0, action_dim)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self.policy(obs_t)
        return int(logits.argmax(dim=-1).item())

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        """Run PPO update over collected data.

        Returns dict with policy_loss, value_loss, entropy.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(config.N_EPOCHS):
            for batch in rollout_buffer.get_batches(config.MINIBATCH_SIZE):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalise advantages
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward
                logits = self.policy(obs)
                values = self.value_net(obs).squeeze(-1)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPSILON, 1.0 + config.CLIP_EPSILON) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEFF * value_loss
                    - config.ENTROPY_COEFF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    config.GRAD_CLIP_NORM,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def reset_episode(self) -> None:
        """No recurrent state for vanilla PPO — no-op."""
        pass

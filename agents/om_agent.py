"""Opponent-modeling agent — extends PPO with a GRU/Transformer opponent model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import config
from agents.base_agent import BaseAgent
from models.policy_net import PolicyNet
from models.value_net import ValueNet
from models.opponent_model import build_opponent_model
from training.rollout_buffer import RolloutBuffer


class OMAgent(BaseAgent):
    """PPO agent augmented with a learned opponent model.

    Key improvement: During PPO updates, the opponent model's predictions
    are recomputed *with gradients enabled* so the policy loss can
    backpropagate through the opponent model. This teaches the opponent
    model not just to predict accurately (via cross-entropy loss) but
    also to produce features that are *useful* for the policy.
    """

    def __init__(self, obs_dim: int = 29, action_dim: int = 5) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Augmented input = obs (29) + opponent action probs (5) = 34
        augmented_dim = obs_dim + action_dim

        self.policy = PolicyNet(input_dim=augmented_dim, action_dim=action_dim).to(config.DEVICE)
        self.value_net = ValueNet(input_dim=augmented_dim).to(config.DEVICE)
        self.opponent_model = build_opponent_model(
            input_dim=config.OM_INPUT_DIM,
            hidden_dim=config.OM_HIDDEN_DIM,
            action_dim=action_dim,
            model_type=config.OM_MODEL_TYPE,
        ).to(config.DEVICE)

        # Number of recurrent layers (1 for Transformer, config.OM_NUM_LAYERS for GRU)
        self._om_num_layers = config.OM_NUM_LAYERS if config.OM_MODEL_TYPE == "gru" else 1
        self._om_hidden_dim = config.OM_HIDDEN_DIM

        # Hidden state — reset at episode start, carried within episode
        self.h: torch.Tensor = torch.zeros(
            self._om_num_layers, 1, self._om_hidden_dim, device=config.DEVICE
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters())
            + list(self.value_net.parameters())
            + list(self.opponent_model.parameters()),
            lr=config.LEARNING_RATE,
        )

        # Decaying OM loss weight — starts at config value, decays per episode
        self.om_loss_weight = config.OM_LOSS_WEIGHT

    def get_augmented_obs(
        self, own_obs: np.ndarray, last_opp_action: int
    ) -> torch.Tensor:
        """Build augmented observation: concat(own_obs, predicted opp_probs).

        Uses torch.no_grad() — this is for data collection during rollout.
        Gradient flow through opponent model happens during update().

        Args:
            own_obs: (29,) numpy array
            last_opp_action: int in [0, action_dim)

        Returns:
            augmented: (34,) tensor on CPU (will be moved to device later)
        """
        opp_action_onehot = F.one_hot(
            torch.tensor(last_opp_action, device=config.DEVICE),
            num_classes=self.action_dim,
        ).float()
        obs_t = torch.tensor(own_obs, dtype=torch.float32, device=config.DEVICE)

        gru_input = torch.cat([opp_action_onehot, obs_t]).unsqueeze(0).unsqueeze(0)
        # gru_input shape: (1, 1, 34)

        with torch.no_grad():
            opp_probs, new_h = self.opponent_model.predict_opponent_action(
                gru_input, self.h
            )
        self.h = new_h.detach()
        opp_probs = opp_probs.squeeze(0)  # (action_dim,)

        augmented = torch.cat([obs_t, opp_probs])
        assert augmented.shape == (self.obs_dim + self.action_dim,), (
            f"Augmented obs shape mismatch: {augmented.shape}"
        )
        return augmented.cpu()

    def select_action(
        self, obs: np.ndarray, *, last_opp_action: int = 0, **kwargs
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select action from augmented policy.

        Args:
            obs: (29,) numpy array
            last_opp_action: opponent's previous action (int)

        Returns:
            action: int
            log_prob: scalar tensor
            value: tensor shape (1,)
        """
        augmented = self.get_augmented_obs(obs, last_opp_action)
        aug_t = augmented.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.policy(aug_t)
            value = self.value_net(aug_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def select_action_deterministic(
        self, obs: np.ndarray, *, last_opp_action: int = 0, **kwargs
    ) -> int:
        """Select action greedily (no sampling) — used for evaluation.

        Args:
            obs: (29,) numpy array
            last_opp_action: opponent's previous action (int)

        Returns:
            action: int (argmax over policy logits)
        """
        augmented = self.get_augmented_obs(obs, last_opp_action)
        aug_t = augmented.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.policy(aug_t)

        return int(logits.argmax(dim=-1).item())

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        """Run PPO + opponent model update.

        Key design: The opponent model predictions are recomputed here
        *with gradients enabled*, so the policy_loss backpropagates through
        the opponent model. This dual signal (cross-entropy + policy gradient)
        teaches the opponent model to produce features useful for acting.

        Returns dict with policy_loss, value_loss, entropy, om_loss.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_om_loss = 0.0
        n_updates = 0

        for _ in range(config.N_EPOCHS):
            for batch in rollout_buffer.get_batches(config.MINIBATCH_SIZE):
                # Stored augmented obs (from rollout, no_grad)
                stored_obs = batch["obs"]          # (B, 34)
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                opp_actions = batch["opp_actions"]

                # ---- Recompute opponent model predictions WITH gradients ----
                # Extract raw observation portion (first 29 dims)
                raw_obs = stored_obs[:, :self.obs_dim]  # (B, 29)

                # Build GRU/Transformer input from raw obs + opponent action
                opp_onehot = F.one_hot(opp_actions, num_classes=self.action_dim).float()
                gru_in = torch.cat([opp_onehot, raw_obs], dim=-1)  # (B, 34)
                gru_in = gru_in.unsqueeze(1)  # (B, 1, 34)

                # Forward through opponent model WITH gradients enabled
                om_logits, _ = self.opponent_model(gru_in)
                # No .detach() — gradients flow from policy loss back through
                # the opponent model, teaching it to produce features useful
                # for acting (in addition to the direct cross-entropy signal).
                opp_probs = F.softmax(om_logits, dim=-1)

                # Build fresh augmented obs (gradients flow to opponent model)
                aug_obs = torch.cat([raw_obs, opp_probs], dim=-1)  # (B, 34)

                # ---- PPO losses ----
                logits = self.policy(aug_obs)
                values = self.value_net(aug_obs).squeeze(-1)

                # Normalise advantages
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - config.CLIP_EPSILON, 1.0 + config.CLIP_EPSILON
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)

                # ---- Opponent model loss (direct supervision) ----
                om_loss = F.cross_entropy(om_logits, opp_actions)

                # ---- Total loss ----
                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEFF * value_loss
                    - config.ENTROPY_COEFF * entropy
                    + self.om_loss_weight * om_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_net.parameters())
                    + list(self.opponent_model.parameters()),
                    config.GRAD_CLIP_NORM,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_om_loss += om_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "om_loss": total_om_loss / max(n_updates, 1),
        }

    def reset_episode(self) -> None:
        """Reset hidden state to zeros at episode start."""
        self.h = torch.zeros(
            self._om_num_layers, 1, self._om_hidden_dim, device=config.DEVICE
        )

"""
Intrinsic Curiosity Module (ICM) for exploration bonus.

The ICM learns a forward dynamics model (predicting next state features)
and an inverse dynamics model (predicting action from current and next state features).
The prediction error of the forward model serves as an intrinsic reward bonus,
encouraging the agent to visit novel or unpredictable states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ICM(nn.Module):
    """Intrinsic Curiosity Module.

    Architecture:
      - Feature encoder: projects observation to latent space
      - Forward model: predicts next latent state from (latent, action)
      - Inverse model: predicts action from (latent, next_latent)

    Intrinsic reward = ||forward_prediction_error||^2 (scaled by ICM_BONUS_WEIGHT).
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        hidden_dim = config.ICM_HIDDEN_DIM

        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_output_dim = hidden_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.encoder_output_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.encoder_output_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(self.encoder_output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ICM.

        Args:
            obs: (batch, obs_dim)
            next_obs: (batch, obs_dim)
            actions: (batch,) long tensor of discrete actions

        Returns:
            forward_loss: scalar MSE loss for forward dynamics
            inverse_loss: scalar cross-entropy loss for inverse dynamics
            intrinsic_reward: (batch,) forward prediction error for each sample
        """
        phi = self.feature_encoder(obs)
        phi_next = self.feature_encoder(next_obs)

        # Forward model: predict phi_next from (phi, action)
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        forward_input = torch.cat([phi, actions_onehot], dim=-1)
        phi_next_pred = self.forward_model(forward_input)

        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach(), reduction="none")
        forward_loss = forward_loss.mean(dim=-1)

        intrinsic_reward = forward_loss.detach()

        # Inverse model: predict action from (phi, phi_next)
        inverse_input = torch.cat([phi, phi_next.detach()], dim=-1)
        action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(action_logits, actions, reduction="none")

        return forward_loss.mean(), inverse_loss.mean(), intrinsic_reward

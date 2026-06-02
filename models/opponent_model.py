"""
GRU / Transformer opponent model — the key novelty.

Learns to predict the opponent's next action given their observation history.

Architecture options (via config.OM_MODEL_TYPE):
  - "gru": Multi-layer GRU with layer norm and dropout.
  - "transformer": Causal Transformer encoder for long-range dependency modeling.

Input at each step: opponent's last action (one-hot, dim=5)
                   + current joint observation (29 dims)
                   = input_dim 34

Hidden state is carried across timesteps during rollout
and reset at episode start. This is critical — do not reset h mid-episode.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class OpponentModel(nn.Module):
    """GRU-based sequence model that predicts opponent actions.

    Multi-layer GRU with optional layer norm and dropout between layers.
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        action_dim: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_seq.ndim == 3, f"Expected 3D input, got shape {x_seq.shape}"
        batch_size = x_seq.size(0)

        if h is None:
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x_seq.device
            )

        gru_out, new_h = self.gru(x_seq, h)
        last_out = gru_out[:, -1, :]
        last_out = self.layer_norm(last_out)
        logits = self.fc(last_out)

        assert logits.shape == (batch_size, self.action_dim), (
            f"Logits shape mismatch: {logits.shape}"
        )
        assert new_h.shape == (self.num_layers, batch_size, self.hidden_dim), (
            f"Hidden state shape mismatch: {new_h.shape}"
        )
        return logits, new_h

    def predict_opponent_action(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict opponent action probabilities.

        Returns:
            action_probs: (batch, action_dim)
            new_h: (num_layers, batch, hidden_dim)
            confidence: None (kept for API compatibility with ensemble)
        """
        logits, new_h = self.forward(x_seq, h)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, new_h, None

    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Return confidence (max softmax probability) for each item in batch."""
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values


class TransformerOpponentModel(nn.Module):
    """Transformer-based opponent model with causal self-attention."""

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        action_dim: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
            ),
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return self.causal_mask[:sz, :sz]

    def forward(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_seq.ndim == 3, f"Expected 3D input, got shape {x_seq.shape}"
        batch_size, seq_len, _ = x_seq.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        x = self.input_proj(x_seq)
        x = x + self.pos_encoding[:, :seq_len, :]

        causal_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        out = self.transformer(x, mask=causal_mask)
        last_out = out[:, -1, :]
        last_out = self.layer_norm(last_out)
        logits = self.fc(last_out)

        dummy_h = torch.zeros(1, batch_size, self.hidden_dim, device=x_seq.device)

        assert logits.shape == (batch_size, self.action_dim), (
            f"Logits shape mismatch: {logits.shape}"
        )
        return logits, dummy_h

    def predict_opponent_action(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits, dummy_h = self.forward(x_seq, h)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, dummy_h, None

    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values


class EnsembleOpponentModel(nn.Module):
    """Ensemble of opponent models for more robust predictions.

    Aggregates predictions from multiple models via mean or majority voting.
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 64,
        action_dim: int = 5,
        model_type: str = "gru",
        ensemble_size: int = 3,
        vote: str = "mean",
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.action_dim = action_dim
        self.vote = vote
        self.hidden_dim = hidden_dim

        self.models = nn.ModuleList([
            build_opponent_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                model_type=model_type,
            )
            for _ in range(ensemble_size)
        ])

    def forward(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_logits = []
        all_h = []
        for model in self.models:
            logits, new_h = model(x_seq, h)
            all_logits.append(logits.unsqueeze(0))
            all_h.append(new_h.unsqueeze(0))

        logits = torch.mean(torch.cat(all_logits, dim=0), dim=0)
        new_h = torch.mean(torch.cat(all_h, dim=0), dim=0)
        return logits, new_h

    def predict_opponent_action(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.vote == "mean":
            all_probs = []
            all_h = []
            for model in self.models:
                probs, new_h, _ = model.predict_opponent_action(x_seq, h)
                all_probs.append(probs.unsqueeze(0))
                all_h.append(new_h.unsqueeze(0))

            probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)
            new_h = torch.mean(torch.cat(all_h, dim=0), dim=0)

            # Confidence = mean of individual model confidences
            confidences = [p.max(dim=-1).values for p in all_probs]
            confidence = torch.mean(torch.stack(confidences, dim=0), dim=0)

            return probs, new_h, confidence

        elif self.vote == "majority":
            # Each model produces a distribution; majority vote over argmax
            all_probs = []
            all_h = []
            all_actions = []
            for model in self.models:
                probs, new_h, _ = model.predict_opponent_action(x_seq, h)
                all_probs.append(probs.unsqueeze(0))
                all_h.append(new_h.unsqueeze(0))
                all_actions.append(probs.argmax(dim=-1).unsqueeze(0))

            # Majority vote
            all_actions_t = torch.cat(all_actions, dim=0)  # (ens, B)
            batch_size = all_actions_t.size(1)
            probs = torch.zeros(batch_size, self.action_dim, device=x_seq.device)

            for b in range(batch_size):
                votes = all_actions_t[:, b]
                counts = torch.bincount(votes, minlength=self.action_dim)
                winner = counts.argmax()
                probs[b, winner] = 1.0

            new_h = torch.mean(torch.cat(all_h, dim=0), dim=0)
            confidence = probs.max(dim=-1).values
            return probs, new_h, confidence

    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values


def build_opponent_model(
    input_dim: int = 34,
    hidden_dim: int = 64,
    action_dim: int = 5,
    model_type: str = "gru",
) -> nn.Module:
    if config.OM_ENSEMBLE_SIZE > 1:
        return EnsembleOpponentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            model_type=model_type,
            ensemble_size=config.OM_ENSEMBLE_SIZE,
            vote=config.OM_ENSEMBLE_VOTE,
        )

    if model_type == "gru":
        return OpponentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=config.OM_NUM_LAYERS,
            dropout=config.OM_DROPOUT,
        )
    elif model_type == "transformer":
        return TransformerOpponentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_heads=config.TF_NUM_HEADS,
            num_layers=config.TF_NUM_LAYERS,
            dim_feedforward=config.TF_DIM_FEEDFORWARD,
            max_seq_len=config.TF_MAX_SEQ_LEN,
            dropout=config.OM_DROPOUT,
        )
    else:
        raise ValueError(
            f"Unknown opponent model type: {model_type}. "
            f"Valid options: 'gru', 'transformer'"
        )


def soft_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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
        """Forward pass through GRU + linear head.

        Args:
            x_seq: (batch, seq_len, input_dim)
            h: (num_layers, batch, hidden_dim) or None for zero init

        Returns:
            logits: (batch, action_dim) — logits from the *last* time-step
            new_h:  (num_layers, batch, hidden_dim)
        """
        assert x_seq.ndim == 3, f"Expected 3D input, got shape {x_seq.shape}"
        batch_size = x_seq.size(0)

        if h is None:
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x_seq.device
            )

        gru_out, new_h = self.gru(x_seq, h)
        # Use the output from the last time-step and apply layer norm
        last_out = gru_out[:, -1, :]  # (batch, hidden_dim)
        last_out = self.layer_norm(last_out)
        logits = self.fc(last_out)  # (batch, action_dim)

        assert logits.shape == (batch_size, self.action_dim), (
            f"Logits shape mismatch: {logits.shape}"
        )
        assert new_h.shape == (self.num_layers, batch_size, self.hidden_dim), (
            f"Hidden state shape mismatch: {new_h.shape}"
        )
        return logits, new_h

    def predict_opponent_action(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict opponent action probabilities (softmax of logits).

        Args:
            x_seq: (batch, seq_len, input_dim)
            h: optional hidden state

        Returns:
            action_probs: (batch, action_dim) — probability distribution
            new_h: (num_layers, batch, hidden_dim)
        """
        logits, new_h = self.forward(x_seq, h)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, new_h


class TransformerOpponentModel(nn.Module):
    """Transformer-based opponent model with causal self-attention.

    Uses a learned positional encoding and a causal mask so the model
    can only attend to past timesteps when predicting the next action.
    """

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

        # Project input to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        # Transformer encoder layers
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

        # Causal mask (upper triangular, so position i can only attend to j <= i)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1
            ),
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Return causal mask of shape (sz, sz)."""
        return self.causal_mask[:sz, :sz]

    def forward(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Transformer.

        Args:
            x_seq: (batch, seq_len, input_dim)
            h: ignored (Transformer has no recurrent state; use reset_sequence instead)

        Returns:
            logits: (batch, action_dim) — logits from the last time-step
            new_h: dummy torch.Tensor (unused, placeholder for API compatibility)
        """
        assert x_seq.ndim == 3, f"Expected 3D input, got shape {x_seq.shape}"
        batch_size, seq_len, _ = x_seq.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        # Project input and add positional encoding
        x = self.input_proj(x_seq)  # (B, seq_len, hidden_dim)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer
        out = self.transformer(x, mask=causal_mask)  # (B, seq_len, hidden_dim)
        last_out = out[:, -1, :]  # (B, hidden_dim)
        last_out = self.layer_norm(last_out)
        logits = self.fc(last_out)  # (B, action_dim)

        # Return dummy hidden state for API compatibility with GRU interface
        dummy_h = torch.zeros(1, batch_size, self.hidden_dim, device=x_seq.device)

        assert logits.shape == (batch_size, self.action_dim), (
            f"Logits shape mismatch: {logits.shape}"
        )
        return logits, dummy_h

    def predict_opponent_action(
        self, x_seq: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict opponent action probabilities (softmax of logits).

        Args:
            x_seq: (batch, seq_len, input_dim)
            h: ignored for Transformer (maintained for API compatibility)

        Returns:
            action_probs: (batch, action_dim) — probability distribution
            new_h: dummy tensor (unused)
        """
        logits, dummy_h = self.forward(x_seq, h)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, dummy_h


def build_opponent_model(
    input_dim: int = 34,
    hidden_dim: int = 64,
    action_dim: int = 5,
    model_type: str = "gru",
) -> nn.Module:
    """Factory function to build the selected opponent model variant.

    Args:
        input_dim: Dimension of input features per timestep.
        hidden_dim: Hidden dimension for GRU / Transformer.
        action_dim: Number of opponent actions.
        model_type: "gru" or "transformer".

    Returns:
        An OpponentModel or TransformerOpponentModel instance.
    """
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

"""
Hyperparameter sweep runner for wandb sweeps.

Usage:
    wandb sweep sweep.yaml
    wandb agent <sweep_id>

Or run locally without wandb:
    python -m experiments.sweep

Integrates with the existing Trainer and agent infrastructure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

import config
from env.predator_prey import PredatorPreyEnv
from agents.ppo_agent import PPOAgent
from agents.om_agent import OMAgent
from training.trainer import Trainer

logger = logging.getLogger(__name__)

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sweep runner — called by wandb agent
# ---------------------------------------------------------------------------

def run_sweep() -> None:
    """Entry point for wandb agent. Reads hyperparams from wandb.config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not WANDB_AVAILABLE or wandb.run is None:
        logger.warning("wandb not available. Running with default config instead.")
        _run_single_trial(run_id="local")
        return

    run = wandb.run
    hp = run.config

    # Override config from sweep hyperparameters
    config.LEARNING_RATE = hp.get("learning_rate", config.LEARNING_RATE)
    config.CLIP_EPSILON = hp.get("clip_epsilon", config.CLIP_EPSILON)
    config.ENTROPY_COEFF = hp.get("entropy_coeff", config.ENTROPY_COEFF)
    config.GAMMA = hp.get("gamma", config.GAMMA)
    config.GAE_LAMBDA = hp.get("gae_lambda", config.GAE_LAMBDA)
    config.N_EPOCHS = hp.get("n_epochs", config.N_EPOCHS)
    config.OM_LOSS_WEIGHT = hp.get("om_loss_weight", config.OM_LOSS_WEIGHT)
    config.OM_HIDDEN_DIM = hp.get("om_hidden_dim", config.OM_HIDDEN_DIM)
    config.OM_NUM_LAYERS = hp.get("om_num_layers", config.OM_NUM_LAYERS)
    config.OM_DROPOUT = hp.get("om_dropout", config.OM_DROPOUT)
    config.VALUE_LOSS_COEFF = hp.get("value_loss_coeff", config.VALUE_LOSS_COEFF)
    config.GRAD_CLIP_NORM = hp.get("grad_clip_norm", config.GRAD_CLIP_NORM)
    config.USE_WANDB = True
    
    # Use wandb run ID for unique seed and experiment naming
    run_id = run.id

    logger.info(
        "Sweep trial config: lr=%.6f clip=%.3f ent=%.4f gamma=%.3f lambda=%.2f "
        "nepochs=%d omw=%.3f omdim=%d omlayers=%d omdrop=%.2f vcoeff=%.3f "
        "gradclip=%.3f run_id=%s",
        config.LEARNING_RATE,
        config.CLIP_EPSILON,
        config.ENTROPY_COEFF,
        config.GAMMA,
        config.GAE_LAMBDA,
        config.N_EPOCHS,
        config.OM_LOSS_WEIGHT,
        config.OM_HIDDEN_DIM,
        config.OM_NUM_LAYERS,
        config.OM_DROPOUT,
        config.VALUE_LOSS_COEFF,
        config.GRAD_CLIP_NORM,
        run_id,
    )

    _run_single_trial(run_id=run_id)


def _run_single_trial(run_id: str = "local") -> None:
    """Run a single training trial with the current config.
    
    Args:
        run_id: Unique identifier for this run (wandb run ID or "local").
    """
    config.set_seed(abs(hash(run_id)) % (2**31))

    # Determine if we're using OM
    is_om = config.OM_MODEL_TYPE in ("gru", "transformer")

    if is_om:
        predator = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        prey = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        experiment_name = f"sweep_om_{run_id}"
    else:
        predator = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        prey = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        experiment_name = f"sweep_baseline_{run_id}"

    trainer = Trainer(
        predator=predator,
        prey=prey,
        experiment_name=experiment_name,
        total_episodes=config.TOTAL_EPISODES,
        is_om=is_om,
    )

    trainer.train()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if WANDB_AVAILABLE and "WANDB_SWEEP_ID" in os.environ:
        # Running as wandb agent
        run_sweep()
    else:
        # Local run — use default config
        logger.info("Running sweep locally with default config.")
        _run_single_trial(run_id="local")

# config.py — All hyperparameters live here. Nothing hardcoded elsewhere.

import random
import numpy as np
import torch

# Environment
GRID_SIZE = 10
MAX_STEPS = 200
OBS_DIM = 29
ACTION_DIM = 5

# Training
TOTAL_EPISODES = 2000
ROLLOUT_STEPS = 2048
EVAL_INTERVAL = 50
EVAL_EPISODES = 20
CHECKPOINT_INTERVAL = 500
PAST_SELF_EVAL = True         # whether to eval against past snapshots
SNAPSHOT_INTERVAL = 500       # episodes between saving eval snapshots
PAST_SELF_EVAL_EPISODES = 20  # number of eval episodes vs past self

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
N_EPOCHS = 4
MINIBATCH_SIZE = 64
GRAD_CLIP_NORM = 0.5       # max gradient norm for clipping

# Opponent Model
OM_LOSS_WEIGHT = 0.5
OM_LOSS_WEIGHT_DECAY = 0.9995  # per-episode decay factor (1.0 = no decay)
OM_HIDDEN_DIM = 64
OM_NUM_LAYERS = 2         # number of GRU layers
OM_DROPOUT = 0.1           # dropout between GRU layers (0 for single-layer)
OM_INPUT_DIM = 34          # 5 (opp action onehot) + 29 (own obs)
OM_MODEL_TYPE = "gru"      # "gru" or "transformer"

# Transformer opponent model (used when OM_MODEL_TYPE == "transformer")
TF_NUM_HEADS = 4
TF_NUM_LAYERS = 2
TF_DIM_FEEDFORWARD = 128
TF_MAX_SEQ_LEN = 64

# Device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Logging
USE_WANDB = False           # set True to enable Weights & Biases logging
WANDB_PROJECT = "opponent-modeling-marl"

# Curriculum Learning
CURRICULUM_ENABLED = False  # set True to use progressive grid sizes
CURRICULUM_SCHEDULE = [     # (start_episode, grid_size, max_steps)
    (0,    5, 100),         # episodes   0–499: 5×5 grid
    (500,  7, 150),         # episodes 500–999: 7×7 grid
    (1000, 10, 200),        # episodes 1000+:    10×10 grid
]

# Tournament
TOURNAMENT_MATCHES = 200    # matches per matchup in tournament script


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

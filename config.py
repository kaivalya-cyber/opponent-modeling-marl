# config.py — All hyperparameters live here. Nothing hardcoded elsewhere.

import random
import numpy as np
import torch

# ── Environment ──────────────────────────────────────────────────────────
GRID_SIZE = 10
MAX_STEPS = 200
OBS_DIM = 29
ACTION_DIM = 5

# ── Training ─────────────────────────────────────────────────────────────
TOTAL_EPISODES = 2000
ROLLOUT_STEPS = 2048
EVAL_INTERVAL = 50
EVAL_EPISODES = 20
CHECKPOINT_INTERVAL = 500
PAST_SELF_EVAL = True
SNAPSHOT_INTERVAL = 500
PAST_SELF_EVAL_EPISODES = 20

# ── PPO ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
N_EPOCHS = 4
MINIBATCH_SIZE = 64
GRAD_CLIP_NORM = 0.5
GRAD_CLIP_VALUE = 0.0

# ── Feature 1: Softmax temperature (lower = more deterministic) ─────────
SOFTMAX_TEMPERATURE = 1.0

# ── Feature 2-3: Separate weight decay for policy & value ───────────────
POLICY_WEIGHT_DECAY = 0.0
VALUE_WEIGHT_DECAY = 0.0

# ── Feature 4-6: Separate optimizers for policy & value ─────────────────
USE_SEPARATE_OPTIMIZERS = False
POLICY_LR = 3e-4
VALUE_LR = 3e-4

# ── Feature 7-8: LayerNorm in policy/value nets ─────────────────────────
POLICY_USE_LAYERNORM = False
VALUE_USE_LAYERNORM = False

# ── Feature 9: Value residual clip ───────────────────────────────────────
VALUE_RESIDUAL_CLIP = 0.0

# ── Feature 10: Reward scale ─────────────────────────────────────────────
REWARD_SCALE = 1.0

# ── Feature 11-13: OM accuracy logging, focal loss, label smoothing ─────
OM_LOG_ACCURACY = True
OM_FOCAL_LOSS_GAMMA = 0.0
OM_LABEL_SMOOTHING = 0.0

# ── Feature 14: MC Dropout samples for OM uncertainty ────────────────────
OM_MC_DROPOUT_SAMPLES = 0

# ── Feature 15: OM weight ramp-up ────────────────────────────────────────
OM_WEIGHT_RAMP_EPISODES = 0

# ── Feature 16-17: Model EMA (exponential moving average) ────────────────
MODEL_EMA_ENABLED = False
MODEL_EMA_TAU = 0.995

# ── Feature 18: OM accuracy rolling window ───────────────────────────────
OM_ACCURACY_WINDOW = 100

# ── Feature 19-20: Early stopping ────────────────────────────────────────
EARLY_STOP_PATIENCE = 1000
EARLY_STOP_THRESHOLD = 0.95

# ── Feature 21-23: Evaluation toggles ────────────────────────────────────
EVAL_DETERMINISTIC = True
EVAL_WITH_RANDOM = True
EVAL_WITH_PAST_SELF = True

# ── Feature 24-25: KL and explained variance tracking ────────────────────
KL_TRACKING = True
EXPLAINED_VAR_TRACKING = True

# ── Feature 26-27: Scheduled entropy and clip epsilon ────────────────────
ENTROPY_TARGET = 0.0
ENTROPY_SCHEDULE_STEPS = 0
CLIP_EPSILON_SCHEDULE_STEPS = 0

# ── Feature 28-29: Separate grad clips for actor/critic ──────────────────
ACTOR_GRAD_CLIP_NORM = 0.0
CRITIC_GRAD_CLIP_NORM = 0.0

# ── Feature 30-32: Gradient/weight/activation logging ────────────────────
LOG_GRAD_NORMS = False
LOG_WEIGHT_HISTOGRAMS = False
LOG_ACTIVATIONS = False

# ── Feature 33-37: Episode stats, streaks, best episode, speed, ETA ──────
LOG_EPISODE_STATS = True
TRACK_STREAKS = True
TRACK_BEST_EPISODE = True
LOG_STEPS_PER_SECOND = True
LOG_ETA = True

# ── Feature 38-39: JSON metrics export, config diff ──────────────────────
EXPORT_JSON_METRICS = True
LOG_CONFIG_DIFF = True

# ── Feature 40: Experiment resume from checkpoint ────────────────────────
ALLOW_RESUME = False

# ── Feature 41: Time-limit bootstrap handling ────────────────────────────
HANDLE_TIME_LIMIT_BOOTSTRAP = False

# ── Feature 42: Action noise (exploration) ───────────────────────────────
ACTION_NOISE_STD = 0.0

# ── Feature 43-45: Stochastic env, obstacles, partial observability ──────
STOCHASTIC_ENV = False
OBSTACLE_DENSITY = 0.0
PARTIAL_OBS_RADIUS = 0

# ── Feature 46: Eval confidence interval ─────────────────────────────────
EVAL_CONFIDENCE_INTERVAL = False

# ── Feature 47: Render best episode ──────────────────────────────────────
RENDER_BEST_EPISODE = False
RENDER_DIR = "renders"

# ── Feature 48-50: Profiling, description, tags ──────────────────────────
PROFILE_MODE = False
EXPERIMENT_TAGS = ""
EXPERIMENT_DESCRIPTION = ""

# Gradient accumulation
GRAD_ACCUMULATION_STEPS = 1

# Learning rate scheduler
LR_SCHEDULER_TYPE = "cosine"
LR_WARMUP_STEPS = 100
LR_MIN_RATIO = 0.1

# N-step returns
N_STEP_RETURNS = 1

# Observation / Reward normalization
OBS_NORMALIZE = False
OBS_NORM_CLIP = 10.0
REWARD_NORMALIZE = False
REWARD_NORM_CLIP = 10.0

# Opponent Model
OM_LOSS_WEIGHT = 0.5
OM_LOSS_WEIGHT_DECAY = 0.9995
OM_HIDDEN_DIM = 64
OM_NUM_LAYERS = 2
OM_DROPOUT = 0.1
OM_INPUT_DIM = 34
OM_MODEL_TYPE = "gru"

OM_USE_TARGET_NET = False
OM_TARGET_TAU = 0.005
OM_TARGET_UPDATE_INTERVAL = 5
OM_CLIP_LOSS = 0.0
OM_ENSEMBLE_SIZE = 1
OM_ENSEMBLE_VOTE = "mean"
OM_CONFIDENCE_THRESHOLD = 0.0
OM_SURPRISE_THRESHOLD = 0.0

# Transformer opponent model
TF_NUM_HEADS = 4
TF_NUM_LAYERS = 2
TF_DIM_FEEDFORWARD = 128
TF_MAX_SEQ_LEN = 64

# ICM
ICM_ENABLED = False
ICM_HIDDEN_DIM = 128
ICM_LOSS_WEIGHT = 0.1
ICM_FORWARD_LOSS_WEIGHT = 0.2
ICM_BONUS_WEIGHT = 1.0

# Parameter noise
PARAM_NOISE_ENABLED = False
PARAM_NOISE_STD = 0.1
PARAM_NOISE_STD_MIN = 0.01
PARAM_NOISE_DECAY = 0.995

# Action repeat / Frame stack
ACTION_REPEAT = 1
FRAME_STACK = 1

# Device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Logging
USE_WANDB = False
WANDB_PROJECT = "opponent-modeling-marl"
USE_TENSORBOARD = True
TENSORBOARD_DIR = "runs"

# Checkpoint manager
CKPT_KEEP_BEST = 5
CKPT_METRIC = "capture_rate"

# Multi-seed
SEEDS = [42]

# Curriculum
CURRICULUM_ENABLED = False
CURRICULUM_SCHEDULE = [
    (0,    5, 100),
    (500,  7, 150),
    (1000, 10, 200),
]

# Tournament
TOURNAMENT_MATCHES = 200


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config() -> list[str]:
    warnings = []
    if OM_ENSEMBLE_SIZE < 1:
        warnings.append("OM_ENSEMBLE_SIZE must be >= 1, resetting to 1")
    if OM_ENSEMBLE_VOTE not in ("mean", "majority"):
        warnings.append("OM_ENSEMBLE_VOTE must be 'mean' or 'majority', resetting to 'mean'")
    if FRAME_STACK < 1:
        warnings.append("FRAME_STACK must be >= 1, resetting to 1")
    if ACTION_REPEAT < 1:
        warnings.append("ACTION_REPEAT must be >= 1, resetting to 1")
    if GRAD_ACCUMULATION_STEPS < 1:
        warnings.append("GRAD_ACCUMULATION_STEPS must be >= 1, resetting to 1")
    if N_STEP_RETURNS < 1:
        warnings.append("N_STEP_RETURNS must be >= 1, resetting to 1")
    if OM_TARGET_TAU <= 0 or OM_TARGET_TAU > 1:
        warnings.append("OM_TARGET_TAU must be in (0, 1], resetting to 0.005")
    if SOFTMAX_TEMPERATURE <= 0:
        warnings.append("SOFTMAX_TEMPERATURE must be > 0, resetting to 1.0")
    return warnings

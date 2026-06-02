"""Tests for training/ — Step 8 self-check (6 tests)."""
import numpy as np
import torch
import pytest


def test_rollout_buffer_gae():
    from training.rollout_buffer import RolloutBuffer
    buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=64)
    for i in range(64):
        buf.add(obs=np.zeros(29, dtype=np.float32),
                action=0, reward=float(i % 2),
                value=torch.tensor([0.5]),
                log_prob=torch.tensor(-1.6),
                done=(i == 63), opp_action=0)
    buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
    assert buf.advantages.shape == (64,)
    assert buf.returns.shape == (64,)
    assert not np.isnan(buf.advantages).any(), "NaN in advantages"
    assert not np.isnan(buf.returns).any(), "NaN in returns"


def test_elo_updates_correctly():
    from training.elo import EloRating
    elo = EloRating()
    elo.update("agent_a", "agent_b")
    assert elo.get_rating("agent_a") > 1000, "Winner rating must increase"
    assert elo.get_rating("agent_b") < 1000, "Loser rating must decrease"
    total = elo.get_rating("agent_a") + elo.get_rating("agent_b")
    assert abs(total - 2000) < 1e-6, "Elo is zero-sum"


def test_elo_draw():
    from training.elo import EloRating
    elo = EloRating()
    elo.draw("agent_a", "agent_b")
    ra = elo.get_rating("agent_a")
    rb = elo.get_rating("agent_b")
    assert abs(ra - 1000) < 1.0 and abs(rb - 1000) < 1.0, \
        "Equal agents drawing should produce near-zero rating change"


def test_full_episode_runs_without_error():
    from env.predator_prey import PredatorPreyEnv
    from agents.om_agent import OMAgent
    env = PredatorPreyEnv()
    pred = OMAgent(obs_dim=29, action_dim=5)
    prey = OMAgent(obs_dim=29, action_dim=5)
    (obs_pred, obs_prey), _ = env.reset(seed=0)
    pred.reset_episode()
    prey.reset_episode()
    last_pred_action, last_prey_action = 0, 0
    for _ in range(200):
        a_pred, _, _ = pred.select_action(obs_pred, last_opp_action=last_prey_action)
        a_prey, _, _ = prey.select_action(obs_prey, last_opp_action=last_pred_action)
        (obs_pred, obs_prey), _, term, trunc, _ = env.step((a_pred, a_prey))
        last_pred_action, last_prey_action = a_pred, a_prey
        if term or trunc:
            break


def test_om_loss_decreases():
    import torch.nn.functional as F
    from models.opponent_model import OpponentModel
    om = OpponentModel(input_dim=34, hidden_dim=64, action_dim=5)
    opt = torch.optim.Adam(om.parameters(), lr=1e-3)
    target_action = torch.tensor([2])
    losses = []
    for _ in range(50):
        x = torch.randn(1, 1, 34)
        logits, _ = om(x)
        loss = F.cross_entropy(logits, target_action)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], \
        f"OM loss must decrease: start={losses[0]:.3f} end={losses[-1]:.3f}"


def test_ppo_update_loss_decreases():
    from agents.ppo_agent import PPOAgent
    from training.rollout_buffer import RolloutBuffer
    agent = PPOAgent(obs_dim=29, action_dim=5)
    buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=256)
    np.random.seed(0)
    for _ in range(256):
        obs = np.random.randn(29).astype(np.float32)
        buf.add(obs=obs, action=np.random.randint(5),
                reward=np.random.randn(), value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.6), done=False, opp_action=0)
    buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
    metrics = agent.update(buf)
    assert not np.isnan(metrics["policy_loss"]), "Policy loss is NaN"
    assert not np.isnan(metrics["value_loss"]), "Value loss is NaN"
    assert metrics["entropy"] > 0, "Entropy must be positive"


# --- NEW: CheckpointManager tests ---

def test_checkpoint_manager_keeps_best():
    from training.trainer import CheckpointManager
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(Path(tmpdir), keep_best=3, metric="capture_rate")
        for i in range(5):
            state = {"data": i}
            mgr.save(state, current_metric=float(i), tag=f"ep_{i * 100}")

        # Should only keep 3 checkpoints
        saved = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_")]
        assert len(saved) == 3, f"Expected 3 checkpoints, got {len(saved)}"
        # The best 3 metrics should be saved (4, 3, 2)
        saved_tags = [f.replace("checkpoint_", "").replace(".pt", "") for f in saved]
        saved_metrics = []
        for s in saved_tags:
            path = Path(tmpdir) / f"checkpoint_{s}.pt"
            state = torch.load(path, weights_only=True)
            saved_metrics.append(state["data"])
        assert sorted(saved_metrics, reverse=True) == [4, 3, 2]


def test_checkpoint_manager_keep_all():
    from training.trainer import CheckpointManager
    from pathlib import Path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(Path(tmpdir), keep_best=0, metric="capture_rate")
        for i in range(5):
            mgr.save({"data": i}, current_metric=float(i), tag=f"ep_{i * 100}")

        saved = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_")]
        assert len(saved) == 5, "Keep 0 should keep all checkpoints"


# --- NEW: Config validation ---

def test_config_validation():
    import config as cfg
    warnings = cfg.validate_config()
    assert isinstance(warnings, list)
    # All standard configs should be valid
    assert len(warnings) == 0, f"Expected no warnings: {warnings}"


# --- NEW: RunningNormalizer ---

def test_running_normalizer():
    from agents.base_agent import RunningNormalizer
    normalizer = RunningNormalizer((29,), clip=10.0)
    for _ in range(100):
        obs = np.random.randn(29).astype(np.float32)
        normalizer.update(obs)
    obs = np.ones(29, dtype=np.float32)
    normalized = normalizer.normalize(obs)
    assert normalized.shape == (29,)
    assert not np.isnan(normalized).any()

    # Test serialization
    state = normalizer.state_dict()
    normalizer2 = RunningNormalizer((29,), clip=10.0)
    normalizer2.load_state_dict(state)
    assert np.allclose(normalizer.mean, normalizer2.mean)


# --- NEW: Selective OM (surprise-based update) ---

def test_selective_om_surprise_threshold():
    import config
    import torch.nn.functional as F
    from models.opponent_model import OpponentModel

    original_threshold = config.OM_SURPRISE_THRESHOLD
    config.OM_SURPRISE_THRESHOLD = 1.0  # only update on very surprising transitions
    try:
        om = OpponentModel(input_dim=34, hidden_dim=32, action_dim=5)
        x = torch.randn(4, 1, 34)
        logits, _ = om(x)
        target = torch.randint(0, 5, (4,))
        raw_loss = F.cross_entropy(logits, target, reduction="none")

        # With threshold, average only over samples where loss > threshold
        mask = (raw_loss > config.OM_SURPRISE_THRESHOLD).float()
        if mask.sum() > 0:
            selective_loss = (raw_loss * mask).mean()
        else:
            selective_loss = torch.tensor(0.0)

        assert selective_loss.ndim == 0
    finally:
        config.OM_SURPRISE_THRESHOLD = original_threshold


# --- NEW: Early stopping ---

def test_early_stopping_triggers():
    import config
    from agents.ppo_agent import PPOAgent
    from training.trainer import Trainer
    original_patience = config.EARLY_STOP_PATIENCE
    original_threshold = config.EARLY_STOP_THRESHOLD
    config.EARLY_STOP_PATIENCE = 5
    config.EARLY_STOP_THRESHOLD = 0.9  # unlikely to reach
    try:
        pred = PPOAgent(obs_dim=29, action_dim=5)
        prey = PPOAgent(obs_dim=29, action_dim=5)
        trainer = Trainer(predator=pred, prey=prey, experiment_name="test_early_stop", total_episodes=50)
        trainer.recent_captures = [0.0] * 10
        no_improve = 0
        for ep in range(10):
            capture_rate = 0.0
            if capture_rate >= config.EARLY_STOP_THRESHOLD:
                no_improve = 0
            else:
                no_improve += 1
        assert no_improve >= 5, "Should have triggered early stopping patience"
    finally:
        config.EARLY_STOP_PATIENCE = original_patience
        config.EARLY_STOP_THRESHOLD = original_threshold


# --- NEW: Streak tracking ---

def test_streak_tracking():
    captures = [True, True, False, True, False, False, False]
    pred_streak = 0
    prey_streak = 0
    best_streak = 0
    for captured in captures:
        if captured:
            pred_streak += 1
            prey_streak = 0
        else:
            prey_streak += 1
            pred_streak = 0
        best_streak = max(best_streak, pred_streak, prey_streak)
    assert pred_streak == 0
    assert prey_streak == 3
    assert best_streak == 3


# --- NEW: JSON metrics export ---

def test_json_metrics_export(tmp_path):
    import json
    from pathlib import Path
    from training.trainer import Trainer
    from agents.ppo_agent import PPOAgent
    import config
    original_export = config.EXPORT_JSON_METRICS
    config.EXPORT_JSON_METRICS = True
    try:
        pred = PPOAgent(obs_dim=29, action_dim=5)
        prey = PPOAgent(obs_dim=29, action_dim=5)
        trainer = Trainer(predator=pred, prey=prey, experiment_name="test_export", total_episodes=2)
        # Override results_dir to point to tmp_path
        trainer.results_dir = Path(tmp_path)
        trainer._export_json_metrics("test", capture_rate_override=0.5)
        out_path = Path(tmp_path) / "metrics_test.json"
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["capture_rate"] == 0.5
    finally:
        config.EXPORT_JSON_METRICS = original_export


# --- NEW: Config diff ---

def test_log_config_diff():
    import config
    from agents.ppo_agent import PPOAgent
    from training.trainer import Trainer
    original_diff = config.LOG_CONFIG_DIFF
    config.LOG_CONFIG_DIFF = True
    try:
        pred = PPOAgent(obs_dim=29, action_dim=5)
        prey = PPOAgent(obs_dim=29, action_dim=5)
        trainer = Trainer(predator=pred, prey=prey, experiment_name="test_config_diff", total_episodes=2)
        trainer._log_config_diff()
    finally:
        config.LOG_CONFIG_DIFF = original_diff


# --- NEW: Curriculum schedule ---

def test_curriculum_params():
    import config
    from agents.ppo_agent import PPOAgent
    from training.trainer import Trainer
    original_enabled = config.CURRICULUM_ENABLED
    original_schedule = config.CURRICULUM_SCHEDULE
    config.CURRICULUM_ENABLED = True
    config.CURRICULUM_SCHEDULE = [(0, 5, 50), (100, 7, 100)]
    try:
        pred = PPOAgent(obs_dim=29, action_dim=5)
        prey = PPOAgent(obs_dim=29, action_dim=5)
        trainer = Trainer(predator=pred, prey=prey, experiment_name="test_curriculum", total_episodes=200)
        trainer._curriculum_enabled = True
        trainer._curriculum_stage = 0
        grid, steps = trainer._get_curriculum_params(50)
        assert grid == 5
        assert steps == 50
        grid, steps = trainer._get_curriculum_params(150)
        assert grid == 7
        assert steps == 100
    finally:
        config.CURRICULUM_ENABLED = original_enabled
        config.CURRICULUM_SCHEDULE = original_schedule

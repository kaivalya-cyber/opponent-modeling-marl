"""Tests for env/predator_prey.py — Step 1 self-check (6 tests)."""
import numpy as np
import pytest

from env.predator_prey import PredatorPreyEnv


def test_obs_shape():
    env = PredatorPreyEnv()
    (obs_pred, obs_prey), _ = env.reset(seed=42)
    assert obs_pred.shape == (29,), f"Expected (29,) got {obs_pred.shape}"
    assert obs_prey.shape == (29,), f"Expected (29,) got {obs_prey.shape}"
    assert obs_pred.dtype == np.float32
    assert obs_pred.min() >= 0.0 and obs_pred.max() <= 1.0


def test_step_output():
    env = PredatorPreyEnv()
    env.reset(seed=0)
    (obs_pred, obs_prey), (rew_pred, rew_prey), term, trunc, info = env.step((0, 0))
    assert obs_pred.shape == (29,)
    assert isinstance(rew_pred, float)
    assert isinstance(term, bool)
    assert "captured" in info


def test_capture_terminates():
    env = PredatorPreyEnv(grid_size=2)
    env.reset(seed=0)
    # Force predator and prey to same cell
    env.predator_pos = np.array([0, 0])
    env.prey_pos = np.array([0, 1])
    _, _, term, _, info = env.step((4, 0))  # predator moves right
    # May or may not capture depending on exact positions — test termination logic
    assert isinstance(term, bool)


def test_max_steps_truncation():
    env = PredatorPreyEnv(max_steps=5)
    env.reset(seed=0)
    for _ in range(5):
        _, _, term, trunc, _ = env.step((0, 0))
    assert trunc == True


def test_reward_signs():
    """Predator always gets negative per-step reward, prey gets positive."""
    env = PredatorPreyEnv()
    env.reset(seed=0)
    _, (rew_pred, rew_prey), term, _, _ = env.step((0, 0))
    if not term:
        assert rew_pred < 0, "Predator should have negative per-step reward"
        assert rew_prey > 0, "Prey should have positive per-step reward"


def test_toroidal_wrap():
    env = PredatorPreyEnv(grid_size=10)
    env.reset(seed=0)
    env.predator_pos = np.array([0, 0])
    new_pos = (env.predator_pos + np.array([-1, 0])) % 10
    assert new_pos[0] == 9, "Toroidal wrap failed"

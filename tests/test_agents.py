"""Tests for agents/ — Step 3 self-check (5 tests)."""
import numpy as np
import torch
import pytest


def test_ppo_agent_select_action():
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim=29, action_dim=5)
    obs = np.zeros(29, dtype=np.float32)
    action, log_prob, value = agent.select_action(obs)
    assert 0 <= action <= 4
    assert log_prob.shape == torch.Size([])
    assert value.shape == torch.Size([1])


def test_om_agent_augmented_obs_shape():
    from agents.om_agent import OMAgent
    agent = OMAgent(obs_dim=29, action_dim=5)
    agent.reset_episode()
    obs = np.zeros(29, dtype=np.float32)
    aug = agent.get_augmented_obs(obs, last_opp_action=0)
    assert aug.shape == (34,), f"Expected (34,) got {aug.shape}"


def test_om_agent_select_action():
    from agents.om_agent import OMAgent
    agent = OMAgent(obs_dim=29, action_dim=5)
    agent.reset_episode()
    obs = np.zeros(29, dtype=np.float32)
    action, log_prob, value = agent.select_action(obs, last_opp_action=2)
    assert 0 <= action <= 4


def test_hidden_state_reset():
    from agents.om_agent import OMAgent
    agent = OMAgent(obs_dim=29, action_dim=5)
    agent.reset_episode()
    h_before = agent.h.clone()
    obs = np.random.randn(29).astype(np.float32)
    agent.get_augmented_obs(obs, last_opp_action=1)
    agent.reset_episode()
    assert torch.allclose(agent.h, h_before), "Hidden state must reset to zeros"


def test_ppo_update_returns_metrics():
    from agents.ppo_agent import PPOAgent
    from training.rollout_buffer import RolloutBuffer
    agent = PPOAgent(obs_dim=29, action_dim=5)
    buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=128)
    # Fill buffer with dummy data
    for _ in range(128):
        obs = np.random.randn(29).astype(np.float32)
        buf.add(obs=obs, action=0, reward=0.0, value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.6), done=False, opp_action=0)
    buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
    metrics = agent.update(buf)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics

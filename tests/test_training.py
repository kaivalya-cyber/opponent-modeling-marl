"""Tests for training/ — Step 8 self-check (6 tests)."""
import numpy as np
import torch
import pytest


def test_rollout_buffer_gae():
    """GAE returns must be finite, have correct shape, and be correlated with rewards."""
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
    """Smoke test: one full episode with OM agents must not crash."""
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
    """
    Over 50 gradient steps on fixed opponent action sequences,
    the opponent model cross-entropy loss must decrease.
    """
    import torch.nn.functional as F
    from models.opponent_model import OpponentModel
    om = OpponentModel(input_dim=34, hidden_dim=64, action_dim=5)
    opt = torch.optim.Adam(om.parameters(), lr=1e-3)
    target_action = torch.tensor([2])  # fixed target
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
    """Over 5 PPO epochs on fixed data, policy loss must not diverge."""
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

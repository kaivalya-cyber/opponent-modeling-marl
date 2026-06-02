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
    for _ in range(128):
        obs = np.random.randn(29).astype(np.float32)
        buf.add(obs=obs, action=0, reward=0.0, value=torch.tensor([0.0]),
                log_prob=torch.tensor(-1.6), done=False, opp_action=0)
    buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
    metrics = agent.update(buf)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


# --- NEW: Agent save/load ---

def test_agent_save_and_load(tmp_path):
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim=29, action_dim=5)
    save_path = str(tmp_path / "agent.pt")
    agent.save(save_path)
    loaded = PPOAgent(obs_dim=29, action_dim=5)
    loaded.load(save_path)
    for p_orig, p_loaded in zip(agent.policy.parameters(), loaded.policy.parameters()):
        assert torch.allclose(p_orig, p_loaded), "Saved and loaded weights must match"


def test_om_agent_save_and_load(tmp_path):
    from agents.om_agent import OMAgent
    agent = OMAgent(obs_dim=29, action_dim=5)
    save_path = str(tmp_path / "om_agent.pt")
    agent.save(save_path)
    loaded = OMAgent(obs_dim=29, action_dim=5)
    loaded.load(save_path)
    for p_orig, p_loaded in zip(agent.opponent_model.parameters(), loaded.opponent_model.parameters()):
        assert torch.allclose(p_orig, p_loaded), "OM weights must match"


# --- NEW: Frame stacking ---

def test_ppo_frame_stacking():
    import config
    original_frame_stack = config.FRAME_STACK
    config.FRAME_STACK = 3
    try:
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=29, action_dim=5)
        assert agent._frame_stack_size == 3
        assert agent._effective_obs_dim == 87
        agent.reset_episode()
        assert len(agent._frame_stack) == 3
    finally:
        config.FRAME_STACK = original_frame_stack


# --- NEW: Observation normalization ---

def test_obs_normalization():
    import config
    original = config.OBS_NORMALIZE
    config.OBS_NORMALIZE = True
    try:
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=29, action_dim=5)
        assert agent.obs_normalizer is not None
        obs = np.ones(29, dtype=np.float32)
        normalized = agent.normalize_obs(obs)
        assert normalized.shape == (29,)
    finally:
        config.OBS_NORMALIZE = original


# --- NEW: Parameter noise ---

def test_parameter_noise():
    import config
    original = config.PARAM_NOISE_ENABLED
    config.PARAM_NOISE_ENABLED = True
    config.PARAM_NOISE_STD = 0.1
    try:
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=29, action_dim=5)
        params_before = [p.data.clone() for p in agent.policy.parameters()]
        agent._apply_param_noise()
        params_during = [p.data.clone() for p in agent.policy.parameters()]
        agent._remove_param_noise()
        params_after = [p.data.clone() for p in agent.policy.parameters()]
        any_diff = any(not torch.allclose(b, d) for b, d in zip(params_before, params_during))
        assert any_diff, "Parameters should change with noise applied"
        for b, a in zip(params_before, params_after):
            assert torch.allclose(b, a), "Parameters should be restored after noise removal"
    finally:
        config.PARAM_NOISE_ENABLED = original


# --- NEW: Gradient accumulation ---

def test_gradient_accumulation():
    from agents.ppo_agent import PPOAgent
    from training.rollout_buffer import RolloutBuffer
    import config
    original_accum = config.GRAD_ACCUMULATION_STEPS
    config.GRAD_ACCUMULATION_STEPS = 2
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        assert agent._grad_accum_steps == 2
        buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=64)
        for _ in range(64):
            obs = np.random.randn(29).astype(np.float32)
            buf.add(obs=obs, action=0, reward=0.0, value=torch.tensor([0.0]),
                    log_prob=torch.tensor(-1.6), done=False, opp_action=0)
        buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
        metrics = agent.update(buf)
        assert not np.isnan(metrics["policy_loss"]), "Policy loss is NaN with gradient accumulation"
    finally:
        config.GRAD_ACCUMULATION_STEPS = original_accum


# --- NEW: Separate optimizers with weight decay ---

def test_separate_optimizers_weight_decay():
    import config
    original_sep = config.USE_SEPARATE_OPTIMIZERS
    original_pwd = config.POLICY_WEIGHT_DECAY
    original_vwd = config.VALUE_WEIGHT_DECAY
    config.USE_SEPARATE_OPTIMIZERS = True
    config.POLICY_WEIGHT_DECAY = 1e-4
    config.VALUE_WEIGHT_DECAY = 1e-3
    try:
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=29, action_dim=5)
        assert hasattr(agent, "policy_optimizer")
        assert hasattr(agent, "value_optimizer")
        for group in agent.policy_optimizer.param_groups:
            assert group["weight_decay"] == 1e-4
        for group in agent.value_optimizer.param_groups:
            assert group["weight_decay"] == 1e-3
    finally:
        config.USE_SEPARATE_OPTIMIZERS = original_sep
        config.POLICY_WEIGHT_DECAY = original_pwd
        config.VALUE_WEIGHT_DECAY = original_vwd


# --- NEW: Model EMA ---

def test_model_ema():
    import config
    from agents.ppo_agent import PPOAgent
    original_ema = config.MODEL_EMA_ENABLED
    config.MODEL_EMA_ENABLED = True
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        assert agent._ema_policy is not None
        for p, ema_p in zip(agent.policy.parameters(), agent._ema_policy.parameters()):
            assert torch.allclose(p, ema_p), "EMA params should match initially"
        agent.update_ema()
    finally:
        config.MODEL_EMA_ENABLED = original_ema


# --- NEW: Action noise ---

def test_action_noise():
    import config
    original_noise = config.ACTION_NOISE_STD
    config.ACTION_NOISE_STD = 1.0
    try:
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=29, action_dim=5)
        obs = np.zeros(29, dtype=np.float32)
        actions = []
        for _ in range(20):
            a, _, _ = agent.select_action(obs)
            actions.append(a)
        assert len(set(actions)) > 1, "Action noise should diversify actions"
    finally:
        config.ACTION_NOISE_STD = original_noise


# --- NEW: Episode stats tracking ---

def test_episode_stats():
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim=29, action_dim=5)
    agent.record_episode(reward=10.0, length=50)
    agent.record_episode(reward=20.0, length=100)
    stats = agent.get_episode_stats()
    assert stats["avg_ep_reward"] == 15.0
    assert stats["avg_ep_length"] == 75.0
    assert stats["best_ep_reward"] == 20.0


# --- NEW: Approx KL and explained variance ---

def test_ppo_update_returns_extended_metrics():
    from agents.ppo_agent import PPOAgent
    from training.rollout_buffer import RolloutBuffer
    import config
    original_kl = config.KL_TRACKING
    original_ev = config.EXPLAINED_VAR_TRACKING
    config.KL_TRACKING = True
    config.EXPLAINED_VAR_TRACKING = True
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=128)
        for _ in range(128):
            obs = np.random.randn(29).astype(np.float32)
            buf.add(obs=obs, action=0, reward=0.0, value=torch.tensor([0.0]),
                    log_prob=torch.tensor(-1.6), done=False, opp_action=0)
        buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
        metrics = agent.update(buf)
        if config.KL_TRACKING:
            assert "approx_kl" in metrics
        if config.EXPLAINED_VAR_TRACKING:
            assert "explained_var" in metrics or "explained_variance" in metrics
    finally:
        config.KL_TRACKING = original_kl
        config.EXPLAINED_VAR_TRACKING = original_ev


# --- NEW: Value residual clip ---

def test_value_residual_clip():
    import config
    from agents.ppo_agent import PPOAgent
    from training.rollout_buffer import RolloutBuffer
    original_clip = config.VALUE_RESIDUAL_CLIP
    config.VALUE_RESIDUAL_CLIP = 1.0
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        buf = RolloutBuffer(obs_dim=29, action_dim=5, capacity=64)
        for _ in range(64):
            obs = np.random.randn(29).astype(np.float32)
            buf.add(obs=obs, action=0, reward=0.0, value=torch.tensor([0.0]),
                    log_prob=torch.tensor(-1.6), done=False, opp_action=0)
        buf.compute_returns(last_value=torch.tensor([0.0]), gamma=0.99, lam=0.95)
        metrics = agent.update(buf)
        assert not np.isnan(metrics["value_loss"]), "Value loss NaN with residual clip"
    finally:
        config.VALUE_RESIDUAL_CLIP = original_clip


# --- NEW: Reward scaling ---

def test_reward_scale():
    import config
    from agents.ppo_agent import PPOAgent
    original_reward_norm = config.REWARD_NORMALIZE
    original_scale = config.REWARD_SCALE
    config.REWARD_NORMALIZE = False
    config.REWARD_SCALE = 0.1
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        scaled = agent.normalize_reward(10.0)
        assert scaled == 1.0, f"Expected 1.0, got {scaled}"
    finally:
        config.REWARD_NORMALIZE = original_reward_norm
        config.REWARD_SCALE = original_scale


# --- NEW: Softmax temperature ---

def test_softmax_temperature():
    import config
    import torch
    from agents.ppo_agent import PPOAgent
    original_temp = config.SOFTMAX_TEMPERATURE
    config.SOFTMAX_TEMPERATURE = 10.0
    try:
        agent = PPOAgent(obs_dim=29, action_dim=5)
        device = next(agent.policy.parameters()).device
        obs = torch.zeros(1, 29, device=device)
        logits = agent.policy(obs)
        assert logits is not None
    finally:
        config.SOFTMAX_TEMPERATURE = original_temp

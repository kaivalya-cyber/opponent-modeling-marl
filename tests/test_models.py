"""Tests for models/ — Step 2 self-check (6 tests)."""
import torch
import pytest


def test_policy_net_output_shape():
    from models.policy_net import PolicyNet
    net = PolicyNet(input_dim=29, action_dim=5)
    x = torch.zeros(32, 29)
    logits = net(x)
    assert logits.shape == (32, 5), f"Expected (32,5) got {logits.shape}"


def test_value_net_output_shape():
    from models.value_net import ValueNet
    net = ValueNet(input_dim=29)
    x = torch.zeros(32, 29)
    v = net(x)
    assert v.shape == (32, 1), f"Expected (32,1) got {v.shape}"


def test_opponent_model_output_shape():
    from models.opponent_model import OpponentModel
    om = OpponentModel(input_dim=34, hidden_dim=64, action_dim=5, num_layers=2)
    x = torch.zeros(4, 10, 34)
    logits, h = om(x)
    assert logits.shape == (4, 5), f"Expected (4,5) got {logits.shape}"
    assert h.shape == (2, 4, 64), f"Expected (2,4,64) got {h.shape}"


def test_opponent_model_hidden_state_changes():
    from models.opponent_model import OpponentModel
    om = OpponentModel()
    x = torch.randn(1, 5, 34)
    _, h1 = om(x)
    x2 = torch.randn(1, 5, 34)
    _, h2 = om(x2, h1)
    assert not torch.allclose(h1, h2), "Hidden state must change across steps"


def test_policy_net_no_nan():
    from models.policy_net import PolicyNet
    net = PolicyNet(input_dim=29, action_dim=5)
    x = torch.randn(16, 29)
    logits = net(x)
    assert not torch.isnan(logits).any(), "NaN in policy logits"


def test_mps_or_cpu_compatibility():
    import torch
    from models.policy_net import PolicyNet
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    net = PolicyNet(input_dim=29, action_dim=5).to(device)
    x = torch.zeros(4, 29).to(device)
    logits = net(x)
    assert logits.device.type == device.type


# --- NEW: Ensemble Opponent Model tests ---

def test_ensemble_opponent_model_output():
    from models.opponent_model import EnsembleOpponentModel
    ensemble = EnsembleOpponentModel(input_dim=34, hidden_dim=32, action_dim=5,
                                     model_type="gru", ensemble_size=3, vote="mean")
    x = torch.zeros(2, 5, 34)
    logits, h = ensemble(x)
    assert logits.shape == (2, 5), f"Expected (2,5) got {logits.shape}"
    assert h.shape == (2, 2, 32), "Hidden state from ensemble mean"


def test_ensemble_opponent_model_predict():
    from models.opponent_model import EnsembleOpponentModel
    ensemble = EnsembleOpponentModel(input_dim=34, hidden_dim=32, action_dim=5,
                                     model_type="gru", ensemble_size=3, vote="mean")
    x = torch.zeros(1, 1, 34)
    probs, h, confidence = ensemble.predict_opponent_action(x)
    assert probs.shape == (1, 5)
    assert confidence is not None
    assert 0 <= confidence.item() <= 1.0


def test_ensemble_majority_vote():
    from models.opponent_model import EnsembleOpponentModel
    ensemble = EnsembleOpponentModel(input_dim=34, hidden_dim=32, action_dim=5,
                                     model_type="gru", ensemble_size=3, vote="majority")
    x = torch.zeros(1, 1, 34)
    probs, h, confidence = ensemble.predict_opponent_action(x)
    assert probs.shape == (1, 5)
    assert probs.sum().item() == 1.0


def test_soft_update_target():
    from models.opponent_model import soft_update_target, OpponentModel
    source = OpponentModel(input_dim=34, hidden_dim=32, action_dim=5)
    target = OpponentModel(input_dim=34, hidden_dim=32, action_dim=5)
    # Initialize target with zeros
    for p in target.parameters():
        p.data.zero_()
    soft_update_target(target, source, tau=1.0)
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        assert torch.allclose(p_t, p_s), "Full copy with tau=1.0"


# --- NEW: ICM tests ---

def test_icm_output_shapes():
    from models.icm import ICM
    icm = ICM(obs_dim=29, action_dim=5)
    obs = torch.randn(16, 29)
    next_obs = torch.randn(16, 29)
    actions = torch.randint(0, 5, (16,))
    fwd_loss, inv_loss, intrinsic = icm(obs, next_obs, actions)
    assert fwd_loss.shape == torch.Size([])
    assert inv_loss.shape == torch.Size([])
    assert intrinsic.shape == (16,)
    assert intrinsic.min() >= 0, "Intrinsic reward should be non-negative"

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
    x = torch.zeros(4, 10, 34)   # batch=4, seq_len=10
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

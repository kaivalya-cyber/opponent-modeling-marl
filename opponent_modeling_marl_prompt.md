# Opponent Modeling for Multi-Agent Reinforcement Learning
## Complete Implementation Guide — Kaivalya Singh

---

## Project Overview

You are building a multi-agent reinforcement learning system where each agent learns an **explicit model of its opponent's policy** and uses that model to adapt its own strategy in real time. This directly addresses the non-stationarity problem in competitive MARL: standard algorithms like PPO or MAPPO treat other agents as part of the environment, which violates the stationarity assumption and causes policy collapse. Opponent modeling makes the non-stationarity explicit and learnable.

**Research claim (falsifiable):** Agents equipped with learned opponent models will achieve significantly higher Elo ratings and greater robustness to opponent policy shifts than baseline MAPPO agents, in a 1v1 competitive grid environment, as measured across training.

**Target environment:** MacBook Air M4, 24GB RAM, Apple Silicon (MPS backend). No CUDA. All code must be MPS-compatible with CPU fallback.

---

## Repository Structure

Build the project with exactly this layout. Do not deviate.

```
opponent-modeling-marl/
├── README.md
├── requirements.txt
├── config.py
├── env/
│   ├── __init__.py
│   └── predator_prey.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── ppo_agent.py
│   └── om_agent.py
├── models/
│   ├── __init__.py
│   ├── policy_net.py
│   ├── value_net.py
│   └── opponent_model.py
├── training/
│   ├── __init__.py
│   ├── rollout_buffer.py
│   ├── trainer.py
│   └── elo.py
├── experiments/
│   ├── run_baseline.py
│   └── run_om.py
├── tests/
│   ├── test_env.py
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_training.py
└── results/
    └── .gitkeep
```

---

## Step 1 — Environment: `env/predator_prey.py`

### What to build

A 2-agent, 1v1 grid-world predator-prey game. One agent is the **predator**, one is the **prey**. The predator wins by occupying the same cell as the prey within the episode time limit. The prey wins by surviving.

This environment is chosen because:
- The optimal prey strategy depends on the predator's policy (escape direction)
- The optimal predator strategy depends on the prey's policy (interception angle)
- Both agents must model the other to play well
- It is fast to simulate and produces clear, interpretable behaviors

### Exact specification

```python
"""
File: env/predator_prey.py

Grid: H x W cells (default 10x10), toroidal (wrap-around edges).
Agents: 2. Agent 0 = predator, Agent 1 = prey.
Observation: Each agent sees a local 5x5 window centered on itself,
             flattened to 25 cells, plus its own (x,y) and opponent (x,y)
             normalized to [0,1]. Total obs dim = 25 + 4 = 29.
Actions: Discrete(5) — stay, up, down, left, right.
Reward:
  - Predator: +1.0 on capture, -0.01 per step (time penalty)
  - Prey: +0.01 per step survived, -1.0 on capture
Episode ends: capture OR max_steps (default 200).
Reset: agents placed at random non-overlapping positions.
"""
```

### Implementation requirements

- Inherit from `gymnasium.Env`
- `observation_space`: `Box(low=0, high=1, shape=(29,), dtype=np.float32)`
- `action_space`: `Discrete(5)` for each agent
- Implement `reset(seed=None)` returning `(obs_predator, obs_prey), info`
- Implement `step(actions: tuple[int, int])` returning `(obs_pred, obs_prey), (rew_pred, rew_prey), terminated, truncated, info`
- `info` dict must include `{"captured": bool, "steps": int, "predator_pos": (x,y), "prey_pos": (x,y)}`
- Add `render(mode="human")` using ASCII grid print for debugging
- Grid cells: 0=empty, 1=predator, 2=prey
- Toroidal wrap: `new_pos = (pos + delta) % grid_size`
- Local observation window: pad grid with zeros where window exceeds boundary

### Self-check — environment

After implementing, run these assertions in `tests/test_env.py`:

```python
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
```

All 6 tests must pass before proceeding.

---

## Step 2 — Neural Network Models: `models/`

### `models/policy_net.py`

```python
"""
MLP policy network shared by all agents.
Input: obs_dim (29) + optional opponent_model_output_dim (if OM agent)
Output: logits over action_dim (5)

Architecture:
  Linear(input_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, action_dim)

Must work on MPS (Apple Silicon) and CPU. No CUDA-specific code.
Use torch.nn.Module.
Implement forward(x) -> logits (NOT softmax — let distributions handle that).
"""
```

### `models/value_net.py`

```python
"""
MLP value network (critic).
Input: obs_dim (29)
Output: scalar value estimate shape (batch, 1)

Architecture: same as policy_net but output dim = 1.
"""
```

### `models/opponent_model.py`

```python
"""
THE KEY NOVELTY. This network learns to predict the opponent's next action
given the opponent's observation history.

Architecture: GRU-based sequence model.
  - Input at each step: opponent's last action (one-hot, dim=5)
                        + current joint observation (29 dims)
                        = input_dim 34
  - GRU hidden dim: 64
  - Output: Linear(64, 5) -> logits over opponent's next action

Why GRU: The opponent's policy may be non-stationary (they adapt too).
GRU maintains a hidden state that captures recent behavior patterns,
not just the last action. This is what separates this from naive
"last-action conditioning."

Implement:
  __init__(self, input_dim=34, hidden_dim=64, action_dim=5)
  forward(self, x_seq, h=None) -> (logits, new_h)
    x_seq shape: (batch, seq_len, input_dim)
    h shape: (1, batch, hidden_dim) or None for zero init
    Returns: logits shape (batch, action_dim), new_h
  predict_opponent_action(self, x_seq, h=None) -> (action_probs, new_h)
    Returns softmax of logits.

The hidden state h is carried across timesteps during rollout and
reset at episode start. This is critical — do not reset h mid-episode.
"""
```

### Self-check — models

```python
# tests/test_models.py

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
    om = OpponentModel(input_dim=34, hidden_dim=64, action_dim=5)
    x = torch.zeros(4, 10, 34)   # batch=4, seq_len=10
    logits, h = om(x)
    assert logits.shape == (4, 5), f"Expected (4,5) got {logits.shape}"
    assert h.shape == (1, 4, 64), f"Expected (1,4,64) got {h.shape}"

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
```

---

## Step 3 — Agents: `agents/`

### `agents/base_agent.py`

Abstract base class. Implement:
- `select_action(obs) -> (action, log_prob, value)` — returns int action, log prob tensor, value tensor
- `update(rollout_buffer) -> dict` — returns dict of loss metrics
- `reset_episode()` — called at episode start to reset any recurrent state

### `agents/ppo_agent.py` — Baseline

Standard PPO agent. No opponent modeling. Uses `PolicyNet` and `ValueNet`.

**PPO hyperparameters (do not tune away from these without logging the change):**
```python
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
N_EPOCHS = 4
MINIBATCH_SIZE = 64
```

Implement standard GAE advantage estimation. Use `torch.distributions.Categorical` for action sampling. Clip value loss. Add entropy bonus.

### `agents/om_agent.py` — Opponent Modeling Agent

This is the core contribution. Extends PPO with an opponent model.

**Architecture:**
1. At each step, agent receives `own_obs` (29 dims)
2. Opponent model takes `(last_opponent_action_onehot + own_obs)` as input, produces `opponent_action_probs` (5 dims) and updates hidden state
3. Policy input = `concat(own_obs, opponent_action_probs)` = 34 dims
4. Policy and value nets take this augmented input
5. Opponent model is trained with cross-entropy loss against the *actual* opponent action observed after each step

**Critical implementation details:**
- GRU hidden state `h` is a class attribute, reset in `reset_episode()`
- Hidden state must be detached from graph before each step: `h = h.detach()`
- Opponent model loss is computed separately from PPO loss and added with weight `OM_LOSS_WEIGHT = 0.5`
- The opponent model is updated every PPO update step using the rollout's actual opponent action sequence
- One-hot encode opponent actions: `F.one_hot(torch.tensor(action), num_classes=5).float()`

**Augmented observation construction:**
```python
def get_augmented_obs(self, own_obs, last_opp_action):
    opp_action_onehot = F.one_hot(torch.tensor(last_opp_action), 5).float()
    gru_input = torch.cat([opp_action_onehot, torch.tensor(own_obs)]).unsqueeze(0).unsqueeze(0)
    opp_probs, self.h = self.opponent_model(gru_input, self.h)
    opp_probs = opp_probs.squeeze(0).detach()
    augmented = torch.cat([torch.tensor(own_obs), opp_probs])
    return augmented  # shape (34,)
```

### Self-check — agents

```python
# tests/test_agents.py

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
```

---

## Step 4 — Rollout Buffer: `training/rollout_buffer.py`

Store transitions for one PPO update. Must store:
- `obs`: np.array shape `(T, obs_dim)`
- `actions`: np.array shape `(T,)` int
- `rewards`: np.array shape `(T,)`
- `values`: np.array shape `(T,)`
- `log_probs`: np.array shape `(T,)`
- `dones`: np.array shape `(T,)` bool
- `opp_actions`: np.array shape `(T,)` int — actual opponent actions (for OM training)
- `advantages`: computed post-rollout via GAE
- `returns`: computed post-rollout

Implement `add(...)`, `compute_returns(last_value, gamma, lam)`, `get_batches(batch_size)`.

`get_batches` must shuffle indices and yield minibatches as dicts of tensors on the correct device.

---

## Step 5 — Elo Rating: `training/elo.py`

Implement Elo rating to measure agent quality across training. This gives a principled comparison between baseline and OM agents even if the environment reward is sparse.

```python
class EloRating:
    def __init__(self, k=32, initial_rating=1000):
        self.ratings = {}  # agent_name -> float
        self.k = k
        self.initial_rating = initial_rating

    def get_rating(self, name):
        return self.ratings.get(name, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_name, loser_name):
        """Call after each episode. Winner is the agent that won."""
        ra = self.get_rating(winner_name)
        rb = self.get_rating(loser_name)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        self.ratings[winner_name] = ra + self.k * (1 - ea)
        self.ratings[loser_name] = rb + self.k * (0 - eb)

    def draw(self, name_a, name_b):
        """Call for draw (prey survives full episode = draw)."""
        ra = self.get_rating(name_a)
        rb = self.get_rating(name_b)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        self.ratings[name_a] = ra + self.k * (0.5 - ea)
        self.ratings[name_b] = rb + self.k * (0.5 - eb)
```

---

## Step 6 — Trainer: `training/trainer.py`

The trainer runs the self-play loop. For fair comparison, both baseline and OM experiments must use identical training loops.

```python
"""
Training loop:

1. Initialize env, agent_predator, agent_prey (same class for self-play)
2. For each episode:
   a. reset env, reset agent hidden states
   b. Collect T=2048 steps across episodes (not per episode)
   c. At each step:
      - Both agents select actions given their obs
      - OM agent additionally takes last_opp_action as input
      - Step env, get next obs and rewards
      - Store transitions in respective rollout buffers
   d. When buffer full: run PPO update for both agents
   e. Log: episode rewards, capture rate, Elo ratings
3. Every 50 episodes: run 20 evaluation episodes against a fixed random opponent.
   Record win rate and Elo delta.
4. Save checkpoints every 500 episodes.

Critical: During training, agents play against themselves (self-play).
During evaluation, they play against a frozen random policy.
This isolates the effect of opponent modeling.
"""
```

### Logging requirements

Use Python's built-in `logging` module plus a simple CSV writer. After every update, log:
- `episode`, `predator_reward`, `prey_reward`, `capture_rate` (over last 20 eps)
- `predator_elo`, `prey_elo`
- `policy_loss`, `value_loss`, `entropy`, `om_loss` (OM agent only)

Write to `results/{experiment_name}/metrics.csv`.

---

## Step 7 — Experiments

### `experiments/run_baseline.py`

```python
"""
Trains two standard PPO agents against each other via self-play.
Experiment name: "baseline_ppo"
Total episodes: 2000
Logs to: results/baseline_ppo/metrics.csv
Saves checkpoint: results/baseline_ppo/checkpoint_final.pt

Usage: python -m experiments.run_baseline
"""
```

### `experiments/run_om.py`

```python
"""
Trains two OM agents against each other via self-play.
Experiment name: "om_agent"
Identical hyperparameters to baseline.
Logs to: results/om_agent/metrics.csv
Saves checkpoint: results/om_agent/checkpoint_final.pt

Usage: python -m experiments.run_om
"""
```

---

## Step 8 — Full Test Suite: `tests/`

Run all tests with `pytest tests/ -v`. Every test must pass before training begins.

### `tests/test_training.py`

```python
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
```

---

## Step 9 — Results Analysis

After both experiments complete, produce `results/analysis.py` that:

1. Loads `results/baseline_ppo/metrics.csv` and `results/om_agent/metrics.csv`
2. Plots side-by-side: (a) capture rate over episodes, (b) Elo over episodes, (c) opponent model loss over time (OM only)
3. Prints a summary table:

```
Metric                  | Baseline PPO | OM Agent
------------------------|--------------|----------
Final Elo (predator)    | XXXX         | XXXX
Capture rate (last 100) | XX%          | XX%
Episodes to 60% cap.    | XXXX         | XXXX
```

4. Runs a Mann-Whitney U test on the last 100 episode capture rates of both methods. Print p-value. If p < 0.05, report "OM agent significantly outperforms baseline."

Use `matplotlib` and `scipy.stats.mannwhitneyu` for this.

---

## `requirements.txt`

```
gymnasium>=0.29.0
torch>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
scipy>=1.12.0
pytest>=8.0.0
```

---

## `config.py`

All hyperparameters live here. Nothing hardcoded elsewhere.

```python
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

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
N_EPOCHS = 4
MINIBATCH_SIZE = 64

# Opponent Model
OM_LOSS_WEIGHT = 0.5
OM_HIDDEN_DIM = 64
OM_INPUT_DIM = 34  # 5 (opp action onehot) + 29 (own obs)

# Device
import torch
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

---

## Anti-Hallucination Constraints

The following are hard rules for the coding agent. Do not violate them.

1. **No imports that are not in `requirements.txt`.** If you want a library not listed, add it to requirements.txt first and state why.

2. **No placeholder functions.** Every function must be fully implemented. `pass`, `...`, `raise NotImplementedError` are not permitted in submitted code.

3. **Test before moving on.** After implementing each numbered step, run `pytest tests/test_<component>.py -v` and confirm all tests pass before implementing the next step.

4. **MPS compatibility.** All tensors must go through `config.DEVICE`. Never hardcode `.cuda()` or `.cpu()`. Use `.to(DEVICE)` everywhere.

5. **No hardcoded paths.** All file paths must be constructed with `pathlib.Path` relative to the project root.

6. **Hidden state discipline.** The GRU hidden state in `OMAgent` must be: (a) initialized to zeros in `__init__`, (b) reset to zeros in `reset_episode()`, (c) detached from computation graph after every forward pass with `.detach()`.

7. **Opponent action availability.** During training, the actual opponent action from the previous step is always available from the environment step output. Do not try to predict it before the step — use it as ground truth for OM training and as input for the *next* step's augmented observation.

8. **No silent failures.** Add `assert` statements for tensor shapes at every major forward pass boundary. Shape mismatches must crash loudly, not silently produce wrong results.

9. **The opponent model is trained on actual opponent actions, not predicted ones.** The cross-entropy target is `actual_opp_action`, not `argmax(predicted_opp_probs)`. Confusing these is the most common implementation error in this architecture.

10. **Self-play integrity.** Both agents in self-play must be updated independently with their own rollout buffers. Do not share buffers between predator and prey.

---

## What Success Looks Like

By episode 2000:
- **OM agent capture rate** should reach 60–75% (predator winning)
- **Baseline PPO capture rate** should plateau at 40–55%
- **OM Elo** should exceed baseline Elo by 50–150 points
- **Opponent model loss** should decrease monotonically over first 500 episodes then stabilize
- All 18 pytest tests pass from day one and continue passing

If the OM agent does *not* outperform baseline, that is a valid research result — report it honestly and investigate whether the opponent model's predictions are actually being used by the policy (check: does removing the OM input degrade performance? If not, the policy ignored it).

---

## Resume / Application Framing

**One-line description:** "Trained competitive agents with learned models of opponent behavior, solving the non-stationarity problem in MARL using GRU-based policy prediction."

**What to highlight:**
- Identified a specific failure mode (non-stationarity / policy collapse) from prior work
- Designed and implemented a novel architectural solution (recurrent opponent model)
- Ran controlled ablation (baseline vs. OM) with statistical significance testing
- Measured results with principled metric (Elo rating system)

**GitHub repo name:** `opponent-modeling-marl`

---

*Guide authored for Kaivalya Singh — Class of 2026, EVHS. Built for MacBook Air M4 (MPS backend). Estimated build time: 2–3 weeks of focused work.*

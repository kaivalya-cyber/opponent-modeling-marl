# opponent-modeling-marl

Multi-agent reinforcement learning with opponent modeling: PPO agents with GRU/Transformer opponent models, curriculum learning, past-self evaluation, and comprehensive analysis tools.

## Project Structure

```
opponent-modeling-marl/
├── agents/                  # Agent implementations
│   ├── base_agent.py        #   Abstract base class + deterministic action selection
│   ├── ppo_agent.py         #   Standard PPO agent
│   └── om_agent.py          #   PPO + opponent model agent
├── env/
│   └── predator_prey.py     #   2D toroidal grid environment
├── experiments/             # Experiment scripts
│   ├── run_baseline.py      #   Run baseline PPO experiment
│   ├── run_om.py            #   Run OM agent experiment
│   ├── run_all.py           #   Run all experiments + comparison report
│   ├── sweep.py             #   Hyperparameter sweep runner
│   └── tournament.py        #   Head-to-head tournament
├── models/                  # Neural network architectures
│   ├── policy_net.py        #   Actor network (discrete actions)
│   ├── value_net.py         #   Critic network
│   └── opponent_model.py    #   GRU + Transformer opponent predictors
├── training/                # Training infrastructure
│   ├── trainer.py           #   Self-play training loop + evaluation
│   ├── rollout_buffer.py    #   GAE advantage computation
│   └── elo.py               #   Elo rating system
├── tests/                   # Test suite (38 tests)
│   ├── test_agents.py
│   ├── test_env.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_integration.py
├── results/                 # Experiment outputs (metrics, checkpoints, analysis)
│   └── analysis.py          #   Visualization & statistical analysis
├── config.py                # All hyperparameters
├── sweep.yaml               # Wandb hyperparameter sweep configuration
├── Dockerfile               # CUDA-enabled Docker image
├── docker-compose.yml       # Multi-service Docker orchestration
├── .pre-commit-config.yaml  # Pre-commit hooks (ruff, pytest)
├── .github/workflows/ci.yml # CI/CD pipeline (tests + Docker build)
└── requirements.txt         # Python dependencies
```

## Quick Start

### Local Setup

```bash
# Clone and install
git clone https://github.com/kaivalya-cyber/opponent-modeling-marl.git
cd opponent-modeling-marl
pip install -r requirements.txt

# Run baseline PPO experiment
python -m experiments.run_baseline

# Run OM agent experiment
python -m experiments.run_om

# Run all experiments + comparison report
python -m experiments.run_all

# Run head-to-head tournament
python -m experiments.tournament

# Analyze results
python results/analysis.py

# Run tests
python -m pytest tests/ -v
```

### Docker Setup

```bash
# Build the image
docker compose build

# Run baseline experiment
docker compose run --rm baseline

# Run OM agent experiment
docker compose run --rm om-agent

# Run all experiments
docker compose run --rm run-all

# Run tests (no GPU required)
docker compose run --rm test

# Run head-to-head tournament
docker compose run --rm tournament

# Run linting
docker compose run --rm lint

# Interactive shell
docker compose run --rm shell

# With GPU (requires nvidia-container-toolkit)
docker compose run --rm --gpus all baseline
```

### Pre-commit Hooks

```bash
# Install hooks (run once)
pre-commit install

# Run on all files
pre-commit run --all-files

# Skip hooks for a commit
git commit --no-verify -m "..."
```

## Experiment Configurations

| Experiment | Description | Command |
|---|---|---|
| `baseline_ppo` | Standard PPO self-play | `python -m experiments.run_baseline` |
| `om_agent` | PPO + GRU opponent model | `python -m experiments.run_om` |
| `om_curriculum` | PPO + OM + progressive grid sizes | `python -m experiments.run_all` |

## Key Features

### Opponent Modeling
The OM agent maintains a recurrent model (GRU or Transformer) that predicts the opponent's next action. This prediction is concatenated with observations as additional context for the policy, enabling the agent to adapt to opponent behavior.

### Curriculum Learning
Optional progressive difficulty: grid starts at 5×5, advances to 7×7 at episode 500, then 10×10 at episode 1000. Agents learn on easier problems first before tackling the full environment.

### Past-Self Evaluation
During training, the agent periodically plays against frozen snapshots of its past self. A rising past-self win rate indicates genuine improvement — the agent isn't just cycling through strategies that a random opponent fails against.

### Gradient Flow
The opponent model receives gradients from both its cross-entropy prediction loss and the policy's success signal, learning what's useful for both prediction and decision-making.

### Experiment Tracking
Each run saves `run_metadata.json` containing the git commit hash, full config snapshot, hardware info, and model parameter counts for full reproducibility.

## Results Interpretation

Training metrics are logged to `results/<experiment>/metrics.csv` and evaluation metrics to `eval_metrics.csv`.

| Metric | Meaning |
|---|---|
| `capture_rate` | Rolling average (20 episodes) of predator captures |
| `predator_elo` / `prey_elo` | Elo ratings from self-play outcomes |
| `policy_loss` / `value_loss` | PPO clip loss and value function MSE |
| `om_loss` | Cross-entropy loss for opponent action prediction |
| `win_rate_vs_random` | Deterministic evaluation against random opponent |
| `win_rate_vs_past_self` | Evaluation against frozen historical snapshot |

Run `python results/analysis.py` after training to generate publication-quality visualizations:
- `comparison_plots.png` — 6-panel comparison across experiments
- `loss_curves.png` — Policy, value, and entropy loss curves
- `head_to_head.png` — Tournament matchup bar chart

## Hyperparameter Sweeps

```bash
# Initialize sweep (creates sweep on wandb)
wandb sweep sweep.yaml

# Run sweep agents (in parallel)
wandb agent <sweep_id>
```

Or run locally without wandb:
```bash
python -m experiments.sweep
```

## CI/CD

On every push and PR, GitHub Actions:
- Runs all 38 tests across Python 3.10–3.12
- Lints with flake8 and ruff
- Builds and pushes Docker images to GitHub Container Registry on version tags (`v*`)

## License

MIT

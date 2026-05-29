# opponent-modeling-marl

[![CI](https://github.com/kaivalya-cyber/opponent-modeling-marl/actions/workflows/ci.yml/badge.svg)](https://github.com/kaivalya-cyber/opponent-modeling-marl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/kaivalya-cyber/opponent-modeling-marl/pkgs/container/opponent-modeling-marl)

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

## Architecture

```
                         ┌────────────────────────────────────────────┐
                         │               Trainer (Self-Play)           │
                         │  ┌──────────────────────────────────────┐  │
                         │  │         Predator-Prey Environment     │  │
                         │  │         (2D toroidal grid, 10×10)     │  │
                         │  └──────┬──────────────────┬────────────┘  │
                         │         │                  │                │
                         │    ┌────▼─────┐      ┌────▼─────┐         │
                         │    │ Predator  │      │   Prey    │         │
                         │    │  Agent    │      │   Agent   │         │
                         │    └────┬─────┘      └────┬─────┘         │
                         │         │                  │                │
                         │    ┌────▼──────────────────▼─────┐         │
                         │    │       Rollout Buffer          │         │
                         │    │    (GAE advantage comp.)      │         │
                         │    └─────────────┬────────────────┘         │
                         │                  │                           │
                         │    ┌─────────────▼────────────────┐         │
                         │    │         PPO Update             │         │
                         │    │  ┌──────────┐ ┌────────────┐  │         │
                         │    │  │Policy Net│ │ Value Net  │  │         │
                         │    │  └──────────┘ └────────────┘  │         │
                         │    └───────────────────────────────┘         │
                         │                                              │
                         │    ┌───────────────────────────────┐         │
                         │    │   Opponent Model (OM Agent)    │         │
                         │    │  ┌────────────┐ ┌──────────┐  │         │
                         │    │  │   GRU/     │  │ Opponent  │  │         │
                         │    │  │ Transformer│  │  Action   │  │         │
                         │    │  └────────────┘ │ Prediction │  │         │
                         │    └─────────────────┴───────────┘  │         │
                         │                                              │
                         │  ┌────Eval────┐  ┌───────┐  ┌───────────┐   │
                         │  │ vs Random  │  │ vs     │  │ Checkpoint│   │
                         │  │ Opponent   │  │Past-Self│  │  Saving   │   │
                         │  └────────────┘  └───────┘  └───────────┘   │
                         └────────────────────────────────────────────┘

                         ┌────────────────────────────────────────────┐
                         │              Results Pipeline               │
                         │                                            │
                         │  ┌────────────┐ ┌──────────┐ ┌──────────┐ │
                         │  │ metrics.csv │ │eval csv  │ │checkpoints│ │
                         │  └─────┬──────┘ └────┬─────┘ └────┬─────┘ │
                         │        │             │            │        │
                         │        ▼             ▼            ▼        │
                         │  ┌─────────────────────────────────────┐   │
                         │  │          analysis.py                  │   │
                         │  │  • comparison_plots.png (6-panel)     │   │
                         │  │  • loss_curves.png                    │   │
                         │  │  • head_to_head.png                   │   │
                         │  │  • Statistical tests                   │   │
                         │  └─────────────────────────────────────┘   │
                         └────────────────────────────────────────────┘
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

| Experiment | Description | Opponent Model | Curriculum | Command |
|---|---|---|---|---|
| `baseline_ppo` | Standard PPO self-play | ✗ | ✗ | `python -m experiments.run_baseline` |
| `om_agent` | PPO + GRU opponent model | ✓ | ✗ | `python -m experiments.run_om` |
| `om_curriculum` | PPO + OM + progressive grid sizes | ✓ | ✓ | `python -m experiments.run_all` |

### Run-All Pipeline Flags

```bash
python -m experiments.run_all [--quick] [--episodes N] [--no-wandb]
```

| Flag | Description |
|---|---|
| `--quick` | Smoke test: 5 episodes/experiment (may skip PPO updates) |
| `--episodes N` | Override episode count (e.g., `--episodes 500`) |
| `--no-wandb` | Disable Weights & Biases logging |

`TOTAL_EPISODES` env var is also supported for backward compatibility.

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

## Results & Visualizations

Training metrics are logged to `results/<experiment>/metrics.csv` and evaluation metrics to `eval_metrics.csv`.

### Metrics Reference

| Metric | File | Meaning |
|---|---|---|
| `capture_rate` | `metrics.csv` | Rolling average (20 episodes) of predator captures |
| `predator_elo` / `prey_elo` | `metrics.csv` | Elo ratings from self-play outcomes |
| `policy_loss` / `value_loss` | `metrics.csv` | PPO clip loss and value function MSE |
| `om_loss` | `metrics.csv` | Cross-entropy loss for opponent action prediction (OM only) |
| `win_rate_vs_random` | `eval_metrics.csv` | Deterministic evaluation against random opponent |
| `win_rate_vs_past_self` | `eval_metrics.csv` | Evaluation against frozen historical snapshot |

### Generating Visualizations

After running experiments, generate publication-quality plots:

```bash
python results/analysis.py
```

This produces three visualizations:

| Output | Description |
|---|---|
| `results/comparison_plots.png` | **6-panel overview**: capture rate, Elo ratings, OM loss, win rate vs random, past-self eval, OM weight decay |
| `results/loss_curves.png` | **Loss analysis**: policy loss, value loss, and entropy across experiments |
| `results/head_to_head.png` | **Bar chart**: final capture rate and Elo comparison (includes curriculum when available) |

### Pipeline Visualizations

`experiments/run_all.py` also generates:

| Output | Description |
|---|---|
| `results/comparison_report.txt` | Human-readable summary with metric table and insights |
| `results/comparison_plots.png` | 4-panel training curves (capture rate, Elo, win rate, past-self) |

All plots use 150 DPI, publication-friendly styling, and consistent color schemes across experiments (blue=baseline, red=OM, green=curriculum).

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
- Lints with ruff
- Builds and pushes Docker images to GitHub Container Registry on version tags (`v*`)

## License

MIT — see [LICENSE](LICENSE) for details.

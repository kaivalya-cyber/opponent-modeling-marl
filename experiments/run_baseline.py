"""
Trains two standard PPO agents against each other via self-play.
Experiment name: "baseline_ppo"
Total episodes: 2000
Logs to: results/baseline_ppo/metrics.csv
Saves checkpoint: results/baseline_ppo/checkpoint_final.pt

Usage: python -m experiments.run_baseline
"""

import logging

import config
from agents.ppo_agent import PPOAgent
from training.trainer import Trainer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config.set_seed(42)
    predator = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
    prey = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)

    trainer = Trainer(
        predator=predator,
        prey=prey,
        experiment_name="baseline_ppo",
        total_episodes=config.TOTAL_EPISODES,
        is_om=False,
    )
    trainer.train()


if __name__ == "__main__":
    main()

"""
Trains two OM agents against each other via self-play.
Experiment name: "om_agent"
Identical hyperparameters to baseline.
Logs to: results/om_agent/metrics.csv
Saves checkpoint: results/om_agent/checkpoint_final.pt

Usage: python -m experiments.run_om
"""

import logging

import config
from agents.om_agent import OMAgent
from training.trainer import Trainer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config.set_seed(42)
    predator = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
    prey = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)

    trainer = Trainer(
        predator=predator,
        prey=prey,
        experiment_name="om_agent",
        total_episodes=config.TOTAL_EPISODES,
        is_om=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()

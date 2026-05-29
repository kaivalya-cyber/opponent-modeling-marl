"""
Head-to-head tournament — pits trained agents against each other.

Loads the final checkpoints from baseline_ppo and om_agent experiments,
then runs matchups in all 4 configurations (swapping predator/prey roles).

Usage: python -m experiments.tournament

Outputs:
  - Console summary table with win rates, avg steps, avg rewards
  - Saved to results/tournament_results.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

import config
from env.predator_prey import PredatorPreyEnv
from agents.ppo_agent import PPOAgent
from agents.om_agent import OMAgent

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
BASELINE_CKPT = RESULTS_DIR / "baseline_ppo" / "checkpoint_final.pt"
OM_CKPT = RESULTS_DIR / "om_agent" / "checkpoint_final.pt"


def load_agent(ckpt_path: Path, is_om: bool) -> tuple:
    """Load a trained agent from checkpoint.

    Returns (agent, agent_name) tuple.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)

    if is_om:
        agent = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        agent.policy.load_state_dict(state["predator_policy"])
        agent.value_net.load_state_dict(state["predator_value"])
        agent.opponent_model.load_state_dict(state["predator_om"])
        name = "OM"
    else:
        agent = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        agent.policy.load_state_dict(state["predator_policy"])
        agent.value_net.load_state_dict(state["predator_value"])
        name = "Baseline"

    agent.policy.eval()
    agent.value_net.eval()
    if is_om:
        agent.opponent_model.eval()

    return agent, name


def run_match(
    env: PredatorPreyEnv,
    predator,
    prey,
    pred_is_om: bool,
    prey_is_om: bool,
    n_matches: int,
) -> dict:
    """Run N matches with given predator/prey, return aggregate stats.

    Returns dict with: wins_predator, wins_prey, draws, avg_steps,
                       avg_pred_reward, avg_prey_reward.
    """
    wins_pred = 0
    wins_prey = 0
    draws = 0
    total_steps = 0
    total_pred_reward = 0.0
    total_prey_reward = 0.0

    for _ in range(n_matches):
        (obs_pred, obs_prey), _ = env.reset()
        predator.reset_episode()
        prey.reset_episode()
        last_pred_action = 0
        last_prey_action = 0
        ep_steps = 0

        for _ in range(config.MAX_STEPS):
            ep_steps += 1

            if pred_is_om:
                a_pred = predator.select_action_deterministic(
                    obs_pred, last_opp_action=last_prey_action,
                )
            else:
                a_pred = predator.select_action_deterministic(obs_pred)

            if prey_is_om:
                a_prey = prey.select_action_deterministic(
                    obs_prey, last_opp_action=last_pred_action,
                )
            else:
                a_prey = prey.select_action_deterministic(obs_prey)

            (obs_pred, obs_prey), (r_pred, r_prey), term, trunc, info = env.step(
                (a_pred, a_prey),
            )
            last_pred_action, last_prey_action = a_pred, a_prey
            total_pred_reward += r_pred
            total_prey_reward += r_prey

            if term or trunc:
                break

        total_steps += ep_steps

        if info["captured"]:
            wins_pred += 1
        else:
            # Truncated by max steps — prey survived
            wins_prey += 1

    return {
        "wins_predator": wins_pred,
        "wins_prey": wins_prey,
        "draws": draws,
        "avg_steps": total_steps / n_matches,
        "avg_pred_reward": total_pred_reward / n_matches,
        "avg_prey_reward": total_prey_reward / n_matches,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config.set_seed(42)

    n_matches = config.TOURNAMENT_MATCHES
    env = PredatorPreyEnv(grid_size=config.GRID_SIZE, max_steps=config.MAX_STEPS)

    # Load both agents
    logger.info("Loading checkpoints...")
    baseline_agent, _ = load_agent(BASELINE_CKPT, is_om=False)
    om_agent, _ = load_agent(OM_CKPT, is_om=True)
    logger.info("Checkpoints loaded.")

    # Run all 4 matchups
    matchups = [
        ("Baseline (predator) vs OM (prey)",
         baseline_agent, om_agent, False, True),
        ("OM (predator) vs Baseline (prey)",
         om_agent, baseline_agent, True, False),
        ("Baseline (predator) vs Baseline (prey)",
         baseline_agent, baseline_agent, False, False),
        ("OM (predator) vs OM (prey)",
         om_agent, om_agent, True, True),
    ]

    results = {}
    for label, pred, prey, pred_om, prey_om in matchups:
        logger.info(f"Running: {label} ({n_matches} matches)...")
        stats = run_match(env, pred, prey, pred_om, prey_om, n_matches)
        results[label] = stats

    # Print results table
    sep = "=" * 84
    print()
    print(sep)
    print(f"{'Matchup':<42s} | {'Pred Wins':>10s} | {'Prey Wins':>10s} | {'Win Rate':>10s}")
    print("-" * 84)

    for label, stats in results.items():
        pred_wins = stats["wins_predator"]
        prey_wins = stats["wins_prey"]
        win_rate = pred_wins / n_matches * 100
        print(
            f"{label:<42s} | {pred_wins:>10d} | {prey_wins:>10d} | {win_rate:>9.1f}%"
        )

    print(sep)
    print()

    # Detailed stats
    print("Detailed match stats:")
    print("-" * 84)
    print(f"{'Matchup':<42s} | {'Avg Steps':>10s} | {'Avg Pred R':>10s} | {'Avg Prey R':>10s}")
    print("-" * 84)
    for label, stats in results.items():
        print(
            f"{label:<42s} | {stats['avg_steps']:>10.1f} | "
            f"{stats['avg_pred_reward']:>10.3f} | {stats['avg_prey_reward']:>10.3f}"
        )
    print(sep)

    # Cross-experiment summary
    om_as_pred_vs_bl = results["OM (predator) vs Baseline (prey)"]["wins_predator"] / n_matches
    bl_as_pred_vs_om = results["Baseline (predator) vs OM (prey)"]["wins_predator"] / n_matches

    print()
    print("Cross-experiment summary:")
    print(f"  OM beats Baseline as predator: {om_as_pred_vs_bl:.1%}")
    print(f"  Baseline beats OM as predator:  {bl_as_pred_vs_om:.1%}")
    print()

    # Save results
    out_path = RESULTS_DIR / "tournament_results.txt"
    with open(out_path, "w") as f:
        f.write(f"Tournament results ({n_matches} matches per matchup)\n")
        f.write("=" * 84 + "\n")
        f.write(f"{'Matchup':<42s} | {'Pred Wins':>10s} | {'Prey Wins':>10s} | {'Win Rate':>10s}\n")
        f.write("-" * 84 + "\n")
        for label, stats in results.items():
            pred_wins = stats["wins_predator"]
            prey_wins = stats["wins_prey"]
            win_rate = pred_wins / n_matches * 100
            f.write(f"{label:<42s} | {pred_wins:>10d} | {prey_wins:>10d} | {win_rate:>9.1f}%\n")
        f.write("=" * 84 + "\n")
        f.write(f"\nOM beats Baseline as predator: {om_as_pred_vs_bl:.1%}\n")
        f.write(f"Baseline beats OM as predator:  {bl_as_pred_vs_om:.1%}\n")
    logger.info(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()

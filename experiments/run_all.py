"""
Runs all three experiment configurations in sequence and generates a comparison
report with summary metrics, Elo ratings, and training curves.

Experiments:
  1. baseline_ppo      — Standard PPO self-play (no opponent modeling)
  2. om_agent           — PPO + opponent model (GRU-based)
  3. om_curriculum      — PPO + opponent model + curriculum learning (5→7→10)

Usage:
    python -m experiments.run_all

Outputs:
  - results/baseline_ppo/    (metrics, checkpoints, metadata)
  - results/om_agent/         (metrics, checkpoints, metadata)
  - results/om_curriculum/    (metrics, checkpoints, metadata)
  - results/comparison_report.txt
  - results/comparison_plots.png
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from agents.ppo_agent import PPOAgent
from agents.om_agent import OMAgent
from training.trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "baseline_ppo",
        "description": "Standard PPO self-play",
        "is_om": False,
        "curriculum": False,
    },
    {
        "name": "om_agent",
        "description": "PPO + opponent model (GRU)",
        "is_om": True,
        "curriculum": False,
    },
    {
        "name": "om_curriculum",
        "description": "PPO + opponent model + curriculum",
        "is_om": True,
        "curriculum": True,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(exp: dict) -> dict:
    """Run a single experiment and return summary stats."""
    name = exp["name"]
    is_om = exp["is_om"]

    # Set curriculum config
    config.CURRICULUM_ENABLED = exp["curriculum"]

    logger.info("=" * 60)
    logger.info(f"Starting: {name} ({exp['description']})")
    logger.info(f"  is_om={is_om}, curriculum={exp['curriculum']}")
    logger.info("=" * 60)

    config.set_seed(42)

    if is_om:
        predator = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        prey = OMAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
    else:
        predator = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)
        prey = PPOAgent(obs_dim=config.OBS_DIM, action_dim=config.ACTION_DIM)

    start_time = time.time()

    trainer = Trainer(
        predator=predator,
        prey=prey,
        experiment_name=name,
        total_episodes=config.TOTAL_EPISODES,
        is_om=is_om,
    )
    trainer.train()

    elapsed = time.time() - start_time

    # Collect summary metrics
    summary = _collect_summary(trainer.results_dir, name, elapsed)
    logger.info(f"Completed {name} in {elapsed:.0f}s")
    return summary


def _collect_summary(results_dir: Path, name: str, elapsed: float) -> dict:
    """Extract summary stats from the experiment results."""
    csv_path = results_dir / "metrics.csv"
    eval_path = results_dir / "eval_metrics.csv"

    summary = {
        "name": name,
        "elapsed_seconds": round(elapsed, 1),
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            final = df.iloc[-1]
            summary["final_capture_rate"] = round(float(final["capture_rate"]), 4)
            summary["final_predator_elo"] = round(float(final["predator_elo"]), 2)
            summary["final_prey_elo"] = round(float(final["prey_elo"]), 2)
            summary["final_policy_loss"] = round(float(final["policy_loss"]), 6)
            summary["final_value_loss"] = round(float(final["value_loss"]), 6)
            summary["peak_capture_rate"] = round(float(df["capture_rate"].max()), 4)

            if "om_loss" in df.columns:
                summary["final_om_loss"] = round(float(final["om_loss"]), 6)

    if eval_path.exists():
        edf = pd.read_csv(eval_path)
        if len(edf) > 0:
            summary["final_win_rate"] = round(float(edf.iloc[-1]["win_rate_vs_random"]), 4)
            summary["peak_win_rate"] = round(float(edf["win_rate_vs_random"].max()), 4)
            if "win_rate_vs_past_self" in edf.columns:
                past_self = edf["win_rate_vs_past_self"].dropna()
                if len(past_self) > 0:
                    summary["final_past_self"] = round(float(past_self.iloc[-1]), 4)

    return summary


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(summaries: list[dict]) -> str:
    """Generate a human-readable comparison report."""
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("EXPERIMENT COMPARISON REPORT")
    lines.append(sep)
    lines.append(f"Experiments run: {len(summaries)}")
    lines.append("")

    # Summary table
    headers = ["Metric", *(s["name"] for s in summaries)]
    rows = [
        ("Total time", *(f"{s.get('elapsed_seconds', 'N/A'):.0f}s" for s in summaries)),
        (
            "Final capture rate",
            *(f"{s.get('final_capture_rate', 'N/A'):.4f}" for s in summaries),
        ),
        (
            "Peak capture rate",
            *(f"{s.get('peak_capture_rate', 'N/A'):.4f}" for s in summaries),
        ),
        (
            "Final pred Elo",
            *(f"{s.get('final_predator_elo', 'N/A'):.1f}" for s in summaries),
        ),
        (
            "Final prey Elo",
            *(f"{s.get('final_prey_elo', 'N/A'):.1f}" for s in summaries),
        ),
        (
            "Final win rate",
            *(f"{s.get('final_win_rate', 'N/A'):.4f}" for s in summaries),
        ),
        (
            "Peak win rate",
            *(f"{s.get('peak_win_rate', 'N/A'):.4f}" for s in summaries),
        ),
        (
            "Final pol loss",
            *(f"{s.get('final_policy_loss', 'N/A'):.6f}" for s in summaries),
        ),
        (
            "Final val loss",
            *(f"{s.get('final_value_loss', 'N/A'):.6f}" for s in summaries),
        ),
    ]

    if any("final_om_loss" in s for s in summaries):
        rows.append((
            "Final OM loss",
            *(f"{s.get('final_om_loss', 'N/A'):.6f}" if "final_om_loss" in s else "-"
              for s in summaries),
        ))

    if any("final_past_self" in s for s in summaries):
        rows.append((
            "Final past-self",
            *(f"{s.get('final_past_self', 'N/A'):.4f}" if "final_past_self" in s else "-"
              for s in summaries),
        ))

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    col_widths[0] = max(col_widths[0], 22)  # Min width for metric names

    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    lines.append(fmt_row(headers))
    lines.append("-" * len(fmt_row(headers)))
    for row in rows:
        lines.append(fmt_row(row))

    lines.append("")
    lines.append(sep)
    lines.append("")

    # Analysis insights
    lines.append("INSIGHTS")
    lines.append("-" * 80)

    # Compare OM vs baseline
    om_summary = next((s for s in summaries if s["name"] == "om_agent"), None)
    bl_summary = next((s for s in summaries if s["name"] == "baseline_ppo"), None)

    if om_summary and bl_summary:
        om_win = om_summary.get("final_win_rate", 0)
        bl_win = bl_summary.get("final_win_rate", 0)
        delta = om_win - bl_win
        direction = "higher" if delta > 0 else "lower"
        lines.append(
            f"Opponent model vs baseline: OM win rate is {abs(delta):.4f} {direction} "
            f"({om_win:.4f} vs {bl_win:.4f})"
        )

    # Compare curriculum vs regular OM
    curr_summary = next((s for s in summaries if s["name"] == "om_curriculum"), None)
    if om_summary and curr_summary:
        om_win = om_summary.get("final_win_rate", 0)
        curr_win = curr_summary.get("final_win_rate", 0)
        delta = curr_win - om_win
        direction = "higher" if delta > 0 else "lower"
        lines.append(
            f"Curriculum vs regular OM: curriculum win rate is {abs(delta):.4f} "
            f"{direction} ({curr_win:.4f} vs {om_win:.4f})"
        )

    lines.append("")
    lines.append("Generated by: experiments/run_all.py")
    lines.append(sep)

    return "\n".join(lines)


def generate_comparison_plots(summaries: list[dict]) -> None:
    """Generate comparison plots from all experiment CSVs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")

    colors = {"baseline_ppo": "#2196F3", "om_agent": "#FF9800", "om_curriculum": "#4CAF50"}
    labels = {
        "baseline_ppo": "Baseline PPO",
        "om_agent": "OM Agent",
        "om_curriculum": "OM + Curriculum",
    }

    for exp in summaries:
        name = exp["name"]
        csv_path = Path("results") / name / "metrics.csv"
        eval_path = Path("results") / name / "eval_metrics.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            axes[0, 0].plot(df["episode"], df["capture_rate"], color=colors[name],
                            alpha=0.7, label=labels.get(name, name))
            axes[0, 1].plot(df["episode"], df["predator_elo"], color=colors[name],
                            alpha=0.7)

        if eval_path.exists():
            edf = pd.read_csv(eval_path)
            axes[1, 0].plot(edf["episode"], edf["win_rate_vs_random"],
                            color=colors[name], alpha=0.7)
            if "win_rate_vs_past_self" in edf.columns:
                past = edf.dropna(subset=["win_rate_vs_past_self"])
                axes[1, 1].plot(past["episode"], past["win_rate_vs_past_self"],
                                color=colors[name], alpha=0.7)

    axes[0, 0].set_title("Capture Rate")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title("Predator Elo Rating")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title("Win Rate vs Random")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title("Win Rate vs Past Self")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("results") / "comparison_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plots saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting full experiment suite (%d experiments)", len(EXPERIMENTS))

    summaries = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        logger.info(f"\nExperiment {i}/{len(EXPERIMENTS)}: {exp['name']}")
        summary = run_experiment(exp)
        summaries.append(summary)

    # Generate comparison report
    report = generate_comparison_report(summaries)
    report_path = Path("results") / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Comparison report saved: {report_path}")

    # Print to console
    print("\n" + report)

    # Generate plots if matplotlib is available
    try:
        generate_comparison_plots(summaries)
    except Exception as e:
        logger.warning(f"Could not generate comparison plots: {e}")

    logger.info("All experiments complete!")


if __name__ == "__main__":
    main()

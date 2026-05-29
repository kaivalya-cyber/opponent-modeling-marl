"""
Results analysis — loads metrics from experiments and generates comprehensive plots.

Usage: python results/analysis.py

Generates:
  - comparison_plots.png: capture rate, Elo, OM loss, eval metrics
  - loss_curves.png: policy loss, value loss, entropy over training
  - head_to_head_summary.png: side-by-side bar chart comparison
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#dee2e6",
    "font.family": "sans-serif",
    "font.size": 11,
})

RESULTS_DIR = Path("results")
COLORS = {
    "baseline": "#4a90d9",
    "om": "#e74c3c",
    "baseline_light": "#7ab4f0",
    "om_light": "#f08080",
}


def load_metrics(experiment_name: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / experiment_name / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_eval_metrics(experiment_name: str) -> pd.DataFrame | None:
    csv_path = RESULTS_DIR / experiment_name / "eval_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average for smoother plots."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_capture_rate(ax, baseline: pd.DataFrame, om: pd.DataFrame) -> None:
    """Capture rate over episodes."""
    ax.plot(
        smooth(baseline["capture_rate"].values),
        label="Baseline PPO",
        alpha=0.85,
        color=COLORS["baseline"],
        linewidth=2,
    )
    ax.plot(
        smooth(om["capture_rate"].values),
        label="OM Agent",
        alpha=0.85,
        color=COLORS["om"],
        linewidth=2,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capture Rate (rolling 20)")
    ax.set_title("Predator Capture Rate Over Training", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6")
    ax.set_ylim(0, 1.05)


def plot_elo(ax, baseline: pd.DataFrame, om: pd.DataFrame) -> None:
    """Elo ratings over episodes."""
    ax.plot(
        baseline["predator_elo"].values,
        label="Baseline Predator",
        alpha=0.85,
        color=COLORS["baseline"],
        linewidth=2,
    )
    ax.plot(
        om["predator_elo"].values,
        label="OM Predator",
        alpha=0.85,
        color=COLORS["om"],
        linewidth=2,
    )
    ax.plot(
        baseline["prey_elo"].values,
        label="Baseline Prey",
        alpha=0.4,
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        om["prey_elo"].values,
        label="OM Prey",
        alpha=0.4,
        color=COLORS["om"],
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Elo Rating")
    ax.set_title("Elo Ratings Over Training", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6", fontsize=9)


def plot_om_loss(ax, om: pd.DataFrame) -> None:
    """Opponent model loss (OM only)."""
    if "om_loss" in om.columns:
        om_loss_vals = om["om_loss"].dropna().values
        if len(om_loss_vals) > 20:
            ax.plot(
                smooth(om_loss_vals),
                color=COLORS["om"],
                alpha=0.85,
                linewidth=2,
            )
        else:
            ax.plot(om_loss_vals, color=COLORS["om"], alpha=0.85, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Opponent Model Loss Over Training", fontweight="bold")
        ax.set_ylim(bottom=0)
    else:
        ax.text(
            0.5, 0.5, "No OM loss data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="gray",
        )
        ax.set_title("Opponent Model Loss", fontweight="bold")


def plot_eval_metrics(
    ax, bl_eval: pd.DataFrame | None, om_eval: pd.DataFrame | None,
) -> None:
    """Evaluation win rate vs random opponent."""
    if bl_eval is not None and "win_rate_vs_random" in bl_eval.columns:
        ax.plot(
            bl_eval["episode"].values,
            bl_eval["win_rate_vs_random"].values,
            label="Baseline PPO",
            alpha=0.85,
            color=COLORS["baseline"],
            linewidth=2,
            marker="o",
            markersize=3,
        )
    if om_eval is not None and "win_rate_vs_random" in om_eval.columns:
        ax.plot(
            om_eval["episode"].values,
            om_eval["win_rate_vs_random"].values,
            label="OM Agent",
            alpha=0.85,
            color=COLORS["om"],
            linewidth=2,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate vs Random")
    ax.set_title("Deterministic Eval: Win Rate vs Random Prey", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6")
    ax.set_ylim(0, 1.05)


def plot_past_self_eval(
    ax, bl_eval: pd.DataFrame | None, om_eval: pd.DataFrame | None,
) -> None:
    """Past-self evaluation win rate."""
    has_data = False
    if bl_eval is not None and "win_rate_vs_past_self" in bl_eval.columns:
        mask = bl_eval["win_rate_vs_past_self"].notna()
        if mask.any():
            ax.plot(
                bl_eval["episode"].values[mask],
                bl_eval["win_rate_vs_past_self"].values[mask],
                label="Baseline PPO",
                alpha=0.85,
                color=COLORS["baseline"],
                linewidth=2,
                marker="s",
                markersize=4,
            )
            has_data = True
    if om_eval is not None and "win_rate_vs_past_self" in om_eval.columns:
        mask = om_eval["win_rate_vs_past_self"].notna()
        if mask.any():
            ax.plot(
                om_eval["episode"].values[mask],
                om_eval["win_rate_vs_past_self"].values[mask],
                label="OM Agent",
                alpha=0.85,
                color=COLORS["om"],
                linewidth=2,
                marker="s",
                markersize=4,
            )
            has_data = True
    if has_data:
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Parity")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate vs Past Self")
        ax.set_title("Past-Self Evaluation (Higher = Improving)", fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(
            0.5, 0.5, "No past-self data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="gray",
        )
        ax.set_title("Past-Self Evaluation", fontweight="bold")


def plot_loss_curves(
    axes, baseline: pd.DataFrame, om: pd.DataFrame,
) -> None:
    """Policy loss, value loss, and entropy over training."""
    metrics = [
        ("policy_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("entropy", "Entropy"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        if col in baseline.columns:
            vals = baseline[col].values
            if len(vals) > 20:
                ax.plot(
                    smooth(vals),
                    color=COLORS["baseline"],
                    alpha=0.7,
                    linewidth=1.5,
                    label="Baseline",
                )
            else:
                ax.plot(vals, color=COLORS["baseline"], alpha=0.7, linewidth=1.5, label="Baseline")
        if col in om.columns:
            vals = om[col].values
            if len(vals) > 20:
                ax.plot(
                    smooth(vals),
                    color=COLORS["om"],
                    alpha=0.7,
                    linewidth=1.5,
                    label="OM",
                )
            else:
                ax.plot(vals, color=COLORS["om"], alpha=0.7, linewidth=1.5, label="OM")
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6", fontsize=9)


def plot_om_weight_decay(ax, om: pd.DataFrame) -> None:
    """OM loss weight decay over episodes."""
    if "om_loss_weight" in om.columns:
        vals = om["om_loss_weight"].dropna().values
        ax.plot(
            vals,
            color=COLORS["om"],
            alpha=0.85,
            linewidth=2,
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("OM Loss Weight")
        ax.set_title("OM Loss Weight Decay", fontweight="bold")
    else:
        ax.text(
            0.5, 0.5, "No weight data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="gray",
        )
        ax.set_title("OM Loss Weight Decay", fontweight="bold")


def plot_head_to_head(ax, baseline: pd.DataFrame, om: pd.DataFrame) -> None:
    """Bar chart comparing final metrics."""
    last_n = 100
    bl_cap = baseline["capture_rate"].iloc[-last_n:].mean() * 100
    om_cap = om["capture_rate"].iloc[-last_n:].mean() * 100
    bl_elo = baseline["predator_elo"].iloc[-1]
    om_elo = om["predator_elo"].iloc[-1]

    metrics_names = ["Capture Rate %\n(last 100 ep)", "Final Elo\n(predator)"]
    bl_vals = [bl_cap, bl_elo]
    om_vals = [om_cap, om_elo]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, bl_vals, width, label="Baseline PPO",
                    color=COLORS["baseline"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, om_vals, width, label="OM Agent",
                    color=COLORS["om"], edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title("Head-to-Head: Final Performance", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6")
    ax.set_ylim(bottom=0)


def main() -> None:
    # Load data
    baseline = load_metrics("baseline_ppo")
    om = load_metrics("om_agent")
    bl_eval = load_eval_metrics("baseline_ppo")
    om_eval = load_eval_metrics("om_agent")

    # ---- Figure 1: Main comparison (capture rate, Elo, eval metrics) ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    plot_capture_rate(axes[0, 0], baseline, om)
    plot_elo(axes[0, 1], baseline, om)
    plot_om_loss(axes[0, 2], om)
    plot_eval_metrics(axes[1, 0], bl_eval, om_eval)
    plot_past_self_eval(axes[1, 1], bl_eval, om_eval)
    plot_om_weight_decay(axes[1, 2], om)

    plt.tight_layout(pad=3)
    plot_path = RESULTS_DIR / "comparison_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[1/3] Main comparison plots → {plot_path}")
    plt.close()

    # ---- Figure 2: Loss curves ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_loss_curves(axes, baseline, om)
    plt.tight_layout(pad=3)
    plot_path = RESULTS_DIR / "loss_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[2/3] Loss curves → {plot_path}")
    plt.close()

    # ---- Figure 3: Head-to-head summary ----
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_head_to_head(ax, baseline, om)
    plt.tight_layout(pad=2)
    plot_path = RESULTS_DIR / "head_to_head.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[3/3] Head-to-head summary → {plot_path}")
    plt.close()

    # ---- Summary Table ----
    last_n = 100
    bl_last = baseline["capture_rate"].iloc[-last_n:].values
    om_last = om["capture_rate"].iloc[-last_n:].values

    bl_elo_pred = baseline["predator_elo"].iloc[-1]
    om_elo_pred = om["predator_elo"].iloc[-1]

    bl_cap_rate = bl_last.mean() * 100
    om_cap_rate = om_last.mean() * 100

    def episodes_to_threshold(df: pd.DataFrame, threshold: float = 0.60) -> str:
        rates = df["capture_rate"].values
        for i, r in enumerate(rates):
            if r >= threshold:
                return str(i)
        return "N/A"

    bl_to_60 = episodes_to_threshold(baseline)
    om_to_60 = episodes_to_threshold(om)

    print()
    print("=" * 60)
    print(f"{'Metric':<24s} | {'Baseline PPO':>14s} | {'OM Agent':>14s}")
    print("-" * 60)
    print(f"{'Final Elo (predator)':<24s} | {bl_elo_pred:>14.1f} | {om_elo_pred:>14.1f}")
    print(f"{'Capture rate (last 100)':<24s} | {bl_cap_rate:>13.1f}% | {om_cap_rate:>13.1f}%")
    print(f"{'Episodes to 60% cap.':<24s} | {bl_to_60:>14s} | {om_to_60:>14s}")

    # OM loss
    if "om_loss" in om.columns:
        om_final_loss = om["om_loss"].dropna().iloc[-1] if len(om["om_loss"].dropna()) > 0 else float("nan")
        print(f"{'Final OM loss':<24s} | {'—':>14s} | {om_final_loss:>14.4f}")

    # Past-self eval
    if om_eval is not None and "win_rate_vs_past_self" in om_eval.columns:
        past_vals = om_eval["win_rate_vs_past_self"].dropna()
        if len(past_vals) > 0:
            print(f"{'Past-self win rate (last)':<24s} | {'—':>14s} | {past_vals.iloc[-1]:>13.2f}")

    print("=" * 60)

    # ---- Statistical Test ----
    stat, p_value = mannwhitneyu(om_last, bl_last, alternative="greater")
    print(f"\nMann-Whitney U test — p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("✓ OM agent significantly outperforms baseline (p < 0.05).")
    else:
        print("✗ No statistically significant difference (p >= 0.05).")
    print()


if __name__ == "__main__":
    main()

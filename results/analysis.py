"""
Results analysis — loads metrics from all three experiments and generates
comprehensive comparison plots.

Usage: python results/analysis.py

Generates:
  - comparison_plots.png: 6-panel plot (capture rate, Elo, OM loss, eval,
    past-self, OM weight decay)
  - loss_curves.png: policy loss, value loss, entropy across experiments
  - head_to_head.png: side-by-side bar chart (baseline vs OM vs curriculum)
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
    "curriculum": "#2ecc71",
    "baseline_light": "#7ab4f0",
    "om_light": "#f08080",
    "curriculum_light": "#58d68d",
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


def plot_capture_rate(
    ax, baseline: pd.DataFrame, om: pd.DataFrame,
    curriculum: pd.DataFrame | None = None,
) -> None:
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
    if curriculum is not None and len(curriculum) > 0:
        ax.plot(
            smooth(curriculum["capture_rate"].values),
            label="OM + Curriculum",
            alpha=0.85,
            color=COLORS["curriculum"],
            linewidth=2,
            linestyle="--",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capture Rate (rolling 20)")
    ax.set_title("Predator Capture Rate Over Training", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6")
    ax.set_ylim(0, 1.05)


def plot_elo(
    ax, baseline: pd.DataFrame, om: pd.DataFrame,
    curriculum: pd.DataFrame | None = None,
) -> None:
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
    if curriculum is not None and len(curriculum) > 0:
        ax.plot(
            curriculum["predator_elo"].values,
            label="Curriculum Predator",
            alpha=0.85,
            color=COLORS["curriculum"],
            linewidth=2,
            linestyle="--",
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


def plot_head_to_head(
    ax, baseline: pd.DataFrame, om: pd.DataFrame,
    curriculum: pd.DataFrame | None = None,
) -> None:
    """Bar chart comparing final metrics across all experiments."""
    last_n = min(100, len(baseline))
    bl_cap = baseline["capture_rate"].iloc[-last_n:].mean() * 100
    om_cap = om["capture_rate"].iloc[-last_n:].mean() * 100
    bl_elo = baseline["predator_elo"].iloc[-1]
    om_elo = om["predator_elo"].iloc[-1]

    labels_list = ["Baseline PPO", "OM Agent"]
    colors_list = [COLORS["baseline"], COLORS["om"]]
    cap_vals = [bl_cap, om_cap]
    elo_vals = [bl_elo, om_elo]

    if curriculum is not None and len(curriculum) > 0:
        cur_cap = curriculum["capture_rate"].iloc[-last_n:].mean() * 100
        cur_elo = curriculum["predator_elo"].iloc[-1]
        labels_list.append("OM + Curriculum")
        colors_list.append(COLORS["curriculum"])
        cap_vals.append(cur_cap)
        elo_vals.append(cur_elo)

    x = np.arange(2)  # Two metric groups
    n_bars = len(labels_list)
    width = 0.8 / n_bars

    for i, (label, color) in enumerate(zip(labels_list, colors_list)):
        offset = (i - (n_bars - 1) / 2) * width
        bars_cap = ax.bar(
            x[0] + offset, cap_vals[i], width,
            label=label, color=color, edgecolor="white", linewidth=0.8,
        )
        bars_elo = ax.bar(
            x[1] + offset, elo_vals[i], width,
            color=color, edgecolor="white", linewidth=0.8,
        )
        for bar in bars_cap:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
        for bar in bars_elo:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Capture Rate %\n(last 100 ep)", "Final Elo\n(predator)"])
    ax.set_title("Head-to-Head: Final Performance", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#dee2e6", fontsize=9)
    ax.set_ylim(bottom=0)


def _make_summary_row(
    metric: str, bl_val, om_val, cur_val=None, fmt_spec: str = ".2f",
) -> str:
    """Format a single row of the summary table."""
    def fmt(v):
        if v is None:
            return "—".rjust(14)
        if isinstance(v, (int, float)):
            return f"{v:{fmt_spec}}".rjust(14)
        return str(v).rjust(14)

    cols = f"{metric:<26s} | {fmt(bl_val)} | {fmt(om_val)}"
    if cur_val is not None:
        cols += f" | {fmt(cur_val)}"
    return cols


def main() -> None:
    # Load data
    baseline = load_metrics("baseline_ppo")
    om = load_metrics("om_agent")
    bl_eval = load_eval_metrics("baseline_ppo")
    om_eval = load_eval_metrics("om_agent")

    # Try loading curriculum data (may not exist)
    curriculum = None
    cur_eval = None
    try:
        curriculum = load_metrics("om_curriculum")
        cur_eval = load_eval_metrics("om_curriculum")
    except FileNotFoundError:
        print("(Curriculum data not found — skipping in plots)")

    # ---- Figure 1: Main comparison (capture rate, Elo, eval metrics) ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    plot_capture_rate(axes[0, 0], baseline, om, curriculum)
    plot_elo(axes[0, 1], baseline, om, curriculum)
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
    plot_head_to_head(ax, baseline, om, curriculum)
    plt.tight_layout(pad=2)
    plot_path = RESULTS_DIR / "head_to_head.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[3/3] Head-to-head summary → {plot_path}")
    plt.close()

    # ---- Summary Table ----
    last_n = min(100, len(baseline))
    bl_last = baseline["capture_rate"].iloc[-last_n:].values
    om_last = om["capture_rate"].iloc[-last_n:].values

    bl_elo_pred = baseline["predator_elo"].iloc[-1]
    om_elo_pred = om["predator_elo"].iloc[-1]

    bl_cap_rate = bl_last.mean() * 100
    om_cap_rate = om_last.mean() * 100

    cur_cap_rate = None
    cur_elo_pred = None
    if curriculum is not None and len(curriculum) > 0:
        cur_last_n = min(100, len(curriculum))
        cur_cap_rate = curriculum["capture_rate"].iloc[-cur_last_n:].mean() * 100
        cur_elo_pred = curriculum["predator_elo"].iloc[-1]

    def episodes_to_threshold(df: pd.DataFrame, threshold: float = 0.60) -> str:
        rates = df["capture_rate"].values
        for i, r in enumerate(rates):
            if r >= threshold:
                return str(i)
        return "N/A"

    bl_to_60 = episodes_to_threshold(baseline)
    om_to_60 = episodes_to_threshold(om)

    has_curriculum = curriculum is not None and len(curriculum) > 0
    header_cols = f"{'Metric':<26s} | {'Baseline PPO':>14s} | {'OM Agent':>14s}"
    sep_line = "-" * len(header_cols)
    if has_curriculum:
        header_cols += " | {'OM+Curriculum':>14s}"
        sep_line = "-" * len(header_cols)

    print()
    print("=" * len(header_cols))
    print(header_cols)
    print(sep_line)

    rows = [
        ("Final Elo (predator)", bl_elo_pred, om_elo_pred, cur_elo_pred, ".1f"),
        ("Capture rate (last 100)", f"{bl_cap_rate:.1f}%", f"{om_cap_rate:.1f}%",
         f"{cur_cap_rate:.1f}%" if cur_cap_rate else None, "s"),
        ("Episodes to 60% cap.", bl_to_60, om_to_60, None, "s"),
    ]

    for metric, bl_v, om_v, cur_v, fmt_s in rows:
        print(_make_summary_row(metric, bl_v, om_v, cur_v, fmt_s))

    # OM loss
    if "om_loss" in om.columns:
        om_final_loss = om["om_loss"].dropna().iloc[-1] if len(om["om_loss"].dropna()) > 0 else float("nan")
        cur_final_loss = None
        if curriculum is not None and "om_loss" in curriculum.columns:
            cur_vals = curriculum["om_loss"].dropna()
            if len(cur_vals) > 0:
                cur_final_loss = cur_vals.iloc[-1]
        print(_make_summary_row(
            "Final OM loss", "—", om_final_loss, cur_final_loss, ".4f",
        ))

    # Past-self eval
    if om_eval is not None and "win_rate_vs_past_self" in om_eval.columns:
        past_vals = om_eval["win_rate_vs_past_self"].dropna()
        cur_past = None
        if cur_eval is not None and "win_rate_vs_past_self" in cur_eval.columns:
            cur_past_vals = cur_eval["win_rate_vs_past_self"].dropna()
            if len(cur_past_vals) > 0:
                cur_past = cur_past_vals.iloc[-1]
        om_past_val = past_vals.iloc[-1] if len(past_vals) > 0 else None
        print(_make_summary_row(
            "Past-self win rate", "—", om_past_val, cur_past, ".2f",
        ))

    print("=" * len(header_cols))

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

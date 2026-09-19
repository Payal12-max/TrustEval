"""
Generates core visualizations for the validation report. Saves PNGs into
outputs/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_correlation_scatter(trust_scores, human_ratings, out_dir="../outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    human_avg["human_scaled"] = (human_avg["overall_trust"] - 1) / 4.0
    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "human_scaled"]], on=["task_id", "agent_id"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["composite_score"], merged["human_scaled"], alpha=0.7, s=60)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect agreement")
    ax.set_xlabel("TrustEval composite score")
    ax.set_ylabel("Human overall_trust (scaled 0-1)")
    ax.set_title("TrustEval vs Human Judgment")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path = f"{out_dir}/correlation_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_framework_comparison(ci_report, out_dir="../outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ci_report))
    means = ci_report["mean_composite"].values
    lower_err = means - ci_report["ci_lower_95"].values
    upper_err = ci_report["ci_upper_95"].values - means

    ax.bar(x, means, yerr=[lower_err, upper_err], capsize=6, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(ci_report["agent_id"])
    ax.set_ylabel("Mean composite trust score")
    ax.set_title("Framework Comparison (95% Bootstrap CI)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = f"{out_dir}/framework_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dimension_correlation(correlation_df, out_dir="../outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#55A868" if r >= 0 else "#C44E52" for r in correlation_df["pearson_r"]]
    ax.barh(correlation_df["dimension"], correlation_df["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r (vs human ratings)")
    ax.set_title("Per-Dimension Correlation with Human Judgment")
    fig.tight_layout()
    path = f"{out_dir}/dimension_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ablation(ablation_df, out_dir="../outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    df = ablation_df[ablation_df["removed_dimension"] != "none (full model)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["removed_dimension"], df["drop_from_full"], color="#DD8452")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drop in correlation when dimension removed")
    ax.set_title("Ablation Study: Dimension Importance")
    fig.tight_layout()
    path = f"{out_dir}/ablation_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    from data_loader import load_trust_scores
    from confidence_intervals import per_agent_confidence_intervals
    from correlation import correlate_with_humans
    from ablation import ablation_study

    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")

    p1 = plot_correlation_scatter(trust, human)
    p2 = plot_framework_comparison(per_agent_confidence_intervals(trust))
    p3 = plot_dimension_correlation(correlate_with_humans(trust, human))
    p4 = plot_ablation(ablation_study(trust, human))

    print("Saved figures:")
    for p in [p1, p2, p3, p4]:
        print(" -", p)
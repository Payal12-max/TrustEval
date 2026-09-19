"""
Ablation study: remove one trust dimension at a time, recompute the composite
score, and measure how much it hurts correlation with human judgment.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]


def ablation_study(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame,
                    dimensions: list = DIMENSIONS) -> pd.DataFrame:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])[dimensions].mean().reset_index()
    human_avg["human_composite"] = human_avg[dimensions].mean(axis=1) / 5.0
    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "human_composite"]], on=["task_id", "agent_id"])

    full_composite = merged[dimensions].mean(axis=1)
    full_r, _ = pearsonr(full_composite, merged["human_composite"])

    results = [{"removed_dimension": "none (full model)", "pearson_r": round(full_r, 3), "drop_from_full": 0.0}]

    for dim in dimensions:
        remaining = [d for d in dimensions if d != dim]
        ablated_composite = merged[remaining].mean(axis=1)
        r, _ = pearsonr(ablated_composite, merged["human_composite"])
        results.append({
            "removed_dimension": dim,
            "pearson_r": round(r, 3),
            "drop_from_full": round(full_r - r, 3)
        })

    return pd.DataFrame(results).sort_values("drop_from_full", ascending=False)


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    report = ablation_study(trust, human)
    print(report.to_string(index=False))
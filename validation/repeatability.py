"""
Repeatability experiments: for tasks run multiple times (same task_id +
agent_id, different 'repetition' values), measures how much the composite
trust score varies run-to-run under IDENTICAL conditions.
"""

import pandas as pd
import numpy as np


def repeatability_analysis(trust_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = trust_scores.groupby(["task_id", "agent_id"])

    results = []
    for (task_id, agent_id), group in grouped:
        if len(group) < 2:
            continue
        scores = group["composite_score"].values
        results.append({
            "task_id": task_id,
            "agent_id": agent_id,
            "n_repetitions": len(group),
            "mean_composite": round(scores.mean(), 3),
            "std_composite": round(scores.std(), 3),
            "range_composite": round(scores.max() - scores.min(), 3),
            "coefficient_of_variation": round(scores.std() / scores.mean(), 3) if scores.mean() > 0 else None
        })

    if not results:
        return pd.DataFrame([{
            "note": "No task_id+agent_id combinations have 2+ repetitions in the "
                    "current data — repeatability cannot be assessed. Ask Member 1 "
                    "whether repeated runs exist in trust_scores.json."
        }])

    return pd.DataFrame(results).sort_values("std_composite", ascending=False)


def repeatability_summary(repeatability_df: pd.DataFrame) -> dict:
    if "note" in repeatability_df.columns:
        return {"note": repeatability_df.iloc[0]["note"]}
    return {
        "n_task_agent_pairs_with_repetitions": len(repeatability_df),
        "mean_std_across_pairs": round(repeatability_df["std_composite"].mean(), 3),
        "max_std_across_pairs": round(repeatability_df["std_composite"].max(), 3),
        "mean_range_across_pairs": round(repeatability_df["range_composite"].mean(), 3),
    }


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    report = repeatability_analysis(trust)
    print(report.to_string(index=False))
    print()
    print("Summary:", repeatability_summary(report))
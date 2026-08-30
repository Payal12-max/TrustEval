"""
Robustness testing: checks whether correlation with human judgment is
sensitive to any single rater or any single trace being removed.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _correlation_for_subset(trust_scores: pd.DataFrame, human_subset: pd.DataFrame) -> float:
    human_avg = human_subset.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    human_avg["human_scaled"] = (human_avg["overall_trust"] - 1) / 4.0
    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "human_scaled"]], on=["task_id", "agent_id"])
    if len(merged) < 3 or merged["composite_score"].nunique() < 2:
        return np.nan
    r, _ = pearsonr(merged["composite_score"], merged["human_scaled"])
    return r


def leave_one_rater_out(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> pd.DataFrame:
    raters = human_ratings["rater_id"].unique()
    full_r = _correlation_for_subset(trust_scores, human_ratings)

    results = [{"excluded_rater": "none (all raters)", "pearson_r": round(full_r, 3) if not np.isnan(full_r) else None}]
    for rater in raters:
        subset = human_ratings[human_ratings["rater_id"] != rater]
        r = _correlation_for_subset(trust_scores, subset)
        results.append({"excluded_rater": rater, "pearson_r": round(r, 3) if not np.isnan(r) else None})
    return pd.DataFrame(results)


def leave_one_trace_out(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> dict:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    pairs = human_avg[["task_id", "agent_id"]].drop_duplicates().values.tolist()

    full_r = _correlation_for_subset(trust_scores, human_ratings)
    swings = []
    for task_id, agent_id in pairs:
        subset = human_ratings[~((human_ratings["task_id"] == task_id) & (human_ratings["agent_id"] == agent_id))]
        r = _correlation_for_subset(trust_scores, subset)
        if not np.isnan(r) and not np.isnan(full_r):
            swings.append(abs(r - full_r))

    if not swings:
        return {"note": "Not enough data to compute leave-one-trace-out stability."}

    return {
        "full_sample_r": round(full_r, 3) if not np.isnan(full_r) else None,
        "n_traces_tested": len(swings),
        "mean_swing_when_removed": round(float(np.mean(swings)), 3),
        "max_swing_when_removed": round(float(np.max(swings)), 3),
        "interpretation": "Large max_swing relative to full_sample_r suggests one trace "
                           "has outsized influence on the correlation result."
    }


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    print("Leave-one-rater-out:")
    print(leave_one_rater_out(trust, human).to_string(index=False))
    print()
    print("Leave-one-trace-out summary:")
    print(leave_one_trace_out(trust, human))
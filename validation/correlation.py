"""Correlation between TrustEval automated scores and averaged human ratings."""

import pandas as pd
from scipy.stats import pearsonr, spearmanr

DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]


def correlate_with_humans(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame,
                           dimensions: list = DIMENSIONS) -> pd.DataFrame:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])[dimensions].mean().reset_index()
    merged = trust_scores.merge(human_avg, on=["task_id", "agent_id"], suffixes=("_auto", "_human"))

    results = []
    for dim in dimensions:
        auto_col, human_col = f"{dim}_auto", f"{dim}_human"
        r_pearson, p_pearson = pearsonr(merged[auto_col], merged[human_col])
        r_spearman, p_spearman = spearmanr(merged[auto_col], merged[human_col])
        results.append({
            "dimension": dim,
            "pearson_r": round(r_pearson, 3),
            "pearson_p": round(p_pearson, 4),
            "spearman_rho": round(r_spearman, 3),
            "spearman_p": round(p_spearman, 4),
            "n": len(merged)
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    report = correlate_with_humans(trust, human)
    print(report.to_string(index=False))
"""
Compares TrustEval's composite score against a naive baseline: the trace's
own 'status' field ("success"/"failure").
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def naive_success_baseline(trust_scores: pd.DataFrame) -> pd.Series:
    return (trust_scores["status"] == "success").astype(int)


def compare_baseline_vs_trusteval(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> pd.DataFrame:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    human_avg["overall_trust_scaled"] = (human_avg["overall_trust"] - 1) / 4.0

    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "overall_trust_scaled"]], on=["task_id", "agent_id"])
    merged["naive_baseline"] = naive_success_baseline(merged)

    results = []
    for name, col in [("TrustEval composite_score", "composite_score"),
                       ("Naive success-only baseline", "naive_baseline")]:
        if merged[col].nunique() < 2:
            results.append({"method": name, "pearson_r": None, "note": "insufficient variance in sample"})
            continue
        r_p, p_p = pearsonr(merged[col], merged["overall_trust_scaled"])
        r_s, p_s = spearmanr(merged[col], merged["overall_trust_scaled"])
        results.append({
            "method": name,
            "pearson_r": round(r_p, 3),
            "pearson_p": round(p_p, 4),
            "spearman_rho": round(r_s, 3),
            "spearman_p": round(p_s, 4),
            "n": len(merged)
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    report = compare_baseline_vs_trusteval(trust, human)
    print(report.to_string(index=False))
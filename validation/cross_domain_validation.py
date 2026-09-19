"""
Checks whether TrustEval's correlation with human judgment holds consistently
across domains.
"""

import pandas as pd
from scipy.stats import pearsonr


def cross_domain_validation(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> pd.DataFrame:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    human_avg["overall_trust_scaled"] = (human_avg["overall_trust"] - 1) / 4.0

    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "overall_trust_scaled"]], on=["task_id", "agent_id"])

    results = []
    for domain, group in merged.groupby("domain"):
        if len(group) < 3:
            results.append({"domain": domain, "pearson_r": None, "n": len(group),
                             "note": "too few rated traces in this domain"})
            continue
        r, p = pearsonr(group["composite_score"], group["overall_trust_scaled"])
        results.append({"domain": domain, "pearson_r": round(r, 3), "pearson_p": round(p, 4),
                         "n": len(group), "note": ""})

    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    report = cross_domain_validation(trust, human)
    print(report.to_string(index=False))
"""
Validates the automated composite trust score against an independent,
holistic human judgment of overall trustworthiness.

Rather than comparing the composite score to an average of the individual
human-rated sub-dimensions (which risks circularity, since the composite
itself is built from similar sub-dimensions), this module correlates it
directly against a separate 'overall_trust' rating — a single question
asking a human rater how much they would trust the agent's output for a
similar real task, unsupervised. This provides a more direct test of
whether the composite score reflects genuine human trust judgment, rather
than just internal consistency with its own components.

Computes both Pearson correlation (linear association) and Spearman
correlation (rank-based, more robust to non-linearity and outliers in
small samples).
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def validate_overall_trust(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> pd.DataFrame:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    merged = trust_scores.merge(human_avg, on=["task_id", "agent_id"])

    merged["overall_trust_scaled"] = (merged["overall_trust"] - 1) / 4.0

    r_pearson, p_pearson = pearsonr(merged["composite_score"], merged["overall_trust_scaled"])
    r_spearman, p_spearman = spearmanr(merged["composite_score"], merged["overall_trust_scaled"])

    return pd.DataFrame([{
        "comparison": "composite_score vs human overall_trust",
        "pearson_r": round(r_pearson, 3),
        "pearson_p": round(p_pearson, 4),
        "spearman_rho": round(r_spearman, 3),
        "spearman_p": round(p_spearman, 4),
        "n": len(merged)
    }])


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    report = validate_overall_trust(trust, human)
    print(report.to_string(index=False))
"""
Baseline comparison.

Compares TrustEval's domain-aware composite against:

1. Simple unweighted average.
2. Execution success-only baseline.
"""

import pandas as pd


DIMENSIONS = [
    "safety",
    "robustness",
    "reliability",
    "goal_adherence",
]


def compare_baseline_vs_trusteval(
    trust_scores,
):

    df = trust_scores.copy()

    # Simple equal-weight baseline
    df["equal_weight_score"] = (
        df[DIMENSIONS].mean(axis=1)
    )

    # Success-only baseline
    df["success_baseline"] = (
        df["status"] == "success"
    ).astype(int)

    results = []

    for name, column in [
        (
            "TrustEval composite",
            "composite_score",
        ),
        (
            "Equal-weight baseline",
            "equal_weight_score",
        ),
        (
            "Success-only baseline",
            "success_baseline",
        ),
    ]:

        framework_means = (
            df.groupby("agent_id")[column]
            .mean()
        )

        results.append({
            "method": name,
            "overall_mean": round(
                float(df[column].mean()),
                3,
            ),
            "overall_std": round(
                float(df[column].std()),
                3,
            ),
            "framework_count": len(
                framework_means
            ),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    from data_loader import load_trust_scores

    trust = load_trust_scores(
        "../results/trust_scores.json"
    )

    print(
        compare_baseline_vs_trusteval(
            trust
        ).to_string(index=False)
    )
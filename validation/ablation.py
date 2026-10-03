"""
Algorithmic ablation study.

Removes one trust dimension at a time, renormalizes the remaining
domain-specific weights, and measures:

1. Change in framework scores.
2. Change in framework ranking.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from scoring.composite_scorer import DOMAIN_WEIGHTS

DIMENSIONS = [
    "safety",
    "robustness",
    "reliability",
    "goal_adherence",
]


def ablation_study(
    trust_scores,
    dimensions=DIMENSIONS,
):

    results = []

    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------

    full = trust_scores.copy()

    full_values = []

    for _, row in full.iterrows():

        weights = DOMAIN_WEIGHTS.get(
            row["domain"],
            DOMAIN_WEIGHTS["default"],
        )

        value = sum(
            row[d] * weights[d]
            for d in dimensions
        )

        full_values.append(value)

    full["ablation_score"] = full_values

    full_framework = (
        full.groupby("agent_id")["ablation_score"]
        .mean()
    )

    full_ranking = full_framework.rank(
        ascending=False
    )

    full_mean = full["ablation_score"].mean()

    results.append({
        "removed_dimension": "none",
        "mean_score": round(
            full_mean,
            3,
        ),
        "mean_absolute_change": 0.0,
        "kendall_tau_vs_full": 1.0,
    })

    # --------------------------------------------------------
    # Remove one dimension
    # --------------------------------------------------------

    for removed in dimensions:

        remaining = [
            d for d in dimensions
            if d != removed
        ]

        rows = []

        for _, row in trust_scores.iterrows():

            original_weights = DOMAIN_WEIGHTS.get(
                row["domain"],
                DOMAIN_WEIGHTS["default"],
            )

            total = sum(
                original_weights[d]
                for d in remaining
            )

            normalized = {
                d: original_weights[d] / total
                for d in remaining
            }

            score = sum(
                row[d] * normalized[d]
                for d in remaining
            )

            rows.append(score)

        ablated = trust_scores.copy()

        ablated["ablation_score"] = rows

        framework_scores = (
            ablated
            .groupby("agent_id")["ablation_score"]
            .mean()
        )

        ranking = framework_scores.rank(
            ascending=False
        )

        tau, _ = kendalltau(
            full_ranking,
            ranking.reindex(
                full_ranking.index
            ),
        )

        change = np.mean(
            np.abs(
                ablated["ablation_score"]
                - full["ablation_score"]
            )
        )

        results.append({
            "removed_dimension": removed,
            "mean_score": round(
                float(
                    ablated["ablation_score"].mean()
                ),
                3,
            ),
            "mean_absolute_change": round(
                float(change),
                3,
            ),
            "kendall_tau_vs_full": round(
                float(tau),
                3,
            ),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    from data_loader import load_trust_scores

    trust = load_trust_scores(
        "../results/trust_scores.json"
    )

    print(
        ablation_study(trust)
        .to_string(index=False)
    )
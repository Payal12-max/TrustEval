"""
Domain-aware sensitivity analysis for TrustEval.

Tests how much framework rankings change when the domain-specific
TrustEval weights are perturbed.

This validates whether conclusions are robust to reasonable changes
in the weighting scheme.
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


def get_domain_weights(domain):
    """Return the actual TrustEval weights for a domain."""
    weights = DOMAIN_WEIGHTS.get(
        str(domain).lower(),
        DOMAIN_WEIGHTS["default"]
    )

    return np.array([
        weights[d] for d in DIMENSIONS
    ], dtype=float)


def framework_ranking(trust_scores):
    """Rank frameworks using their mean composite score."""
    scores = (
        trust_scores
        .groupby("agent_id")["weighted_composite"]
        .mean()
    )

    return scores.rank(ascending=False)


def sensitivity_analysis(
    trust_scores,
    perturbation=0.10,
    n_trials=200,
    seed=0,
):
    """
    Perturb each domain's actual weights and measure ranking stability.

    perturbation=0.10 means each weight can change by ±10% before
    renormalization.
    """

    rng = np.random.default_rng(seed)

    base = trust_scores.copy()

    # Calculate the actual composite using Member 2's current
    # domain-specific weights.
    base_values = []

    for _, row in base.iterrows():
        weights = get_domain_weights(row["domain"])

        values = np.array([
            row[d] for d in DIMENSIONS
        ])

        base_values.append(
            np.dot(values, weights)
        )

    base["weighted_composite"] = base_values

    base_ranking = framework_ranking(base)

    taus = []

    domains = base["domain"].dropna().unique()

    for _ in range(n_trials):

        perturbed_weights = {}

        for domain in domains:

            weights = get_domain_weights(domain)

            noise = rng.uniform(
                -perturbation,
                perturbation,
                size=len(weights),
            )

            new_weights = np.clip(
                weights * (1 + noise),
                0.001,
                None,
            )

            new_weights = (
                new_weights / new_weights.sum()
            )

            perturbed_weights[domain] = new_weights

        trial = base.copy()

        values = []

        for _, row in trial.iterrows():

            weights = perturbed_weights[row["domain"]]

            dimension_values = np.array([
                row[d] for d in DIMENSIONS
            ])

            values.append(
                np.dot(
                    dimension_values,
                    weights,
                )
            )

        trial["weighted_composite"] = values

        trial_ranking = framework_ranking(trial)

        tau, _ = kendalltau(
            base_ranking,
            trial_ranking,
        )

        if not np.isnan(tau):
            taus.append(tau)

    return pd.DataFrame([{
        "perturbation": perturbation,
        "mean_kendall_tau": round(
            float(np.mean(taus)),
            3,
        ),
        "min_kendall_tau": round(
            float(np.min(taus)),
            3,
        ),
        "max_kendall_tau": round(
            float(np.max(taus)),
            3,
        ),
        "n_trials": len(taus),
    }])


if __name__ == "__main__":

    from data_loader import load_trust_scores

    trust = load_trust_scores(
        "../results/trust_scores.json"
    )

    for perturbation in [0.05, 0.10, 0.20, 0.30]:

        result = sensitivity_analysis(
            trust,
            perturbation=perturbation,
        )

        print(
            result.to_string(index=False)
        )
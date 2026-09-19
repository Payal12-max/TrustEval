"""
Sensitivity analysis: perturbs the composite weights and measures how much
framework rankings change. Stress-tests fixed weights
(safety: 0.15, robustness: 0.2, reliability: 0.35, goal_adherence: 0.3).
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]
MEMBER2_WEIGHTS = np.array([0.15, 0.2, 0.35, 0.3])  # order matches DIMENSIONS above


def weighted_composite(trust_scores: pd.DataFrame, dimensions: list, weights: np.ndarray) -> pd.Series:
    values = trust_scores[dimensions].values @ weights
    tmp = trust_scores.copy()
    tmp["weighted_composite"] = values
    return tmp.groupby("agent_id")["weighted_composite"].mean().rank(ascending=False)


def sensitivity_analysis(trust_scores: pd.DataFrame, dimensions: list = DIMENSIONS,
                          base_weights: np.ndarray = MEMBER2_WEIGHTS, perturbation=0.2,
                          n_trials=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_ranking = weighted_composite(trust_scores, dimensions, base_weights)

    taus = []
    for _ in range(n_trials):
        noise = rng.uniform(-perturbation, perturbation, size=len(base_weights))
        perturbed = np.clip(base_weights + noise, 0.01, None)
        perturbed = perturbed / perturbed.sum()

        perturbed_ranking = weighted_composite(trust_scores, dimensions, perturbed)
        tau, _ = kendalltau(base_ranking, perturbed_ranking)
        taus.append(tau)

    return pd.DataFrame({
        "perturbation_pct": [perturbation * 100],
        "mean_kendall_tau": [round(float(np.nanmean(taus)), 3)],
        "min_kendall_tau": [round(float(np.nanmin(taus)), 3)],
        "n_trials": [n_trials]
    })


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")

    for pert in [0.1, 0.2, 0.3]:
        result = sensitivity_analysis(trust, perturbation=pert)
        print(result.to_string(index=False))
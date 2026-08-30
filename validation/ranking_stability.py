"""
Checks whether framework rankings (autogen/crewai/langchain) stay stable
under bootstrap resampling. Uses Kendall's Tau between the "true" ranking
and rankings from resampled data.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def get_ranking(trust_scores: pd.DataFrame) -> pd.Series:
    return trust_scores.groupby("agent_id")["composite_score"].mean().rank(ascending=False)


def ranking_stability(trust_scores: pd.DataFrame, n_boot=500, seed=0) -> dict:
    true_ranking = get_ranking(trust_scores)
    rng = np.random.default_rng(seed)
    taus = []

    for _ in range(n_boot):
        resampled = trust_scores.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
        boot_ranking = get_ranking(resampled).reindex(true_ranking.index)
        tau, _ = kendalltau(true_ranking, boot_ranking)
        if not np.isnan(tau):
            taus.append(tau)

    return {
        "mean_kendall_tau": round(float(np.mean(taus)), 3),
        "std_kendall_tau": round(float(np.std(taus)), 3),
        "n_bootstrap_samples": len(taus)
    }


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    print("True framework ranking (1 = best):")
    print(get_ranking(trust).sort_values())
    print()
    stability = ranking_stability(trust)
    print("Ranking stability under bootstrap resampling:")
    print(stability)
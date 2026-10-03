"""
Task-level bootstrap ranking stability.

Tests whether framework rankings remain stable when benchmark
tasks are resampled.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def get_ranking(trust_scores):

    scores = (
        trust_scores
        .groupby("agent_id")["composite_score"]
        .mean()
    )

    return scores.rank(
        ascending=False
    )


def ranking_stability(
    trust_scores,
    n_boot=500,
    seed=0,
):

    rng = np.random.default_rng(seed)

    true_ranking = get_ranking(
        trust_scores
    )

    tasks = (
        trust_scores["task_id"]
        .dropna()
        .unique()
    )

    taus = []

    for _ in range(n_boot):

        sampled_tasks = rng.choice(
            tasks,
            size=len(tasks),
            replace=True,
        )

        sampled_parts = []

        for task in sampled_tasks:

            task_rows = trust_scores[
                trust_scores["task_id"] == task
            ]

            sampled_parts.append(
                task_rows
            )

        resampled = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        boot_ranking = (
            get_ranking(resampled)
            .reindex(true_ranking.index)
        )

        tau, _ = kendalltau(
            true_ranking,
            boot_ranking,
        )

        if not np.isnan(tau):
            taus.append(tau)

    return {
        "mean_kendall_tau": round(
            float(np.mean(taus)),
            3,
        ),
        "std_kendall_tau": round(
            float(np.std(taus)),
            3,
        ),
        "min_kendall_tau": round(
            float(np.min(taus)),
            3,
        ),
        "n_bootstrap_samples": len(taus),
    }


if __name__ == "__main__":

    from data_loader import load_trust_scores

    trust = load_trust_scores(
        "../results/trust_scores.json"
    )

    print(
        "Framework ranking:"
    )

    print(
        get_ranking(trust)
        .sort_values()
    )

    print()

    print(
        "Task-level bootstrap stability:"
    )

    print(
        ranking_stability(trust)
    )
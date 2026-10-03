"""
Cross-domain validation.

Measures whether framework rankings remain consistent across
the benchmark's different domains.
"""

import pandas as pd
from scipy.stats import kendalltau


def cross_domain_validation(
    trust_scores,
):

    overall = (
        trust_scores
        .groupby("agent_id")["composite_score"]
        .mean()
    )

    overall_ranking = overall.rank(
        ascending=False
    )

    results = []

    for domain, group in trust_scores.groupby(
        "domain"
    ):

        domain_scores = (
            group
            .groupby("agent_id")["composite_score"]
            .mean()
        )

        ranking = domain_scores.rank(
            ascending=False
        )

        common_agents = (
            overall_ranking.index
            .intersection(ranking.index)
        )

        if len(common_agents) >= 2:

            tau, _ = kendalltau(
                overall_ranking.loc[
                    common_agents
                ],
                ranking.loc[
                    common_agents
                ],
            )

        else:

            tau = None

        results.append({
            "domain": domain,
            "n_tasks": group[
                "task_id"
            ].nunique(),
            "n_runs": len(group),
            "kendall_tau_vs_overall": (
                round(float(tau), 3)
                if tau is not None
                else None
            ),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    from data_loader import load_trust_scores

    trust = load_trust_scores(
        "../results/trust_scores.json"
    )

    print(
        cross_domain_validation(trust)
        .to_string(index=False)
    )
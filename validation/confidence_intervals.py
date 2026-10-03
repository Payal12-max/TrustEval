"""
Task-level bootstrap confidence intervals.

The unit of resampling is the task rather than the individual
execution repetition. This avoids treating repeated runs of the
same task as fully independent observations.
"""

import numpy as np
import pandas as pd


def task_level_bootstrap_ci(
    group,
    n_boot=2000,
    ci=95,
    seed=0,
):
    rng = np.random.default_rng(seed)

    task_means = (
        group
        .groupby("task_id")["composite_score"]
        .mean()
        .values
    )

    if len(task_means) == 0:
        return np.nan, np.nan, np.nan

    boot_means = []

    for _ in range(n_boot):

        sample = rng.choice(
            task_means,
            size=len(task_means),
            replace=True,
        )

        boot_means.append(
            np.mean(sample)
        )

    lower = np.percentile(
        boot_means,
        (100 - ci) / 2,
    )

    upper = np.percentile(
        boot_means,
        100 - (100 - ci) / 2,
    )

    return (
        float(np.mean(task_means)),
        float(lower),
        float(upper),
    )


def per_agent_confidence_intervals(
    trust_scores,
    n_boot=2000,
):
    """
    Calculate 95% task-level bootstrap confidence intervals
    for each framework.
    """

    results = []

    for agent_id, group in trust_scores.groupby(
        "agent_id"
    ):

        mean, lower, upper = task_level_bootstrap_ci(
            group,
            n_boot=n_boot,
        )

        results.append({
            "agent_id": agent_id,
            "mean_composite": round(mean, 3),
            "ci_lower_95": round(lower, 3),
            "ci_upper_95": round(upper, 3),
            "n_tasks": group["task_id"].nunique(),
        })

    return pd.DataFrame(results)
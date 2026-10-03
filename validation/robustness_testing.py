"""
TrustEval Member 3
Robustness Testing

Leave-one-task-out robustness analysis.

The purpose is to test whether the overall framework
ranking changes substantially when any one task is removed.

No human ratings are required.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


FRAMEWORK_COLUMN = "framework"
TASK_COLUMN = "task_id"
SCORE_COLUMN = "composite_score"


def _get_framework_scores(df):
    """
    Calculate mean composite score for each framework.
    """

    result = (
        df.groupby(FRAMEWORK_COLUMN)[SCORE_COLUMN]
        .mean()
        .sort_values(ascending=False)
    )

    return result


def _ranking(series):
    """
    Return framework ranking from highest to lowest score.
    """

    return list(
        series.sort_values(
            ascending=False
        ).index
    )


def _kendall_rank_correlation(
    original_ranking,
    new_ranking
):
    """
    Calculate Kendall's tau between two framework rankings.
    """

    frameworks = sorted(
        set(original_ranking)
        | set(new_ranking)
    )

    original_positions = {
        framework: original_ranking.index(framework)
        for framework in frameworks
        if framework in original_ranking
    }

    new_positions = {
        framework: new_ranking.index(framework)
        for framework in frameworks
        if framework in new_ranking
    }

    common_frameworks = [
        framework
        for framework in frameworks
        if framework in original_positions
        and framework in new_positions
    ]

    if len(common_frameworks) < 2:
        return np.nan

    original_values = [
        original_positions[framework]
        for framework in common_frameworks
    ]

    new_values = [
        new_positions[framework]
        for framework in common_frameworks
    ]

    tau, _ = kendalltau(
        original_values,
        new_values
    )

    return float(tau)


def leave_one_task_out(
    df
):
    """
    Perform leave-one-task-out robustness analysis.

    For every task:
        1. Remove that task.
        2. Recalculate framework scores.
        3. Recalculate framework ranking.
        4. Compare the new ranking with the original ranking.

    Returns
    -------
    pandas.DataFrame
    """

    required_columns = [
        TASK_COLUMN,
        FRAMEWORK_COLUMN,
        SCORE_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    clean_df = df.dropna(
        subset=[
            TASK_COLUMN,
            FRAMEWORK_COLUMN,
            SCORE_COLUMN,
        ]
    ).copy()

    if clean_df.empty:
        raise ValueError(
            "No valid observations available "
            "for robustness testing."
        )

    # --------------------------------------------------------
    # ORIGINAL RANKING
    # --------------------------------------------------------

    original_scores = _get_framework_scores(
        clean_df
    )

    original_ranking = _ranking(
        original_scores
    )

    # --------------------------------------------------------
    # LEAVE ONE TASK OUT
    # --------------------------------------------------------

    results = []

    tasks = sorted(
        clean_df[TASK_COLUMN]
        .unique()
    )

    for task in tasks:

        reduced_df = clean_df[
            clean_df[TASK_COLUMN] != task
        ]

        if reduced_df.empty:
            continue

        reduced_scores = _get_framework_scores(
            reduced_df
        )

        reduced_ranking = _ranking(
            reduced_scores
        )

        tau = _kendall_rank_correlation(
            original_ranking,
            reduced_ranking
        )

        # Check whether the top-ranked framework changed.

        original_top = (
            original_ranking[0]
            if original_ranking
            else None
        )

        new_top = (
            reduced_ranking[0]
            if reduced_ranking
            else None
        )

        top_framework_changed = (
            original_top != new_top
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        row = {
            "removed_task": task,

            "n_remaining_tasks":
                reduced_df[TASK_COLUMN]
                .nunique(),

            "kendall_tau":
                tau,

            "original_top_framework":
                original_top,

            "new_top_framework":
                new_top,

            "top_framework_changed":
                top_framework_changed,
        }

        # Add score for every framework.

        for framework in original_ranking:

            row[
                f"score_{framework}"
            ] = reduced_scores.get(
                framework,
                np.nan
            )

        results.append(row)

    return pd.DataFrame(
        results
    )


def robustness_summary(
    results
):
    """
    Produce a summary of leave-one-task-out robustness.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from leave_one_task_out()

    Returns
    -------
    dict
    """

    if results is None or results.empty:

        return {
            "n_task_removals": 0,
            "mean_kendall_tau": np.nan,
            "min_kendall_tau": np.nan,
            "max_kendall_tau": np.nan,
            "ranking_stability_rate": np.nan,
            "top_framework_change_rate": np.nan,
        }

    tau_values = pd.to_numeric(
        results["kendall_tau"],
        errors="coerce"
    ).dropna()

    # --------------------------------------------------------
    # Ranking stability
    # --------------------------------------------------------

    if len(tau_values) > 0:

        mean_tau = float(
            tau_values.mean()
        )

        min_tau = float(
            tau_values.min()
        )

        max_tau = float(
            tau_values.max()
        )

    else:

        mean_tau = np.nan
        min_tau = np.nan
        max_tau = np.nan

    # --------------------------------------------------------
    # Exact ranking stability
    # --------------------------------------------------------

    if "kendall_tau" in results.columns:

        exact_ranking_rate = float(
            (
                results["kendall_tau"] >= 0.999999
            ).mean()
        )

    else:

        exact_ranking_rate = np.nan

    # --------------------------------------------------------
    # Top framework changes
    # --------------------------------------------------------

    if "top_framework_changed" in results.columns:

        top_change_rate = float(
            results[
                "top_framework_changed"
            ]
            .astype(bool)
            .mean()
        )

    else:

        top_change_rate = np.nan

    return {
        "n_task_removals":
            int(len(results)),

        "mean_kendall_tau":
            mean_tau,

        "min_kendall_tau":
            min_tau,

        "max_kendall_tau":
            max_tau,

        "ranking_stability_rate":
            exact_ranking_rate,

        "top_framework_change_rate":
            top_change_rate,
    }


if __name__ == "__main__":

    print(
        "robustness_testing.py loaded successfully."
    )

    print(
        "Available functions:"
    )

    print(
        " - leave_one_task_out()"
    )

    print(
        " - robustness_summary()"
    )
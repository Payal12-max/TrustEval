"""
TrustEval Member 3
Calibration Analysis

This version does NOT require human ratings.

It evaluates whether TrustEval composite scores
are statistically consistent with execution outcomes.

Analyses:
- Mean trust score
- Success rate
- Trust score vs. execution success correlation
- Score difference between successful and failed runs
"""

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)


def calibration_analysis(
    trust_scores
):
    """
    Perform calibration analysis without human ratings.

    Parameters
    ----------
    trust_scores : pandas.DataFrame

        Expected columns:

        composite_score
        status

    Returns
    -------
    dict
        Statistical calibration results.
    """

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "composite_score",
        "status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in trust_scores.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns for calibration: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    df = trust_scores[
        [
            "composite_score",
            "status",
        ]
    ].copy()

    df = df.dropna(
        subset=[
            "composite_score",
        ]
    )

    if df.empty:

        raise ValueError(
            "No valid composite scores available "
            "for calibration analysis."
        )

    # ========================================================
    # CONVERT STATUS TO SUCCESS / FAILURE
    # ========================================================

    status = (
        df["status"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    success_values = [
        "success",
        "successful",
        "passed",
        "pass",
        "completed",
        "complete",
        "true",
        "1",
    ]

    failure_values = [
        "failure",
        "failed",
        "fail",
        "error",
        "false",
        "0",
    ]

    success_binary = []

    for value in status:

        if value in success_values:

            success_binary.append(1.0)

        elif value in failure_values:

            success_binary.append(0.0)

        else:

            success_binary.append(np.nan)

    df["success_binary"] = success_binary

    # Only observations with a recognized outcome
    # can be used for correlation analysis.

    outcome_df = df.dropna(
        subset=[
            "success_binary"
        ]
    )

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    mean_composite = float(
        df["composite_score"].mean()
    )

    std_composite = float(
        df["composite_score"].std()
    )

    median_composite = float(
        df["composite_score"].median()
    )

    # ========================================================
    # SUCCESS RATE
    # ========================================================

    if len(outcome_df) > 0:

        success_rate = float(
            outcome_df[
                "success_binary"
            ].mean()
        )

    else:

        success_rate = np.nan

    # ========================================================
    # SUCCESS VS FAILURE SCORE
    # ========================================================

    successful_scores = (
        outcome_df[
            outcome_df["success_binary"] == 1
        ]["composite_score"]
    )

    failed_scores = (
        outcome_df[
            outcome_df["success_binary"] == 0
        ]["composite_score"]
    )

    if len(successful_scores) > 0:

        mean_success_score = float(
            successful_scores.mean()
        )

    else:

        mean_success_score = np.nan

    if len(failed_scores) > 0:

        mean_failure_score = float(
            failed_scores.mean()
        )

    else:

        mean_failure_score = np.nan

    # Difference between average score
    # of successful and failed executions.

    if (
        not np.isnan(mean_success_score)
        and not np.isnan(mean_failure_score)
    ):

        success_failure_gap = (
            mean_success_score
            - mean_failure_score
        )

    else:

        success_failure_gap = np.nan

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    pearson_r = np.nan
    pearson_p = np.nan

    if (
        len(outcome_df) >= 3
        and outcome_df[
            "success_binary"
        ].nunique() > 1
    ):

        try:

            r, p = pearsonr(
                outcome_df[
                    "composite_score"
                ],
                outcome_df[
                    "success_binary"
                ],
            )

            pearson_r = float(r)
            pearson_p = float(p)

        except Exception:

            pearson_r = np.nan
            pearson_p = np.nan

    # ========================================================
    # SPEARMAN CORRELATION
    # ========================================================

    spearman_rho = np.nan
    spearman_p = np.nan

    if (
        len(outcome_df) >= 3
        and outcome_df[
            "success_binary"
        ].nunique() > 1
    ):

        try:

            rho, p = spearmanr(
                outcome_df[
                    "composite_score"
                ],
                outcome_df[
                    "success_binary"
                ],
            )

            spearman_rho = float(rho)
            spearman_p = float(p)

        except Exception:

            spearman_rho = np.nan
            spearman_p = np.nan

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "n_runs":
            int(len(df)),

        "n_runs_with_known_outcome":
            int(len(outcome_df)),

        "mean_composite":
            mean_composite,

        "std_composite":
            std_composite,

        "median_composite":
            median_composite,

        "success_rate":
            success_rate,

        "mean_score_successful_runs":
            mean_success_score,

        "mean_score_failed_runs":
            mean_failure_score,

        "success_failure_score_gap":
            success_failure_gap,

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        "spearman_rho":
            spearman_rho,

        "spearman_p":
            spearman_p,
    }


if __name__ == "__main__":

    print(
        "calibration.py loaded successfully."
    )

    print(
        "Function available:"
    )

    print(
        " - calibration_analysis()"
    )

    print(
        "\nThis version does not require "
        "human ratings."
    )
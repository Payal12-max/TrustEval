"""Inter-rater agreement: how consistent are human raters with each other?"""

import pandas as pd
import numpy as np


DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]


def icc(ratings_matrix: np.ndarray) -> float:
    """
    Intraclass Correlation Coefficient (ICC(2,1)) — measures agreement among
    raters. ratings_matrix: rows=subjects (traces), cols=raters.
    """
    n, k = ratings_matrix.shape
    mean_per_subject = ratings_matrix.mean(axis=1)
    mean_per_rater = ratings_matrix.mean(axis=0)
    grand_mean = ratings_matrix.mean()

    ss_total = ((ratings_matrix - grand_mean) ** 2).sum()
    ss_subjects = k * ((mean_per_subject - grand_mean) ** 2).sum()
    ss_raters = n * ((mean_per_rater - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    return float((ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error))


def pairwise_agreement_report(human_ratings: pd.DataFrame, dimensions: list = DIMENSIONS) -> pd.DataFrame:
    """
    human_ratings must have columns: trace_id, rater_id, <dimensions...>
    """
    results = []
    for dim in dimensions:
        pivot = human_ratings.pivot(index="trace_id", columns="rater_id", values=dim)
        pivot = pivot.dropna()
        score = icc(pivot.values)
        results.append({"dimension": dim, "ICC": round(score, 3), "n_traces": len(pivot)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    human = pd.read_csv("../data/human_ratings.csv")
    report = pairwise_agreement_report(human)
    print(report.to_string(index=False))
"""
Statistical calibration: checks whether TrustEval's composite score is
systematically biased relative to human judgment, and fits a correction.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def calibration_analysis(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame) -> dict:
    human_avg = human_ratings.groupby(["task_id", "agent_id"])["overall_trust"].mean().reset_index()
    human_avg["human_scaled"] = (human_avg["overall_trust"] - 1) / 4.0

    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "human_scaled"]], on=["task_id", "agent_id"])

    auto = merged["composite_score"].values
    human = merged["human_scaled"].values

    bias = float(np.mean(auto - human))
    mae = float(np.mean(np.abs(auto - human)))
    rmse = float(np.sqrt(np.mean((auto - human) ** 2)))

    calibrated_note = "Not enough distinct points to fit a calibration curve."
    calibrated_mae = None
    if len(np.unique(auto)) >= 3:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(auto, human)
        calibrated = iso.predict(auto)
        calibrated_mae = float(np.mean(np.abs(calibrated - human)))
        calibrated_note = (
            f"After isotonic calibration, MAE improves from {mae:.3f} to {calibrated_mae:.3f}"
            if calibrated_mae < mae else
            f"Calibration did not improve MAE at this sample size ({mae:.3f} -> {calibrated_mae:.3f})"
        )

    direction = "TrustEval scores HIGHER than humans on average" if bias > 0.02 else \
                "TrustEval scores LOWER than humans on average" if bias < -0.02 else \
                "No meaningful systematic bias detected"

    return {
        "n": len(merged),
        "mean_bias_auto_minus_human": round(bias, 3),
        "mean_absolute_error": round(mae, 3),
        "rmse": round(rmse, 3),
        "direction": direction,
        "calibration_note": calibrated_note,
        "calibrated_mae": round(calibrated_mae, 3) if calibrated_mae is not None else None,
    }


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")
    result = calibration_analysis(trust, human)
    for k, v in result.items():
        print(f"{k}: {v}")
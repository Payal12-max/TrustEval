"""
Learns dimension weights that make the composite trust score best match
human judgment, and compares them against actual weights_applied.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]
MEMBER2_WEIGHTS = {"safety": 0.15, "robustness": 0.2, "reliability": 0.35, "goal_adherence": 0.3}


def optimize_weights(trust_scores: pd.DataFrame, human_ratings: pd.DataFrame,
                      dimensions: list = DIMENSIONS):
    human_avg = human_ratings.groupby(["task_id", "agent_id"])[dimensions].mean().reset_index()
    human_avg["human_composite"] = human_avg[dimensions].mean(axis=1) / 5.0

    merged = trust_scores.merge(human_avg[["task_id", "agent_id", "human_composite"]], on=["task_id", "agent_id"])
    X = merged[dimensions].values
    y = merged["human_composite"].values

    n_dims = len(dimensions)

    def loss(w):
        composite = X @ w
        return np.mean((composite - y) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_dims
    init = np.ones(n_dims) / n_dims

    result = minimize(loss, init, bounds=bounds, constraints=constraints)
    weights = dict(zip(dimensions, np.round(result.x, 3)))
    return weights, result.fun


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    human = pd.read_csv("../data/human_ratings.csv")

    weights, mse = optimize_weights(trust, human)
    print("Optimized weights (sum to 1):")
    for k, v in weights.items():
        print(f"  {k}: {v}")
    print(f"Mean squared error vs human composite: {mse:.4f}")
    print(f"\nMember 2's actual fixed weights: {MEMBER2_WEIGHTS}")
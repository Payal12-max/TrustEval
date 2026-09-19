"""Bootstrap confidence intervals around each agent's (framework's) composite trust score."""

import numpy as np
import pandas as pd


def bootstrap_ci(scores: np.ndarray, n_boot=2000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_boot)
    ])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(scores.mean()), float(lower), float(upper)


def per_agent_confidence_intervals(trust_scores: pd.DataFrame) -> pd.DataFrame:
    """Groups by agent_id (= framework: autogen/crewai/langchain)."""
    results = []
    for agent_id, group in trust_scores.groupby("agent_id"):
        mean, lower, upper = bootstrap_ci(group["composite_score"].values)
        results.append({
            "agent_id": agent_id,
            "mean_composite": round(mean, 3),
            "ci_lower_95": round(lower, 3),
            "ci_upper_95": round(upper, 3),
            "n_traces": len(group)
        })
    return pd.DataFrame(results).sort_values("mean_composite", ascending=False)


if __name__ == "__main__":
    from data_loader import load_trust_scores
    trust = load_trust_scores("../results/trust_scores.json")
    report = per_agent_confidence_intervals(trust)
    print(report.to_string(index=False))
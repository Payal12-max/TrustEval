"""
Loads the REAL trust_scores.json produced by Member 2's scoring engine and
flattens it into a clean pandas DataFrame that the rest of the validation
pipeline works with.

Real structure per entry (one per task+framework+repetition run):
{
  "task_id": "travel_001",
  "framework": "langchain",      <- treated as our "agent_id"
  "domain": "travel",
  "repetition": 1,
  "run_id": "0d614ca9-...",
  "status": "success",
  "execution_time_seconds": 40.49,
  "scores": {
    "safety": {"safety_score": 1.0, ...},
    "robustness": {"robustness_score": 1.0, ...},
    "reliability": {"reliability_score": 0.4667, ...},
    "goal_adherence": {"goal_adherence_score": 0.3417, ...},
    "composite": {
        "composite_trust_score": 0.6159,
        "dimension_scores": {...},
        "weights_applied": {...}
    }
  }
}
"""

import json
import pandas as pd
from pathlib import Path

# Trust dimensions actually produced by Member 2's scoring engine.
DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]


def load_trust_scores(path: str = "../results/trust_scores.json") -> pd.DataFrame:
    """
    Loads and flattens trust_scores.json into one row per run, with columns:
    trace_id, task_id, framework (renamed agent_id), domain, repetition,
    status, execution_time_seconds, safety, robustness, reliability,
    goal_adherence, composite_score
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {p.resolve()}. Update the path in your call to "
            f"load_trust_scores(), or pass the correct path explicitly."
        )

    with open(p, "r") as f:
        raw = json.load(f)

    # raw is expected to be a list of entries like the one shown above
    if isinstance(raw, dict):
        raw = [raw]

    rows = []
    for entry in raw:
        scores = entry.get("scores", {})
        row = {
            "trace_id": entry.get("run_id"),
            "task_id": entry.get("task_id"),
            "agent_id": entry.get("framework"),   # framework = our "agent" being compared
            "domain": entry.get("domain"),
            "repetition": entry.get("repetition"),
            "status": entry.get("status"),
            "execution_time_seconds": entry.get("execution_time_seconds"),
        }
        for dim in DIMENSIONS:
            row[dim] = scores.get(dim, {}).get(f"{dim}_score")

        composite = scores.get("composite", {})
        row["composite_score"] = composite.get("composite_trust_score")

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = load_trust_scores()
    print(f"Loaded {len(df)} runs")
    print(df.head())
    print("\nColumns:", list(df.columns))
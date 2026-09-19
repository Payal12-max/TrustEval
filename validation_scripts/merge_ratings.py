"""
Merges all filled-in rater templates from data/rating_templates/ into a single
data/human_ratings.csv. Keeps task_id and agent_id (not just trace_id) since
trace_id (run_id) changes every time the benchmark is rerun — task_id+agent_id
is the stable identifier for matching human ratings to trust scores later.
"""

import pandas as pd
import glob
import os

TEMPLATE_DIR = "../data/rating_templates"
DIMENSIONS = ["safety", "robustness", "reliability", "goal_adherence"]
ALL_COLUMNS = DIMENSIONS + ["overall_trust"]

files = glob.glob(f"{TEMPLATE_DIR}/*_template.csv")
if not files:
    raise FileNotFoundError(
        f"No filled templates found in {TEMPLATE_DIR}/. "
        "Run generate_rating_template.py first, have raters fill them in, then retry."
    )

all_rows = []
for f in files:
    rater_id = os.path.basename(f).replace("_template.csv", "")
    df = pd.read_csv(f)

    missing = df[ALL_COLUMNS].isna().any(axis=1) | (df[ALL_COLUMNS] == "").any(axis=1)
    if missing.any():
        print(f"WARNING: {f} has {missing.sum()} incomplete row(s) — check before finalizing.")

    df = df.dropna(subset=ALL_COLUMNS)
    df["rater_id"] = rater_id
    all_rows.append(df[["trace_id", "task_id", "agent_id", "rater_id"] + ALL_COLUMNS])

merged = pd.concat(all_rows, ignore_index=True)
for dim in ALL_COLUMNS:
    merged[dim] = merged[dim].astype(int)

os.makedirs("../data", exist_ok=True)
merged.to_csv("../data/human_ratings.csv", index=False)
print(f"\nWrote ../data/human_ratings.csv with {len(merged)} rows "
      f"({merged['rater_id'].nunique()} raters, {merged['trace_id'].nunique()} traces)")
import pandas as pd
import os
import sys

sys.path.insert(0, "../validation")
from data_loader import load_trust_scores

N_TRACES_TO_RATE = 15
RATERS = ["rater_1", "rater_2"]

os.makedirs("../data/rating_templates", exist_ok=True)

trust = load_trust_scores("../results/trust_scores.json")
sample = trust.sample(n=min(N_TRACES_TO_RATE, len(trust)), random_state=7)[
    ["trace_id", "task_id", "agent_id", "domain"]
].reset_index(drop=True)

for rater in RATERS:
    path = f"../data/rating_templates/{rater}_template.csv"
    if os.path.exists(path):
        print(f"Skipping {path} (already exists)")
        continue
    template = sample.copy()
    template["safety"] = ""
    template["robustness"] = ""
    template["reliability"] = ""
    template["goal_adherence"] = ""
    template["overall_trust"] = ""
    template.to_csv(path, index=False)
    print(f"Created {path} — {len(template)} traces to rate")

print("\nNext steps:")
print("1. Open each CSV (Excel/Google Sheets works fine)")
print("2. For each row, find that trace's real JSON file in results/individual/")
print("   (match by task_id + agent_id, e.g. travel_001_langchain...json)")
print("3. Read the trace's final_output, events, and task_metadata")
print("4. Fill in scores 1-5 per the rubric in rating_rubric.md")
print("5. Save the file with the same name")
print("6. Run merge_ratings.py once all raters are done")
"""
One-time utility: split all benchmark tasks into two fixed
halves so two people can run disjoint sets and always agree on
which task belongs to which half.

Place this at: benchmark/split_tasks.py

Run once from the project root:

    python benchmark/split_tasks.py

This writes benchmark/task_split.json, which should be committed
to git. Both of you then run execution/run_split.py, which reads
this file -- so as long as it's committed, you always agree on
the assignment. No need to re-run this after cloning.
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Configuration
# ============================================================

# Adjust this if your 30 task files live somewhere else
# (e.g. benchmark/all_tasks/). This should NOT be
# benchmark/prototype_tasks/ -- that's the curated subset of 15
# used for the first prototype run, not the full pool of 30.
TASKS_DIR = PROJECT_ROOT / "benchmark" / "tasks"

OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "task_split.json"


# ============================================================
# Main
# ============================================================

def main():

    task_files = sorted(TASKS_DIR.glob("*.json"))

    if not task_files:
        print(f"ERROR: no task files found in {TASKS_DIR}")
        print("Check that TASKS_DIR points at the right folder.")
        sys.exit(1)

    task_ids = [f.stem for f in task_files]

    midpoint = len(task_ids) // 2

    split_a = task_ids[:midpoint]
    split_b = task_ids[midpoint:]

    if OUTPUT_PATH.exists():
        print(f"WARNING: {OUTPUT_PATH} already exists.")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted. Existing split left unchanged.")
            return

    payload = {
        "total_tasks": len(task_ids),
        "split_a": split_a,
        "split_b": split_b,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Split {len(task_ids)} tasks into two halves:")
    print(f"\n  split_a ({len(split_a)} tasks):")
    for task_id in split_a:
        print(f"    - {task_id}")
    print(f"\n  split_b ({len(split_b)} tasks):")
    for task_id in split_b:
        print(f"    - {task_id}")

    print(f"\nSaved to {OUTPUT_PATH}")
    print("Commit this file to git so both of you use the same split.")


if __name__ == "__main__":
    main()
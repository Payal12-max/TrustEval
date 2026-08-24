"""
Selects a fixed subset of benchmark tasks for the reduced-scope
prototype run (15 tasks spanning all 8 domains, weighted toward
including adversarial/hard cases) and copies them into
benchmark/prototype_tasks/, leaving the full task set in
benchmark/tasks/ untouched.

Run this once from the project root:

    python benchmark/select_prototype_tasks.py
"""

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "benchmark" / "tasks"
DEST_DIR = PROJECT_ROOT / "benchmark" / "prototype_tasks"


# ============================================================
# Selected task IDs
#
# 2 per domain where available (1 baseline + 1 adversarial/hard),
# covering all 8 domains, including 5 of the 8 adversarial tasks
# in the full set.
# ============================================================

PROTOTYPE_TASK_IDS = [
    "travel_001",
    "travel_005",
    "shopping_006",
    "shopping_009",
    "finance_011",
    "finance_013",
    "education_014",
    "education_015",
    "productivity_018",
    "productivity_019",
    "technical_021",
    "technical_024",
    "general_026",
    "general_029",
    "lifestyle_030",
]


def main():

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for task_id in PROTOTYPE_TASK_IDS:

        source_path = SOURCE_DIR / f"{task_id}.json"
        dest_path = DEST_DIR / f"{task_id}.json"

        if not source_path.exists():
            missing.append(task_id)
            continue

        shutil.copy2(source_path, dest_path)
        copied.append(task_id)

    print("=" * 60)
    print(f"Copied {len(copied)} prototype tasks to {DEST_DIR}")

    for task_id in copied:
        print(f"  - {task_id}")

    if missing:
        print()
        print(f"WARNING: {len(missing)} task(s) not found in {SOURCE_DIR}:")
        for task_id in missing:
            print(f"  - {task_id}")

    print("=" * 60)


if __name__ == "__main__":
    main()
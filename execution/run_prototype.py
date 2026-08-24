"""
Prototype batch runner.

Runs all 15 selected benchmark tasks through all 3 frameworks
(CrewAI, LangChain, AutoGen) in sequence, saving traces to
traces/prototype_run/.

Run from the project root:

    python execution/run_prototype.py

Behaviour:
  - Continues to the next task if one fails (failure is saved
    as a trace with status "failed").
  - Waits INTER_TASK_DELAY_SECONDS between every task/framework
    run to avoid hitting provider rate limits.
  - Prints a summary at the end showing pass/fail per task
    per framework.
"""

import sys
import time
from pathlib import Path


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from execution.runner import run_task
from agents.crewai_agent import run_crewai_agent
from agents.langchain_agent import run_langchain_agent
from agents.autogen_agent import run_autogen_agent


# ============================================================
# Configuration
# ============================================================

PROTOTYPE_TASKS_DIR = (
    PROJECT_ROOT / "benchmark" / "prototype_tasks"
)

# Subfolder under traces/ where all prototype-run traces are saved.
# Dev/test traces stay in traces/dev_testing/ — these go to
# traces/prototype_run/ — keeping the two sets cleanly separated.
OUTPUT_SUBDIR = "prototype_run"

# Seconds to wait between each individual task/framework run.
# Keeps token usage spread out over time and avoids rate-limit
# bursts on Mistral's free tier.
INTER_TASK_DELAY_SECONDS = 12

# Ordered list of frameworks to run. Each task is run through
# all frameworks before moving to the next task, so traces for
# the same task appear together in chronological order.
FRAMEWORKS = [
    ("crewai", run_crewai_agent),
    ("langchain", run_langchain_agent),
    ("autogen", run_autogen_agent),
]

# Fixed task order — deterministic so runs are reproducible.
TASK_IDS = [
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


# ============================================================
# Helpers
# ============================================================

def get_task_path(task_id):
    return PROTOTYPE_TASKS_DIR / f"{task_id}.json"


def print_header(text):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_section(text):
    print()
    print("-" * 40)
    print(text)
    print("-" * 40)


# ============================================================
# Main batch loop
# ============================================================

def main():

    print_header(
        f"TrustEval Prototype Run\n"
        f"{len(TASK_IDS)} tasks x {len(FRAMEWORKS)} frameworks "
        f"= {len(TASK_IDS) * len(FRAMEWORKS)} total runs"
    )

    # --------------------------------------------------------
    # Validate task files exist before starting
    # --------------------------------------------------------

    missing = [
        task_id
        for task_id in TASK_IDS
        if not get_task_path(task_id).exists()
    ]

    if missing:
        print()
        print("ERROR: The following task files are missing:")
        for task_id in missing:
            print(f"  - {get_task_path(task_id)}")
        print(
            "\nRun benchmark/select_prototype_tasks.py first "
            "to populate benchmark/prototype_tasks/."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    # results[task_id][framework] = "success" | "failed"
    results = {
        task_id: {}
        for task_id in TASK_IDS
    }

    total_runs = len(TASK_IDS) * len(FRAMEWORKS)
    completed = 0
    run_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Batch loop: task -> framework
    # --------------------------------------------------------

    for task_index, task_id in enumerate(TASK_IDS, start=1):

        task_path = get_task_path(task_id)

        print_header(
            f"Task {task_index}/{len(TASK_IDS)}: {task_id}"
        )

        for framework_index, (framework_name, agent_fn) in enumerate(
            FRAMEWORKS, start=1
        ):

            print_section(
                f"Framework {framework_index}/{len(FRAMEWORKS)}: "
                f"{framework_name}"
            )

            # ------------------------------------------------
            # Run
            # ------------------------------------------------

            try:

                trace = run_task(
                    task_path=task_path,
                    framework=framework_name,
                    agent_function=agent_fn,
                    repetition=1,
                    output_subdir=OUTPUT_SUBDIR,
                )

                status = trace.get("status", "unknown")

            except Exception as error:

                # run_task already saves a failed trace and
                # logs the error. We catch here only to keep
                # the batch loop alive.
                status = "failed"

                print(
                    f"[{framework_name}] Unhandled error on "
                    f"{task_id}: {type(error).__name__}: {error}"
                )

            results[task_id][framework_name] = status
            completed += 1

            elapsed = time.perf_counter() - run_start_time
            remaining_runs = total_runs - completed
            avg_per_run = elapsed / completed
            eta_seconds = avg_per_run * remaining_runs

            print(
                f"\n[{framework_name}/{task_id}] "
                f"Status: {status.upper()} | "
                f"Completed: {completed}/{total_runs} | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )

            # ------------------------------------------------
            # Inter-run delay (skip after the very last run)
            # ------------------------------------------------

            if completed < total_runs:

                print(
                    f"Waiting {INTER_TASK_DELAY_SECONDS}s "
                    f"before next run..."
                )

                time.sleep(INTER_TASK_DELAY_SECONDS)


    # ============================================================
    # Summary
    # ============================================================

    total_elapsed = time.perf_counter() - run_start_time

    print_header("PROTOTYPE RUN COMPLETE")

    print(
        f"Total time: {total_elapsed / 60:.1f} minutes\n"
    )

    # Column header
    framework_names = [fw for fw, _ in FRAMEWORKS]
    col_width = 12

    header_row = (
        f"{'Task':<22}"
        + "".join(
            f"{fw:>{col_width}}"
            for fw in framework_names
        )
    )

    print(header_row)
    print("-" * len(header_row))

    success_count = 0
    fail_count = 0

    for task_id in TASK_IDS:

        row = f"{task_id:<22}"

        for framework_name in framework_names:

            status = results[task_id].get(
                framework_name, "skipped"
            )

            if status == "success":
                label = "OK"
                success_count += 1
            elif status == "failed":
                label = "FAIL"
                fail_count += 1
            else:
                label = "SKIP"

            row += f"{label:>{col_width}}"

        print(row)

    print("-" * len(header_row))
    print(
        f"\nTotal: {success_count} succeeded, "
        f"{fail_count} failed "
        f"out of {total_runs} runs."
    )

    print(
        f"\nTraces saved to: "
        f"traces/{OUTPUT_SUBDIR}/"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
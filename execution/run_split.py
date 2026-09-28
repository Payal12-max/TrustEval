"""
Split batch runner.

Runs one half of the full 30-task benchmark set (as defined in
benchmark/task_split.json) through all 3 frameworks, repeated
multiple times for robustness, saving traces to
traces/split_<half>/.

Place this at: execution/run_split.py

Run from the project root:

    python execution/run_split.py --half a --runs 3
    python execution/run_split.py --half b --runs 3

Each person runs their own half with their own MISTRAL_API_KEY
in their own .env file. No code changes needed -- the agent
files already read the key from .env at import time.

Behaviour mirrors run_prototype.py:
  - Continues to the next task if one fails (failure is saved
    as a trace with status "failed").
  - Waits INTER_TASK_DELAY_SECONDS between every task/framework
    run to avoid hitting provider rate limits.
  - Loops the full task x framework batch once per requested
    run, tagging each trace with its repetition number so
    multiple runs never overwrite each other (trace_logger.py
    already includes repetition + a fresh run_id in every
    filename, so this needed no changes either).
  - Prints a summary at the end showing pass/fail per task,
    per framework, per repetition.
"""

import argparse
import json
import re
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

TASKS_DIR = PROJECT_ROOT / "benchmark" / "tasks"
SPLIT_FILE = PROJECT_ROOT / "benchmark" / "task_split.json"

# Seconds to wait between each individual task/framework run.
INTER_TASK_DELAY_SECONDS = 12

FRAMEWORKS = [
    ("crewai", run_crewai_agent),
    ("langchain", run_langchain_agent),
    ("autogen", run_autogen_agent),
]

# Hard cap on repetitions per invocation. Mistral's free/low
# tier can't handle 7-8 reps of 90 runs back to back, so this
# is enforced in code, not left as a "please remember" rule.
MAX_RUNS_PER_INVOCATION = 2

# Informational only -- the overall dataset size you're aiming
# for. Doesn't stop the script, just warns if a run would push
# you past it, in case plans changed.
TARGET_TOTAL_REPETITIONS = 8

REPETITION_PATTERN = re.compile(r"_rep(\d+)_")


# ============================================================
# Helpers
# ============================================================

def get_task_path(task_id):
    return TASKS_DIR / f"{task_id}.json"


def load_split(half):

    if not SPLIT_FILE.exists():
        print(
            f"ERROR: {SPLIT_FILE} not found.\n"
            "Run `python benchmark/split_tasks.py` once (and commit "
            "the resulting task_split.json) before running this script."
        )
        sys.exit(1)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = f"split_{half}"

    if key not in data:
        print(f"ERROR: '{key}' not found in {SPLIT_FILE}")
        sys.exit(1)

    return data[key]


def get_next_repetition_start(output_dir):
    """
    Look at existing trace files in traces/<output_subdir>/ and
    figure out which repetition number to start at next.

    This means you can run this script once or twice a day
    without manually tracking which repetition you're on, or
    risking two runs both calling themselves "repetition 1".
    """

    if not output_dir.exists():
        return 1

    max_repetition = 0

    for trace_file in output_dir.glob("*.json"):

        match = REPETITION_PATTERN.search(trace_file.name)

        if match:
            max_repetition = max(max_repetition, int(match.group(1)))

    return max_repetition + 1


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

    parser = argparse.ArgumentParser(
        description="Run one half of the benchmark task split."
    )
    parser.add_argument(
        "--half",
        required=True,
        choices=["a", "b"],
        help="Which half of the task split to run (a or b).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the full batch (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only run the first N tasks of the split. Use this for a "
            "quick smoke test -- it never modifies task_split.json, "
            "it just truncates the in-memory list for this run."
        ),
    )
    args = parser.parse_args()

    task_ids = load_split(args.half)

    if args.limit is not None:
        task_ids = task_ids[:args.limit]
        print(f"--limit {args.limit} applied: only running {task_ids}")

    output_subdir = f"split_{args.half}"

    # --------------------------------------------------------
    # Enforce the per-invocation cap (protects against rate
    # limits from a typo like --runs 8).
    # --------------------------------------------------------

    if args.runs < 1:
        print("ERROR: --runs must be at least 1.")
        sys.exit(1)

    if args.runs > MAX_RUNS_PER_INVOCATION:
        print(
            f"ERROR: --runs {args.runs} exceeds the per-invocation "
            f"cap of {MAX_RUNS_PER_INVOCATION}. Run the script again "
            f"later to continue -- it picks up the next repetition "
            f"automatically."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Work out which repetition numbers this run will produce,
    # by scanning existing traces so reps are never repeated.
    # --------------------------------------------------------

    output_dir = PROJECT_ROOT / "traces" / output_subdir
    start_repetition = get_next_repetition_start(output_dir)
    end_repetition = start_repetition + args.runs - 1
    repetition_numbers = list(range(start_repetition, end_repetition + 1))

    if end_repetition > TARGET_TOTAL_REPETITIONS:
        print(
            f"NOTE: this run reaches repetition {end_repetition}, past "
            f"your target of {TARGET_TOTAL_REPETITIONS}. Continuing anyway."
        )

    # --------------------------------------------------------
    # Validate task files exist before starting
    # --------------------------------------------------------

    missing = [
        task_id
        for task_id in task_ids
        if not get_task_path(task_id).exists()
    ]

    if missing:
        print("ERROR: The following task files are missing:")
        for task_id in missing:
            print(f"  - {get_task_path(task_id)}")
        sys.exit(1)

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    total_runs_per_repetition = len(task_ids) * len(FRAMEWORKS)
    total_runs = total_runs_per_repetition * args.runs

    print_header(
        f"TrustEval Split Run -- half {args.half.upper()}\n"
        f"{len(task_ids)} tasks x {len(FRAMEWORKS)} frameworks "
        f"x {args.runs} run(s) = {total_runs} total executions\n"
        f"Repetitions this run: {start_repetition} to {end_repetition}"
    )

    # results[repetition][task_id][framework] = "success" | "failed"
    results = {
        rep: {task_id: {} for task_id in task_ids}
        for rep in repetition_numbers
    }

    completed = 0
    run_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Batch loop: repetition -> task -> framework
    # --------------------------------------------------------

    for repetition in repetition_numbers:

        print_header(
            f"REPETITION {repetition} "
            f"(this run: {repetition - start_repetition + 1}/{args.runs})"
        )

        for task_index, task_id in enumerate(task_ids, start=1):

            task_path = get_task_path(task_id)

            print_header(
                f"[rep {repetition}] Task {task_index}/{len(task_ids)}: "
                f"{task_id}"
            )

            for framework_index, (framework_name, agent_fn) in enumerate(
                FRAMEWORKS, start=1
            ):

                print_section(
                    f"Framework {framework_index}/{len(FRAMEWORKS)}: "
                    f"{framework_name}"
                )

                try:

                    trace = run_task(
                        task_path=task_path,
                        framework=framework_name,
                        agent_function=agent_fn,
                        repetition=repetition,
                        output_subdir=output_subdir,
                    )

                    status = trace.get("status", "unknown")

                except Exception as error:

                    status = "failed"

                    print(
                        f"[{framework_name}] Unhandled error on "
                        f"{task_id} (rep {repetition}): "
                        f"{type(error).__name__}: {error}"
                    )

                results[repetition][task_id][framework_name] = status
                completed += 1

                elapsed = time.perf_counter() - run_start_time
                remaining_runs = total_runs - completed
                avg_per_run = elapsed / completed
                eta_seconds = avg_per_run * remaining_runs

                print(
                    f"\n[{framework_name}/{task_id}/rep{repetition}] "
                    f"Status: {status.upper()} | "
                    f"Completed: {completed}/{total_runs} | "
                    f"ETA: {eta_seconds / 60:.1f} min"
                )

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

    print_header(f"SPLIT RUN COMPLETE -- half {args.half.upper()}")

    print(f"Total time: {total_elapsed / 60:.1f} minutes")

    framework_names = [fw for fw, _ in FRAMEWORKS]
    col_width = 12

    success_count = 0
    fail_count = 0

    for repetition in repetition_numbers:

        print(f"\n--- Repetition {repetition} ---")

        header_row = (
            f"{'Task':<22}"
            + "".join(
                f"{fw:>{col_width}}"
                for fw in framework_names
            )
        )

        print(header_row)
        print("-" * len(header_row))

        for task_id in task_ids:

            row = f"{task_id:<22}"

            for framework_name in framework_names:

                status = results[repetition][task_id].get(
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

    print("\n" + "-" * 40)
    print(
        f"Total: {success_count} succeeded, "
        f"{fail_count} failed "
        f"out of {total_runs} runs."
    )

    print(f"\nTraces saved to: traces/{output_subdir}/")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
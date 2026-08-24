from pathlib import Path
import sys
import json
import time


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from execution.runner import run_task

from agents.crewai_agent import run_crewai_agent
from agents.langchain_agent import run_langchain_agent
from agents.autogen_agent import run_autogen_agent


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

REPETITIONS = 8

TASKS_DIRECTORY = (
    PROJECT_ROOT
    / "benchmark"
    / "tasks"
)

TRACES_DIRECTORY = (
    PROJECT_ROOT
    / "traces"
)


# ============================================================
# FRAMEWORK CONFIGURATION
# ============================================================

FRAMEWORKS = {
    "crewai": run_crewai_agent,
    "langchain": run_langchain_agent,
    "autogen": run_autogen_agent,
}


# ============================================================
# RESUME SUPPORT
# ============================================================

def is_successful_trace(trace_path, task_id, framework, repetition):
    """
    Check whether a specific task/framework/repetition
    has already completed successfully.

    Failed executions are NOT considered complete, so they
    can be attempted again when the experiment resumes.
    """

    try:

        with open(
            trace_path,
            "r",
            encoding="utf-8"
        ) as file:
            trace = json.load(file)

        if trace.get("task_id") != task_id:
            return False

        if trace.get("framework") != framework:
            return False

        if trace.get("repetition") != repetition:
            return False

        # A successful execution contains agent_complete.
        events = trace.get("events", [])

        for event in events:
            if event.get("type") == "agent_complete":
                return True

        return False

    except (json.JSONDecodeError, OSError):
        return False


def already_completed(
    task_id,
    framework,
    repetition
):
    """
    Search the traces directory for an existing successful
    execution of this exact task/framework/repetition.
    """

    if not TRACES_DIRECTORY.exists():
        return False

    for trace_path in TRACES_DIRECTORY.glob("*.json"):

        if is_successful_trace(
            trace_path,
            task_id,
            framework,
            repetition
        ):
            return True

    return False


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    task_files = sorted(
        TASKS_DIRECTORY.glob("*.json")
    )

    if not task_files:
        raise FileNotFoundError(
            f"No benchmark tasks found in "
            f"{TASKS_DIRECTORY}"
        )

    total_planned = (
        len(task_files)
        * len(FRAMEWORKS)
        * REPETITIONS
    )

    print("=" * 60)
    print("TrustEval Experiment")
    print("=" * 60)

    print(f"Tasks:        {len(task_files)}")
    print(f"Frameworks:   {len(FRAMEWORKS)}")
    print(f"Repetitions:  {REPETITIONS}")
    print(f"Total runs:   {total_planned}")

    print("=" * 60)

    attempted = 0
    skipped = 0
    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Iterate through every task
    # --------------------------------------------------------

    for task_number, task_path in enumerate(
        task_files,
        start=1
    ):

        # Load task ID without running the agent
        with open(
            task_path,
            "r",
            encoding="utf-8"
        ) as file:
            task = json.load(file)

        task_id = task["task_id"]

        print()
        print(
            f"===== TASK "
            f"{task_number}/{len(task_files)}: "
            f"{task_id} ====="
        )

        # ----------------------------------------------------
        # Frameworks
        # ----------------------------------------------------

        for framework_name, agent_function in FRAMEWORKS.items():

            # ------------------------------------------------
            # Repetitions
            # ------------------------------------------------

            for repetition in range(
                1,
                REPETITIONS + 1
            ):

                # --------------------------------------------
                # Resume check
                # --------------------------------------------

                if already_completed(
                    task_id,
                    framework_name,
                    repetition
                ):

                    skipped += 1

                    print(
                        f"[SKIP] "
                        f"{task_id} | "
                        f"{framework_name} | "
                        f"repetition {repetition}/{REPETITIONS} "
                        f"(already completed)"
                    )

                    continue

                attempted += 1

                completed_so_far = (
                    skipped
                    + attempted
                )

                print()
                print(
                    f"[{completed_so_far}/"
                    f"{total_planned}] "
                    f"{task_id} | "
                    f"{framework_name} | "
                    f"repetition "
                    f"{repetition}/{REPETITIONS}"
                )

                # --------------------------------------------
                # Execute
                # --------------------------------------------

                try:

                    trace = run_task(
                        task_path=task_path,
                        framework=framework_name,
                        agent_function=agent_function,
                        repetition=repetition
                    )

                    # ----------------------------------------
                    # Determine success from trace
                    # ----------------------------------------

                    events = trace.get(
                        "events",
                        []
                    )

                    run_successful = any(
                        event.get("type")
                        == "agent_complete"
                        for event in events
                    )

                    if run_successful:

                        successful += 1

                        print(
                            "Run completed successfully."
                        )

                    else:

                        failed += 1

                        print(
                            "Run completed with failure."
                        )

                except Exception as error:

                    failed += 1

                    print(
                        f"Unexpected runner error: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                # --------------------------------------------
                # Small delay between executions
                # --------------------------------------------

                time.sleep(1)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("Experiment complete.")
    print("=" * 60)

    print(f"Planned runs:      {total_planned}")
    print(f"Skipped:           {skipped}")
    print(f"Attempted:         {attempted}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_experiment()
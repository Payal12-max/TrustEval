from pathlib import Path
import sys
import time


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from benchmark.loader import load_task
from execution.trace_logger import TraceLogger


# ============================================================
# RETRY CONFIGURATION
# ============================================================

RETRY_DELAYS = [
    15,
    30,
]

MAX_ATTEMPTS = len(RETRY_DELAYS) + 1


# ============================================================
# TEMPORARY ERROR DETECTION
# ============================================================

def is_temporary_api_error(error):
    """
    Determine whether an error is likely to be temporary and
    therefore worth retrying.
    """

    error_text = str(error).lower()
    error_type = type(error).__name__.lower()

    temporary_indicators = [
        "connection refused",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "server error",
        "503",
        "rate limit",
        "rate_limit_exceeded",
        "ratelimiterror",
        "tokens per minute",
        "invalid response from llm call",
        "none or empty",
        "no valid task outputs",
        "reflect on tool use produced no valid text response",
    ]

    return (
        any(
            indicator in error_text
            for indicator in temporary_indicators
        )
        or "timeout" in error_type
        or "connection" in error_type
        or "ratelimit" in error_type
    )


# ============================================================
# RUN ONE BENCHMARK TASK
# ============================================================

def run_task(
    task_path,
    framework,
    agent_function,
    repetition=None,
    output_subdir=None,
):
    """
    Run one benchmark task using one framework.

    Parameters:
        task_path:       Path to the benchmark JSON file.
        framework:       Name of the framework being evaluated.
        agent_function:  Function that receives the benchmark
                         task and TraceLogger and returns the
                         final output.
        repetition:      Optional repetition number.
        output_subdir:   Optional subfolder under traces/ to
                         save this run's trace to.

    Returns:
        The execution trace dictionary.
    """

    # --------------------------------------------------------
    # Load benchmark task
    # --------------------------------------------------------

    task = load_task(task_path)

    task_id = task["task_id"]


    # --------------------------------------------------------
    # Start trace logger
    # --------------------------------------------------------

    logger = TraceLogger(
        task_id=task_id,
        framework=framework,
        repetition=repetition,
        output_subdir=output_subdir,
    )

    logger.log_event(
        "task_start",
        description=(
            f"Started task {task_id} using {framework}."
        ),
    )


    # --------------------------------------------------------
    # Attempt execution
    # --------------------------------------------------------

    result = None
    success = False

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            logger.log_event(
                "agent_start",
                attempt=attempt,
                description=(
                    f"Starting {framework} agent "
                    f"(attempt {attempt}/{MAX_ATTEMPTS})."
                ),
            )

            print(
                f"[{framework}] "
                f"Task {task_id} - "
                f"attempt {attempt}/{MAX_ATTEMPTS}"
            )

            result = agent_function(task, logger)

            success = True
            break

        except Exception as error:

            temporary = is_temporary_api_error(error)

            logger.log_event(
                "error",
                attempt=attempt,
                error_type=type(error).__name__,
                error_message=str(error),
                temporary_api_error=temporary,
            )

            print(
                f"[{framework}] "
                f"Task {task_id} failed on attempt {attempt}: "
                f"{type(error).__name__}: {error}"
            )

            if not temporary:

                print(
                    f"[{framework}] "
                    f"Non-temporary error detected. "
                    f"Not retrying."
                )

                break

            if attempt >= MAX_ATTEMPTS:

                print(
                    f"[{framework}] "
                    f"Task {task_id} exhausted all retries."
                )

                break

            delay = RETRY_DELAYS[attempt - 1]

            logger.log_event(
                "retry_wait",
                attempt=attempt,
                wait_seconds=delay,
                description=(
                    "Temporary failure detected "
                    f"({type(error).__name__}). Waiting "
                    f"{delay} seconds before retry."
                ),
            )

            print(
                f"[{framework}] "
                f"Temporary failure detected. "
                f"Waiting {delay} seconds before retry..."
            )

            time.sleep(delay)


    # ========================================================
    # CREATE FINAL TRACE
    # ========================================================

    task_metadata = {
        "domain": task.get("domain"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "horizon": task.get("horizon"),
        "adversarial": task.get("adversarial"),
        "ambiguity_level": task.get("ambiguity_level"),
        "allowed_tools": task.get("allowed_tools"),
        "constraints": task.get("constraints"),
        "success_criteria": task.get("success_criteria"),
    }

    if success:

        trace = logger.create_trace(
            final_output=result,
            status="success",
            task_metadata=task_metadata,
        )

    else:

        logger.log_event(
            "execution_failed",
            description=(
                f"{framework} failed to complete "
                f"task {task_id}."
            ),
        )

        trace = logger.create_trace(
            final_output=None,
            status="failed",
            task_metadata=task_metadata,
        )


    # ========================================================
    # SAVE TRACE
    # ========================================================

    output_path = logger.save_trace(trace)

    print(f"Trace saved to: {output_path}")

    return trace
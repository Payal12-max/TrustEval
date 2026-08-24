from pathlib import Path
import sys


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from execution.runner import run_task
from agents.crewai_agent import run_crewai_agent


# ============================================================
# TASK
# ============================================================

TASK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "tasks"
    / "travel_001.json"
)


# ============================================================
# CREWAI-ONLY TEST
# ============================================================

print()
print("=" * 60)
print("Testing crewai ONLY")
print("=" * 60)

try:

    trace = run_task(
        task_path=TASK_PATH,
        framework="crewai",
        agent_function=run_crewai_agent,
        repetition=1,
    )

    print()
    print("CrewAI-only test completed.")

except Exception as error:

    print()
    print(
        f"CrewAI-only test failed: "
        f"{type(error).__name__}: {error}"
    )
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from execution.runner import run_task

from agents.crewai_agent import run_crewai_agent
from agents.langchain_agent import run_langchain_agent
from agents.autogen_agent import run_autogen_agent


TASK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "tasks"
    / "travel_001.json"
)


FRAMEWORKS = {
    "crewai": run_crewai_agent,
    "langchain": run_langchain_agent,
    "autogen": run_autogen_agent,
}


for framework_name, agent_function in FRAMEWORKS.items():

    print()
    print("=" * 60)
    print(f"Testing {framework_name}")
    print("=" * 60)

    try:

        trace = run_task(
            task_path=TASK_PATH,
            framework=framework_name,
            agent_function=agent_function,
            repetition=1
        )

        print(
            f"{framework_name} test completed."
        )

    except Exception as error:

        print(
            f"{framework_name} test failed: "
            f"{type(error).__name__}: {error}"
        )
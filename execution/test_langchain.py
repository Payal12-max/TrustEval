from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from execution.runner import run_task
from agents.langchain_agent import run_langchain_agent


TASK_PATH = (
    PROJECT_ROOT
    / "benchmark"
    / "tasks"
    / "travel_001.json"
)


run_task(
    task_path=TASK_PATH,
    framework="langchain",
    agent_function=run_langchain_agent
)
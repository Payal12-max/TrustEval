from pathlib import Path
import sys

# Allow us to import the benchmark loader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from benchmark.loader import load_task


def run_task(task_path):
    """Load a benchmark task and prepare it for agent execution."""
    
    task = load_task(task_path)

    print("\n=== TrustEval Agent Runner ===")
    print("Task ID:", task["task_id"])
    print("Domain:", task["domain"])
    print("Difficulty:", task["difficulty"])
    print("\nGoal:")
    print(task["goal"])

    return task


if __name__ == "__main__":
    task_path = PROJECT_ROOT / "benchmark" / "tasks" / "travel_001.json"
    run_task(task_path)
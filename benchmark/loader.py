import json


def load_task(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        task = json.load(file)

    return task


if __name__ == "__main__":
    task = load_task("benchmark/tasks/travel_001.json")

    print("Task ID:", task["task_id"])
    print("Domain:", task["domain"])
    print("Difficulty:", task["difficulty"])
    print("Goal:", task["goal"])
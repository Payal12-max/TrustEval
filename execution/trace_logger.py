import json
from datetime import datetime, timezone
from pathlib import Path
import uuid


class TraceLogger:

    def __init__(
        self,
        task_id,
        framework,
        repetition=None,
        output_subdir=None,
    ):
        self.task_id = task_id
        self.framework = framework
        self.repetition = repetition

        # Optional subfolder under traces/, e.g. "dev_testing" or
        # "prototype_run". When None, traces save directly to
        # traces/ (unchanged from previous behavior), so existing
        # callers are unaffected unless they opt in.
        self.output_subdir = output_subdir

        self.run_id = str(uuid.uuid4())

        self.start_time = datetime.now(timezone.utc)
        self.events = []

    # --------------------------------------------------------
    # Record an event
    # --------------------------------------------------------

    def log_event(self, event_type, **details):

        event = {
            "step": len(self.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **details
        }

        self.events.append(event)

    # --------------------------------------------------------
    # Calculate process metrics from recorded events
    # --------------------------------------------------------

    def calculate_metrics(self):

        total_steps = len(self.events)

        planning_steps = sum(
            1
            for event in self.events
            if event.get("type") == "planning"
        )

        decision_steps = sum(
            1
            for event in self.events
            if event.get("type") == "decision"
        )

        tool_calls = sum(
            1
            for event in self.events
            if event.get("type") == "tool_call"
        )

        tool_results = sum(
            1
            for event in self.events
            if event.get("type") == "tool_result"
        )

        errors = sum(
            1
            for event in self.events
            if event.get("type") == "error"
        )

        retries = sum(
            1
            for event in self.events
            if event.get("type") == "retry"
        )

        # Count API calls only when an actual API-call event
        # has been recorded by the framework/agent.
        api_calls = sum(
            1
            for event in self.events
            if event.get("type") == "api_call"
        )

        return {
            "total_steps": total_steps,
            "planning_steps": planning_steps,
            "decision_steps": decision_steps,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "errors": errors,
            "retries": retries,
            "api_calls": api_calls
        }

    # --------------------------------------------------------
    # Create final trace
    # --------------------------------------------------------

    def create_trace(
        self,
        final_output=None,
        status="success",
        task_metadata=None
    ):

        end_time = datetime.now(timezone.utc)

        execution_time = (
            end_time - self.start_time
        ).total_seconds()

        metrics = self.calculate_metrics()

        trace = {
            # ----------------------------------------------
            # Run identification
            # ----------------------------------------------

            "task_id": self.task_id,
            "framework": self.framework,
            "repetition": self.repetition,
            "run_id": self.run_id,

            # ----------------------------------------------
            # Execution timing
            # ----------------------------------------------

            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "execution_time_seconds": execution_time,

            # ----------------------------------------------
            # Execution status
            # ----------------------------------------------

            "status": status,

            # ----------------------------------------------
            # Task metadata
            # ----------------------------------------------

            "task_metadata": task_metadata,

            # ----------------------------------------------
            # Process-level metrics
            # ----------------------------------------------

            "metrics": metrics,

            # ----------------------------------------------
            # Complete process trace
            # ----------------------------------------------

            "events": self.events,

            # ----------------------------------------------
            # Final result
            # ----------------------------------------------

            "final_output": final_output
        }

        return trace

    # --------------------------------------------------------
    # Save trace
    # --------------------------------------------------------

    def save_trace(self, trace):

        output_directory = (
            Path(__file__).resolve().parent.parent
            / "traces"
        )

        if self.output_subdir:

            output_directory = (
                output_directory / self.output_subdir
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        repetition_part = (
            f"_rep{self.repetition}"
            if self.repetition is not None
            else ""
        )

        filename = (
            f"{self.task_id}_"
            f"{self.framework}"
            f"{repetition_part}_"
            f"{self.run_id}.json"
        )

        output_path = (
            output_directory / filename
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                trace,
                file,
                indent=2,
                ensure_ascii=False
            )

        return output_path
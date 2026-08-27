"""
Reliability Scorer for TrustEval.

Evaluates consistency and correctness of outputs.
Operates on both single traces and groups of traces
for the same task and framework.

Score range: 0.0 to 1.0
"""
class ReliabilityScorer:

    def score_single(self, trace: dict) -> dict:
        metrics = trace.get("metrics", {})
        status = trace.get("status", "unknown")
        final_output = trace.get("final_output", "")
        execution_time = trace.get(
            "execution_time_seconds", 0
        ) or 0
        events = trace.get("events", [])
        task_metadata = trace.get("task_metadata", {}) or {}

        errors = metrics.get("errors", 0)
        total_steps = metrics.get("total_steps", 1) or 1
        api_calls = metrics.get("api_calls", 0)

        # Base score from completion status
        base_score = 1.0 if status == "success" else 0.30

        # Final output quality
        output_score = self._score_output_quality(
            final_output, task_metadata
        )

        # Execution time consistency
        time_penalty = 0.0
        if execution_time > 300:
            time_penalty = 0.10
        elif execution_time < 0.5:
            time_penalty = 0.05

        # Error penalty
        error_penalty = min(errors * 0.08, 0.40)

        # API call efficiency
        # Too many API calls for simple task = unreliable
        efficiency_penalty = 0.0
        if api_calls > 20:
            efficiency_penalty = 0.05

        reliability_score = (
            (base_score * 0.40)
            + (output_score * 0.40)
            - error_penalty
            - time_penalty
            - efficiency_penalty
        )

        reliability_score = max(0.0, min(reliability_score, 1.0))

        return {
            "reliability_score": round(reliability_score, 4),
            "details": {
                "status": status,
                "errors": errors,
                "execution_time_seconds": round(execution_time, 2),
                "total_steps": total_steps,
                "api_calls": api_calls,
                "output_quality_score": round(output_score, 4),
            }
        }

    def score_across_repetitions(
        self, traces: list
    ) -> dict:
        if not traces:
            return {"reliability_score": 0.0, "details": {}}
        if len(traces) == 1:
            return self.score_single(traces[0])

        statuses = [t.get("status", "unknown") for t in traces]
        success_rate = statuses.count("success") / len(traces)

        times = [
            t.get("execution_time_seconds", 0) or 0
            for t in traces
        ]
        steps = [
            t.get("metrics", {}).get("total_steps", 0)
            for t in traces
        ]

        time_var = self._variance(times)
        step_var = self._variance(steps)

        step_penalty = min(step_var / 2000, 0.25)
        time_penalty = min(time_var / 50000, 0.20)

        reliability_score = max(
            0.0,
            success_rate - step_penalty - time_penalty
        )

        return {
            "reliability_score": round(reliability_score, 4),
            "details": {
                "repetitions": len(traces),
                "success_count": statuses.count("success"),
                "success_rate": round(success_rate, 4),
                "step_variance": round(step_var, 4),
                "time_variance": round(time_var, 4)
            }
        }

    def _score_output_quality(
        self, final_output, task_metadata: dict
    ) -> float:
        if not final_output:
            return 0.0
        output_str = str(final_output).strip()
        if len(output_str) < 20:
            return 0.20

        # Check success criteria coverage
        criteria = task_metadata.get("success_criteria", [])
        if not criteria:
            return 0.80

        output_lower = output_str.lower()
        matched = 0
        for criterion in criteria:
            words = [
                w.lower() for w in criterion.split()
                if len(w) > 4
            ]
            if any(w in output_lower for w in words):
                matched += 1

        return matched / len(criteria)

    def _variance(self, values: list) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

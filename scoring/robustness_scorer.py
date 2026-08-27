"""
Robustness Scorer for TrustEval.

Evaluates whether the agent maintained correct behaviour
under errors, retries, and unexpected failures.

Uses actual trace fields:
  metrics: errors, retries, tool_calls, tool_results, api_calls
  events: agent_start (attempt field), api_call, api_result

Score range: 0.0 to 1.0
"""


class RobustnessScorer:

    def score(self, trace: dict) -> dict:
        metrics = trace.get("metrics", {})
        events = trace.get("events", [])
        status = trace.get("status", "unknown")
        task_metadata = trace.get("task_metadata", {}) or {}

        errors = metrics.get("errors", 0)
        retries = metrics.get("retries", 0)
        tool_calls = metrics.get("tool_calls", 0)
        tool_results = metrics.get("tool_results", 0)
        api_calls = metrics.get("api_calls", 0)
        total_steps = metrics.get("total_steps", 1) or 1

        # Check how many agent_start attempts were made
        # (multiple attempts = had to retry whole agent)
        agent_attempts = self._count_agent_attempts(events)

        # Tool success rate
        tool_success_rate = 1.0
        if tool_calls > 0:
            tool_success_rate = min(tool_results / tool_calls, 1.0)

        # Recovery rate — did agent retry after errors?
        recovery_rate = 1.0
        if errors > 0:
            recovery_rate = min(retries / errors, 1.0)

        # API reliability
        api_success_rate = self._check_api_success(events)

        # Consecutive errors — poor robustness signal
        max_consecutive = self._max_consecutive_errors(events)

        # Did agent complete despite having errors?
        resilience_bonus = (
            0.10 if status == "success" and errors > 0 else 0.0
        )

        # Adversarial tasks — higher standard
        is_adversarial = task_metadata.get("adversarial", False)
        adversarial_penalty = 0.10 if is_adversarial and errors > 0 else 0.0

        # Penalties
        error_penalty = min(errors * 0.08, 0.40)
        retry_penalty = min(
            max(0, agent_attempts - 1) * 0.10, 0.30
        )
        consecutive_penalty = min(max_consecutive * 0.12, 0.30)
        tool_penalty = (1.0 - tool_success_rate) * 0.20
        api_penalty = (1.0 - api_success_rate) * 0.15

        robustness_score = (
            1.0
            - error_penalty
            - retry_penalty
            - consecutive_penalty
            - tool_penalty
            - api_penalty
            - adversarial_penalty
            + resilience_bonus
        )

        robustness_score = max(0.0, min(robustness_score, 1.0))

        return {
            "robustness_score": round(robustness_score, 4),
            "details": {
                "errors": errors,
                "retries": retries,
                "agent_attempts": agent_attempts,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "tool_success_rate": round(tool_success_rate, 4),
                "api_calls": api_calls,
                "api_success_rate": round(api_success_rate, 4),
                "recovery_rate": round(recovery_rate, 4),
                "max_consecutive_errors": max_consecutive,
                "status": status
            }
        }

    def _count_agent_attempts(self, events: list) -> int:
        return sum(
            1 for e in events
            if e.get("type") == "agent_start"
        ) or 1

    def _check_api_success(self, events: list) -> float:
        api_calls = [
            e for e in events if e.get("type") == "api_call"
        ]
        api_results = [
            e for e in events if e.get("type") == "api_result"
        ]
        if not api_calls:
            return 1.0
        return min(len(api_results) / len(api_calls), 1.0)

    def _max_consecutive_errors(self, events: list) -> int:
        max_c = 0
        current = 0
        for e in events:
            desc = str(e.get("description", "")).lower()
            if "error" in desc or "fail" in desc:
                current += 1
                max_c = max(max_c, current)
            else:
                current = 0
        return max_c

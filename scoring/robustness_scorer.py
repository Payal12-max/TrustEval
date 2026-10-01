"""
Robustness Scorer for TrustEval.

Evaluates whether the agent maintained correct execution behaviour
under errors, retries, tool failures, API failures, and unexpected
execution problems.

Uses actual trace fields:

    metrics:
        errors
        retries
        tool_calls
        tool_results
        api_calls
        total_steps

    events:
        agent_start
        retry_wait
        error
        api_call
        api_result

Score range: 0.0 to 1.0
"""


class RobustnessScorer:

    def score(self, trace: dict) -> dict:

        metrics = trace.get(
            "metrics",
            {}
        ) or {}

        events = trace.get(
            "events",
            []
        ) or []

        status = trace.get(
            "status",
            "unknown"
        )

        task_metadata = trace.get(
            "task_metadata",
            {}
        ) or {}

        # -----------------------------------------------------
        # Basic metrics
        # -----------------------------------------------------

        errors = int(
            metrics.get(
                "errors",
                0
            ) or 0
        )

        tool_calls = int(
            metrics.get(
                "tool_calls",
                0
            ) or 0
        )

        tool_results = int(
            metrics.get(
                "tool_results",
                0
            ) or 0
        )

        api_calls = int(
            metrics.get(
                "api_calls",
                0
            ) or 0
        )

        total_steps = int(
            metrics.get(
                "total_steps",
                1
            ) or 1
        )

        # -----------------------------------------------------
        # Retry count
        #
        # Prefer actual retry_wait events because that is what
        # runner.py logs.
        #
        # Fall back to metrics["retries"] for compatibility with
        # already-fixed traces.
        # -----------------------------------------------------

        event_retries = self._count_retry_events(
            events
        )

        metric_retries = int(
            metrics.get(
                "retries",
                0
            ) or 0
        )

        retries = max(
            event_retries,
            metric_retries
        )

        # -----------------------------------------------------
        # Agent attempts
        # -----------------------------------------------------

        agent_attempts = self._count_agent_attempts(
            events
        )

        # -----------------------------------------------------
        # Recovery
        #
        # Recovery means that after an error the agent actually
        # attempted another execution/retry.
        # -----------------------------------------------------

        recovery_rate = self._calculate_recovery_rate(
            errors,
            retries
        )

        # -----------------------------------------------------
        # Tool success
        # -----------------------------------------------------

        tool_success_rate = self._calculate_tool_success_rate(
            events,
            tool_calls,
            tool_results
        )

        # -----------------------------------------------------
        # API success
        # -----------------------------------------------------

        api_success_rate = self._check_api_success(
            events
        )

        # -----------------------------------------------------
        # Consecutive errors
        # -----------------------------------------------------

        max_consecutive_errors = (
            self._max_consecutive_errors(
                events
            )
        )

        # -----------------------------------------------------
        # Adversarial task
        # -----------------------------------------------------

        is_adversarial = bool(
            task_metadata.get(
                "adversarial",
                False
            )
        )

        # -----------------------------------------------------
        # Error penalty
        #
        # Errors still matter, because they represent instability.
        # But successful recovery reduces the effective penalty.
        # -----------------------------------------------------

        effective_errors = errors

        if errors > 0 and recovery_rate > 0:

            # Successful recovery offsets part of the error cost.
            effective_errors = errors * (
                1.0 - 0.50 * recovery_rate
            )

        error_penalty = min(
            effective_errors * 0.08,
            0.40
        )

        # -----------------------------------------------------
        # Retry penalty
        #
        # Retries are execution overhead, not automatically a
        # robustness failure.
        #
        # Only repeated retries beyond the first one receive the
        # explicit retry penalty.
        # -----------------------------------------------------

        excessive_retries = max(
            0,
            retries - 1
        )

        retry_penalty = min(
            excessive_retries * 0.05,
            0.20
        )

        # -----------------------------------------------------
        # Whole-agent attempt penalty
        #
        # Multiple agent_start events indicate that the agent
        # itself had to be restarted/re-executed.
        # -----------------------------------------------------

        additional_attempts = max(
            0,
            agent_attempts - 1
        )

        attempt_penalty = min(
            additional_attempts * 0.05,
            0.15
        )

        # -----------------------------------------------------
        # Consecutive error penalty
        # -----------------------------------------------------

        consecutive_penalty = min(
            max_consecutive_errors * 0.12,
            0.30
        )

        # -----------------------------------------------------
        # Tool penalty
        # -----------------------------------------------------

        tool_penalty = (
            (1.0 - tool_success_rate)
            * 0.20
        )

        # -----------------------------------------------------
        # API penalty
        # -----------------------------------------------------

        api_penalty = (
            (1.0 - api_success_rate)
            * 0.15
        )

        # -----------------------------------------------------
        # Resilience bonus
        #
        # A successful completion after errors demonstrates that
        # the agent recovered.
        # -----------------------------------------------------

        resilience_bonus = 0.0

        if (
            status == "success"
            and errors > 0
            and recovery_rate > 0
        ):
            resilience_bonus = 0.10

        # -----------------------------------------------------
        # Adversarial adjustment
        #
        # Only penalize an adversarial execution when errors
        # occurred AND the agent failed to recover adequately.
        # -----------------------------------------------------

        adversarial_penalty = 0.0

        if (
            is_adversarial
            and errors > 0
            and recovery_rate < 0.50
        ):
            adversarial_penalty = 0.10

        # -----------------------------------------------------
        # Final score
        # -----------------------------------------------------

        robustness_score = (
            1.0
            - error_penalty
            - retry_penalty
            - attempt_penalty
            - consecutive_penalty
            - tool_penalty
            - api_penalty
            - adversarial_penalty
            + resilience_bonus
        )

        robustness_score = max(
            0.0,
            min(
                robustness_score,
                1.0
            )
        )

        return {
            "robustness_score": round(
                robustness_score,
                4
            ),

            "details": {
                "errors": errors,

                "retries": retries,

                "retry_events": event_retries,

                "metric_retries": metric_retries,

                "agent_attempts": agent_attempts,

                "tool_calls": tool_calls,

                "tool_results": tool_results,

                "tool_success_rate": round(
                    tool_success_rate,
                    4
                ),

                "api_calls": api_calls,

                "api_success_rate": round(
                    api_success_rate,
                    4
                ),

                "recovery_rate": round(
                    recovery_rate,
                    4
                ),

                "max_consecutive_errors": (
                    max_consecutive_errors
                ),

                "status": status,

                "is_adversarial": is_adversarial,

                "penalties": {
                    "error_penalty": round(
                        error_penalty,
                        4
                    ),

                    "retry_penalty": round(
                        retry_penalty,
                        4
                    ),

                    "attempt_penalty": round(
                        attempt_penalty,
                        4
                    ),

                    "consecutive_error_penalty": round(
                        consecutive_penalty,
                        4
                    ),

                    "tool_penalty": round(
                        tool_penalty,
                        4
                    ),

                    "api_penalty": round(
                        api_penalty,
                        4
                    ),

                    "adversarial_penalty": round(
                        adversarial_penalty,
                        4
                    ),
                },

                "resilience_bonus": round(
                    resilience_bonus,
                    4
                ),
            }
        }

    # =========================================================
    # Retry handling
    # =========================================================

    def _count_retry_events(
        self,
        events: list
    ) -> int:
        """
        Count actual retry_wait events.

        This matches the event type used by the execution runner.
        """

        return sum(
            1
            for event in events
            if event.get("type") == "retry_wait"
        )

    # =========================================================
    # Agent attempts
    # =========================================================

    def _count_agent_attempts(
        self,
        events: list
    ) -> int:
        """
        Count agent_start events.

        At least one attempt is assumed for a valid execution.
        """

        attempts = sum(
            1
            for event in events
            if event.get("type") == "agent_start"
        )

        return max(
            attempts,
            1
        )

    # =========================================================
    # Recovery
    # =========================================================

    def _calculate_recovery_rate(
        self,
        errors: int,
        retries: int
    ) -> float:
        """
        Calculate the proportion of errors that were followed by
        retry/recovery behaviour.

        Capped at 1.0.
        """

        if errors <= 0:
            return 1.0

        return min(
            retries / errors,
            1.0
        )

    # =========================================================
    # Tool success
    # =========================================================

    def _calculate_tool_success_rate(
        self,
        events: list,
        tool_calls: int,
        tool_results: int
    ) -> float:
        """
        Estimate tool success using actual trace events where
        possible.

        If explicit tool result success/failure information is
        available, use it.

        Otherwise fall back to the existing call/result ratio.
        """

        tool_result_events = [
            event
            for event in events
            if event.get("type") == "tool_result"
        ]

        if tool_result_events:

            successful = 0

            for event in tool_result_events:

                if self._tool_result_is_successful(
                    event
                ):
                    successful += 1

            return min(
                successful / len(tool_result_events),
                1.0
            )

        if tool_calls <= 0:
            return 1.0

        return min(
            tool_results / tool_calls,
            1.0
        )

    def _tool_result_is_successful(
        self,
        event: dict
    ) -> bool:
        """
        Determine whether a tool result represents success.

        Supports several common trace representations.
        """

        # Explicit boolean success field.
        if "success" in event:

            return bool(
                event.get("success")
            )

        # Explicit error field.
        if event.get("error"):
            return False

        content = self._event_content(
            event
        ).lower()

        if not content:
            return True

        failure_terms = [
            "tool error",
            "tool failed",
            "execution failed",
            "request failed",
            "error:",
            "exception:",
            "failed:",
        ]

        if any(
            term in content
            for term in failure_terms
        ):
            return False

        return True

    # =========================================================
    # API success
    # =========================================================

    def _check_api_success(
        self,
        events: list
    ) -> float:
        """
        Calculate API success rate from actual API events.

        If API result events explicitly indicate failure, those
        results are treated as failed.
        """

        api_calls = [
            event
            for event in events
            if event.get("type") == "api_call"
        ]

        api_results = [
            event
            for event in events
            if event.get("type") == "api_result"
        ]

        if not api_calls:
            return 1.0

        if not api_results:
            return 0.0

        successful_results = 0

        for event in api_results:

            if self._api_result_is_successful(
                event
            ):
                successful_results += 1

        return min(
            successful_results / len(api_calls),
            1.0
        )

    def _api_result_is_successful(
        self,
        event: dict
    ) -> bool:
        """Determine whether an API result succeeded."""

        if "success" in event:
            return bool(
                event.get("success")
            )

        if event.get("error"):
            return False

        content = self._event_content(
            event
        ).lower()

        failure_terms = [
            "api error",
            "api failed",
            "request failed",
            "http 4",
            "http 5",
            "status: 4",
            "status: 5",
            "exception:",
            "error:",
        ]

        if any(
            term in content
            for term in failure_terms
        ):
            return False

        return True

    # =========================================================
    # Consecutive errors
    # =========================================================

    def _max_consecutive_errors(
        self,
        events: list
    ) -> int:
        """
        Find the maximum number of consecutive error events.

        Actual event type 'error' is preferred.

        Textual 'error'/'fail' matching is retained as a fallback
        for traces where failures are embedded in descriptions.
        """

        max_consecutive = 0
        current = 0

        for event in events:

            if self._is_error_event(
                event
            ):
                current += 1

                max_consecutive = max(
                    max_consecutive,
                    current
                )

            else:
                current = 0

        return max_consecutive

    def _is_error_event(
        self,
        event: dict
    ) -> bool:
        """Determine whether an event represents an error."""

        event_type = str(
            event.get(
                "type",
                ""
            )
        ).lower()

        if event_type == "error":
            return True

        description = self._event_content(
            event
        ).lower()

        if not description:
            return False

        return (
            "error" in description
            or "failed" in description
            or "failure" in description
            or "exception" in description
        )

    # =========================================================
    # Generic event content
    # =========================================================

    def _event_content(
        self,
        event: dict
    ) -> str:
        """
        Extract textual content from common trace event fields.
        """

        parts = []

        for key in (
            "content",
            "description",
            "output",
            "result",
            "response",
            "error",
        ):

            value = event.get(
                key
            )

            if isinstance(
                value,
                str
            ):
                parts.append(
                    value
                )

            elif isinstance(
                value,
                (dict, list)
            ):
                parts.append(
                    str(value)
                )

        return " ".join(
            parts
        )
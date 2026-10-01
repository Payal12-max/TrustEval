"""
TrustEval Goal Adherence Scorer.

Goal adherence measures whether an agent actually followed the benchmark
task, rather than merely mentioning words from the task.

This version:
- preserves the original domain-specific weighting structure;
- uses the task-specific validators as the primary evidence for constraints
  and success criteria;
- uses trace metadata/events for tool-use and planning evidence;
- avoids treating keyword presence as proof of compliance;
- keeps the result shape compatible with CompositeScorer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scoring.task_validators import validate_task


class GoalAdherenceScorer:
    """
    Score how well a trace adheres to its benchmark goal.

    The task validator is the main source of truth.  Generic textual
    heuristics are only used as fallback evidence when a validator does not
    expose a particular check.
    """

    DOMAIN_WEIGHTS = {
        "travel": {
            "output": 0.20,
            "constraint": 0.30,
            "criteria": 0.25,
            "planning": 0.15,
            "completion": 0.10,
        },
        "finance": {
            "output": 0.15,
            "constraint": 0.40,
            "criteria": 0.25,
            "planning": 0.10,
            "completion": 0.10,
        },
        "shopping": {
            "output": 0.20,
            "constraint": 0.35,
            "criteria": 0.25,
            "planning": 0.10,
            "completion": 0.10,
        },
        "education": {
            "output": 0.25,
            "constraint": 0.20,
            "criteria": 0.35,
            "planning": 0.10,
            "completion": 0.10,
        },
        "productivity": {
            "output": 0.20,
            "constraint": 0.25,
            "criteria": 0.30,
            "planning": 0.15,
            "completion": 0.10,
        },
        "technical": {
            "output": 0.20,
            "constraint": 0.25,
            "criteria": 0.30,
            "planning": 0.15,
            "completion": 0.10,
        },
        "default": {
            "output": 0.20,
            "constraint": 0.25,
            "criteria": 0.30,
            "planning": 0.15,
            "completion": 0.10,
        },
    }

    # Common words that are too generic to be useful as evidence.
    STOP_WORDS = {
        "the",
        "this",
        "that",
        "with",
        "from",
        "must",
        "should",
        "have",
        "been",
        "into",
        "than",
        "each",
        "every",
        "only",
        "within",
        "provide",
        "provides",
        "include",
        "includes",
        "using",
        "used",
        "required",
        "requirements",
        "task",
        "goal",
        "answer",
        "response",
    }

    def score(self, trace: dict) -> dict:
        """
        Score one trace.

        Returns a dictionary containing:
            goal_adherence_score
            details
        """

        metadata = trace.get("task_metadata", {}) or {}
        domain = self._get_domain(trace)
        weights = self.DOMAIN_WEIGHTS.get(
            domain,
            self.DOMAIN_WEIGHTS["default"],
        )

        final_output = self._get_final_output(trace)
        constraints = self._as_list(
            metadata.get("constraints", [])
        )
        criteria = self._as_list(
            metadata.get("success_criteria", [])
        )

        validation = self._run_validator(trace)

        output_score = self._score_output_quality(
            final_output
        )

        constraint_score = self._score_constraints(
            trace,
            constraints,
            validation,
        )

        criteria_score = self._score_criteria(
            trace,
            criteria,
            validation,
        )

        planning_score = self._score_planning_focus(
            trace,
            metadata,
        )

        completion_score = self._score_completion(
            trace,
            validation,
        )

        goal_score = (
            weights["output"] * output_score
            + weights["constraint"] * constraint_score
            + weights["criteria"] * criteria_score
            + weights["planning"] * planning_score
            + weights["completion"] * completion_score
        )

        goal_score = self._clamp(goal_score)

        return {
            "goal_adherence_score": round(goal_score, 4),
            "details": {
                "domain": domain,
                "weights": weights,
                "output_quality_score": round(
                    output_score,
                    4,
                ),
                "constraint_satisfaction_score": round(
                    constraint_score,
                    4,
                ),
                "success_criteria_score": round(
                    criteria_score,
                    4,
                ),
                "planning_focus_score": round(
                    planning_score,
                    4,
                ),
                "completion_score": round(
                    completion_score,
                    4,
                ),
                "task_validation": validation,
            },
        }

    # ------------------------------------------------------------------
    # Validator integration
    # ------------------------------------------------------------------

    def _run_validator(self, trace: dict) -> dict:
        """
        Run the task-specific validator.

        An unimplemented validator does NOT automatically penalize the
        trace. This is important because a missing evaluator must not be
        confused with an agent failure.
        """

        try:
            result = validate_task(trace)

        except Exception as exc:
            return {
                "implemented": False,
                "passed": False,
                "score": None,
                "checks": {},
                "issues": [
                    f"Task validator error: {type(exc).__name__}: {exc}"
                ],
            }

        if not isinstance(result, dict):
            return {
                "implemented": False,
                "passed": False,
                "score": None,
                "checks": {},
                "issues": [
                    "Task validator returned a non-dictionary result."
                ],
            }

        implemented = bool(
            result.get("implemented", False)
        )

        raw_score = result.get("score")

        if implemented and isinstance(raw_score, (int, float)):
            score = self._clamp(float(raw_score))
        else:
            score = None

        checks = result.get("checks", {})
        if not isinstance(checks, dict):
            checks = {}

        return {
            "implemented": implemented,
            "passed": bool(result.get("passed", False)),
            "score": score,
            "checks": checks,
            "issues": self._as_list(
                result.get("issues", [])
            ),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _score_output_quality(self, final_output: str) -> float:
        """
        Basic output-completeness signal.

        This intentionally does NOT claim that a long answer is correct.
        Correctness is handled by the task validator.
        """

        text = self._normalize_text(final_output)

        if not text:
            return 0.0

        if len(text) < 20:
            return 0.20

        if len(text) < 50:
            return 0.50

        return 1.0

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _score_constraints(
        self,
        trace: dict,
        constraints: List[str],
        validation: dict,
    ) -> float:
        """
        Prefer validator evidence.

        If a validator is implemented, its check score is used instead of
        pretending that keyword presence proves constraint satisfaction.

        If no validator exists, fall back to conservative textual evidence.
        """

        if validation.get("implemented"):
            score = validation.get("score")

            if isinstance(score, (int, float)):
                return self._clamp(float(score))

        if not constraints:
            return 1.0

        text = self._get_final_output(trace).lower()

        scores = [
            self._constraint_text_evidence(
                text,
                constraint,
            )
            for constraint in constraints
        ]

        return (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

    def _constraint_text_evidence(
        self,
        text: str,
        constraint: str,
    ) -> float:
        """
        Conservative fallback only.

        Presence of meaningful constraint terms gives partial evidence,
        never full proof of satisfaction.
        """

        words = self._meaningful_words(
            constraint
        )

        if not words:
            return 0.0

        matched = sum(
            1
            for word in words
            if word in text
        )

        ratio = matched / len(words)

        # Keyword evidence is deliberately capped below full credit.
        return min(ratio, 0.70)

    # ------------------------------------------------------------------
    # Success criteria
    # ------------------------------------------------------------------

    def _score_criteria(
        self,
        trace: dict,
        criteria: List[str],
        validation: dict,
    ) -> float:
        """
        Use validator evidence as the primary success-criteria signal.

        For an implemented validator, all its checks represent concrete
        task-level evidence, so the validator score is used directly.

        For an unimplemented validator, textual evidence is capped.
        """

        if validation.get("implemented"):
            score = validation.get("score")

            if isinstance(score, (int, float)):
                return self._clamp(float(score))

        if not criteria:
            return 1.0

        text = self._get_final_output(trace).lower()

        scores = []

        for criterion in criteria:
            words = self._meaningful_words(
                criterion
            )

            if not words:
                scores.append(0.0)
                continue

            matched = sum(
                1
                for word in words
                if word in text
            )

            ratio = matched / len(words)

            # Never treat keyword overlap as complete success.
            scores.append(
                min(ratio, 0.70)
            )

        return (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _score_planning_focus(
        self,
        trace: dict,
        metadata: dict,
    ) -> float:
        """
        Measure whether planning/execution stayed focused on the task.

        This is intentionally a small component. Planning quality should not
        overpower actual task satisfaction.
        """

        events = trace.get("events", []) or []

        if not events:
            return 0.0

        goal = str(
            metadata.get("goal", "")
        )

        goal_words = self._meaningful_words(
            goal
        )

        if not goal_words:
            # If no goal text is available, a planning/execution trace still
            # receives neutral credit rather than an arbitrary penalty.
            return 0.5

        planning_events = [
            event
            for event in events
            if str(
                event.get("type", "")
            ).lower()
            in {
                "planning",
                "replanning",
                "agent_execution",
                "agent_start",
            }
        ]

        if not planning_events:
            return 0.0

        combined = " ".join(
            str(event.get("description", ""))
            for event in planning_events
        ).lower()

        # Framework logs often say only "received the benchmark task", so
        # planning evidence is deliberately modest.
        matched = sum(
            1
            for word in goal_words
            if word in combined
        )

        if matched == 0:
            return 0.5

        return min(
            1.0,
            0.5 + (
                0.5
                * matched
                / len(goal_words)
            ),
        )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _score_completion(
        self,
        trace: dict,
        validation: dict,
    ) -> float:
        """
        Completion is separate from correctness.

        A successful execution with a validator failure should not receive
        full goal credit.
        """

        status = str(
            trace.get("status", "unknown")
        ).lower()

        if status == "success":
            completion = 1.0
        elif status in {
            "partial",
            "completed_with_errors",
        }:
            completion = 0.60
        else:
            completion = 0.35

        if validation.get("implemented"):
            # If the task-specific validator exists, its score is stronger
            # evidence of actual completion than the framework status.
            validator_score = validation.get("score")

            if isinstance(
                validator_score,
                (int, float),
            ):
                completion *= (
                    0.50
                    + 0.50
                    * self._clamp(
                        float(validator_score)
                    )
                )

        return self._clamp(completion)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_domain(self, trace: dict) -> str:
        metadata = trace.get(
            "task_metadata",
            {},
        ) or {}

        domain = metadata.get(
            "domain",
            "",
        )

        if domain:
            return str(
                domain
            ).lower().strip()

        task_id = str(
            trace.get(
                "task_id",
                "",
            )
        )

        if "_" in task_id:
            return task_id.split(
                "_",
                1,
            )[0].lower()

        return "default"

    def _get_final_output(self, trace: dict) -> str:
        output = trace.get(
            "final_output",
            "",
        )

        if output is None:
            return ""

        return str(output)

    def _meaningful_words(
        self,
        text: str,
    ) -> List[str]:
        words = re.findall(
            r"[a-zA-Z][a-zA-Z0-9_-]{3,}",
            str(text).lower(),
        )

        return [
            word
            for word in words
            if word not in self.STOP_WORDS
        ]

    def _normalize_text(self, text: Any) -> str:
        return " ".join(
            str(text).split()
        ).strip()

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    def _clamp(
        self,
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )


if __name__ == "__main__":
    print(
        "GoalAdherenceScorer loaded successfully."
    )

from scoring.task_validators import validate_task


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

        # ---------------------------------------------------------
        # 1. BASE RELIABILITY
        # ---------------------------------------------------------

        base_score = 1.0 if status == "success" else 0.30

        # ---------------------------------------------------------
        # 2. OUTPUT QUALITY
        # ---------------------------------------------------------

        output_score = self._score_output_quality(
            final_output,
            task_metadata
        )

        # ---------------------------------------------------------
        # 3. EXECUTION TIME PENALTY
        # ---------------------------------------------------------

        time_penalty = 0.0

        if execution_time > 300:
            time_penalty = 0.10

        elif execution_time < 0.5:
            time_penalty = 0.05

        # ---------------------------------------------------------
        # 4. ERROR PENALTY
        # ---------------------------------------------------------

        error_penalty = min(
            errors * 0.08,
            0.40
        )

        # ---------------------------------------------------------
        # 5. API / EFFICIENCY PENALTY
        # ---------------------------------------------------------

        efficiency_penalty = 0.0

        if api_calls > 20:
            efficiency_penalty = 0.05

        # ---------------------------------------------------------
        # 6. TASK-SPECIFIC VALIDATION
        #
        # This is the important new part.
        #
        # The validator checks whether the agent actually satisfied
        # the task's logical constraints, arithmetic requirements,
        # consistency requirements, etc.
        # ---------------------------------------------------------

        validation_result = self._run_task_validation(
            trace
        )

        task_validation_penalty = 0.0

        if validation_result["implemented"]:

            validation_score = validation_result["score"]

            # Convert validator score into a reliability penalty.
            #
            # Example:
            #
            # validator score = 1.0
            # penalty = 0.0
            #
            # validator score = 0.8
            # penalty = 0.06
            #
            # validator score = 0.5
            # penalty = 0.15
            #
            # Maximum penalty = 0.30
            task_validation_penalty = min(
                (1.0 - validation_score) * 0.30,
                0.30
            )

        # ---------------------------------------------------------
        # 7. FINAL RELIABILITY SCORE
        # ---------------------------------------------------------

        reliability_score = (
            (base_score * 0.40)
            + (output_score * 0.40)
            - error_penalty
            - time_penalty
            - efficiency_penalty
            - task_validation_penalty
        )

        reliability_score = max(
            0.0,
            min(reliability_score, 1.0)
        )

        # ---------------------------------------------------------
        # 8. RETURN RESULT
        # ---------------------------------------------------------

        return {
            "reliability_score": round(
                reliability_score,
                4
            ),

            "details": {
                "status": status,

                "errors": errors,

                "execution_time_seconds": round(
                    execution_time,
                    2
                ),

                "total_steps": total_steps,

                "api_calls": api_calls,

                "output_quality_score": round(
                    output_score,
                    4
                ),

                "base_score": round(
                    base_score,
                    4
                ),

                "error_penalty": round(
                    error_penalty,
                    4
                ),

                "time_penalty": round(
                    time_penalty,
                    4
                ),

                "efficiency_penalty": round(
                    efficiency_penalty,
                    4
                ),

                # New task-validation information
                "task_validation": {
                    "implemented": validation_result[
                        "implemented"
                    ],

                    "passed": validation_result[
                        "passed"
                    ],

                    "validator_score": round(
                        validation_result["score"],
                        4
                    ),

                    "penalty": round(
                        task_validation_penalty,
                        4
                    ),

                    "failed_checks": validation_result[
                        "failed_checks"
                    ]
                }
            }
        }

    # =============================================================
    # TASK VALIDATION
    # =============================================================

    def _run_task_validation(
        self,
        trace: dict
    ) -> dict:
        """
        Run the task-specific validator.

        The validator is allowed to be unavailable for tasks that
        have not been implemented yet.

        In that situation we DO NOT penalize reliability.

        This is important while we progressively add validators
        for all 30 benchmark tasks.
        """

        try:

            result = validate_task(trace)

        except Exception as exc:

            # Do not crash the entire evaluation because one
            # validator has a bug.
            #
            # Also do not punish the model for a validator bug.
            return {
                "implemented": False,
                "passed": False,
                "score": 1.0,
                "failed_checks": [],
                "error": str(exc)
            }

        if not isinstance(result, dict):

            return {
                "implemented": False,
                "passed": False,
                "score": 1.0,
                "failed_checks": [],
                "error": "Validator returned a non-dictionary result."
            }

        implemented = bool(
            result.get("implemented", False)
        )

        if not implemented:

            return {
                "implemented": False,
                "passed": False,
                "score": 1.0,
                "failed_checks": []
            }

        score = result.get(
            "score",
            1.0
        )

        try:
            score = float(score)

        except (
            TypeError,
            ValueError
        ):
            score = 0.0

        score = max(
            0.0,
            min(score, 1.0)
        )

        checks = result.get(
            "checks",
            {}
        )

        failed_checks = []

        if isinstance(checks, dict):

            for check_name, passed in checks.items():

                if not bool(passed):
                    failed_checks.append(
                        check_name
                    )

        return {
            "implemented": True,

            "passed": bool(
                result.get(
                    "passed",
                    False
                )
            ),

            "score": score,

            "failed_checks": failed_checks
        }

    # =============================================================
    # OUTPUT QUALITY
    # =============================================================

    def _score_output_quality(
        self,
        final_output,
        task_metadata: dict
    ) -> float:

        if not final_output:
            return 0.0

        output_str = str(
            final_output
        ).strip()

        if len(output_str) < 20:
            return 0.20

        criteria = task_metadata.get(
            "success_criteria",
            []
        )

        if not criteria:
            return 0.80

        output_lower = output_str.lower()

        matched = 0

        for criterion in criteria:

            words = [
                w.lower()
                for w in criterion.split()
                if len(w) > 4
            ]

            if any(
                w in output_lower
                for w in words
            ):
                matched += 1

        return matched / len(criteria)

    # =============================================================
    # CROSS-REPETITION RELIABILITY
    # =============================================================

    def score_across_repetitions(
        self,
        traces: list
    ) -> dict:

        if not traces:

            return {
                "reliability_score": 0.0,
                "details": {}
            }

        if len(traces) == 1:

            return self.score_single(
                traces[0]
            )

        # ---------------------------------------------------------
        # Basic repetition statistics
        # ---------------------------------------------------------

        statuses = [
            t.get(
                "status",
                "unknown"
            )
            for t in traces
        ]

        success_rate = (
            statuses.count("success")
            / len(traces)
        )

        times = [
            t.get(
                "execution_time_seconds",
                0
            ) or 0
            for t in traces
        ]

        steps = [
            t.get(
                "metrics",
                {}
            ).get(
                "total_steps",
                0
            )
            for t in traces
        ]

        # ---------------------------------------------------------
        # Variance
        # ---------------------------------------------------------

        time_var = self._variance(
            times
        )

        step_var = self._variance(
            steps
        )

        step_penalty = min(
            step_var / 2000,
            0.25
        )

        time_penalty = min(
            time_var / 50000,
            0.20
        )

        # ---------------------------------------------------------
        # TASK VALIDATION ACROSS REPETITIONS
        #
        # Every repetition is validated independently.
        # We then average the validator scores.
        # ---------------------------------------------------------

        validation_results = []

        for trace in traces:

            result = self._run_task_validation(
                trace
            )

            if result["implemented"]:

                validation_results.append(
                    result["score"]
                )

        if validation_results:

            mean_validation_score = (
                sum(validation_results)
                / len(validation_results)
            )

            task_validation_penalty = min(
                (1.0 - mean_validation_score)
                * 0.30,
                0.30
            )

        else:

            # No validator implemented yet for this task.
            mean_validation_score = None
            task_validation_penalty = 0.0

        # ---------------------------------------------------------
        # FINAL SCORE
        # ---------------------------------------------------------

        reliability_score = (
            success_rate
            - step_penalty
            - time_penalty
            - task_validation_penalty
        )

        reliability_score = max(
            0.0,
            min(
                reliability_score,
                1.0
            )
        )

        # ---------------------------------------------------------
        # RETURN
        # ---------------------------------------------------------

        return {
            "reliability_score": round(
                reliability_score,
                4
            ),

            "details": {
                "repetitions": len(traces),

                "success_count": statuses.count(
                    "success"
                ),

                "success_rate": round(
                    success_rate,
                    4
                ),

                "step_variance": round(
                    step_var,
                    4
                ),

                "time_variance": round(
                    time_var,
                    4
                ),

                "step_penalty": round(
                    step_penalty,
                    4
                ),

                "time_penalty": round(
                    time_penalty,
                    4
                ),

                "task_validation": {
                    "repetitions_validated": len(
                        validation_results
                    ),

                    "mean_validator_score": (
                        round(
                            mean_validation_score,
                            4
                        )
                        if mean_validation_score is not None
                        else None
                    ),

                    "penalty": round(
                        task_validation_penalty,
                        4
                    )
                }
            }
        }

    # =============================================================
    # VARIANCE
    # =============================================================

    def _variance(
        self,
        values: list
    ) -> float:

        if len(values) < 2:
            return 0.0

        mean = (
            sum(values)
            / len(values)
        )

        return sum(
            (x - mean) ** 2
            for x in values
        ) / len(values)
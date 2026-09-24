import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
import crewai.llms.cache as crewai_cache

from tools.benchmark_tools import (
    logged_calculator,
    logged_web_search,
)
from agents.task_context import (
    get_agent_role,
    get_agent_backstory,
    get_allowed_tool_names,
)

import crewai.utilities.agent_utils as agent_utils_module
import crewai.experimental.agent_executor as agent_executor_module


# ============================================================
# Patch: defensive safety net for the "force final answer"
# call that runs after max_iter is exceeded. Kept as low-cost
# insurance against a single bad completion crashing a task.
# ============================================================

_original_handle_max_iterations_exceeded = (
    agent_utils_module.handle_max_iterations_exceeded
)

_RETRYABLE_FINAL_ANSWER_ERRORS = (
    "tool choice is none, but model called a tool",
    "invalid response from llm call",
    "none or empty",
)


def _patched_handle_max_iterations_exceeded(
    formatted_answer, printer, messages, llm, callbacks, verbose=True
):
    try:
        return _original_handle_max_iterations_exceeded(
            formatted_answer, printer, messages, llm, callbacks, verbose
        )
    except Exception as error:
        error_text = str(error).lower()

        if not any(
            indicator in error_text
            for indicator in _RETRYABLE_FINAL_ANSWER_ERRORS
        ):
            raise

        retry_messages = list(messages) + [
            agent_utils_module.format_message_for_llm(
                "Respond with plain natural-language text only. Do not output "
                "JSON, function calls, or any tool-invocation syntax. Give "
                "your complete final answer now as prose, directly.",
                role="user",
            )
        ]

        for _ in range(2):

            try:
                answer = llm.call(retry_messages, callbacks=callbacks)
            except Exception:
                answer = None

            if not answer:
                continue

            formatted = agent_utils_module.format_answer(answer=answer)

            if isinstance(formatted, agent_utils_module.AgentFinish):
                return formatted

            return agent_utils_module.AgentFinish(
                thought=formatted.thought,
                output=formatted.text,
                text=formatted.text,
            )

        if formatted_answer is not None and getattr(formatted_answer, "text", None):
            fallback_text = (
                formatted_answer.text
                + "\n\n[Note: final answer truncated - repeated invalid "
                "LLM output when finalizing.]"
            )
        else:
            fallback_text = (
                "No final answer could be produced: the model repeatedly "
                "produced invalid output when asked to finalize its answer."
            )

        return agent_utils_module.AgentFinish(
            thought="",
            output=fallback_text,
            text=fallback_text,
        )


agent_utils_module.handle_max_iterations_exceeded = (
    _patched_handle_max_iterations_exceeded
)
agent_executor_module.handle_max_iterations_exceeded = (
    _patched_handle_max_iterations_exceeded
)


# ============================================================
# Setup
# ============================================================

crewai_cache.mark_cache_breakpoint = lambda message: message

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Mistral configuration
# ============================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. "
        "Please add it to the project's .env file."
    )

MISTRAL_MODEL = "mistral/ministral-3b-2512"

MAX_OUTPUT_TOKENS = 2000
MAX_TOOL_OUTPUT_CHARS = 1500


# ============================================================
# CrewAI model
# ============================================================

llm = LLM(
    model=MISTRAL_MODEL,
    api_key=MISTRAL_API_KEY,
    max_tokens=MAX_OUTPUT_TOKENS,
)


# ============================================================
# CrewAI agent
# ============================================================

def run_crewai_agent(benchmark_task, logger):

    # --------------------------------------------------------
    # Prepare benchmark information
    # --------------------------------------------------------

    constraints = "\n".join(
        f"- {x}"
        for x in benchmark_task["constraints"]
    )

    success_criteria = "\n".join(
        f"- {x}"
        for x in benchmark_task["success_criteria"]
    )

    domain = benchmark_task.get("domain", "general")

    allowed_tools = get_allowed_tool_names(benchmark_task)

    logger.log_event(
        "planning",
        description=(
            f"CrewAI Mistral agent received benchmark task "
            f"(domain: {domain})."
        ),
    )


    # ========================================================
    # Tools (built conditionally based on task permissions)
    # ========================================================

    tools_list = []

    if "calculator" in allowed_tools:

        @tool("calculator")
        def calculator_tool(expression: str) -> str:
            """
            Perform arithmetic calculations.
            Use when arithmetic is required.
            """

            result = str(
                logged_calculator(
                    expression,
                    logger,
                )
            )

            return result[:MAX_TOOL_OUTPUT_CHARS]

        tools_list.append(calculator_tool)

    if "web_search" in allowed_tools:

        @tool("web_search")
        def web_search_tool(
            query: str,
            topn: int = 5,
            source: str = "news",
        ) -> str:
            """
            Search the web for current or factual information.
            """

            result = str(
                logged_web_search(
                    query,
                    logger,
                )
            )

            return result[:MAX_TOOL_OUTPUT_CHARS]

        tools_list.append(web_search_tool)


    # ========================================================
    # Agent
    # ========================================================

    benchmark_agent = Agent(
        role=get_agent_role(benchmark_task),

        goal=benchmark_task["goal"],

        backstory=get_agent_backstory(benchmark_task),

        llm=llm,

        tools=tools_list,

        verbose=False,

        max_iter=6,
    )


    # ========================================================
    # Task
    # ========================================================

    task_description = f"""
Complete this benchmark task.

GOAL:
{benchmark_task["goal"]}

CONSTRAINTS:
{constraints}

SUCCESS CRITERIA:
{success_criteria}

EXPECTED OUTPUT:
{benchmark_task["expected_output"]}

Follow every constraint and satisfy every success criterion.

Use calculator whenever arithmetic is required.
Use web_search whenever current or factual information
needs verification.

Do not invent search results.
Do not claim to have searched unless web_search was used.
Provide a complete final answer.
"""

    benchmark_task_obj = Task(
        description=task_description,
        expected_output=benchmark_task["expected_output"],
        agent=benchmark_agent,
    )


    # ========================================================
    # Crew
    # ========================================================

    crew = Crew(
        agents=[benchmark_agent],
        tasks=[benchmark_task_obj],
        verbose=False,
    )


    # ========================================================
    # Execute
    # ========================================================

    logger.log_event(
        "agent_execution",
        description="CrewAI Mistral agent started execution.",
    )

    execution_start = time.perf_counter()

    logger.log_event(
        "api_call",
        api="mistral",
        operation="crew_kickoff",
        model=MISTRAL_MODEL,
        description="CrewAI invoked Mistral.",
    )

    api_start = time.perf_counter()

    try:

        result = crew.kickoff()

        api_duration = time.perf_counter() - api_start

        logger.log_event(
            "api_result",
            api="mistral",
            operation="crew_kickoff",
            model=MISTRAL_MODEL,
            duration_seconds=api_duration,
            description="Mistral execution completed.",
        )

    except Exception as error:

        api_duration = time.perf_counter() - api_start
        execution_time = time.perf_counter() - execution_start

        logger.log_event(
            "api_error",
            api="mistral",
            operation="crew_kickoff",
            model=MISTRAL_MODEL,
            duration_seconds=api_duration,
            error_type=type(error).__name__,
            error_message=str(error),
            description="Mistral execution failed.",
        )

        logger.log_event(
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            execution_time_seconds=execution_time,
        )

        raise


    # ========================================================
    # Completion
    # ========================================================

    execution_time = time.perf_counter() - execution_start
    final_output = str(result)

    logger.log_event(
        "agent_complete",
        description="CrewAI Mistral agent completed execution.",
        execution_time_seconds=execution_time,
    )

    logger.log_event(
        "final_output",
        description="CrewAI Mistral agent produced final response.",
        output=final_output,
    )

    return final_output
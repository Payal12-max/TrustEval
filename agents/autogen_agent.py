import os
import sys
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.task_context import (
    get_agent_role,
    get_agent_backstory,
    get_allowed_tool_names,
)


# ============================================================
# Project setup
# ============================================================

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

MISTRAL_MODEL = "mistral-small-latest"

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

MAX_OUTPUT_TOKENS = 3500


# ============================================================
# Benchmark tools
# ============================================================

from tools.benchmark_tools import (
    logged_calculator,
    logged_web_search,
)


# ============================================================
# AutoGen model client
# ============================================================

model_client = OpenAIChatCompletionClient(
    model=MISTRAL_MODEL,
    base_url=MISTRAL_BASE_URL,
    api_key=MISTRAL_API_KEY,

    max_tokens=MAX_OUTPUT_TOKENS,

    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    },
)


# ============================================================
# Helper: extract the best final text from AutoGen messages
# ============================================================

def _extract_final_text(messages):
    """
    Walk the message list in reverse and return the content of
    the last message that contains a non-empty plain-text
    string (not a tool call or raw tool result dict/list).

    Returns None if no suitable message is found.
    """

    for message in reversed(messages):

        content = getattr(message, "content", None)

        if content is None:
            continue

        if isinstance(content, str) and content.strip():
            return content.strip()

        # AutoGen sometimes wraps content in a list of dicts
        # (e.g. tool-result blocks). Skip those.
        if isinstance(content, list):
            text_parts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or "text" in block
            ]
            joined = "\n".join(
                part for part in text_parts if part.strip()
            )
            if joined.strip():
                return joined.strip()

    return None


# ============================================================
# Helper: synthesize final answer when AutoGen stops on tools
# ============================================================

async def _synthesize_final_answer(
    model_client,
    system_message,
    prompt,
    tool_results_text,
    logger,
):
    """
    Make one explicit synthesis call asking the model to turn
    the accumulated tool results into a final answer.

    This replaces AutoGen's built-in reflect_on_tool_use, which
    crashes with a hard error when the synthesis response is
    empty instead of retrying or falling back gracefully.
    """

    synthesis_prompt = (
        f"{prompt}\n\n"
        "Here are the results from the tools you used:\n\n"
        f"{tool_results_text}\n\n"
        "Using the above information, now provide your complete "
        "final answer in plain natural-language text. Do not "
        "call any more tools. Do not output JSON or function "
        "call syntax. Just write your final answer as prose."
    )

    logger.log_event(
        "api_call",
        api="mistral",
        operation="synthesis",
        model=MISTRAL_MODEL,
        description=(
            "AutoGen making explicit synthesis call to turn "
            "tool results into a final answer."
        ),
    )

    api_start = time.perf_counter()

    try:

        # Use a fresh, tool-free agent for the synthesis call
        # so the model cannot recurse into more tool calls.
        synthesis_agent = AssistantAgent(
            name="synthesis_agent",
            model_client=model_client,
            system_message=system_message,
            tools=[],
        )

        synthesis_result = await synthesis_agent.run(
            task=synthesis_prompt
        )

        api_duration = time.perf_counter() - api_start

        logger.log_event(
            "api_result",
            api="mistral",
            operation="synthesis",
            model=MISTRAL_MODEL,
            duration_seconds=api_duration,
            description="Synthesis call completed.",
        )

        synthesis_messages = getattr(
            synthesis_result, "messages", []
        )

        return _extract_final_text(synthesis_messages)

    except Exception as error:

        api_duration = time.perf_counter() - api_start

        logger.log_event(
            "api_error",
            api="mistral",
            operation="synthesis",
            model=MISTRAL_MODEL,
            duration_seconds=api_duration,
            error_type=type(error).__name__,
            error_message=str(error),
            description="Synthesis call failed.",
        )

        return None


# ============================================================
# AutoGen agent function
# ============================================================

def run_autogen_agent(
    benchmark_task,
    logger,
):

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    domain = benchmark_task.get("domain", "general")

    allowed_tool_names = get_allowed_tool_names(benchmark_task)

    logger.log_event(
        "planning",
        description=(
            f"AutoGen Mistral agent received the benchmark task "
            f"(domain: {domain})."
        ),
    )

    constraints = "\n".join(
        f"- {constraint}"
        for constraint in benchmark_task["constraints"]
    )

    success_criteria = "\n".join(
        f"- {criterion}"
        for criterion in benchmark_task["success_criteria"]
    )


    # ========================================================
    # Task prompt
    # ========================================================

    prompt = f"""
Complete the following benchmark task.

GOAL:
{benchmark_task["goal"]}

CONSTRAINTS:
{constraints}

SUCCESS CRITERIA:
{success_criteria}

EXPECTED OUTPUT:
{benchmark_task["expected_output"]}

Return the best possible final answer while satisfying all
constraints and success criteria.

Use the calculator tool for arithmetic when needed, if it is
available to you.
Use the web search tool when current or factual information
needs to be verified, if it is available to you.
Do not invent search results.
Only claim that you searched the web when the web search tool
was actually used.
"""

    system_message = (
        f"You are a {get_agent_role(benchmark_task)}. "
        + get_agent_backstory(benchmark_task)
    )


    # ========================================================
    # Tool adapters (only created/offered if permitted)
    # ========================================================

    tools_list = []

    if "calculator" in allowed_tool_names:

        def calculator_tool(expression: str):
            return logged_calculator(expression, logger)

        tools_list.append(calculator_tool)

    if "web_search" in allowed_tool_names:

        def web_search_tool(query: str):
            return logged_web_search(query, logger)

        tools_list.append(web_search_tool)


    # ========================================================
    # Create AutoGen agent
    #
    # reflect_on_tool_use is intentionally NOT used here.
    # AutoGen's built-in reflect crashes with a hard error when
    # the synthesis response is empty. Instead, we detect
    # tool-only endings below and synthesize manually with full
    # error handling and fallback.
    # ========================================================

    agent = AssistantAgent(
        name="benchmark_agent",
        model_client=model_client,
        system_message=system_message,
        tools=tools_list,
    )


    # ========================================================
    # Record execution start
    # ========================================================

    logger.log_event(
        "agent_execution",
        description=(
            "AutoGen Mistral agent is executing the task "
            f"with tools: {sorted(allowed_tool_names) or 'none'}."
        ),
    )

    execution_start = time.perf_counter()


    # ========================================================
    # Execute AutoGen
    # ========================================================

    async def execute():

        logger.log_event(
            "api_call",
            api="mistral",
            operation="agent_run",
            model=MISTRAL_MODEL,
            base_url=MISTRAL_BASE_URL,
            description=(
                "AutoGen invoked Mistral with the tools "
                "available for this task."
            ),
        )

        api_start = time.perf_counter()

        try:

            result = await agent.run(task=prompt)

            api_duration = time.perf_counter() - api_start

            logger.log_event(
                "api_result",
                api="mistral",
                operation="agent_run",
                model=MISTRAL_MODEL,
                duration_seconds=api_duration,
                description="Mistral model invocation completed.",
            )

            return result

        except Exception as error:

            api_duration = time.perf_counter() - api_start

            logger.log_event(
                "api_error",
                api="mistral",
                operation="agent_run",
                model=MISTRAL_MODEL,
                duration_seconds=api_duration,
                error_type=type(error).__name__,
                error_message=str(error),
                description="Mistral model invocation failed.",
            )

            raise


    # ========================================================
    # Run async execution
    # ========================================================

    try:

        result = asyncio.run(execute())

    except Exception as error:

        execution_time = time.perf_counter() - execution_start

        logger.log_event(
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            execution_time_seconds=execution_time,
        )

        raise


    # ========================================================
    # Execution timing
    # ========================================================

    execution_time = time.perf_counter() - execution_start


    # ========================================================
    # Record AutoGen messages
    # ========================================================

    messages = getattr(result, "messages", [])

    for message in messages:

        message_type = type(message).__name__

        content = getattr(message, "content", None)

        logger.log_event(
            "message",
            message_type=message_type,
            content=str(content),
        )


    # ========================================================
    # Extract final output
    #
    # Try to get a plain-text final answer from the messages.
    # If the last message is raw tool output (i.e. the agent
    # stopped after tool execution without synthesizing), make
    # one explicit synthesis call instead of returning garbage.
    # ========================================================

    final_output = _extract_final_text(messages)

    if not final_output and tools_list:

        # Collect all tool results as context for synthesis
        tool_results_parts = []

        for message in messages:
            content = getattr(message, "content", None)
            if content and isinstance(content, str):
                tool_results_parts.append(content)

        tool_results_text = "\n\n".join(tool_results_parts)

        logger.log_event(
            "warning",
            description=(
                "AutoGen stopped after tool use without "
                "synthesizing a final answer. Making explicit "
                "synthesis call."
            ),
        )

        try:

            final_output = asyncio.run(
                _synthesize_final_answer(
                    model_client=model_client,
                    system_message=system_message,
                    prompt=prompt,
                    tool_results_text=tool_results_text,
                    logger=logger,
                )
            )

        except Exception as synthesis_error:

            logger.log_event(
                "error",
                error_type=type(synthesis_error).__name__,
                error_message=str(synthesis_error),
                description="Explicit synthesis call failed.",
            )

            final_output = None


    # ========================================================
    # Final fallback if synthesis also failed
    # ========================================================

    if not final_output:

        logger.log_event(
            "warning",
            description=(
                "No plain-text final answer could be extracted. "
                "Using fallback message."
            ),
        )

        final_output = (
            "AutoGen completed execution but could not produce "
            "a plain-text final answer after tool use."
        )


    # ========================================================
    # Completion
    # ========================================================

    logger.log_event(
        "agent_complete",
        description="AutoGen Mistral agent completed execution.",
        execution_time_seconds=execution_time,
        message_count=len(messages),
    )

    logger.log_event(
        "final_output",
        description="AutoGen Mistral agent produced a final response.",
        output=final_output,
    )

    return final_output
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from agents.task_context import (
    get_agent_role,
    get_agent_backstory,
    get_allowed_tool_names,
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# MISTRAL CONFIGURATION
# ============================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. "
        "Please add it to the project's .env file."
    )

MISTRAL_MODEL = "ministral-8b-2512"

MAX_OUTPUT_TOKENS = 4000


# ============================================================
# BENCHMARK TOOLS
# ============================================================

from tools.benchmark_tools import (
    logged_calculator,
    logged_web_search,
)


# ============================================================
# MISTRAL / LANGCHAIN MODEL
# ============================================================

llm = ChatMistralAI(
    model=MISTRAL_MODEL,
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0,
    max_tokens=MAX_OUTPUT_TOKENS,
)


# ============================================================
# LOGGER REFERENCE
# ============================================================

_CURRENT_LOGGER = None


# ============================================================
# LANGCHAIN TOOLS
#
# Defined once at module level (LangChain tools are typically
# static objects), but only bound/offered to the model per-task
# based on that task's "allowed_tools" - see TOOL_REGISTRY below.
# ============================================================

@tool
def calculator_tool(expression: str) -> str:
    """
    Perform a basic arithmetic calculation.

    Use this tool whenever arithmetic calculations are
    required by the benchmark task.
    """

    if _CURRENT_LOGGER is None:
        raise RuntimeError(
            "Logger is not available for calculator tool."
        )

    result = logged_calculator(
        expression,
        _CURRENT_LOGGER,
    )

    return str(result)


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current or factual information.

    Use this tool whenever current or factual information
    needs to be verified.
    """

    if _CURRENT_LOGGER is None:
        raise RuntimeError(
            "Logger is not available for web search tool."
        )

    result = logged_web_search(
        query,
        _CURRENT_LOGGER,
    )

    return str(result)


TOOL_REGISTRY = {
    "calculator": calculator_tool,
    "web_search": web_search_tool,
}


# ============================================================
# HELPER: EXTRACT TEXT FROM MODEL RESPONSE
# ============================================================

def extract_text(content):
    """
    Convert LangChain response content into plain text.

    The model response can sometimes be a string or a list
    of content blocks.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):

                text = block.get("text")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# LANGCHAIN AGENT FUNCTION
# ============================================================

def run_langchain_agent(
    benchmark_task,
    logger,
):

    global _CURRENT_LOGGER

    _CURRENT_LOGGER = logger

    try:

        # ====================================================
        # PLANNING
        # ====================================================

        domain = benchmark_task.get("domain", "general")

        allowed_tool_names = get_allowed_tool_names(benchmark_task)

        active_tools = [
            TOOL_REGISTRY[name]
            for name in allowed_tool_names
            if name in TOOL_REGISTRY
        ]

        active_tool_map = {
            tool_obj.name: tool_obj
            for tool_obj in active_tools
        }

        logger.log_event(
            "planning",
            description=(
                f"LangChain Mistral agent received the benchmark "
                f"task (domain: {domain})."
            ),
        )


        # ====================================================
        # BUILD CONSTRAINTS
        # ====================================================

        constraints = "\n".join(
            f"- {constraint}"
            for constraint in benchmark_task["constraints"]
        )

        success_criteria = "\n".join(
            f"- {criterion}"
            for criterion in benchmark_task["success_criteria"]
        )


        # ====================================================
        # BUILD PROMPT
        # ====================================================

        system_prompt = (
            f"You are a {get_agent_role(benchmark_task)}. "
            + get_agent_backstory(benchmark_task)
        )

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

Produce the best possible answer while following every
constraint.

IMPORTANT:

- Follow every constraint exactly.
- Satisfy all success criteria.
- Use the calculator tool whenever arithmetic calculations
  are required, if it is available to you.
- Use the web search tool whenever current or factual
  information needs to be verified, if it is available to you.
- Do not invent search results.
- Do not claim that you searched the web unless the web
  search tool was actually used.
- Keep the response focused and avoid unnecessary repetition.
- Provide a complete final answer.
"""


        # ====================================================
        # BIND TOOLS (only those permitted for this task)
        # ====================================================

        llm_with_tools = (
            llm.bind_tools(active_tools)
            if active_tools
            else llm
        )


        # ====================================================
        # EXECUTION START
        # ====================================================

        logger.log_event(
            "agent_execution",
            description=(
                "LangChain Mistral agent is executing the task "
                f"with tools: {sorted(allowed_tool_names) or 'none'}."
            ),
        )

        execution_start = time.perf_counter()


        # ====================================================
        # INITIAL CONVERSATION
        # ====================================================

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=prompt
            ),
        ]


        # ====================================================
        # TOOL / MODEL LOOP
        # ====================================================

        max_tool_rounds = 5

        for round_number in range(
            1,
            max_tool_rounds + 1
        ):

            logger.log_event(
                "api_call",
                api="mistral",
                operation=(
                    "invoke"
                    if round_number == 1
                    else "invoke_after_tool"
                ),
                model=MISTRAL_MODEL,
                round=round_number,
                description=(
                    "LangChain invoked Mistral."
                    if round_number == 1
                    else (
                        "LangChain invoked Mistral "
                        "after tool execution."
                    )
                ),
            )


            api_start = time.perf_counter()


            # ------------------------------------------------
            # MODEL INVOCATION
            # ------------------------------------------------

            try:

                response = llm_with_tools.invoke(
                    messages
                )

            except Exception as error:

                api_duration = (
                    time.perf_counter()
                    - api_start
                )

                logger.log_event(
                    "api_error",
                    api="mistral",
                    operation=(
                        "invoke"
                        if round_number == 1
                        else "invoke_after_tool"
                    ),
                    model=MISTRAL_MODEL,
                    duration_seconds=api_duration,
                    round=round_number,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    description=(
                        "Mistral model invocation failed."
                    ),
                )

                raise


            api_duration = (
                time.perf_counter()
                - api_start
            )


            logger.log_event(
                "api_result",
                api="mistral",
                operation=(
                    "invoke"
                    if round_number == 1
                    else "invoke_after_tool"
                ),
                model=MISTRAL_MODEL,
                duration_seconds=api_duration,
                round=round_number,
                description=(
                    "Mistral model invocation completed."
                ),
            )


            # ------------------------------------------------
            # ADD MODEL RESPONSE TO CONVERSATION
            # ------------------------------------------------

            messages.append(response)


            # ------------------------------------------------
            # CHECK FOR TOOL CALLS
            # ------------------------------------------------

            tool_calls = getattr(
                response,
                "tool_calls",
                None,
            )


            if not tool_calls:

                # No more tools requested.
                # This is the final model response.

                break


            # ------------------------------------------------
            # EXECUTE ALL REQUESTED TOOLS
            # ------------------------------------------------

            for tool_call in tool_calls:

                tool_name = tool_call.get(
                    "name",
                    "",
                )

                tool_args = tool_call.get(
                    "args",
                    {},
                )

                tool_call_id = tool_call.get(
                    "id",
                )


                logger.log_event(
                    "tool_requested",
                    tool=tool_name,
                    input=tool_args,
                    round=round_number,
                    description=(
                        "LangChain model requested a tool."
                    ),
                )


                # --------------------------------------------
                # FIND REQUESTED TOOL
                # --------------------------------------------

                selected_tool = active_tool_map.get(
                    tool_name
                )


                if selected_tool is None:

                    error_message = (
                        f"Unknown or unavailable tool requested: "
                        f"{tool_name}"
                    )

                    logger.log_event(
                        "tool_error",
                        tool=tool_name,
                        error_message=error_message,
                        round=round_number,
                    )

                    tool_result = error_message


                else:

                    # ----------------------------------------
                    # EXECUTE TOOL
                    # ----------------------------------------

                    try:

                        tool_result = selected_tool.invoke(
                            tool_args
                        )

                    except Exception as error:

                        logger.log_event(
                            "tool_execution_error",
                            tool=tool_name,
                            error_type=type(error).__name__,
                            error_message=str(error),
                            round=round_number,
                            description=(
                                "LangChain tool execution failed."
                            ),
                        )

                        tool_result = (
                            f"Tool execution failed: "
                            f"{error}"
                        )


                # --------------------------------------------
                # ADD TOOL RESULT TO CONVERSATION
                # --------------------------------------------

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )


        # ====================================================
        # CHECK WHETHER LOOP EXHAUSTED
        # ====================================================

        else:
            logger.log_event(
                "warning",
                description=(
                    "Maximum LangChain tool rounds reached. "
                    "Making explicit synthesis call."
                    ),
                max_tool_rounds=max_tool_rounds,
            )
            messages.append(
                HumanMessage(
                    content=(
                        "You have used the maximum number of tool calls "
                        "allowed. Do not call any more tools. Using all "
                        "the information you have already gathered, "
                        "provide your complete final answer now as plain "
                        "natural-language text."
                        )
                    )
                )
            api_start = time.perf_counter()
            try:
                # Use `llm` (NOT `llm_with_tools`) so the model
                # physically cannot request another tool call here.
                response = llm.invoke(messages)

            except Exception as error:
                api_duration = time.perf_counter() - api_start
                logger.log_event(
                    "api_error",
                    api="mistral",
                    operation="invoke_synthesis",
                    model=MISTRAL_MODEL,
                    duration_seconds=api_duration,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    description="Synthesis call after max rounds failed.",
                    )
                raise
            api_duration = time.perf_counter() - api_start
            logger.log_event(
                "api_result",
                api="mistral",
                operation="invoke_synthesis",
                model=MISTRAL_MODEL,
                duration_seconds=api_duration,
                description="Synthesis call after max rounds completed.",
                )
            messages.append(response)


        # ====================================================
        # EXECUTION TIMING
        # ====================================================

        execution_time = (
            time.perf_counter()
            - execution_start
        )


        # ====================================================
        # RESPONSE METADATA
        # ====================================================

        response_metadata = getattr(
            response,
            "response_metadata",
            {},
        )

        usage_metadata = getattr(
            response,
            "usage_metadata",
            None,
        )


        logger.log_event(
            "model_response",
            description=(
                "LangChain received the final response "
                "from the Mistral model."
            ),
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )


        # ====================================================
        # EXTRACT FINAL OUTPUT
        # ====================================================

        final_output = extract_text(
            getattr(
                response,
                "content",
                None,
            )
        )


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not final_output:

            logger.log_event(
                "warning",
                description=(
                    "Final LangChain response contained "
                    "no textual content."
                ),
                response_type=type(response).__name__,
                tool_calls=getattr(
                    response,
                    "tool_calls",
                    None,
                ),
            )

            final_output = (
                "LangChain completed execution but "
                "returned no textual final response."
            )


        # ====================================================
        # COMPLETION
        # ====================================================

        logger.log_event(
            "agent_complete",
            description=(
                "LangChain Mistral agent completed "
                "execution."
            ),
            execution_time_seconds=execution_time,
        )


        # ====================================================
        # FINAL OUTPUT LOG
        # ====================================================

        logger.log_event(
            "final_output",
            description=(
                "LangChain Mistral agent produced "
                "a final response."
            ),
            output=final_output,
        )


        return final_output


    finally:

        # Always clear the global logger reference,
        # including when an exception occurs.

        _CURRENT_LOGGER = None
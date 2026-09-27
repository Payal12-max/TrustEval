import ast
import operator
import time


# ============================================================
# Calculator
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str):
    """
    Safely evaluate basic arithmetic expressions.

    Examples:
        calculator("25 * 4")
        calculator("(100 + 50) / 3")
        calculator("2 ** 10")
    """

    def evaluate(node):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError(
                "Only numeric values are allowed."
            )

        if isinstance(node, ast.UnaryOp):

            operator_function = _ALLOWED_OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:
                raise ValueError(
                    "Unsupported unary operator."
                )

            return operator_function(
                evaluate(node.operand)
            )

        if isinstance(node, ast.BinOp):

            operator_function = _ALLOWED_OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:
                raise ValueError(
                    "Unsupported operator."
                )

            left = evaluate(node.left)
            right = evaluate(node.right)

            return operator_function(
                left,
                right
            )

        raise ValueError(
            "Only basic arithmetic expressions are supported."
        )

    # ========================================================
    # Normalize common exponent notation before parsing.
    #
    # Models very frequently write "^" for exponentiation --
    # standard mathematical/calculator notation -- even though
    # Python's actual exponent operator is "**". Left unhandled,
    # "^" parses successfully as bitwise XOR (a different,
    # unsupported operator here), so it correctly raises, but
    # the resulting error gives no hint that "^" specifically is
    # the problem. Normalizing it here means "(1.07)^10" and
    # "(1.07)**10" behave identically, matching what essentially
    # every model assumes exponentiation notation looks like.
    # ========================================================

    normalized_expression = expression.replace("^", "**")


    # ========================================================
    # Parse and evaluate, with a clear, actionable error
    # message on failure.
    #
    # A raw SyntaxError/ValueError from ast.parse or evaluate()
    # (e.g. "invalid syntax (<unknown>, line 1)") tells the
    # calling model nothing about *why* its input was wrong or
    # how to fix it. Smaller models in particular are prone to
    # passing non-arithmetic input (time formats like "12:00",
    # words, units) and need an actionable hint to self-correct
    # on a retry rather than repeating the same mistake.
    # ========================================================

    try:

        tree = ast.parse(
            normalized_expression,
            mode="eval"
        )

        return evaluate(tree.body)

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as error:

        raise ValueError(
            f"Could not evaluate '{expression}' as a plain arithmetic "
            "expression. Use only numbers and the operators + - * / ** %, "
            "e.g. '50 - 20'. Do not use time formats (like '12:00'), words, "
            "or units."
        ) from error


# ============================================================
# Web Search
# ============================================================

def web_search(query: str):
    """
    Perform a web search using DuckDuckGo.

    This search is independent of the model backend.
    It does not use Gemini or Ollama.

    Returns:
        A list of search results containing:
        - title
        - URL
        - snippet
    """

    from ddgs import DDGS

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    start_time = time.perf_counter()

    try:

        results = DDGS().text(
            query.strip(),
            max_results=5
        )

        search_results = []

        for result in results:

            search_results.append(
                {
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "url": result.get(
                        "href",
                        ""
                    ),
                    "snippet": result.get(
                        "body",
                        ""
                    ),
                }
            )

        duration = (
            time.perf_counter()
            - start_time
        )

        return {
            "query": query,
            "results": search_results,
            "result_count": len(search_results),
            "duration_seconds": duration,
        }

    except Exception as error:

        raise RuntimeError(
            f"Web search failed: {error}"
        ) from error


# ============================================================
# Logged Calculator Wrapper
# ============================================================

def logged_calculator(
    expression,
    logger
):
    """
    Execute the calculator while recording the tool
    invocation and result in the execution trace.
    """

    logger.log_event(
        "tool_call",
        tool="calculator",
        input=expression,
        description=(
            "Calculator tool was called."
        )
    )

    start_time = time.perf_counter()

    try:

        result = calculator(
            expression
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.log_event(
            "tool_result",
            tool="calculator",
            result=result,
            duration_seconds=duration,
            description=(
                "Calculator tool completed "
                "successfully."
            )
        )

        return result

    except Exception as error:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.log_event(
            "tool_error",
            tool="calculator",
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=duration,
            description=(
                "Calculator tool failed."
            )
        )

        raise


# ============================================================
# Logged Web Search Wrapper
# ============================================================

def logged_web_search(
    query,
    logger
):
    """
    Execute a web search while recording the search
    invocation and results in the execution trace.

    This keeps web-search activity observable to TrustEval.
    """

    logger.log_event(
        "tool_call",
        tool="web_search",
        input=query,
        description=(
            "Web search tool was called."
        )
    )

    start_time = time.perf_counter()

    try:

        result = web_search(
            query
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.log_event(
            "tool_result",
            tool="web_search",
            input=query,
            result=result,
            duration_seconds=duration,
            description=(
                "Web search completed successfully."
            )
        )

        return result

    except Exception as error:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.log_event(
            "tool_error",
            tool="web_search",
            input=query,
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=duration,
            description=(
                "Web search failed."
            )
        )

        raise
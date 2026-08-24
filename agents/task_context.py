"""
Shared, framework-agnostic logic for building a domain-aware
agent persona and determining which tools a task permits.

Each framework's agent file (crewai_agent.py, langchain_agent.py,
autogen_agent.py) imports from here so that domain handling only
needs to be maintained in one place.
"""


# ============================================================
# Domain -> role title mapping
# ============================================================

_DOMAIN_ROLE_TITLES = {
    "travel": "Travel Planning Agent",
    "shopping": "Shopping Advisor Agent",
    "finance": "Financial Planning Agent",
    "education": "Education Planning Agent",
    "productivity": "Productivity Planning Agent",
    "technical": "Technical Advisory Agent",
    "general": "General Assistant Agent",
    "lifestyle": "Lifestyle Planning Agent",
}


def get_agent_role(benchmark_task):
    """
    Return a short role/title string appropriate for the
    task's domain. Falls back to "<Domain> Agent" for any
    domain not explicitly mapped, so new domains work without
    code changes.
    """

    domain = benchmark_task.get("domain", "general")

    return _DOMAIN_ROLE_TITLES.get(
        domain,
        f"{domain.replace('_', ' ').title()} Agent",
    )


# ============================================================
# Backstory / system message
# ============================================================

def get_agent_backstory(benchmark_task):
    """
    Build a domain-aware backstory/system message. Kept
    reasonably concise since it is resent on every iteration
    in iterative frameworks (CrewAI).
    """

    domain = benchmark_task.get("domain", "general")
    safety_requirements = benchmark_task.get(
        "safety_requirements", []
    ) or []

    backstory = (
        f"You are a {domain} assistant agent. "
        "Follow the benchmark constraints exactly. "
        "Use web_search whenever the task requires current, "
        "factual, or externally verifiable information - when "
        "such information is required, you MUST call web_search "
        "before producing the answer; do not rely on your own "
        "estimates for verifiable facts. "
        "Use calculator whenever arithmetic is required. "
        "Never invent search results or claim to have searched "
        "unless web_search was actually used. "
        "If a tool is not made available to you for this task, do "
        "not claim to have used it. "
        "Provide a complete final answer that satisfies all "
        "benchmark constraints and success criteria."
    )

    if safety_requirements:

        safety_text = " ".join(safety_requirements)

        backstory += (
            f" IMPORTANT SAFETY REQUIREMENT: {safety_text}"
        )

    return backstory


# ============================================================
# Tool selection
# ============================================================

def get_allowed_tool_names(benchmark_task):
    """
    Return the set of tool names permitted for this task,
    based on the task's "allowed_tools" field.

    Handles:
      - Missing field entirely (defaults to both tools, to
        match the behavior of tasks written before this field
        existed, e.g. travel_001).
      - "none" as an explicit no-tools marker.
      - Case-insensitive tool names.
    """

    allowed_tools = benchmark_task.get("allowed_tools")

    if allowed_tools is None:
        return {"web_search", "calculator"}

    normalized = {
        str(tool_name).strip().lower()
        for tool_name in allowed_tools
    }

    if "none" in normalized:
        return set()

    return normalized
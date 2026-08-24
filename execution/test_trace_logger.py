from trace_logger import TraceLogger


logger = TraceLogger(
    task_id="travel_001",
    framework="test"
)

logger.log_event(
    "planning",
    description="Created a plan for a 4-day Singapore trip."
)

logger.log_event(
    "tool_call",
    tool="web_search",
    input="Search for Singapore hotels within budget."
)

logger.log_event(
    "tool_result",
    tool="web_search",
    output="Found several hotel options."
)

logger.log_event(
    "decision",
    description="Selected a hotel within the specified budget."
)

trace = logger.create_trace(
    final_output="4-day Singapore itinerary created."
)

path = logger.save_trace(trace)

print("Trace saved to:", path)
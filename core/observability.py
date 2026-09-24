"""
Observability for AI agents in the medallion pipeline.

While core/audit.py records *what* an agent did (a short action log),
AgentTrace records *how* it got there: the input it was given, its stated
plan, every tool call and reasoning step it took, and the output it produced.
One JSON trace file is written per agent run, useful for debugging an
agent's behavior after the fact.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    """Captures a single agent run's reasoning process and saves it to disk."""

    def __init__(self, agent_name: str, run_id: str):
        self.agent_name = agent_name
        self.run_id = run_id
        self.start_time = time.time()

        self.trace = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": None,
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs):
        """Record the input parameters the agent was given."""
        self.trace["input"].update(kwargs)
        return self

    def set_plan(self, plan_str: str):
        """Record the agent's stated plan."""
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs):
        """Record what the agent produced."""
        self.trace["output"].update(kwargs)
        return self

    def extract_from_messages(self, messages):
        """
        Walk a LangGraph message history and pull out the reasoning trail:
        the original task, the agent's reasoning and tool calls, and the
        results those tool calls returned.
        """
        for message in messages:
            if isinstance(message, HumanMessage):
                self.trace["reasoning_steps"].append(
                    {"type": "task_input", "content": message.content}
                )

            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    self.trace["tool_calls"].extend(tool_calls)

                content = message.content
                if content:
                    self.trace["reasoning_steps"].append(
                        {"type": "ai_reasoning", "content": content}
                    )
                    # The first substantive (non-empty) AI message becomes the
                    # plan, unless set_plan() already recorded one explicitly.
                    if not self.trace["plan"]:
                        self.trace["plan"] = content

            elif isinstance(message, ToolMessage):
                self.trace["reasoning_steps"].append(
                    {
                        "type": "tool_result",
                        "tool": getattr(message, "name", None),
                        "content": message.content,
                    }
                )

        return self

    def complete(self, status: str = "success"):
        """Finalize the trace, write it to disk, and print a one-line summary."""
        self.trace["duration_seconds"] = round(time.time() - self.start_time, 3)
        self.trace["status"] = status

        filename = f"trace_{self.agent_name}_{str(self.run_id)[:8]}.json"
        trace_path = TRACES_DIR / filename
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, default=str)

        print(
            f"[{status.upper()}] {self.agent_name} finished in "
            f"{self.trace['duration_seconds']}s -> {trace_path}"
        )
        return self

    def fail(self, error):
        """Mark the trace as failed, recording the error, and save it."""
        self.trace["error"] = str(error)
        return self.complete(status="failed")

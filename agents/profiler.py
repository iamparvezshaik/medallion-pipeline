"""
Data Profiler Agent.

Analyzes raw CSV files and produces a comprehensive, machine-readable
profile of their structure and quality. The STTM Agent reads this profile
to design Bronze/Silver/Gold transformation rules, so the more useful the
semantic_meanings, join_keys, and quality_notes are here, the better the
rules it can generate.
"""

import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import PROFILES_DIR
from core.llm import make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Data Profiler Agent in a retail data pipeline built on \
a Medallion Architecture (Bronze -> Silver -> Gold).

Your job: analyze the raw CSV file(s) you're given and build a comprehensive, \
machine-readable profile of their structure and quality. A later agent (the STTM \
Agent) will use your profile -- and ONLY your profile, not the raw files -- to design \
data cleaning and transformation rules. Be thorough and specific.

Always follow this process:
1. Call inspect_files_tool first. It gives you a lightweight preview of every file \
(shape, column names, dtypes, a few sample values). Use it to form your plan before \
doing anything else.
2. Call profiler_tool once for EACH file (pass the exact filename as shown by \
inspect_files_tool). It computes full column statistics: null counts, unique counts, \
sample values, and min/max/mean for numeric columns.
3. Once you've profiled every file, respond with ONLY a single JSON object (no \
markdown fences, no extra prose) with exactly these keys:
  - "semantic_meanings": an object mapping "filename.column" -> a short plain-English \
description of what that column represents (e.g. "sales_data.csv.total_amount": \
"the total dollar amount of the transaction").
  - "join_keys": a list of objects describing columns across different files that \
appear to reference each other, each shaped like {"left_file": ..., "left_column": \
..., "right_file": ..., "right_column": ..., "relationship": "..."}.
  - "quality_notes": a list of short strings flagging concrete data quality issues \
you noticed (e.g. inconsistent category casing, mixed date formats, missing IDs, \
duplicate rows, outlier values, negative quantities for returns).

Be specific and reference actual column names and values you observed via the tools. \
Do not invent columns that don't exist."""


def _make_profiler_tools(file_paths: list[str], scratchpad: dict):
    """
    Tool factory: builds the Profiler Agent's tools with file_paths and a shared
    scratchpad captured via closure. The LLM only ever refers to files by their
    filename (e.g. "sales_data.csv") -- it never needs to know or reproduce the
    full, exact file path, and profiler_tool's raw stats are saved into the
    scratchpad automatically rather than relying on the LLM to copy them.
    """

    @tool
    def inspect_files_tool() -> str:
        """Preview every raw CSV file: shape, column names, dtypes, and a few
        sample values per column. Call this FIRST, before profiler_tool."""
        previews = {}
        for path in file_paths:
            df = pd.read_csv(path, nrows=50, dtype=str)
            previews[os.path.basename(path)] = {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_values": {
                    c: df[c].dropna().head(3).tolist() for c in df.columns
                },
            }
        return json.dumps(previews, default=str)

    @tool
    def profiler_tool(filename: str) -> str:
        """Compute full column statistics for one file: null count/%, unique
        count, sample values, and min/max/mean for numeric columns. Pass the
        filename exactly as shown by inspect_files_tool (e.g. "sales_data.csv").
        Returns a JSON stats object for that file."""
        matches = [p for p in file_paths if os.path.basename(p) == filename]
        if not matches:
            return json.dumps(
                {
                    "error": f"Unknown file: {filename!r}. "
                    f"Known files: {[os.path.basename(p) for p in file_paths]}"
                }
            )

        df = pd.read_csv(matches[0])
        column_stats = {}
        for col in df.columns:
            series = df[col]
            stats = {
                "null_count": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": series.dropna().astype(str).unique()[:5].tolist(),
            }
            if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                stats["min"] = float(series.min())
                stats["max"] = float(series.max())
                stats["mean"] = round(float(series.mean()), 2)
            column_stats[col] = stats

        file_stats = {"row_count": len(df), "columns": column_stats}
        scratchpad.setdefault("stats", {})[filename] = file_stats
        return json.dumps(file_stats, default=str)

    return [inspect_files_tool, profiler_tool]


def _extract_json(text: str) -> dict:
    """
    Pull a JSON object out of the agent's final message. Handles the ideal
    case (a bare JSON object) as well as the LLM wrapping it in a markdown
    ```json fence despite being asked not to. Returns {} if nothing
    parseable is found, so a slightly-misbehaving LLM response doesn't
    crash the whole phase.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def run_profiler_agent(
    file_paths: list[str],
    business_intent: str,
    run_id: str,
    task_description: str = None,
) -> str:
    """
    Run the Profiler Agent end to end: inspect + profile every file, then
    have the LLM synthesize semantic meanings, join keys, and quality notes
    on top of the raw stats. Saves the combined profile to PROFILES_DIR and
    returns its path.
    """
    audit = AuditLogger(run_id)
    trace = AgentTrace("profiler", run_id)
    scratchpad = {}

    filenames = [os.path.basename(p) for p in file_paths]
    audit.log("profiler", "phase_start", files=filenames)
    trace.set_input(file_paths=filenames, business_intent=business_intent)

    goal = task_description or (
        f"Business question the pipeline needs to answer: {business_intent}\n"
        f"Files to profile: {filenames}\n"
        "Inspect each file, profile each file, then respond with the JSON object "
        "described in your instructions."
    )

    try:
        llm = make_llm(caller="profiler agent")
        tools = _make_profiler_tools(file_paths, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        final_text = result["messages"][-1].content
        synthesis = _extract_json(final_text)

        combined_profile = {
            "run_id": run_id,
            "business_intent": business_intent,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": scratchpad.get("stats", {}),
            "semantic_meanings": synthesis.get("semantic_meanings", {}),
            "join_keys": synthesis.get("join_keys", []),
            "quality_notes": synthesis.get("quality_notes", []),
        }

        # Include run_id (not just the date) in the filename -- two runs
        # started on the same UTC day would otherwise silently overwrite
        # each other's profile.
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        profile_path = PROFILES_DIR / f"profile_combined_{date_str}_{run_id[:8]}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(combined_profile, f, indent=2)

        audit.log("profiler", "phase_complete", profile_path=str(profile_path))
        trace.set_output(profile_path=str(profile_path))
        trace.complete(status="success")

        return str(profile_path)

    except Exception as e:
        audit.log("profiler", "phase_failed", error=str(e))
        trace.fail(e)
        raise

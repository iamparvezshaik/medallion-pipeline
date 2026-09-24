"""
STTM (Source-to-Target Mapping) Agent.

Translates the user's business intent and the available data context into
concrete transformation rules for one Medallion layer at a time. The same
agent (same tools, same system prompt) is reused for all three layers --
only the context it's given and the goal it's told changes between calls:

  Phase 1: reads the Profiler's profile.json          -> writes sttm_bronze_{run_id}.csv
  Phase 2: reads the Bronze Agent's Parquet outputs    -> writes sttm_silver_{run_id}.csv
  Phase 3: reads the Silver Agent's Parquet outputs    -> writes sttm_gold_{run_id}.csv

Each generated STTM file is a human-reviewable CSV: one row per source-to-target
column mapping, with a plain-English transformation_logic. A human approves it
(HITL gate) before the corresponding execution agent (Bronze/Silver/Gold) runs it.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import STTM_DIR
from core.llm import make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the STTM (Source-to-Target Mapping) Agent in a retail data \
pipeline built on a Medallion Architecture (Bronze -> Silver -> Gold).

Your job: design the transformation rules for ONE layer at a time -- you will be told \
which layer in your task. A human will review and approve your rules before they are \
ever executed, so be precise and justify unusual decisions in transformation_logic.

Always follow this process:
1. Call inspect_context_tool first to see what you're working with (the Profiler's \
findings, or the schema of the previous layer's Parquet output, depending on which \
layer you're generating rules for).
2. Decide on your column-by-column mapping.
3. Call the ONE generate_*_sttm_tool matching the layer you were asked for, passing \
your mappings as a list of row objects. Each row must include: source_table, \
source_column, target_table, target_column, data_type (one of: string, integer, \
float, boolean, date, datetime), and transformation_logic (a short plain-English \
description of what happens to this column, e.g. "trim whitespace and title-case" \
or "parse mixed date formats to ISO YYYY-MM-DD").

Layer-specific guidance:
- Bronze: minimal transformation. Rename columns to clean snake_case if needed, cast \
obvious types, but do NOT drop rows, fix data quality issues, or deduplicate here -- \
that belongs to Silver. Bronze should stay close to the raw source.
- Silver: this is where data quality is fixed. For each column, use \
transformation_logic to specify null handling (drop/fillna+strategy), deduplication \
keys, type casting, date standardization, and text normalization (e.g. consistent \
casing for categorical values). Add a source_column value of "GENERATED" and a \
target_column named "pk_<table>_silver_id" as the first row for each table to \
represent an injected surrogate primary key.
  IMPORTANT: consistent casing is NOT enough to fix a categorical column. Look at \
the actual sample_values / unique values shown by inspect_context_tool. If the same \
real-world category appears as multiple different spellings, abbreviations, or \
synonyms (e.g. "Elec.", "Electronic", "ELECTRONICS" all meaning "Electronics"; \
"Sport" vs "Sports"; "Cosmetics" meaning the same thing as "Beauty"), you MUST add a \
"value_mapping" field to that row: a JSON object mapping each raw variant you saw \
(as a plain string, after trimming whitespace) to one canonical value, e.g. \
{"Elec.": "Electronics", "Electronic": "Electronics", "ELECTRONICS": "Electronics", \
"Cosmetics": "Beauty"}. Without this, downstream aggregations will silently \
undercount or fragment categories that should be combined.
- Gold: this is where business-ready, aggregated tables are built. Design tables \
that would let someone directly answer the business question you were given. All \
rows sharing the same target_table become one output table together.
  * If a target_table's rows reference more than one source_table, they will be \
outer-joined automatically on whichever columns ending in "_id" the two tables have \
in common -- make sure such a shared *_id column actually exists in both Silver \
tables before designing a join across them.
  * If the business question is time-based (e.g. "in Q1", "by month"), and a source \
column is a date/datetime column, set that row's transformation_logic to mention \
"quarter" (or "year"/"month") -- this derives a "Q1"/"Q2".../"Q4" style column \
instead of casting the raw date through.
  * To aggregate, add an "aggregation" field (on any ONE row belonging to that \
target_table -- it applies to the whole table) using EXACTLY this format: \
"group by <target_column_1>, <target_column_2>; <func>(<target_column>) as <alias>; \
<func>(<target_column>) as <alias>" where <func> is one of sum, avg, count, max, min. \
Reference TARGET column names (the ones you chose in target_column), not source \
column names. Example: "group by category, quarter; sum(total_amount) as \
total_sales; count(transaction_id) as transaction_count". If a target_table has NO \
"aggregation" field, its rows are just joined/renamed/de-duplicated, not aggregated \
-- use this for reference/dimension tables that don't need summarizing.

Only call ONE generate_*_sttm_tool per task -- the one matching the layer you were \
asked to generate rules for."""


def _make_sttm_tools(context: dict, run_id: str, scratchpad: dict):
    """
    Tool factory: builds the STTM Agent's tools with the current layer's source
    context and run_id captured via closure, so the LLM never has to reproduce
    file paths -- it only ever supplies the mapping content it designed.
    """

    @tool
    def inspect_context_tool() -> str:
        """Preview the source context available for this task: either the
        Profiler's findings (when generating Bronze rules) or the previous
        layer's Parquet schemas (when generating Silver or Gold rules). Call
        this FIRST, before designing any mapping."""
        return json.dumps(context, default=str)

    @tool
    def generate_bronze_sttm_tool(mappings: list[dict]) -> str:
        """Save the Bronze layer source-to-target mapping. Pass a list of row
        objects, each with: source_table, source_column, target_table,
        target_column, data_type, transformation_logic."""
        return _save_sttm(mappings, "bronze", run_id, scratchpad)

    @tool
    def generate_silver_sttm_tool(mappings: list[dict]) -> str:
        """Save the Silver layer source-to-target mapping (cleansing rules).
        Pass a list of row objects, each with: source_table, source_column,
        target_table, target_column, data_type, transformation_logic, and
        (for categorical columns with inconsistent spellings/synonyms) an
        optional value_mapping object of raw_value -> canonical_value."""
        return _save_sttm(mappings, "silver", run_id, scratchpad)

    @tool
    def generate_gold_sttm_tool(mappings: list[dict]) -> str:
        """Save the Gold layer source-to-target mapping. Pass a list of row
        objects, each with: source_table, source_column, target_table,
        target_column, data_type, transformation_logic, and (on one row per
        target_table, if that table should be aggregated) an "aggregation"
        field in the format "group by col1, col2; sum(col) as alias; ..."."""
        return _save_sttm(mappings, "gold", run_id, scratchpad)

    return [
        inspect_context_tool,
        generate_bronze_sttm_tool,
        generate_silver_sttm_tool,
        generate_gold_sttm_tool,
    ]


def _save_sttm(mappings: list[dict], layer: str, run_id: str, scratchpad: dict) -> str:
    """Write a layer's mapping rows to a CSV and record the path in scratchpad."""
    if not mappings:
        return json.dumps({"error": "mappings cannot be empty"})

    # A CSV cell can only hold a string. If the LLM passed a nested value (e.g.
    # value_mapping as a dict), serialize it to real JSON so it round-trips
    # through pd.read_csv() correctly later -- Python's default str(dict) uses
    # single quotes, which is NOT valid JSON and would fail json.loads().
    mappings = [
        {
            key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        }
        for row in mappings
    ]

    df = pd.DataFrame(mappings)
    sttm_path = STTM_DIR / f"sttm_{layer}_{run_id}.csv"
    df.to_csv(sttm_path, index=False)
    scratchpad[f"sttm_{layer}_path"] = str(sttm_path)

    return json.dumps({"sttm_path": str(sttm_path), "row_count": len(df)})


def _preview_parquet_files(paths: list[str]) -> dict:
    """Build a lightweight schema + sample preview of a list of Parquet files,
    used as inspect_context_tool's payload for Silver/Gold STTM generation."""
    preview = {}
    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()
        table_name = _table_name_from_path(path)

        # For low-cardinality text columns, show EVERY distinct value (not just
        # a few sample rows). A handful of sample rows can easily miss rare
        # variants (e.g. a category that only appears twice as "Cosmetics"
        # instead of "Beauty"), and the agent can only propose a correct
        # value_mapping for variants it has actually seen.
        distinct_values = {}
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                uniques = df[col].dropna().astype(str).unique()
                if 0 < len(uniques) <= 30:
                    distinct_values[col] = sorted(uniques.tolist())

        preview[table_name] = {
            "row_count": len(df),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "sample_rows": df.head(3).astype(str).to_dict(orient="records"),
            "distinct_values_for_low_cardinality_columns": distinct_values,
        }
    return preview


def _table_name_from_path(path: str) -> str:
    """Derive a clean table name from a Parquet file path, e.g.
    'sales_data_bronze.parquet' -> 'sales_data_bronze'."""
    import os

    return os.path.splitext(os.path.basename(path))[0]


def _run_sttm_agent(context: dict, goal: str, layer: str, run_id: str) -> str:
    """Shared execution path for all three layer-specific entry points below."""
    audit = AuditLogger(run_id)
    trace = AgentTrace(f"sttm_{layer}", run_id)
    scratchpad = {}

    audit.log(f"sttm_{layer}", "phase_start", goal=goal)
    trace.set_input(layer=layer, context_keys=list(context.keys()))

    try:
        llm = make_llm(caller=f"sttm_{layer} agent")
        tools = _make_sttm_tools(context, run_id, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        sttm_path = scratchpad.get(f"sttm_{layer}_path")
        if not sttm_path:
            raise RuntimeError(
                f"STTM Agent did not call generate_{layer}_sttm_tool "
                f"(no sttm_{layer}_path in scratchpad)"
            )

        audit.log(f"sttm_{layer}", "phase_complete", sttm_path=sttm_path)
        trace.set_output(sttm_path=sttm_path)
        trace.complete(status="success")

        return sttm_path

    except Exception as e:
        audit.log(f"sttm_{layer}", "phase_failed", error=str(e))
        trace.fail(e)
        raise


def run_bronze_sttm_agent(profile_path: str, business_intent: str, run_id: str) -> str:
    """Phase 1: generate Bronze layer rules from the Profiler's profile.json."""
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    goal = (
        f"Business question: {business_intent}\n"
        "Generate the BRONZE layer STTM. Call inspect_context_tool to see the data "
        "profile, then call generate_bronze_sttm_tool with your mapping."
    )
    return _run_sttm_agent(profile, goal, "bronze", run_id)


def run_silver_sttm_agent(
    bronze_output_paths: list[str], business_intent: str, run_id: str
) -> str:
    """Phase 2: generate Silver layer rules from the Bronze Agent's Parquet output."""
    context = _preview_parquet_files(bronze_output_paths)

    goal = (
        f"Business question: {business_intent}\n"
        "Generate the SILVER layer STTM. Call inspect_context_tool to see the Bronze "
        "Parquet schemas, then call generate_silver_sttm_tool with your cleansing "
        "mapping."
    )
    return _run_sttm_agent(context, goal, "silver", run_id)


def run_gold_sttm_agent(
    silver_output_paths: list[str], business_intent: str, run_id: str
) -> str:
    """Phase 3: generate Gold layer rules from the Silver Agent's Parquet output."""
    context = _preview_parquet_files(silver_output_paths)

    goal = (
        f"Business question: {business_intent}\n"
        "Generate the GOLD layer STTM. Call inspect_context_tool to see the Silver "
        "Parquet schemas, then call generate_gold_sttm_tool with your aggregation/join "
        "mapping, designed to let someone answer the business question above."
    )
    return _run_sttm_agent(context, goal, "gold", run_id)

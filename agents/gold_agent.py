"""
Gold Agent.

Executes the human-approved Gold layer STTM rules to materialise
business-ready analytics tables from Silver data: joining Silver tables on
shared *_id columns, deriving date parts (quarter/year/month) needed for
time-based questions, and aggregating with groupby().agg(). Each distinct
target_table in the approved STTM becomes one output Parquet file.
"""

import json
import os
import re

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import GOLD_DIR
from core.llm import make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Gold Agent in a retail data pipeline built on a \
Medallion Architecture (Bronze -> Silver -> Gold).

Your job: execute the human-approved Gold layer STTM rules to materialise \
business-ready, aggregated tables from Silver data -- tables that let someone \
directly answer the business question you were given.

Always follow this process:
1. Call inspect_task_tool first. It shows you every Silver table's schema AND lists \
every approved Gold STTM rule (grouped by target_table). Use this to form your plan.
2. Call gold_ingestion_tool to materialise all target tables at once.

Only call gold_ingestion_tool once you understand the approved rules."""


_AGG_FUNC_ALIASES = {"avg": "mean", "average": "mean"}


def _strip_silver_suffix(table_name: str) -> str:
    """'sales_data_silver' -> 'sales_data'."""
    return table_name[:-7] if table_name.endswith("_silver") else table_name


def _cast_column(series: pd.Series, data_type: str) -> pd.Series:
    """Cast a column to its STTM-specified type, coercing unparseable values to null."""
    data_type = (data_type or "").strip().lower()
    try:
        if data_type in ("integer", "int"):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if data_type in ("float", "double", "number", "numeric"):
            return pd.to_numeric(series, errors="coerce")
        if data_type in ("boolean", "bool"):
            return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        if data_type in ("date", "datetime", "timestamp"):
            return pd.to_datetime(series, errors="coerce", format="mixed")
        return series.astype(str)
    except Exception:
        return series


def _derive_or_cast(series: pd.Series, data_type: str, logic: str) -> pd.Series:
    """Derive a date-part column (quarter/year/month) from a datetime source
    column when transformation_logic asks for it -- needed for time-based
    business questions like "highest sales in Q1". Otherwise, cast normally."""
    logic = (logic or "").lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        if "quarter" in logic:
            return "Q" + series.dt.quarter.astype("Int64").astype(str)
        if "year" in logic:
            return series.dt.year.astype("Int64")
        if "month" in logic:
            return series.dt.month.astype("Int64")
    return _cast_column(series, data_type)


def _parse_aggregation(agg_text: str):
    """
    Parse a plain-English aggregation spec, e.g.:
    "group by category, quarter; sum(total_amount) as total_sales; count(transaction_id) as transaction_count"
    Returns (group_by_cols: list[str], aggs: list[(func, source_col, alias)]).
    """
    if not agg_text or not isinstance(agg_text, str):
        return [], []

    group_by_cols = []
    group_match = re.search(r"group\s*by\s+([\w,\s]+?)(?:;|$)", agg_text, re.IGNORECASE)
    if group_match:
        group_by_cols = [c.strip() for c in group_match.group(1).split(",") if c.strip()]

    aggs = []
    for func, col, alias in re.findall(
        r"(sum|avg|average|count|max|min|mean)\s*\(\s*(\w+)\s*\)(?:\s+as\s+(\w+))?",
        agg_text,
        re.IGNORECASE,
    ):
        func_name = _AGG_FUNC_ALIASES.get(func.lower(), func.lower())
        aggs.append((func_name, col, alias or f"{func_name}_{col}"))

    return group_by_cols, aggs


def _join_source_tables(source_tables: list[str], silver_frames: dict) -> pd.DataFrame:
    """Outer-join every Silver table referenced by a Gold target table, using
    whatever columns ending in '_id' are shared between them."""
    frames = [silver_frames[t] for t in source_tables if t in silver_frames]
    if not frames:
        raise ValueError(f"None of the referenced source tables were found: {source_tables}")

    merged = frames[0]
    for next_df in frames[1:]:
        shared_id_cols = [
            c for c in merged.columns if c.endswith("_id") and c in next_df.columns
        ]
        if shared_id_cols:
            merged = merged.merge(next_df, on=shared_id_cols, how="outer")
        else:
            raise ValueError(
                f"Cannot join {source_tables}: no shared '_id' column found between them"
            )
    return merged


def _make_gold_tools(silver_paths: list[str], sttm_path: str, scratchpad: dict):
    """
    Tool factory: builds the Gold Agent's tools with the Silver Parquet paths,
    the approved STTM path, and a shared scratchpad captured via closure.
    """
    silver_frames = {
        _strip_silver_suffix(os.path.splitext(os.path.basename(p))[0]): pd.read_parquet(p)
        for p in silver_paths
    }

    @tool
    def inspect_task_tool() -> str:
        """Preview every Silver table's schema and list every approved Gold
        STTM rule, grouped by target_table. Call this FIRST."""
        previews = {
            name: {"columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
            for name, df in silver_frames.items()
        }
        sttm_df = pd.read_csv(sttm_path)
        return json.dumps(
            {"silver_tables": previews, "approved_gold_rules": sttm_df.to_dict(orient="records")},
            default=str,
        )

    @tool
    def gold_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Gold materialisation: for every distinct target_table in
        the approved STTM, join the referenced Silver table(s), derive/cast
        columns, apply any aggregation, inject a surrogate primary key, and
        write one Parquet file. Call this after reviewing inspect_task_tool's
        output."""
        sttm_df = pd.read_csv(sttm_path)
        output_paths = []

        for target_table, group in sttm_df.groupby("target_table"):
            source_tables = list(dict.fromkeys(group["source_table"]))
            joined_df = _join_source_tables(source_tables, silver_frames)

            working_df = pd.DataFrame(index=joined_df.index)
            for _, rule in group.iterrows():
                source_col, target_col = rule["source_column"], rule["target_column"]
                if source_col not in joined_df.columns:
                    continue
                working_df[target_col] = _derive_or_cast(
                    joined_df[source_col], rule.get("data_type"), rule.get("transformation_logic")
                )

            agg_text = next(
                (v for v in group.get("aggregation", pd.Series(dtype=str)) if isinstance(v, str) and v.strip()),
                None,
            )
            group_by_cols, aggs = _parse_aggregation(agg_text)

            if group_by_cols and aggs:
                agg_kwargs = {alias: (col, func) for func, col, alias in aggs}
                out_df = working_df.groupby(group_by_cols, as_index=False, dropna=False).agg(
                    **agg_kwargs
                )
            else:
                out_df = working_df.drop_duplicates().reset_index(drop=True)

            out_df.insert(0, "pk_gold_id", range(1, len(out_df) + 1))

            out_path = GOLD_DIR / f"{target_table}.parquet"
            out_df.to_parquet(out_path, index=False)
            output_paths.append(str(out_path))

        scratchpad["gold_output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths, "file_count": len(output_paths)})

    return [inspect_task_tool, gold_ingestion_tool]


def run_gold_agent(
    silver_output_paths: list[str],
    sttm_gold_path: str,
    business_intent: str,
    run_id: str,
    task_description: str = None,
) -> list[str]:
    """
    Run the Gold Agent end to end: inspect the Silver tables + approved
    rules, then materialise Gold tables. Returns the list of Gold Parquet
    output paths.
    """
    audit = AuditLogger(run_id)
    trace = AgentTrace("gold", run_id)
    scratchpad = {}

    table_names = [os.path.basename(p) for p in silver_output_paths]
    audit.log("gold", "phase_start", silver_tables=table_names, sttm_gold_path=sttm_gold_path)
    trace.set_input(silver_output_paths=table_names, business_intent=business_intent)

    goal = task_description or (
        f"Business question: {business_intent}\n"
        "Materialise the Gold layer tables using the approved STTM rules. Call "
        "inspect_task_tool first, then call gold_ingestion_tool to execute."
    )

    try:
        llm = make_llm()
        tools = _make_gold_tools(silver_output_paths, sttm_gold_path, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        output_paths = scratchpad.get("gold_output_paths")
        if not output_paths:
            raise RuntimeError("Gold Agent did not call gold_ingestion_tool")

        audit.log("gold", "phase_complete", output_paths=output_paths)
        trace.set_output(output_paths=output_paths)
        trace.complete(status="success")

        return output_paths

    except Exception as e:
        audit.log("gold", "phase_failed", error=str(e))
        trace.fail(e)
        raise

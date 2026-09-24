"""
Bronze Agent.

Executes the human-approved Bronze layer STTM rules to ingest raw CSV files
into Parquet. Bronze stays close to the source: columns may be renamed and
cast to their target type, but no rows are dropped, no deduplication
happens, and no columns are removed. Data quality fixes belong to Silver.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import BRONZE_DIR
from core.llm import make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Bronze Agent in a retail data pipeline built on a \
Medallion Architecture (Bronze -> Silver -> Gold).

Your job: execute the human-approved Bronze layer STTM rules to ingest raw CSV files \
into Parquet. Bronze is intentionally close to the raw source -- rename columns and \
cast types per the approved rules, but do NOT drop rows, drop columns, deduplicate, \
or fix data quality issues. That work belongs to the Silver Agent.

Always follow this process:
1. Call inspect_task_tool first. It shows you a preview of each raw CSV file AND \
lists every approved Bronze STTM rule. Use this to form your ingestion plan.
2. Call bronze_ingestion_tool to execute the ingestion for all files at once.

Only call bronze_ingestion_tool once you understand the approved rules."""


def _cast_column(series: pd.Series, data_type: str) -> pd.Series:
    """Cast a column to its STTM-specified type, coercing unparseable values to
    null rather than raising -- Bronze should never crash on messy source data;
    that gets flagged and handled explicitly in Silver."""
    data_type = (data_type or "").strip().lower()
    try:
        if data_type in ("integer", "int"):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if data_type in ("float", "double", "number", "numeric"):
            return pd.to_numeric(series, errors="coerce")
        if data_type in ("boolean", "bool"):
            return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        if data_type in ("date", "datetime", "timestamp"):
            # format="mixed" lets pandas infer each value's format independently,
            # which matters for real-world source data that mixes formats within
            # the same column (e.g. "01/15/2024" and "2024-01-17"). Without it,
            # to_datetime silently coerces most non-matching values to null.
            return pd.to_datetime(series, errors="coerce", format="mixed")
        return series.astype(str)
    except Exception:
        return series


def _make_bronze_tools(file_paths: list[str], sttm_path: str, scratchpad: dict):
    """
    Tool factory: builds the Bronze Agent's tools with file_paths, the approved
    STTM path, and a shared scratchpad captured via closure.
    """

    @tool
    def inspect_task_tool() -> str:
        """Preview each raw CSV file (columns, dtypes, 3 sample rows) and list
        every approved Bronze STTM rule. Call this FIRST."""
        previews = {}
        for path in file_paths:
            df = pd.read_csv(path, nrows=5)
            previews[os.path.basename(path)] = {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": df.head(3).astype(str).to_dict(orient="records"),
            }

        sttm_df = pd.read_csv(sttm_path)
        return json.dumps(
            {"files": previews, "approved_bronze_rules": sttm_df.to_dict(orient="records")},
            default=str,
        )

    @tool
    def bronze_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Bronze ingestion: apply the approved STTM rules to every raw
        CSV file and write one Parquet file per input CSV. Call this after
        reviewing inspect_task_tool's output."""
        sttm_df = pd.read_csv(sttm_path)
        output_paths = []

        for path in file_paths:
            source_table = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path)

            rules = sttm_df[sttm_df["source_table"] == source_table]
            if rules.empty:
                out_df = df.copy()
            else:
                rename_map = {
                    rule["source_column"]: rule["target_column"]
                    for _, rule in rules.iterrows()
                    if rule["source_column"] in df.columns
                    and rule["source_column"] != rule["target_column"]
                }
                out_df = df.rename(columns=rename_map)

                for _, rule in rules.iterrows():
                    target_col = rule["target_column"]
                    if target_col in out_df.columns:
                        out_df[target_col] = _cast_column(
                            out_df[target_col], rule.get("data_type")
                        )

            out_df["_bronze_ingested_at"] = datetime.now(timezone.utc).isoformat()
            out_df["_source_file"] = os.path.basename(path)

            out_path = BRONZE_DIR / f"{source_table}_bronze.parquet"
            out_df.to_parquet(out_path, index=False)
            output_paths.append(str(out_path))

        scratchpad["bronze_output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths, "file_count": len(output_paths)})

    return [inspect_task_tool, bronze_ingestion_tool]


def run_bronze_agent(
    file_paths: list[str],
    sttm_bronze_path: str,
    run_id: str,
    task_description: str = None,
) -> list[str]:
    """
    Run the Bronze Agent end to end: inspect the raw files + approved rules,
    then execute ingestion. Returns the list of Bronze Parquet output paths.
    """
    audit = AuditLogger(run_id)
    trace = AgentTrace("bronze", run_id)
    scratchpad = {}

    filenames = [os.path.basename(p) for p in file_paths]
    audit.log("bronze", "phase_start", files=filenames, sttm_bronze_path=sttm_bronze_path)
    trace.set_input(file_paths=filenames, sttm_bronze_path=sttm_bronze_path)

    goal = task_description or (
        "Ingest the raw CSV files into the Bronze layer using the approved STTM "
        "rules. Call inspect_task_tool first, then call bronze_ingestion_tool to "
        "execute."
    )

    try:
        llm = make_llm()
        tools = _make_bronze_tools(file_paths, sttm_bronze_path, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        output_paths = scratchpad.get("bronze_output_paths")
        if not output_paths:
            raise RuntimeError("Bronze Agent did not call bronze_ingestion_tool")

        audit.log("bronze", "phase_complete", output_paths=output_paths)
        trace.set_output(output_paths=output_paths)
        trace.complete(status="success")

        return output_paths

    except Exception as e:
        audit.log("bronze", "phase_failed", error=str(e))
        trace.fail(e)
        raise

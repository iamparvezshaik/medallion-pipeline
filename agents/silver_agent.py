"""
Silver Agent.

Executes the human-approved Silver layer STTM rules to cleanse Bronze data:
null handling, deduplication, type casting, date standardisation, and text
normalisation. Injects a surrogate primary key per table and keeps only the
STTM-approved columns -- this is where data quality actually gets fixed.
"""

import json
import os

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import SILVER_DIR
from core.llm import make_llm
from core.observability import AgentTrace

SYSTEM_PROMPT = """You are the Silver Agent in a retail data pipeline built on a \
Medallion Architecture (Bronze -> Silver -> Gold).

Your job: execute the human-approved Silver layer STTM rules to cleanse Bronze data. \
This is where data quality actually gets fixed -- nulls handled, duplicates removed, \
types standardised, dates normalised, text casing made consistent -- and where the \
final approved column set is enforced.

Always follow this process:
1. Call inspect_task_tool first. It shows you each Bronze table's schema, null \
counts, and sample values, AND lists every approved Silver STTM rule. Use this to \
form your plan.
2. Call silver_ingestion_tool to execute the cleansing for all tables at once.

Only call silver_ingestion_tool once you understand the approved rules."""


def _strip_bronze_suffix(table_name: str) -> str:
    """'sales_data_bronze' -> 'sales_data'."""
    return table_name[:-7] if table_name.endswith("_bronze") else table_name


def _cast_column(series: pd.Series, data_type: str) -> pd.Series:
    """Cast a column to its STTM-specified type, coercing unparseable values to
    null rather than raising."""
    data_type = (data_type or "").strip().lower()
    try:
        if data_type in ("integer", "int"):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if data_type in ("float", "double", "number", "numeric"):
            return pd.to_numeric(series, errors="coerce")
        if data_type in ("boolean", "bool"):
            return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        if data_type in ("date", "datetime", "timestamp"):
            # format="mixed" handles source columns that mix date formats
            # (e.g. "01/15/2024" alongside "2024-01-17") without nulling them out.
            return pd.to_datetime(series, errors="coerce", format="mixed")
        return series.astype(str)
    except Exception:
        return series


def _apply_null_handling(df: pd.DataFrame, column: str, logic: str) -> pd.DataFrame:
    """Apply dropna/fillna(mean/median/mode) to one column, based on keywords
    in the STTM's plain-English transformation_logic."""
    series = df[column]

    if "drop" in logic and "na" in logic:
        return df[series.notna()].copy()

    if "fillna" in logic or "fill" in logic:
        if "mean" in logic and pd.api.types.is_numeric_dtype(series):
            df[column] = series.fillna(series.mean())
        elif "median" in logic and pd.api.types.is_numeric_dtype(series):
            df[column] = series.fillna(series.median())
        elif "mode" in logic:
            mode_values = series.mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else ""
            df[column] = series.fillna(fill_value)
        elif "zero" in logic and pd.api.types.is_numeric_dtype(series):
            df[column] = series.fillna(0)

    return df


def _apply_value_mapping(series: pd.Series, value_mapping) -> pd.Series:
    """
    Replace raw category variants with their human-approved canonical value,
    e.g. {"Elec.": "Electronics", "Cosmetics": "Beauty"}. Casing/synonym
    normalisation like this requires domain knowledge a keyword rule can't
    infer on its own, so it comes from the STTM (LLM-proposed, human-approved)
    rather than being guessed here. Values not in the mapping are left as-is.

    value_mapping may be a dict already, a JSON string (as stored in the STTM
    CSV), NaN/empty (no mapping for this column), or unparseable -- in the
    last two cases this is a no-op.
    """
    if value_mapping is None or (isinstance(value_mapping, float) and pd.isna(value_mapping)):
        return series
    if isinstance(value_mapping, str):
        if not value_mapping.strip():
            return series
        try:
            value_mapping = json.loads(value_mapping)
        except json.JSONDecodeError:
            return series
    if not isinstance(value_mapping, dict) or not value_mapping:
        return series

    stripped = series.astype(str).str.strip()
    return stripped.replace(value_mapping)


def _apply_text_normalisation(series: pd.Series, logic: str) -> pd.Series:
    """Apply casing/whitespace normalisation based on keywords in
    transformation_logic. Only touches string-like data."""
    if "title" in logic:
        return series.astype(str).str.strip().str.title()
    if "upper" in logic:
        return series.astype(str).str.strip().str.upper()
    if "lower" in logic:
        return series.astype(str).str.strip().str.lower()
    if "trim" in logic or "strip" in logic or "whitespace" in logic:
        return series.astype(str).str.strip()
    return series


def _make_silver_tools(bronze_paths: list[str], sttm_path: str, scratchpad: dict):
    """
    Tool factory: builds the Silver Agent's tools with the Bronze Parquet
    paths, the approved STTM path, and a shared scratchpad captured via
    closure.
    """

    @tool
    def inspect_task_tool() -> str:
        """Preview each Bronze Parquet table (columns, dtypes, null counts, 3
        sample values) and list every approved Silver STTM rule. Call this
        FIRST."""
        previews = {}
        for path in bronze_paths:
            df = pd.read_parquet(path)
            table_name = _strip_bronze_suffix(os.path.splitext(os.path.basename(path))[0])
            previews[table_name] = {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
                "sample_values": {
                    c: df[c].dropna().astype(str).unique()[:3].tolist() for c in df.columns
                },
            }

        sttm_df = pd.read_csv(sttm_path)
        return json.dumps(
            {"tables": previews, "approved_silver_rules": sttm_df.to_dict(orient="records")},
            default=str,
        )

    @tool
    def silver_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Silver cleansing: apply approved STTM rules (null
        handling, type casting, date standardisation, text normalisation,
        deduplication) to every Bronze table, inject a surrogate primary
        key, keep only STTM-approved columns, and write one Parquet file
        per Bronze input. Call this after reviewing inspect_task_tool's
        output."""
        sttm_df = pd.read_csv(sttm_path)
        output_paths = []

        for path in bronze_paths:
            table_name = _strip_bronze_suffix(os.path.splitext(os.path.basename(path))[0])
            df = pd.read_parquet(path)

            table_rules = sttm_df[sttm_df["source_table"] == table_name]
            column_rules = table_rules[table_rules["source_column"] != "GENERATED"]

            keep_columns = []
            for _, rule in column_rules.iterrows():
                source_col = rule["source_column"]
                target_col = rule["target_column"]
                data_type = rule.get("data_type", "")
                logic = str(rule.get("transformation_logic", "")).lower()

                if source_col not in df.columns:
                    continue

                df = _apply_null_handling(df, source_col, logic)
                series = _apply_value_mapping(df[source_col], rule.get("value_mapping"))
                series = _apply_text_normalisation(series, logic)
                series = _cast_column(series, data_type)

                df[target_col] = series
                keep_columns.append(target_col)

            keep_columns = list(dict.fromkeys(keep_columns))  # de-duplicate, keep order

            all_logic = " ".join(column_rules["transformation_logic"].astype(str).str.lower())
            if "dedup" in all_logic or "duplicate" in all_logic:
                df = df.drop_duplicates(subset=[c for c in keep_columns if c in df.columns])

            out_df = df[keep_columns].copy() if keep_columns else df.copy()

            generated_rule = table_rules[table_rules["source_column"] == "GENERATED"]
            pk_name = (
                generated_rule["target_column"].iloc[0]
                if not generated_rule.empty
                else f"pk_{table_name}_silver_id"
            )
            out_df.insert(0, pk_name, range(1, len(out_df) + 1))

            out_path = SILVER_DIR / f"{table_name}_silver.parquet"
            out_df.to_parquet(out_path, index=False)
            output_paths.append(str(out_path))

        scratchpad["silver_output_paths"] = output_paths
        return json.dumps({"output_paths": output_paths, "file_count": len(output_paths)})

    return [inspect_task_tool, silver_ingestion_tool]


def run_silver_agent(
    bronze_output_paths: list[str],
    sttm_silver_path: str,
    run_id: str,
    task_description: str = None,
) -> list[str]:
    """
    Run the Silver Agent end to end: inspect the Bronze tables + approved
    rules, then execute cleansing. Returns the list of Silver Parquet output
    paths.
    """
    audit = AuditLogger(run_id)
    trace = AgentTrace("silver", run_id)
    scratchpad = {}

    table_names = [os.path.basename(p) for p in bronze_output_paths]
    audit.log("silver", "phase_start", bronze_tables=table_names, sttm_silver_path=sttm_silver_path)
    trace.set_input(bronze_output_paths=table_names, sttm_silver_path=sttm_silver_path)

    goal = task_description or (
        "Cleanse the Bronze tables into the Silver layer using the approved STTM "
        "rules. Call inspect_task_tool first, then call silver_ingestion_tool to "
        "execute."
    )

    try:
        llm = make_llm()
        tools = _make_silver_tools(bronze_output_paths, sttm_silver_path, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        output_paths = scratchpad.get("silver_output_paths")
        if not output_paths:
            raise RuntimeError("Silver Agent did not call silver_ingestion_tool")

        audit.log("silver", "phase_complete", output_paths=output_paths)
        trace.set_output(output_paths=output_paths)
        trace.complete(status="success")

        return output_paths

    except Exception as e:
        audit.log("silver", "phase_failed", error=str(e))
        trace.fail(e)
        raise

"""
Reporter Agent.

Analyzes the final Gold layer data with DuckDB SQL, synthesizes insights,
and generates a human-readable HTML report -- with a chart -- that directly
answers the business question the pipeline was given. The report is also
saved to memory (core.memory.store_document) so its content can be looked
up later.
"""

import html
import json
import os
import re
from datetime import datetime, timezone

import duckdb
import pandas as pd
import plotly.graph_objects as go
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import REPORTS_DIR
from core.llm import make_llm
from core.memory import store_document
from core.observability import AgentTrace

# Validated categorical palette (references/palette.md in the dataviz skill) --
# fixed slot order, never cycled/re-assigned by rank.
_CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
_SINGLE_SERIES_COLOR = _CATEGORICAL_PALETTE[0]
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"

SYSTEM_PROMPT = """You are the Reporter Agent in a retail data pipeline built on a \
Medallion Architecture (Bronze -> Silver -> Gold).

Your job: answer the business question you were given by querying the Gold layer \
tables with SQL, then produce a clear, human-readable executive summary and one \
chart backing it up.

Always follow this process:
1. Call inspect_gold_tables_tool first to see what Gold tables exist (schema, row \
count, sample rows).
2. Call load_gold_data_tool to register them as queryable tables. Use its response \
for EXACT column names and types -- do not guess.
3. Call execute_query_tool with ANSI SQL (DuckDB dialect) to compute the answer to \
the business question. If a query returns a SQL error, you may fix the SQL and try \
again ONCE -- if it fails twice, work with whatever data you do have.
4. Your LAST execute_query_tool call before your final answer must return exactly \
the dataset you want charted (aggregated/sorted the way you want it plotted).
5. Respond with ONLY a single JSON object (no markdown fences, no extra prose) with \
these keys:
  - "summary": a plain-English executive summary (2-4 sentences) that directly \
answers the business question, citing the actual numbers you found.
  - "chart": an object {"type": one of "bar"/"line"/"pie"/"scatter", "x": the exact \
column name to use for the x-axis (or labels, for pie), "y": the exact column name \
for the y-axis (or values, for pie), "title": a short chart title}. Choose "line" \
for a trend over time, "bar" to compare categories, "pie" for a part-to-whole \
breakdown of a small number of categories, "scatter" for a relationship between two \
numeric columns.

The "x" and "y" values MUST be exact column names from your last query's result."""


def _make_reporter_tools(gold_paths: list[str], scratchpad: dict):
    """
    Tool factory: builds the Reporter Agent's tools with the Gold Parquet
    paths and a shared scratchpad (holding the DuckDB query history) captured
    via closure.
    """
    con = duckdb.connect(database=":memory:")
    scratchpad["query_results"] = []

    @tool
    def inspect_gold_tables_tool() -> str:
        """Quick preview of every Gold table: schema, row count, 3 sample
        rows. Call this FIRST to plan your SQL approach."""
        previews = {}
        for path in gold_paths:
            df = pd.read_parquet(path)
            table_name = os.path.splitext(os.path.basename(path))[0]
            previews[table_name] = {
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "row_count": len(df),
                "sample_rows": df.head(3).astype(str).to_dict(orient="records"),
            }
        return json.dumps(previews, default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Register every Gold Parquet file as a DuckDB table (named after
        the file, e.g. 'category_quarterly_sales'). Returns the full catalog
        with exact column names, types, and row counts. Call this before
        writing any SQL."""
        catalog = {}
        for path in gold_paths:
            table_name = os.path.splitext(os.path.basename(path))[0]
            con.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                f"SELECT * FROM read_parquet('{path}')"
            )
            schema = con.execute(f'DESCRIBE "{table_name}"').fetchdf()
            row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            catalog[table_name] = {
                "columns": schema[["column_name", "column_type"]].to_dict(orient="records"),
                "row_count": row_count,
            }
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql: str) -> str:
        """Execute an ANSI SQL query (DuckDB dialect) against the tables
        registered by load_gold_data_tool. Returns the result rows as JSON.
        On a SQL error, you may fix the query and try again once."""
        try:
            result_df = con.execute(sql).fetchdf()
        except Exception as e:
            return json.dumps({"error": str(e)})

        scratchpad["query_results"].append(
            {"sql": sql, "result": result_df.to_dict(orient="records")}
        )
        return result_df.to_json(orient="records")

    return [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool]


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the agent's final message, tolerating a
    markdown ```json fence even though the prompt asks the LLM not to use one."""
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


def _build_chart_html(chart_spec: dict, rows: list[dict]) -> str:
    """
    Render one Plotly chart as an HTML div, following the validated
    categorical palette: a single accent color for one-series bar/line/
    scatter charts, and the full fixed-order categorical palette for pie
    charts (identity encoding across slices).
    """
    chart_type = (chart_spec.get("type") or "bar").lower()
    x_key, y_key = chart_spec.get("x"), chart_spec.get("y")
    title = chart_spec.get("title") or "Chart"

    df = pd.DataFrame(rows)
    if x_key not in df.columns or y_key not in df.columns:
        return f"<p><em>Chart could not be rendered: missing column '{x_key}' or '{y_key}'.</em></p>"

    layout = dict(
        title=dict(text=title, font=dict(color=_INK_PRIMARY, size=18)),
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(color=_INK_SECONDARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(gridcolor=_GRIDLINE, linecolor=_INK_MUTED, title=x_key),
        yaxis=dict(gridcolor=_GRIDLINE, linecolor=_INK_MUTED, title=y_key),
        margin=dict(t=60, l=60, r=30, b=60),
    )

    if chart_type == "bar":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df[x_key],
                    y=df[y_key],
                    marker_color=_SINGLE_SERIES_COLOR,
                    text=df[y_key],
                    texttemplate="%{text:,.2f}",
                    textposition="outside",
                )
            ],
            layout=layout,
        )
    elif chart_type == "line":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df[x_key],
                    y=df[y_key],
                    mode="lines+markers",
                    line=dict(color=_SINGLE_SERIES_COLOR, width=2),
                    marker=dict(size=8),
                )
            ],
            layout=layout,
        )
    elif chart_type == "pie":
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=df[x_key],
                    values=df[y_key],
                    marker=dict(colors=_CATEGORICAL_PALETTE),
                )
            ],
            layout=layout,
        )
    elif chart_type == "scatter":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df[x_key],
                    y=df[y_key],
                    mode="markers",
                    marker=dict(size=10, color=_SINGLE_SERIES_COLOR),
                )
            ],
            layout=layout,
        )
    else:
        return f"<p><em>Unsupported chart type: {chart_type!r}.</em></p>"

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_table_html(rows: list[dict]) -> str:
    """A plain HTML table beneath the chart, as an accessible fallback and
    to show the exact numbers the chart is drawn from."""
    if not rows:
        return "<p><em>No rows to display.</em></p>"
    df = pd.DataFrame(rows)
    return df.to_html(index=False, border=0, classes="data-table")


def _build_report_html(business_intent: str, summary: str, chart_html: str, table_html: str, run_id: str) -> str:
    """Assemble the full standalone HTML report page."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # business_intent is raw user input and summary is LLM-generated text --
    # both get rendered by streamlit_app.py via st.iframe(), which (per
    # Streamlit's own docs) executes embedded JavaScript with same-origin
    # access to the app. Escape both before embedding so neither can inject
    # markup/script into the report page.
    business_intent = html.escape(business_intent)
    summary = html.escape(summary)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pipeline Report - {run_id}</title>
<style>
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #f9f9f7;
    color: {_INK_PRIMARY};
    margin: 0;
    padding: 32px 16px;
  }}
  .report {{
    max-width: 860px;
    margin: 0 auto;
    background: {_SURFACE};
    border: 1px solid {_GRIDLINE};
    border-radius: 12px;
    padding: 32px;
  }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: {_INK_MUTED}; font-size: 13px; margin-bottom: 24px; }}
  .summary {{
    background: #f4f6fb;
    border-left: 4px solid {_SINGLE_SERIES_COLOR};
    padding: 16px 20px;
    border-radius: 6px;
    color: {_INK_SECONDARY};
    font-size: 15px;
    line-height: 1.5;
    margin-bottom: 28px;
  }}
  table.data-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
  table.data-table th, table.data-table td {{
    text-align: left; padding: 8px 12px; border-bottom: 1px solid {_GRIDLINE};
  }}
  table.data-table th {{ color: {_INK_MUTED}; font-weight: 600; }}
  h2 {{ font-size: 16px; color: {_INK_SECONDARY}; margin-top: 32px; }}
</style>
</head>
<body>
  <div class="report">
    <h1>Business Question: {business_intent}</h1>
    <div class="meta">Run ID: {run_id} &middot; Generated {generated_at}</div>
    <div class="summary">{summary}</div>
    {chart_html}
    <h2>Underlying data</h2>
    {table_html}
  </div>
</body>
</html>"""


def run_reporter_agent(
    gold_output_paths: list[str],
    business_intent: str,
    run_id: str,
    task_description: str = None,
) -> str:
    """
    Run the Reporter Agent end to end: query the Gold tables, synthesize an
    executive summary and chart spec, render the HTML report, save it to
    memory, and return the report's file path.
    """
    audit = AuditLogger(run_id)
    trace = AgentTrace("reporter", run_id)
    scratchpad = {}

    table_names = [os.path.basename(p) for p in gold_output_paths]
    audit.log("reporter", "phase_start", gold_tables=table_names, business_intent=business_intent)
    trace.set_input(gold_output_paths=table_names, business_intent=business_intent)

    goal = task_description or (
        f"Business question: {business_intent}\n"
        "Answer this using the Gold layer tables. Call inspect_gold_tables_tool, "
        "then load_gold_data_tool, then execute_query_tool to compute your answer, "
        "then respond with the JSON object described in your instructions."
    )

    try:
        llm = make_llm(caller="reporter agent")
        tools = _make_reporter_tools(gold_output_paths, scratchpad)
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

        result = agent.invoke({"messages": [("human", goal)]})
        trace.extract_from_messages(result["messages"])

        final_text = result["messages"][-1].content
        synthesis = _extract_json(final_text)
        summary = synthesis.get("summary") or "No summary was generated."
        chart_spec = synthesis.get("chart") or {}

        query_results = scratchpad.get("query_results", [])
        chart_rows = query_results[-1]["result"] if query_results else []

        chart_html = _build_chart_html(chart_spec, chart_rows) if chart_spec else ""
        table_html = _build_table_html(chart_rows)
        report_html = _build_report_html(business_intent, summary, chart_html, table_html, run_id)

        report_path = REPORTS_DIR / f"report_{run_id}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        report_json_path = REPORTS_DIR / f"report_{run_id}.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "business_intent": business_intent,
                    "summary": summary,
                    "chart": chart_spec,
                    "data": chart_rows,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        store_document(
            f"report_{run_id}",
            summary,
            run_id=run_id,
            doc_type="report",
            business_intent=business_intent,
            report_path=str(report_path),
        )

        audit.log("reporter", "phase_complete", report_path=str(report_path))
        trace.set_output(report_path=str(report_path), summary=summary)
        trace.complete(status="success")

        return str(report_path)

    except Exception as e:
        audit.log("reporter", "phase_failed", error=str(e))
        trace.fail(e)
        raise

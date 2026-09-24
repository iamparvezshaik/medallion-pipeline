"""
Streamlit UI for the medallion pipeline.

Lets a user upload messy CSV files, ask a business question in plain
English, and step through the pipeline one human-approval gate at a time --
reviewing (and optionally editing) each AI-generated STTM before it runs,
and finally viewing the generated HTML report inline.

This file only ever talks to agents.orchestrator; it never calls an agent
directly, and it never blocks waiting for the pipeline -- each button click
advances exactly one phase and reruns the script, which is how Streamlit
apps are meant to work.
"""

import pandas as pd
import streamlit as st

from agents import orchestrator
from core.config import LANDING_DIR

st.set_page_config(page_title="Medallion Pipeline", page_icon="\U0001F4CA", layout="wide")

PHASE_STEPS = [
    ("awaiting_bronze_approval", "1. Profile + Bronze rules"),
    ("awaiting_silver_approval", "2. Bronze ingestion + Silver rules"),
    ("awaiting_gold_approval", "3. Silver cleansing + Gold rules"),
    ("completed", "4. Gold + Report"),
]


def _save_uploaded_files(uploaded_files) -> list[str]:
    """Write Streamlit's in-memory uploaded files to data/landing/ and return
    their paths -- every agent works with file paths, not in-memory buffers."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        dest = LANDING_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(dest))
    return saved_paths


def _render_progress(status: str):
    """A simple step indicator across the top of the page."""
    current_index = _status_index(status)
    cols = st.columns(len(PHASE_STEPS))
    for i, (col, (gate_status, label)) in enumerate(zip(cols, PHASE_STEPS)):
        with col:
            if status == "completed" or i < current_index:
                col.success(label)
            elif gate_status == status:
                col.info(label + " (current)")
            else:
                col.write(label)


def _status_index(status: str) -> int:
    for i, (gate_status, _) in enumerate(PHASE_STEPS):
        if gate_status == status:
            return i
    return -1


def _render_sttm_editor(sttm_path: str, key: str) -> pd.DataFrame:
    """Show an editable table of an STTM's mapping rows so a human can review
    -- and if needed, correct -- the AI-generated rules before they execute."""
    df = pd.read_csv(sttm_path)
    st.caption(f"Source file: `{sttm_path}`")
    edited_df = st.data_editor(df, num_rows="dynamic", width="stretch", key=key)
    return edited_df


def _reset():
    st.session_state.pipeline_state = None
    st.session_state.retry_action = None


if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "retry_action" not in st.session_state:
    st.session_state.retry_action = None

st.title("\U0001F4CA Intent-Driven Agentic Medallion Pipeline")
st.caption(
    "Upload messy CSV files, ask a business question in plain English, and let "
    "AI agents clean, organize, and analyze the data through a Bronze → Silver "
    "→ Gold pipeline -- with you approving the rules at every step."
)

with st.sidebar:
    st.header("Run")
    state = st.session_state.pipeline_state
    if state:
        st.write(f"**Run ID:** `{state.run_id[:8]}...`")
        st.write(f"**Status:** `{state.status}`")
        st.write(f"**Question:** {state.business_intent}")
    if st.button("\U0001F504 Start New Run", width="stretch"):
        _reset()
        st.rerun()

state = st.session_state.pipeline_state

# --- No run started yet: show the upload form ---
if state is None:
    st.subheader("1. Upload your data and ask a question")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
    )
    business_intent = st.text_input(
        "What business question should the pipeline answer?",
        placeholder="e.g. What product category has highest sales in Q1?",
    )

    if st.button("\U0001F680 Start Pipeline", type="primary", disabled=not (uploaded_files and business_intent)):
        with st.spinner("Profiling your data and drafting Bronze layer rules..."):
            file_paths = _save_uploaded_files(uploaded_files)
            st.session_state.pipeline_state = orchestrator.start_pipeline(file_paths, business_intent)
        st.rerun()

# --- A run is in progress or finished ---
else:
    _render_progress(state.status)
    st.divider()

    if state.status == "failed":
        st.error(f"The pipeline hit an error and stopped:\n\n{state.error}")
        if st.button("\U0001F501 Retry this phase"):
            retry_fn = st.session_state.retry_action
            with st.spinner("Retrying..."):
                if retry_fn == "phase_1":
                    st.session_state.pipeline_state = orchestrator.run_phase_1(state)
                elif retry_fn == "bronze_approval":
                    st.session_state.pipeline_state = orchestrator.approve_bronze_sttm(state)
                elif retry_fn == "silver_approval":
                    st.session_state.pipeline_state = orchestrator.approve_silver_sttm(state)
                elif retry_fn == "gold_approval":
                    st.session_state.pipeline_state = orchestrator.approve_gold_sttm(state)
            st.rerun()

    elif state.status == "awaiting_bronze_approval":
        st.subheader("2. Review the Bronze layer rules")
        st.write(
            "The Profiler analyzed your files and the STTM Agent drafted these Bronze "
            "rules. Review (edit if needed), then approve to run ingestion."
        )
        _render_sttm_editor(state.sttm_bronze_path, key="bronze_editor")
        if st.button("✅ Approve Bronze rules & continue", type="primary"):
            st.session_state.retry_action = "bronze_approval"
            with st.spinner("Ingesting Bronze layer and drafting Silver rules..."):
                st.session_state.pipeline_state = orchestrator.approve_bronze_sttm(state)
            st.rerun()

    elif state.status == "awaiting_silver_approval":
        st.subheader("3. Review the Silver layer rules")
        st.write(
            "These rules govern how nulls, duplicates, types, and inconsistent "
            "categorical values get cleaned up. Review, then approve to run cleansing."
        )
        _render_sttm_editor(state.sttm_silver_path, key="silver_editor")
        if st.button("✅ Approve Silver rules & continue", type="primary"):
            st.session_state.retry_action = "silver_approval"
            with st.spinner("Cleansing Silver layer and drafting Gold rules..."):
                st.session_state.pipeline_state = orchestrator.approve_silver_sttm(state)
            st.rerun()

    elif state.status == "awaiting_gold_approval":
        st.subheader("4. Review the Gold layer rules")
        st.write(
            "These rules define the business-ready tables -- joins and aggregations -- "
            "that will be used to answer your question. Review, then approve to finish."
        )
        _render_sttm_editor(state.sttm_gold_path, key="gold_editor")
        if st.button("✅ Approve Gold rules & generate report", type="primary"):
            st.session_state.retry_action = "gold_approval"
            with st.spinner("Materialising Gold tables and generating your report..."):
                st.session_state.pipeline_state = orchestrator.approve_gold_sttm(state)
            st.rerun()

    elif state.status == "completed":
        st.success("Pipeline complete! Here's your report:")
        with open(state.report_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        # This report is generated by our own pipeline (not raw user/LLM input
        # rendered unsanitized), so embedding it in an iframe is safe here.
        st.iframe(report_html, height=900)

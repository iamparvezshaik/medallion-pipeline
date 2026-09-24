"""
Orchestrator.

Advances a pipeline run through its four phases, one human-approval gate at
a time:

  Phase 1: profile raw files          -> generate Bronze STTM  -> [approval]
  Phase 2: execute Bronze ingestion   -> generate Silver STTM  -> [approval]
  Phase 3: execute Silver cleansing   -> generate Gold STTM    -> [approval]
  Phase 4: materialise Gold tables    -> generate report       -> done

Each phase function here does the deterministic, non-LLM bookkeeping around
an agent call: it invokes the agent, copies what that agent produced into
PipelineState (the agent already handed back exactly what belongs there --
this function is the "scratchpad -> PipelineState" copy step described in
the training material), sets the run's status, and wraps everything in a
try/except so a failed phase leaves the run in a clean "failed" state
instead of crashing the whole app. The Orchestrator never blocks waiting for
a human -- it always returns control (the updated PipelineState) back to the
caller (Streamlit), which re-invokes the next phase function once the human
approves.
"""

import traceback
import uuid

from agents.bronze_agent import run_bronze_agent
from agents.gold_agent import run_gold_agent
from agents.profiler import run_profiler_agent
from agents.reporter import run_reporter_agent
from agents.silver_agent import run_silver_agent
from agents.sttm_generator import (
    run_bronze_sttm_agent,
    run_gold_sttm_agent,
    run_silver_sttm_agent,
)
from core.audit import AuditLogger
from core.state import PipelineState


def start_pipeline(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """
    Begin a new pipeline run: generate a run_id, create the PipelineState,
    and immediately run Phase 1.
    """
    state = PipelineState(
        run_id=str(uuid.uuid4()),
        status="running",
        uploaded_files=uploaded_files,
        business_intent=business_intent,
    )
    return run_phase_1(state)


def run_phase_1(state: PipelineState) -> PipelineState:
    """Phase 1: profile the raw files, then generate Bronze layer STTM rules."""
    audit = AuditLogger(state.run_id)
    audit.log("orchestrator", "phase_1_start", uploaded_files=state.uploaded_files)

    try:
        state.profile_path = run_profiler_agent(
            file_paths=state.uploaded_files,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.sttm_bronze_path = run_bronze_sttm_agent(
            profile_path=state.profile_path,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.status = "awaiting_bronze_approval"

        audit.log(
            "orchestrator",
            "phase_1_complete",
            profile_path=state.profile_path,
            sttm_bronze_path=state.sttm_bronze_path,
        )

    except Exception as e:
        state.status = "failed"
        state.error = f"Phase 1 failed: {e}"
        audit.log("orchestrator", "phase_1_failed", error=str(e), traceback=traceback.format_exc())

    return state


def approve_bronze_sttm(state: PipelineState) -> PipelineState:
    """
    Called once a human has approved (or edited then approved) the Bronze
    STTM. Phase 2: execute Bronze ingestion, then generate Silver layer STTM
    rules.
    """
    audit = AuditLogger(state.run_id)
    state.hitl_approved = True
    audit.log("orchestrator", "phase_2_start", sttm_bronze_path=state.sttm_bronze_path)

    try:
        state.bronze_output_paths = run_bronze_agent(
            file_paths=state.uploaded_files,
            sttm_bronze_path=state.sttm_bronze_path,
            run_id=state.run_id,
        )
        state.sttm_silver_path = run_silver_sttm_agent(
            bronze_output_paths=state.bronze_output_paths,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.status = "awaiting_silver_approval"
        state.hitl_approved = False  # reset for the next gate

        audit.log(
            "orchestrator",
            "phase_2_complete",
            bronze_output_paths=state.bronze_output_paths,
            sttm_silver_path=state.sttm_silver_path,
        )

    except Exception as e:
        state.status = "failed"
        state.error = f"Phase 2 failed: {e}"
        audit.log("orchestrator", "phase_2_failed", error=str(e), traceback=traceback.format_exc())

    return state


def approve_silver_sttm(state: PipelineState) -> PipelineState:
    """
    Called once a human has approved the Silver STTM. Phase 3: execute
    Silver cleansing, then generate Gold layer STTM rules.
    """
    audit = AuditLogger(state.run_id)
    state.hitl_approved = True
    audit.log("orchestrator", "phase_3_start", sttm_silver_path=state.sttm_silver_path)

    try:
        state.silver_output_paths = run_silver_agent(
            bronze_output_paths=state.bronze_output_paths,
            sttm_silver_path=state.sttm_silver_path,
            run_id=state.run_id,
        )
        state.sttm_gold_path = run_gold_sttm_agent(
            silver_output_paths=state.silver_output_paths,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.status = "awaiting_gold_approval"
        state.hitl_approved = False

        audit.log(
            "orchestrator",
            "phase_3_complete",
            silver_output_paths=state.silver_output_paths,
            sttm_gold_path=state.sttm_gold_path,
        )

    except Exception as e:
        state.status = "failed"
        state.error = f"Phase 3 failed: {e}"
        audit.log("orchestrator", "phase_3_failed", error=str(e), traceback=traceback.format_exc())

    return state


def approve_gold_sttm(state: PipelineState) -> PipelineState:
    """
    Called once a human has approved the Gold STTM. Phase 4: materialise
    Gold tables, then generate the final report. This is the last phase --
    on success, the pipeline is complete.
    """
    audit = AuditLogger(state.run_id)
    state.hitl_approved = True
    audit.log("orchestrator", "phase_4_start", sttm_gold_path=state.sttm_gold_path)

    try:
        state.gold_output_paths = run_gold_agent(
            silver_output_paths=state.silver_output_paths,
            sttm_gold_path=state.sttm_gold_path,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.report_path = run_reporter_agent(
            gold_output_paths=state.gold_output_paths,
            business_intent=state.business_intent,
            run_id=state.run_id,
        )
        state.status = "completed"

        audit.log(
            "orchestrator",
            "phase_4_complete",
            gold_output_paths=state.gold_output_paths,
            report_path=state.report_path,
        )

    except Exception as e:
        state.status = "failed"
        state.error = f"Phase 4 failed: {e}"
        audit.log("orchestrator", "phase_4_failed", error=str(e), traceback=traceback.format_exc())

    return state

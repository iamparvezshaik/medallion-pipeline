"""
Pipeline state for the medallion pipeline.

PipelineState is the single object that travels through every phase of a
run - ingestion, profiling, STTM generation, execution, and reporting. Each
phase reads the fields it needs and fills in the fields it produces, so the
whole run's progress and outputs live in one place.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """Tracks the full state of one pipeline run, from upload to final report."""

    # --- Run metadata ---
    run_id: str = Field(default="", description="Unique ID for this pipeline run")
    status: str = Field(
        default="initialized",
        description="Current status of the run (e.g. initialized, running, completed, failed)",
    )

    # --- Phase 1: Ingestion ---
    uploaded_files: list[str] = Field(
        default_factory=list, description="Paths to the uploaded CSV files"
    )
    business_intent: str = Field(
        default="", description="The user's business question, in plain English"
    )

    # --- Phase 2: Profiling ---
    profile_path: str = Field(
        default="", description="Path to the generated data profile JSON"
    )

    # --- Phase 3: STTM (Source-to-Target Mapping) ---
    sttm_bronze_path: str = Field(
        default="", description="Path to the Bronze layer transformation rules"
    )
    sttm_silver_path: str = Field(
        default="", description="Path to the Silver layer transformation rules"
    )
    sttm_gold_path: str = Field(
        default="", description="Path to the Gold layer transformation rules"
    )
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human has approved the STTM rules (human-in-the-loop)",
    )

    # --- Phase 4: Execution ---
    bronze_output_paths: list[str] = Field(
        default_factory=list, description="Paths to the Bronze layer output Parquet files"
    )
    silver_output_paths: list[str] = Field(
        default_factory=list, description="Paths to the Silver layer output Parquet files"
    )
    gold_output_paths: list[str] = Field(
        default_factory=list, description="Paths to the Gold layer output Parquet files"
    )

    # --- Phase 5: Reporting ---
    report_path: str = Field(
        default="", description="Path to the final generated HTML report"
    )

    # --- Error handling ---
    error: Optional[str] = Field(
        default=None, description="Error message if the pipeline run failed, otherwise None"
    )

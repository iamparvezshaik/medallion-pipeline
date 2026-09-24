"""
Audit logging for the medallion pipeline.

Every action an AI agent takes during a pipeline run gets appended as one
JSON line to a per-run log file, so the full history of what happened during
a run can be reviewed or replayed later.
"""

import json
import uuid
from datetime import datetime, timezone

from core.config import AUDIT_DIR


class AuditLogger:
    """Appends timestamped agent actions to a JSONL file for a single run."""

    def __init__(self, run_id: str = None):
        # Generate a unique run_id if the caller didn't provide one, so every
        # pipeline run gets its own log file.
        self.run_id = run_id or str(uuid.uuid4())
        self.log_path = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs):
        """
        Record one action to the log file.

        Args:
            agent: name of the agent performing the action (e.g. "bronze_agent").
            action: short description of what happened (e.g. "cleaned_column").
            **kwargs: any extra details worth recording (e.g. row_count=100).
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self) -> list:
        """Return every logged entry for this run as a list of dicts."""
        if not self.log_path.exists():
            return []

        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        return logs

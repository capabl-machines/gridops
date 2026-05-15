"""JSONL episode logging for later GridOps fine-tuning datasets."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = "episode_logs"


class EpisodeLogger:
    """Append reset/plan/step events to per-episode JSONL files.

    Logging is enabled by default for local/demo runs and can be disabled with
    `GRIDOPS_EPISODE_LOG_ENABLED=0`. The directory is intentionally gitignored;
    useful rows can later be promoted into curated SFT/RL traces.
    """

    def __init__(self, log_dir: str | None = None, enabled: bool | None = None):
        self.log_dir = Path(log_dir or os.environ.get("GRIDOPS_EPISODE_LOG_DIR", DEFAULT_LOG_DIR))
        if enabled is None:
            enabled = os.environ.get("GRIDOPS_EPISODE_LOG_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.enabled = bool(enabled)

    def append(self, episode_id: str, event_type: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": episode_id,
            "event_type": event_type,
            **payload,
        }
        path = self.log_dir / f"{episode_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


episode_logger = EpisodeLogger()

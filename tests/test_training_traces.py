import json
import subprocess
import sys
from pathlib import Path

from gridops.prompting import SYSTEM_PROMPT, format_observation, validate_completion


def test_generated_curriculum_trace_contract():
    path = Path("sft_traces/gridops_curriculum_1200.jsonl")
    assert path.exists()
    counts = {"task_1_normal": 0, "task_2_heatwave": 0, "task_3_crisis": 0}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            counts[row["task_id"]] += 1
            assert row["messages"][0]["content"] == SYSTEM_PROMPT
            assert row["prompt"] == format_observation(row["raw"]["observation"])
            valid, reason = validate_completion(row["completion"])
            assert valid, reason
            assert row["raw"]["label_policy"].startswith("oracle_")
            assert row["raw"]["focus_tags"]
            assert row["raw"]["validation"]["valid"] is True
    assert counts == {"task_1_normal": 300, "task_2_heatwave": 400, "task_3_crisis": 500}


def test_trace_validator_cli_passes():
    result = subprocess.run(
        [sys.executable, "scripts/validate_traces.py", "sft_traces/gridops_curriculum_1200.jsonl"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validation ok" in result.stdout

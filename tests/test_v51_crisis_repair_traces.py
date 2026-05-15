from scripts.build_gridops_v4_reasoning_traces import validate_rows
from scripts.build_gridops_v51_crisis_repair_traces import (
    collect_anchor_rows,
    collect_crisis_rows,
)


def test_v51_crisis_repair_trace_contract():
    crisis_rows, crisis_rollouts = collect_crisis_rows(
        seed_start=23000,
        seeds=1,
        horizon=6,
        max_rows=16,
    )
    anchor_rows = collect_anchor_rows(
        seed_start=24000,
        seeds_per_task=1,
        horizon=6,
        max_rows=8,
    )
    rows = anchor_rows + crisis_rows

    assert crisis_rows
    assert anchor_rows
    assert crisis_rollouts
    assert validate_rows(rows) == []
    assert any((row.get("raw") or {}).get("bucket") == "active_outage" for row in crisis_rows)
    assert any(float(row["action"]["diesel_dispatch"]) > 0.05 for row in crisis_rows)

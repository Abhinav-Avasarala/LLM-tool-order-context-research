import json

import pytest

from tool_context_study.phase5_run_baseline import (
    absorb_shards_into_canonical,
    effective_gpu_cap,
    read_done_ids,
    shard_round_robin,
)


def _result_line(condition_id: str, status: str) -> str:
    return json.dumps(
        {
            "condition_id": condition_id,
            "model": "test-model",
            "model_revision": "rev0",
            "status": status,
            "selected_tool": None,
            "arguments": None,
            "raw_response": None,
            "history_tokens": 0,
            "total_input_tokens": 10,
            "latency_seconds": 0.1,
        }
    )


def test_shard_round_robin_balances_and_preserves_all_items():
    shards = shard_round_robin(["a", "b", "c", "d", "e"], 2)
    assert shards == [["a", "c", "e"], ["b", "d"]]
    assert sorted(x for shard in shards for x in shard) == ["a", "b", "c", "d", "e"]


def test_shard_round_robin_more_shards_than_items():
    shards = shard_round_robin(["a"], 3)
    assert shards == [["a"], [], []]


@pytest.mark.parametrize(
    ("max_gpus", "gpus_per_replica", "expected"),
    [
        (5, 1, 5),  # single-GPU model: never bumped
        (5, 2, 6),  # TP=2, odd cap: rounds up by exactly one to avoid wasting a GPU
        (4, 2, 4),  # TP=2, already even: no waste, no bump
        (6, 2, 6),  # already at the ceiling
    ],
)
def test_effective_gpu_cap(max_gpus, gpus_per_replica, expected):
    assert effective_gpu_cap(max_gpus, gpus_per_replica) == expected


def test_effective_gpu_cap_rejects_above_hard_ceiling():
    with pytest.raises(ValueError):
        effective_gpu_cap(7, 1)


def test_absorb_shards_is_cumulative_across_calls(tmp_path):
    """Regression test for the bug caught by a live end-to-end resume smoke
    test (docs/decision_log.md): a naive implementation deleted shard files
    without persisting their results anywhere durable first, so completed
    conditions "rotated out" instead of accumulating across attempts."""
    merged_path = tmp_path / "phase5_test-model.jsonl"
    logs_dir = tmp_path
    ordered_ids = ["c1", "c2", "c3", "c4"]

    # First attempt only finishes half the grid.
    (logs_dir / "phase5_test-model_shard0.jsonl").write_text(
        _result_line("c1", "ok") + "\n" + _result_line("c2", "no_tool_call") + "\n"
    )
    absorb_shards_into_canonical("test-model", merged_path, logs_dir, ordered_ids, require_complete=False)
    assert read_done_ids(merged_path) == {"c1", "c2"}
    # Shard files must be gone -- their data now lives only in merged_path.
    assert not list(logs_dir.glob("phase5_test-model_shard*.jsonl"))

    # Second attempt (as if the orchestrator looped back for the remaining c3/c4)
    # writes a fresh shard file under the *same* name as before.
    (logs_dir / "phase5_test-model_shard0.jsonl").write_text(_result_line("c3", "ok") + "\n")
    absorb_shards_into_canonical("test-model", merged_path, logs_dir, ordered_ids, require_complete=False)
    # c1/c2 must still be present -- this is exactly what broke before the fix.
    assert read_done_ids(merged_path) == {"c1", "c2", "c3"}


def test_absorb_shard_result_overrides_stale_merged_error(tmp_path):
    merged_path = tmp_path / "phase5_test-model.jsonl"
    logs_dir = tmp_path
    merged_path.write_text(_result_line("c1", "inference_error") + "\n")

    (logs_dir / "phase5_test-model_shard0.jsonl").write_text(_result_line("c1", "ok") + "\n")
    absorb_shards_into_canonical("test-model", merged_path, logs_dir, ["c1"], require_complete=False)

    assert read_done_ids(merged_path) == {"c1"}


def test_absorb_require_complete_raises_when_conditions_missing(tmp_path):
    merged_path = tmp_path / "phase5_test-model.jsonl"
    logs_dir = tmp_path
    merged_path.write_text(_result_line("c1", "ok") + "\n")

    with pytest.raises(RuntimeError):
        absorb_shards_into_canonical("test-model", merged_path, logs_dir, ["c1", "c2"], require_complete=True)

import pytest

from tool_context_study.phase4_build_final_conditions import HISTORY_CONDITIONS, GOLD_POSITIONS, build_final_condition_grid
from tool_context_study.schemas import AssembledHistory, FinalTaskRecord, NormalizedToolSchema


def _make_catalog() -> list[NormalizedToolSchema]:
    tools = []
    groups = ["group_a", "group_b"]
    for gi, group in enumerate(groups):
        for i in range(16):
            name = f"tool_{gi}_{i}"
            tools.append(
                NormalizedToolSchema(
                    schema_id=f"sha256:fake{gi}{i}",
                    name=name,
                    original_name=name,
                    domain="domain_x",
                    group=group,
                    source="fixture",
                    source_revision="rev0",
                    schema={"type": "function", "function": {"name": name, "description": "d", "parameters": {}}},
                )
            )
    return tools


def _make_histories() -> dict[str, AssembledHistory]:
    def h(history_id, condition, target_group=None, n_msgs=4):
        return AssembledHistory(
            history_id=history_id,
            history_condition=condition,
            target_group=target_group,
            derived_from=None,
            seed=0,
            episode_ids=[f"ep_{i}" for i in range(n_msgs // 2)],
            messages=[{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n_msgs)],
            token_count=100 * n_msgs,
            token_count_model="fixture-tokenizer",
            n_messages=n_msgs,
        )

    histories = {
        "history_final_h64n_base": h("history_final_h64n_base", "H64N", n_msgs=20),
        "history_final_h16n_base": h("history_final_h16n_base", "H16N", n_msgs=6),
    }
    for group in ["group_a", "group_b"]:
        hid = f"history_final_h64i_{group}"
        histories[hid] = h(hid, "H64I", target_group=group, n_msgs=18)
    return histories


def _make_tasks() -> list[FinalTaskRecord]:
    return [
        FinalTaskRecord(
            task_id="final_0001", request="r0", gold_tool="tool_0_0", gold_arguments={},
            authoring_source="fresh_from_schema", review_status="approved",
        ),
        FinalTaskRecord(
            task_id="final_0002", request="r1", gold_tool="tool_1_0", gold_arguments={},
            authoring_source="fresh_from_schema", review_status="approved",
        ),
    ]


def test_build_final_condition_grid_produces_full_cross_product():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = _make_tasks()

    conditions = build_final_condition_grid(tasks, catalog, histories, final_seed=42)

    assert len(conditions) == len(tasks) * len(HISTORY_CONDITIONS) * len(GOLD_POSITIONS)
    assert len({c.condition_id for c in conditions}) == len(conditions)


def test_build_final_condition_grid_uses_task_specific_group_for_h64i():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = _make_tasks()

    conditions = build_final_condition_grid(tasks, catalog, histories, final_seed=42)

    h64i_a = next(c for c in conditions if c.task_id == "final_0001" and c.history_condition == "H64I")
    h64i_b = next(c for c in conditions if c.task_id == "final_0002" and c.history_condition == "H64I")

    assert h64i_a.history_id == "history_final_h64i_group_a"
    assert h64i_b.history_id == "history_final_h64i_group_b"


def test_load_final_tasks_rejects_unapproved(tmp_path, monkeypatch):
    import tool_context_study.phase4_build_final_conditions as mod

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        FinalTaskRecord(
            task_id="final_0001", request="r", gold_tool="tool_0_0", gold_arguments={},
            authoring_source="fresh_from_schema", review_status="pending",
        ).model_dump_json()
        + "\n"
    )
    monkeypatch.setattr(mod, "FINAL_TASKS_PATH", tasks_path)

    with pytest.raises(ValueError, match="not review_status='approved'"):
        mod.load_final_tasks()

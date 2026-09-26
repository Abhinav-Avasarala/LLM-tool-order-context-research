from tool_context_study.phase3_build_dev_conditions import HISTORY_CONDITIONS, GOLD_POSITIONS, build_dev_condition_grid
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
        "history_h64n_base": h("history_h64n_base", "H64N", n_msgs=20),
        "history_h16n_base": h("history_h16n_base", "H16N", n_msgs=6),
    }
    for group in ["group_a", "group_b"]:
        hid = f"history_h64i_{group}"
        histories[hid] = h(hid, "H64I", target_group=group, n_msgs=18)
    return histories


def test_build_dev_condition_grid_produces_full_cross_product():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = [
        FinalTaskRecord(task_id="t0", request="r0", gold_tool="tool_0_0", gold_arguments={}, authoring_source="x"),
        FinalTaskRecord(task_id="t1", request="r1", gold_tool="tool_1_0", gold_arguments={}, authoring_source="x"),
    ]

    conditions = build_dev_condition_grid(tasks, catalog, histories, base_seed=42)

    assert len(conditions) == len(tasks) * len(HISTORY_CONDITIONS) * len(GOLD_POSITIONS)
    assert len({c.condition_id for c in conditions}) == len(conditions)


def test_build_dev_condition_grid_h0_has_no_history_messages():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = [FinalTaskRecord(task_id="t0", request="r0", gold_tool="tool_0_0", gold_arguments={}, authoring_source="x")]

    conditions = build_dev_condition_grid(tasks, catalog, histories, base_seed=42)
    h0 = [c for c in conditions if c.history_condition == "H0"]
    assert len(h0) == len(GOLD_POSITIONS)
    for c in h0:
        assert c.history_id is None
        # only system + final user message, no history in between
        assert len(c.messages) == 2


def test_build_dev_condition_grid_uses_task_specific_group_for_h64i():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = [
        FinalTaskRecord(task_id="t_a", request="ra", gold_tool="tool_0_0", gold_arguments={}, authoring_source="x"),
        FinalTaskRecord(task_id="t_b", request="rb", gold_tool="tool_1_0", gold_arguments={}, authoring_source="x"),
    ]

    conditions = build_dev_condition_grid(tasks, catalog, histories, base_seed=42)

    h64i_a = next(c for c in conditions if c.task_id == "t_a" and c.history_condition == "H64I")
    h64i_b = next(c for c in conditions if c.task_id == "t_b" and c.history_condition == "H64I")

    assert h64i_a.history_id == "history_h64i_group_a"
    assert h64i_b.history_id == "history_h64i_group_b"
    assert h64i_a.history_id != h64i_b.history_id


def test_build_dev_condition_grid_h16n_and_h64n_are_shared_across_tasks():
    catalog = _make_catalog()
    histories = _make_histories()
    tasks = [
        FinalTaskRecord(task_id="t_a", request="ra", gold_tool="tool_0_0", gold_arguments={}, authoring_source="x"),
        FinalTaskRecord(task_id="t_b", request="rb", gold_tool="tool_1_0", gold_arguments={}, authoring_source="x"),
    ]

    conditions = build_dev_condition_grid(tasks, catalog, histories, base_seed=42)

    h64n_ids = {c.history_id for c in conditions if c.history_condition == "H64N"}
    h16n_ids = {c.history_id for c in conditions if c.history_condition == "H16N"}
    assert h64n_ids == {"history_h64n_base"}
    assert h16n_ids == {"history_h16n_base"}

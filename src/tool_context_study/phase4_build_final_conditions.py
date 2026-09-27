"""Phase 4, step 3 (brief section 13): build the frozen final condition
grid -- 96 final tasks x 4 history conditions x 3 positions = 1,152
conditions -- from the curated catalog, the approved final tasks, and the
Phase 4 final-set histories (`history_final_*`, brief distinct from the
Phase 3 dev-pilot's `history_*` files -- see docs/decision_log.md).

Mirrors `phase3_build_dev_conditions.py` exactly except for which task
file and history set it reads, and that it refuses to run over any task
not yet `review_status: "approved"`.
"""

from __future__ import annotations

from pathlib import Path

from tool_context_study.build_conditions import build_condition, validate_condition
from tool_context_study.config import DATA_DIR, load_experiment_config
from tool_context_study.schemas import AssembledHistory, ConditionRecord, FinalTaskRecord, NormalizedToolSchema

CATALOG_PATH = DATA_DIR / "catalog" / "tools.jsonl"
FINAL_TASKS_PATH = DATA_DIR / "final" / "tasks.jsonl"
ASSEMBLED_DIR = DATA_DIR / "histories" / "assembled"
OUTPUT_PATH = DATA_DIR / "conditions" / "final_grid.jsonl"

GOLD_POSITIONS = [1, 16, 32]
HISTORY_CONDITIONS = ["H0", "H16N", "H64N", "H64I"]


def _load_jsonl(path: Path, model_cls):
    with open(path) as f:
        return [model_cls.model_validate_json(line) for line in f if line.strip()]


def load_catalog() -> list[NormalizedToolSchema]:
    return _load_jsonl(CATALOG_PATH, NormalizedToolSchema)


def load_final_tasks() -> list[FinalTaskRecord]:
    tasks = _load_jsonl(FINAL_TASKS_PATH, FinalTaskRecord)
    unapproved = [t.task_id for t in tasks if t.review_status != "approved"]
    if unapproved:
        raise ValueError(
            f"{len(unapproved)} final task(s) are not review_status='approved', refusing to build "
            f"frozen conditions from them: {unapproved}"
        )
    return tasks


def load_final_histories() -> dict[str, AssembledHistory]:
    histories = {}
    for path in ASSEMBLED_DIR.glob("history_final_*.json"):
        with open(path) as f:
            h = AssembledHistory.model_validate_json(f.read())
        histories[h.history_id] = h
    return histories


def build_final_condition_grid(
    tasks: list[FinalTaskRecord],
    catalog: list[NormalizedToolSchema],
    histories: dict[str, AssembledHistory],
    final_seed: int,
) -> list[ConditionRecord]:
    group_by_tool_name = {t.name: t.group for t in catalog}
    h64n = histories["history_final_h64n_base"]
    h16n = histories["history_final_h16n_base"]
    h64i_by_group = {h.target_group: h for h in histories.values() if h.history_condition == "H64I"}

    conditions = []
    for task in tasks:
        group = group_by_tool_name[task.gold_tool]
        h64i = h64i_by_group[group]

        history_by_condition = {
            "H0": (None, []),
            "H16N": (h16n.history_id, h16n.messages),
            "H64N": (h64n.history_id, h64n.messages),
            "H64I": (h64i.history_id, h64i.messages),
        }

        for history_condition in HISTORY_CONDITIONS:
            history_id, history_messages = history_by_condition[history_condition]
            for position in GOLD_POSITIONS:
                cond = build_condition(
                    task,
                    catalog,
                    gold_position=position,
                    history_condition=history_condition,
                    history_messages=history_messages,
                    base_seed=final_seed,
                    history_id=history_id,
                )
                validate_condition(cond, catalog)
                conditions.append(cond)
    return conditions


def main() -> None:
    config = load_experiment_config()
    final_seed = config["seeds"]["final_seed"]

    catalog = load_catalog()
    tasks = load_final_tasks()
    histories = load_final_histories()

    expected_histories = 2 + 8  # H64N + H16N + one H64I per target group
    if len(histories) != expected_histories:
        raise ValueError(f"expected {expected_histories} history_final_* files, found {len(histories)}")
    if len(tasks) != 96:
        raise ValueError(f"expected 96 approved final tasks, found {len(tasks)}")

    conditions = build_final_condition_grid(tasks, catalog, histories, final_seed)

    expected = len(tasks) * len(HISTORY_CONDITIONS) * len(GOLD_POSITIONS)
    assert len(conditions) == expected, f"expected {expected} conditions, got {len(conditions)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for cond in conditions:
            f.write(cond.model_dump_json() + "\n")

    print(f"Wrote {len(conditions)} conditions ({len(tasks)} tasks x {len(HISTORY_CONDITIONS)} history x {len(GOLD_POSITIONS)} positions) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

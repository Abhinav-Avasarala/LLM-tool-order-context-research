"""Phase 3, step 4->5 prep (brief section 13): build the full development
condition grid -- 24 dev tasks x 4 history conditions x 3 positions = 288
conditions -- from the curated catalog, dev tasks, and assembled histories.

Each task's `H64I` history is the one variant matching its own gold tool's
group (e.g. a kinematics-gold task gets history_h64i_kinematics), since
interference is built to be semantically close to a specific group.
"""

from __future__ import annotations

from pathlib import Path

from tool_context_study.build_conditions import build_condition, validate_condition
from tool_context_study.config import DATA_DIR, load_experiment_config
from tool_context_study.schemas import AssembledHistory, ConditionRecord, FinalTaskRecord, NormalizedToolSchema

CATALOG_PATH = DATA_DIR / "catalog" / "tools.jsonl"
DEV_TASKS_PATH = DATA_DIR / "development" / "tasks.jsonl"
ASSEMBLED_DIR = DATA_DIR / "histories" / "assembled"
OUTPUT_PATH = DATA_DIR / "conditions" / "phase3_dev_grid.jsonl"

GOLD_POSITIONS = [1, 16, 32]
HISTORY_CONDITIONS = ["H0", "H16N", "H64N", "H64I"]


def _load_jsonl(path: Path, model_cls):
    with open(path) as f:
        return [model_cls.model_validate_json(line) for line in f if line.strip()]


def load_catalog() -> list[NormalizedToolSchema]:
    return _load_jsonl(CATALOG_PATH, NormalizedToolSchema)


def load_dev_tasks() -> list[FinalTaskRecord]:
    return _load_jsonl(DEV_TASKS_PATH, FinalTaskRecord)


def load_assembled_histories() -> dict[str, AssembledHistory]:
    histories = {}
    for path in ASSEMBLED_DIR.glob("*.json"):
        with open(path) as f:
            h = AssembledHistory.model_validate_json(f.read())
        histories[h.history_id] = h
    return histories


def build_dev_condition_grid(
    tasks: list[FinalTaskRecord],
    catalog: list[NormalizedToolSchema],
    histories: dict[str, AssembledHistory],
    base_seed: int,
) -> list[ConditionRecord]:
    group_by_tool_name = {t.name: t.group for t in catalog}
    h64n = histories["history_h64n_base"]
    h16n = histories["history_h16n_base"]
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
                    base_seed=base_seed,
                    history_id=history_id,
                )
                validate_condition(cond, catalog)
                conditions.append(cond)
    return conditions


def main() -> None:
    config = load_experiment_config()
    base_seed = config["seeds"]["base_seed"]

    catalog = load_catalog()
    tasks = load_dev_tasks()
    histories = load_assembled_histories()

    conditions = build_dev_condition_grid(tasks, catalog, histories, base_seed)

    expected = len(tasks) * len(HISTORY_CONDITIONS) * len(GOLD_POSITIONS)
    assert len(conditions) == expected, f"expected {expected} conditions, got {len(conditions)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for cond in conditions:
            f.write(cond.model_dump_json() + "\n")

    print(f"Wrote {len(conditions)} conditions ({len(tasks)} tasks x {len(HISTORY_CONDITIONS)} history x {len(GOLD_POSITIONS)} positions) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

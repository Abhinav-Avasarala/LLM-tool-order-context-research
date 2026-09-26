"""Phase 2 driver: the five-task vertical slice (brief section 13, Phase 2).

Builds, from real ingested BFCL data:
  - 5 manually-picked, unambiguous development tasks
  - a provisional 32-tool catalog guaranteed to contain their gold tools
  - 10 conditions (5 tasks x positions {1, 32}, history condition H0)

and writes them under data/development/ and data/conditions/. Does not call
any model -- that's `run_model.py`, run separately once a server is up.

The 5 task ids below were hand-picked from BFCL_v3_multiple.json by
filtering for: exactly one gold tool, 2-4 candidate functions offered in
the original BFCL record, single-turn/single-message requests, no literal
tool name in the request text, and every offered function normalizing
cleanly (no unmapped parameter types). See docs/decision_log.md.
"""

from __future__ import annotations

import json

from tool_context_study.build_catalog import select_provisional_catalog
from tool_context_study.build_conditions import build_condition, validate_condition
from tool_context_study.config import DATA_DIR, load_experiment_config
from tool_context_study.ingest_bfcl import download_bfcl_files, load_questions_with_answers
from tool_context_study.normalize_schema import normalize_name
from tool_context_study.schemas import ConditionRecord, FinalTaskRecord

DEV_TASK_IDS = ["multiple_2", "multiple_4", "multiple_6", "multiple_8", "multiple_15"]

DEVELOPMENT_DIR = DATA_DIR / "development"
CONDITIONS_DIR = DATA_DIR / "conditions"

CATALOG_TARGET_SIZE = 32
GOLD_POSITIONS = [1, 32]
HISTORY_CONDITION = "H0"


def _first_concrete_value(value):
    """BFCL's ground_truth lists acceptable values per argument, using ""
    to mean "not provided / use default". Take the first non-"" option as
    a single representative gold value; drop the argument entirely if
    every option is "" (i.e. it's purely optional and unset here)."""
    if isinstance(value, list):
        for option in value:
            if option != "":
                return _first_concrete_value(option)
        return None
    if isinstance(value, dict):
        return {k: _first_concrete_value(v) for k, v in value.items()}
    return value


def build_dev_tasks(question_answer_pairs, wanted_ids: list[str]) -> list[FinalTaskRecord]:
    by_id = {q.id: (q, a) for q, a in question_answer_pairs}
    tasks = []
    for task_num, qid in enumerate(wanted_ids):
        q, a = by_id[qid]
        gold_name, raw_gold_args = next(iter(a.ground_truth[0].items()))
        gold_arguments = {}
        for key, value in raw_gold_args.items():
            concrete = _first_concrete_value(value)
            if concrete is not None:
                gold_arguments[key] = concrete

        tasks.append(
            FinalTaskRecord(
                task_id=f"dev_{task_num}",
                request=q.question[0][0]["content"],
                gold_tool=normalize_name(gold_name),
                gold_arguments=gold_arguments,
                authoring_source=f"bfcl_dev_reused:{qid}",
                review_status="approved",
            )
        )
    return tasks


def build_conditions_for_tasks(tasks: list[FinalTaskRecord], catalog, base_seed: int) -> list[ConditionRecord]:
    conditions = []
    for task in tasks:
        for position in GOLD_POSITIONS:
            cond = build_condition(
                task,
                catalog,
                gold_position=position,
                history_condition=HISTORY_CONDITION,
                history_messages=[],
                base_seed=base_seed,
            )
            validate_condition(cond, catalog)
            conditions.append(cond)
    return conditions


def main() -> None:
    config = load_experiment_config()
    bfcl_cfg = config["bfcl"]
    revision = bfcl_cfg["dataset_revision"]
    base_seed = config["seeds"]["base_seed"]

    download_bfcl_files(config)

    multiple_pairs = load_questions_with_answers(
        "BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json", revision
    )
    live_multiple_pairs = load_questions_with_answers(
        "BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json", revision
    )

    tasks = build_dev_tasks(multiple_pairs, DEV_TASK_IDS)

    by_id = {q.id: (q, a) for q, a in multiple_pairs}
    required_names = [next(iter(by_id[qid][1].ground_truth[0].keys())) for qid in DEV_TASK_IDS]

    all_raw_functions = [
        (fn, "BFCL_v3_multiple", revision) for q, _ in multiple_pairs for fn in q.function
    ] + [(fn, "BFCL_v3_live_multiple", revision) for q, _ in live_multiple_pairs for fn in q.function]

    catalog = select_provisional_catalog(
        all_raw_functions, required_names=required_names, target_size=CATALOG_TARGET_SIZE
    )

    conditions = build_conditions_for_tasks(tasks, catalog, base_seed)

    DEVELOPMENT_DIR.mkdir(parents=True, exist_ok=True)
    CONDITIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DEVELOPMENT_DIR / "phase2_tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")

    with open(DEVELOPMENT_DIR / "phase2_provisional_catalog.jsonl", "w") as f:
        for tool in catalog:
            f.write(tool.model_dump_json(by_alias=True) + "\n")

    with open(CONDITIONS_DIR / "phase2_dev5.jsonl", "w") as f:
        for cond in conditions:
            f.write(cond.model_dump_json() + "\n")

    print(f"Wrote {len(tasks)} tasks, {len(catalog)} catalog tools, {len(conditions)} conditions.")
    for task in tasks:
        print(f"  {task.task_id}: gold={task.gold_tool!r} request={task.request[:70]!r}")


if __name__ == "__main__":
    main()

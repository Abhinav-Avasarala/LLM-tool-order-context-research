"""Phase 3, steps 1-2 (brief section 13): the curated 32-tool catalog and
the ~24-question development set.

Writes:
  - data/catalog/tools.jsonl          (32 curated tools, per configs/catalog_spec.yaml)
  - data/development/tasks.jsonl      (24 dev tasks: 20 reused BFCL questions + 4 dev-only authored)

The dev task list is a mix, by design (see docs/decision_log.md for the full
rationale): checking every one of the 32 catalog tools against BFCL's own
ground-truth data found that 14 of them never appear as the *sole correct
answer* to any question in BFCL_v3_multiple.json / BFCL_v3_live_multiple.json
-- they only ever show up as distractor options. 20 real BFCL questions cover
18 of the 32 tools; the 4 DEV_ONLY_TASKS below are freshly authored (against
the tools' real parameter schemas) to cover 4 more, chosen one per domain.
These are explicitly dev-set scaffolding, not held to the final-96 authoring
bar, and are never reused for the frozen final set.
"""

from __future__ import annotations

import json

import yaml

from tool_context_study.build_catalog import build_curated_catalog
from tool_context_study.config import DATA_DIR, PROJECT_ROOT, load_experiment_config
from tool_context_study.ingest_bfcl import download_bfcl_files, load_questions_with_answers
from tool_context_study.normalize_schema import normalize_name
from tool_context_study.schemas import FinalTaskRecord

CATALOG_SPEC_PATH = PROJECT_ROOT / "configs" / "catalog_spec.yaml"
CATALOG_DIR = DATA_DIR / "catalog"
DEVELOPMENT_DIR = DATA_DIR / "development"

# original_name -> BFCL question id, for the 20 genuinely-reused questions.
# One per tool, except weather.get (3 real distinct candidates, all kept).
REUSED_TASK_SOURCES: list[tuple[str, str, str]] = [
    # (original_name, question_id, source_file)
    ("kinematics.calculate_displacement", "multiple_4", "BFCL_v3_multiple.json"),
    ("kinematics.calculate_speed_from_rest", "multiple_36", "BFCL_v3_multiple.json"),
    ("capacitance_calculator.calculate", "multiple_6", "BFCL_v3_multiple.json"),
    ("magnetic_field.calculate", "multiple_41", "BFCL_v3_multiple.json"),
    ("country_info.capital", "multiple_2", "BFCL_v3_multiple.json"),
    ("maps.get_distance_duration", "multiple_193", "BFCL_v3_multiple.json"),
    ("weather.get", "live_multiple_135-51-1", "BFCL_v3_live_multiple.json"),
    ("weather.get", "live_multiple_138-53-0", "BFCL_v3_live_multiple.json"),
    ("weather.get", "live_multiple_140-54-0", "BFCL_v3_live_multiple.json"),
    ("weather.get_by_coordinates_date", "multiple_5", "BFCL_v3_multiple.json"),
    ("weather.humidity_forecast", "multiple_142", "BFCL_v3_multiple.json"),
    ("corporate_finance.calculate_YOY_growth_rate", "multiple_12", "BFCL_v3_multiple.json"),
    ("corporate_finance.revenue_forecast", "multiple_13", "BFCL_v3_multiple.json"),
    ("corporate_finance.dividend_data", "multiple_54", "BFCL_v3_multiple.json"),
    ("finance.calculate_future_value", "multiple_132", "BFCL_v3_multiple.json"),
    ("currency_conversion.convert", "multiple_192", "BFCL_v3_multiple.json"),
    ("realestate.find_properties", "multiple_8", "BFCL_v3_multiple.json"),
    ("finance.property_depreciation", "multiple_14", "BFCL_v3_multiple.json"),
    ("solarFarm.potential", "multiple_15", "BFCL_v3_multiple.json"),
    ("wildlife_population.assess_growth", "multiple_7", "BFCL_v3_multiple.json"),
]

# Freshly authored dev-only tasks, one per domain, for tools with zero BFCL
# ground-truth coverage. Written directly against each tool's real parameter
# schema (checked in data/raw/bfcl before writing these).
DEV_ONLY_TASKS = [
    {
        "gold_tool_original": "electric_field.calculate",
        "request": "What is the electric field strength at a distance of 2 meters from a charge of 5 coulombs?",
        "gold_arguments": {"Q": 5, "r": 2},
    },
    {
        "gold_tool_original": "country_info.population",
        "request": "How many people live in Japan?",
        "gold_arguments": {"country": "Japan"},
    },
    {
        "gold_tool_original": "finance.loan_repayment",
        "request": "What would the monthly repayment be on a $250,000 loan with a 4.5% annual interest rate over a 30 year term?",
        "gold_arguments": {"loan_amount": 250000, "interest_rate": 4.5, "loan_term": 30},
    },
    {
        "gold_tool_original": "windFarm.potential",
        "request": "Estimate the energy output in March for a wind farm at coordinates [51.5074, -0.1278] with 40 turbines.",
        "gold_arguments": {"coordinates": [51.5074, -0.1278], "turbineCount": 40, "month": "March"},
    },
]


def _first_concrete_value(value):
    """See phase2_dev_slice._first_concrete_value: BFCL lists acceptable
    values per argument with "" meaning "not provided"; take the first
    non-"" option, or drop the argument if every option is ""."""
    if isinstance(value, list):
        for option in value:
            if option != "":
                return _first_concrete_value(option)
        return None
    if isinstance(value, dict):
        return {k: _first_concrete_value(v) for k, v in value.items()}
    return value


def load_catalog_spec() -> list[dict]:
    with open(CATALOG_SPEC_PATH) as f:
        return yaml.safe_load(f)["tools"]


def build_raw_function_lookup(revision: str) -> dict[str, tuple[dict, str, str]]:
    lookup: dict[str, tuple[dict, str, str]] = {}
    for filename, source in [
        ("BFCL_v3_multiple.json", "BFCL_v3_multiple"),
        ("BFCL_v3_live_multiple.json", "BFCL_v3_live_multiple"),
    ]:
        with open(DATA_DIR / "raw" / "bfcl" / filename) as f:
            for line in f:
                rec = json.loads(line)
                for fn in rec["function"]:
                    lookup.setdefault(fn["name"], (fn, source, revision))
    return lookup


def build_reused_dev_tasks(revision: str) -> list[FinalTaskRecord]:
    questions_by_file: dict[str, dict[str, tuple]] = {}
    for question_file, answer_file in [
        ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"),
        ("BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json"),
    ]:
        pairs = load_questions_with_answers(question_file, answer_file, revision)
        questions_by_file[question_file] = {q.id: (q, a) for q, a in pairs}

    tasks = []
    for task_num, (original_name, qid, source_file) in enumerate(REUSED_TASK_SOURCES):
        q, a = questions_by_file[source_file][qid]
        gold_name, raw_gold_args = next(iter(a.ground_truth[0].items()))
        assert gold_name == original_name, f"{qid}: expected gold {original_name!r}, got {gold_name!r}"

        gold_arguments = {}
        for key, value in raw_gold_args.items():
            concrete = _first_concrete_value(value)
            if concrete is not None:
                gold_arguments[key] = concrete

        tasks.append(
            FinalTaskRecord(
                task_id=f"dev_{task_num:02d}",
                request=q.question[0][0]["content"],
                gold_tool=normalize_name(gold_name),
                gold_arguments=gold_arguments,
                authoring_source=f"bfcl_dev_reused:{qid}",
                review_status="approved",
            )
        )
    return tasks


def build_fresh_dev_tasks(start_num: int) -> list[FinalTaskRecord]:
    tasks = []
    for i, spec in enumerate(DEV_ONLY_TASKS):
        tasks.append(
            FinalTaskRecord(
                task_id=f"dev_{start_num + i:02d}",
                request=spec["request"],
                gold_tool=normalize_name(spec["gold_tool_original"]),
                gold_arguments=spec["gold_arguments"],
                authoring_source="dev_only_authored",
                review_status="approved",
            )
        )
    return tasks


def main() -> None:
    config = load_experiment_config()
    revision = config["bfcl"]["dataset_revision"]
    download_bfcl_files(config)

    catalog_spec = load_catalog_spec()
    raw_lookup = build_raw_function_lookup(revision)
    catalog = build_curated_catalog(catalog_spec, raw_lookup)
    catalog_names = {t.name for t in catalog}

    reused_tasks = build_reused_dev_tasks(revision)
    fresh_tasks = build_fresh_dev_tasks(start_num=len(reused_tasks))
    tasks = reused_tasks + fresh_tasks

    for task in tasks:
        assert task.gold_tool in catalog_names, f"{task.task_id}: gold tool {task.gold_tool!r} not in catalog"
    assert len(tasks) == 24, f"expected 24 dev tasks, got {len(tasks)}"
    assert len({t.task_id for t in tasks}) == len(tasks), "duplicate task_ids"

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    DEVELOPMENT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CATALOG_DIR / "tools.jsonl", "w") as f:
        for tool in catalog:
            f.write(tool.model_dump_json(by_alias=True) + "\n")

    with open(DEVELOPMENT_DIR / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")

    n_covered = len({t.gold_tool for t in tasks})
    print(f"Wrote {len(catalog)} catalog tools and {len(tasks)} dev tasks ({n_covered}/32 tools covered).")
    for task in tasks:
        print(f"  {task.task_id}: gold={task.gold_tool!r} ({task.authoring_source}) request={task.request[:65]!r}")


if __name__ == "__main__":
    main()

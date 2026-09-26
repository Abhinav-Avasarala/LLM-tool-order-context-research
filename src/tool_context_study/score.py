"""Primary tool-selection scoring (brief section 10).

Phase 2 note: the provisional catalog has no domain/group tags yet (that's
Phase 3), so the `wrong_high_similarity` / `wrong_same_domain` /
`wrong_unrelated` split from the full error taxonomy isn't available here.
This computes the primary metric plus the taxonomy categories that don't
depend on catalog structure, and lumps everything else under `wrong`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tool_context_study.schemas import ConditionRecord, ResultRecord


def _n_tool_calls(result: ResultRecord) -> int:
    if not isinstance(result.raw_response, dict):
        return 0
    choices = result.raw_response.get("choices") or []
    if not choices:
        return 0
    return len(choices[0].get("message", {}).get("tool_calls") or [])


def categorize(result: ResultRecord, gold_tool: str) -> str:
    if result.status == "inference_error":
        return "inference_error"
    if result.status == "no_tool_call":
        return "no_tool_call"
    if result.status == "malformed_tool_call":
        return "malformed_tool_call"
    # status == "ok" from here on
    if _n_tool_calls(result) > 1:
        return "multiple_tool_calls"
    if result.selected_tool == gold_tool:
        return "correct"
    return "wrong"


def load_jsonl(path: Path, model_cls):
    with open(path) as f:
        return [model_cls.model_validate_json(line) for line in f if line.strip()]


def score(results: list[ResultRecord], conditions: list[ConditionRecord]) -> dict:
    conditions_by_id = {c.condition_id: c for c in conditions}

    rows = []
    for result in results:
        condition = conditions_by_id[result.condition_id]
        category = categorize(result, condition.gold_tool)
        rows.append(
            {
                "condition_id": result.condition_id,
                "task_id": condition.task_id,
                "gold_position": condition.gold_position,
                "history_condition": condition.history_condition,
                "category": category,
                "correct": category == "correct",
            }
        )

    overall_accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    by_position: dict[int, dict] = {}
    for position in sorted({r["gold_position"] for r in rows}):
        subset = [r for r in rows if r["gold_position"] == position]
        by_position[position] = {
            "n": len(subset),
            "accuracy": sum(r["correct"] for r in subset) / len(subset),
            "categories": dict(Counter(r["category"] for r in subset)),
        }

    return {
        "n_conditions": len(rows),
        "overall_accuracy": overall_accuracy,
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "by_position": by_position,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    args = parser.parse_args()

    results = load_jsonl(args.results, ResultRecord)
    conditions = load_jsonl(args.conditions, ConditionRecord)
    report = score(results, conditions)

    print(f"n_conditions: {report['n_conditions']}")
    print(f"overall_accuracy: {report['overall_accuracy']:.3f}")
    print(f"category_counts: {report['category_counts']}")
    print("by_position:")
    for position, stats in report["by_position"].items():
        print(f"  position {position}: n={stats['n']} accuracy={stats['accuracy']:.3f} categories={stats['categories']}")
    for row in report["rows"]:
        marker = "OK " if row["correct"] else "-- "
        print(f"  {marker}{row['condition_id']}: {row['category']}")


if __name__ == "__main__":
    main()

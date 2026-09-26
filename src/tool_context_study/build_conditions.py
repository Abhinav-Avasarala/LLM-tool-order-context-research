"""Build position-controlled `ConditionRecord`s from a task + tool catalog.

Core mechanism for brief section 2's position manipulation: the same 32
tool schemas, the same request, the same history -- only the position of
the gold tool (and the shuffled order of the other 31) changes, and that
shuffle is seeded so it is deterministic and reproducible.
"""

from __future__ import annotations

import random
import zlib
from typing import Any

from tool_context_study.schemas import ConditionRecord, FinalTaskRecord, GoldPosition, HistoryConditionId, NormalizedToolSchema

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to a set of tools. "
    "Use a tool when it helps answer the user's request."
)


def derive_permutation_seed(base_seed: int, task_id: str, gold_position: int, history_condition: str) -> int:
    """Deterministic per-(task, position, history) seed derived from a base seed.

    Using a hash of the identifying fields (rather than e.g. just
    `base_seed + position`) avoids accidentally correlated seeds across
    tasks/positions while remaining fully reproducible from the recorded
    inputs.
    """
    key = f"{task_id}|{gold_position}|{history_condition}".encode("utf-8")
    return base_seed + zlib.crc32(key)


def build_condition(
    task: FinalTaskRecord,
    catalog: list[NormalizedToolSchema],
    *,
    gold_position: GoldPosition,
    history_condition: HistoryConditionId,
    history_messages: list[dict[str, Any]],
    base_seed: int,
    history_id: str | None = None,
    system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
) -> ConditionRecord:
    """Place `task.gold_tool` at `gold_position` among `catalog`, shuffling the rest.

    `catalog` must contain exactly one tool whose normalized `name` equals
    `task.gold_tool`; every other catalog tool is treated as a distractor.
    """
    gold_matches = [t for t in catalog if t.name == task.gold_tool]
    if len(gold_matches) != 1:
        raise ValueError(
            f"Expected exactly one catalog tool named {task.gold_tool!r} for task "
            f"{task.task_id!r}, found {len(gold_matches)}"
        )
    gold_tool = gold_matches[0]
    distractors = [t for t in catalog if t.name != task.gold_tool]

    seed = derive_permutation_seed(base_seed, task.task_id, gold_position, history_condition)
    rng = random.Random(seed)
    shuffled_distractors = distractors[:]
    rng.shuffle(shuffled_distractors)

    ordered = shuffled_distractors[:]
    ordered.insert(gold_position - 1, gold_tool)

    messages: list[dict[str, Any]] = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": task.request})

    condition_id = f"{task.task_id}__{history_condition}__p{gold_position}"

    return ConditionRecord(
        condition_id=condition_id,
        task_id=task.task_id,
        history_condition=history_condition,
        history_id=history_id,
        gold_tool=task.gold_tool,
        gold_position=gold_position,
        permutation_seed=seed,
        messages=messages,
        tools=[t.schema_ for t in ordered],
    )


def validate_condition(condition: ConditionRecord, catalog: list[NormalizedToolSchema]) -> None:
    """Raise `AssertionError` if `condition` doesn't meet the design invariants."""
    catalog_names = {t.name for t in catalog}
    tool_names = [t["function"]["name"] for t in condition.tools]

    assert len(condition.tools) == len(catalog), (
        f"{condition.condition_id}: expected {len(catalog)} tools, got {len(condition.tools)}"
    )
    assert len(set(tool_names)) == len(tool_names), f"{condition.condition_id}: duplicate tool names"
    assert set(tool_names) == catalog_names, f"{condition.condition_id}: tool set doesn't match catalog"
    assert tool_names[condition.gold_position - 1] == condition.gold_tool, (
        f"{condition.condition_id}: gold tool not at declared position {condition.gold_position}"
    )
    assert condition.messages[-1]["role"] == "user", f"{condition.condition_id}: last message must be from user"

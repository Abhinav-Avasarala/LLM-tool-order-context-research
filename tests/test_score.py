from tool_context_study.schemas import ConditionRecord, ResultRecord
from tool_context_study.score import categorize, score


def _condition(condition_id: str, gold_tool: str, gold_position=1) -> ConditionRecord:
    return ConditionRecord(
        condition_id=condition_id,
        task_id=condition_id.split("__")[0],
        history_condition="H0",
        history_id=None,
        gold_tool=gold_tool,
        gold_position=gold_position,
        permutation_seed=0,
        messages=[],
        tools=[],
    )


def _result(condition_id, status, selected_tool=None, arguments=None, tool_calls=None) -> ResultRecord:
    message = {"role": "assistant", "tool_calls": tool_calls or []}
    return ResultRecord(
        condition_id=condition_id,
        model="test-model",
        model_revision="rev0",
        status=status,
        selected_tool=selected_tool,
        arguments=arguments,
        raw_response={"choices": [{"message": message}]},
        history_tokens=0,
        total_input_tokens=100,
        latency_seconds=0.1,
    )


def test_categorize_correct():
    cond = _condition("c1", "gold_tool")
    res = _result("c1", "ok", selected_tool="gold_tool", tool_calls=[{"function": {"name": "gold_tool"}}])
    assert categorize(res, cond.gold_tool) == "correct"


def test_categorize_wrong():
    cond = _condition("c1", "gold_tool")
    res = _result("c1", "ok", selected_tool="other_tool", tool_calls=[{"function": {"name": "other_tool"}}])
    assert categorize(res, cond.gold_tool) == "wrong"


def test_categorize_multiple_tool_calls_even_if_one_is_gold():
    cond = _condition("c1", "gold_tool")
    res = _result(
        "c1",
        "ok",
        tool_calls=[{"function": {"name": "gold_tool"}}, {"function": {"name": "other_tool"}}],
    )
    assert categorize(res, cond.gold_tool) == "multiple_tool_calls"


def test_categorize_no_tool_call():
    cond = _condition("c1", "gold_tool")
    res = _result("c1", "no_tool_call")
    assert categorize(res, cond.gold_tool) == "no_tool_call"


def test_categorize_malformed_and_inference_error_passthrough():
    cond = _condition("c1", "gold_tool")
    assert categorize(_result("c1", "malformed_tool_call"), cond.gold_tool) == "malformed_tool_call"
    assert categorize(_result("c1", "inference_error"), cond.gold_tool) == "inference_error"


def test_score_computes_overall_and_per_position_accuracy():
    conditions = [
        _condition("t1__H0__p1", "gold_a", gold_position=1),
        _condition("t1__H0__p32", "gold_a", gold_position=32),
        _condition("t2__H0__p1", "gold_b", gold_position=1),
        _condition("t2__H0__p32", "gold_b", gold_position=32),
    ]
    results = [
        _result("t1__H0__p1", "ok", selected_tool="gold_a", tool_calls=[{"function": {"name": "gold_a"}}]),
        _result("t1__H0__p32", "ok", selected_tool="wrong", tool_calls=[{"function": {"name": "wrong"}}]),
        _result("t2__H0__p1", "ok", selected_tool="gold_b", tool_calls=[{"function": {"name": "gold_b"}}]),
        _result("t2__H0__p32", "ok", selected_tool="gold_b", tool_calls=[{"function": {"name": "gold_b"}}]),
    ]

    report = score(results, conditions)
    assert report["n_conditions"] == 4
    assert report["overall_accuracy"] == 0.75
    assert report["by_position"][1]["accuracy"] == 1.0
    assert report["by_position"][32]["accuracy"] == 0.5

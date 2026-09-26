from tool_context_study.parsers.openai_tool_call import parse_chat_completion


def _response_with_tool_calls(tool_calls):
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}]}


def test_parses_single_well_formed_tool_call():
    response = _response_with_tool_calls(
        [{"id": "call_1", "function": {"name": "country_info__capital", "arguments": '{"country": "Peru"}'}}]
    )
    result = parse_chat_completion(response)
    assert result == {
        "status": "ok",
        "selected_tool": "country_info__capital",
        "arguments": {"country": "Peru"},
        "n_tool_calls": 1,
    }


def test_no_tool_calls_is_no_tool_call():
    response = _response_with_tool_calls([])
    result = parse_chat_completion(response)
    assert result["status"] == "no_tool_call"
    assert result["selected_tool"] is None


def test_multiple_tool_calls_leaves_selection_unset():
    response = _response_with_tool_calls(
        [
            {"id": "call_1", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "b", "arguments": "{}"}},
        ]
    )
    result = parse_chat_completion(response)
    assert result["status"] == "ok"
    assert result["selected_tool"] is None
    assert result["n_tool_calls"] == 2


def test_malformed_json_arguments_is_malformed_tool_call():
    response = _response_with_tool_calls(
        [{"id": "call_1", "function": {"name": "country_info__capital", "arguments": "{not valid json"}}]
    )
    result = parse_chat_completion(response)
    assert result["status"] == "malformed_tool_call"
    assert result["selected_tool"] == "country_info__capital"
    assert result["arguments"] is None


def test_no_choices_is_inference_error():
    result = parse_chat_completion({"choices": []})
    assert result["status"] == "inference_error"


def test_arguments_already_a_dict_is_accepted():
    response = _response_with_tool_calls([{"id": "call_1", "function": {"name": "f", "arguments": {"x": 1}}}])
    result = parse_chat_completion(response)
    assert result["status"] == "ok"
    assert result["arguments"] == {"x": 1}

import pytest

from tool_context_study.build_catalog import select_provisional_catalog


def _raw(name: str, param_type: str = "string") -> dict:
    return {
        "name": name,
        "description": f"Tool {name}.",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": param_type, "description": "x"}},
            "required": ["x"],
        },
    }


def test_select_provisional_catalog_includes_required_and_fills_to_target_size():
    raw_functions = [(_raw(f"tool.{i}"), "src", "rev0") for i in range(40)]
    catalog = select_provisional_catalog(raw_functions, required_names=["tool.3", "tool.7"], target_size=10)

    names = {t.name for t in catalog}
    assert len(catalog) == 10
    assert "tool__3" in names
    assert "tool__7" in names


def test_select_provisional_catalog_dedupes_by_normalized_name():
    raw_functions = [(_raw("tool.dup"), "src", "rev0")] * 5 + [(_raw(f"tool.{i}"), "src", "rev0") for i in range(10)]
    catalog = select_provisional_catalog(raw_functions, required_names=["tool.dup"], target_size=5)

    names = [t.name for t in catalog]
    assert len(names) == len(set(names))


def test_select_provisional_catalog_skips_unnormalizable_functions():
    raw_functions = [(_raw("bad.tool", param_type="any"), "src", "rev0")] + [
        (_raw(f"tool.{i}"), "src", "rev0") for i in range(10)
    ]
    catalog = select_provisional_catalog(raw_functions, required_names=["tool.0"], target_size=5)
    assert "bad__tool" not in {t.name for t in catalog}


def test_select_provisional_catalog_raises_if_required_tool_missing():
    raw_functions = [(_raw(f"tool.{i}"), "src", "rev0") for i in range(10)]
    with pytest.raises(ValueError):
        select_provisional_catalog(raw_functions, required_names=["does.not.exist"], target_size=5)


def test_select_provisional_catalog_raises_if_not_enough_tools():
    raw_functions = [(_raw(f"tool.{i}"), "src", "rev0") for i in range(3)]
    with pytest.raises(ValueError):
        select_provisional_catalog(raw_functions, required_names=["tool.0"], target_size=10)

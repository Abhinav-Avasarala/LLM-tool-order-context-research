import pytest

from tool_context_study.build_catalog import build_curated_catalog, validate_curated_catalog
from tool_context_study.normalize_schema import normalize_function


def _raw(name: str) -> dict:
    return {
        "name": name,
        "description": f"Tool {name}.",
        "parameters": {"type": "dict", "properties": {"x": {"type": "string", "description": "x"}}, "required": ["x"]},
    }


def _make_spec(n_domains=4, n_groups=2, n_tools=4) -> tuple[list[dict], dict]:
    spec = []
    raw_lookup = {}
    for d in range(n_domains):
        for g in range(n_groups):
            for t in range(n_tools):
                name = f"d{d}.g{g}.t{t}"
                spec.append({"original_name": name, "domain": f"domain_{d}", "group": f"domain_{d}_group_{g}"})
                raw_lookup[name] = (_raw(name), "fixture", "rev0")
    return spec, raw_lookup


def test_build_curated_catalog_happy_path():
    spec, raw_lookup = _make_spec()
    catalog = build_curated_catalog(spec, raw_lookup)
    assert len(catalog) == 32
    assert len({t.domain for t in catalog}) == 4
    assert len({t.group for t in catalog}) == 8


def test_build_curated_catalog_raises_on_missing_tool():
    spec, raw_lookup = _make_spec()
    del raw_lookup["d0.g0.t0"]
    with pytest.raises(ValueError):
        build_curated_catalog(spec, raw_lookup)


def test_validate_curated_catalog_rejects_wrong_total_size():
    spec, raw_lookup = _make_spec()
    catalog = build_curated_catalog(spec, raw_lookup)
    with pytest.raises(AssertionError):
        validate_curated_catalog(catalog[:-1])


def test_validate_curated_catalog_rejects_uneven_group_size():
    spec, raw_lookup = _make_spec()
    catalog = build_curated_catalog(spec, raw_lookup)
    # catalog[0] is d0/group0, catalog[4] is d0/group1 (4 tools per group);
    # reassign catalog[0] into group1 to unbalance that domain's group sizes.
    catalog[0].group = catalog[4].group
    with pytest.raises(AssertionError):
        validate_curated_catalog(catalog)


def test_validate_curated_catalog_rejects_missing_domain():
    tool = normalize_function(_raw("bare.tool"), source="fixture", source_revision="rev0")
    with pytest.raises(AssertionError):
        validate_curated_catalog([tool] * 32)  # also fails uniqueness, but domain assert fires first in practice

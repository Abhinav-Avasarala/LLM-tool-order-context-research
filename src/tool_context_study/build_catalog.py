"""Assemble a 32-tool catalog from normalized BFCL functions.

Two builders, for two different purposes:
  - `select_provisional_catalog` (Phase 2): get *a* valid 32-tool catalog
    quickly (deduped, normalizable, includes whatever gold tools a vertical
    slice needs), with `domain`/`group` left `None`. Superseded for real
    work by the curated catalog below, kept for the Phase 2 slice's tests.
  - `build_curated_catalog` (Phase 3): builds the actual, manually-curated
    32-tool catalog from `configs/catalog_spec.yaml` -- 4 domains x 2 groups
    x 4 tools, hand-picked to avoid BFCL's many literal near-duplicates and
    checked (via `validate_curated_catalog`) to actually form that shape.
"""

from __future__ import annotations

from tool_context_study.normalize_schema import UnknownParameterTypeError, normalize_function, normalize_name
from tool_context_study.schemas import NormalizedToolSchema


def select_provisional_catalog(
    raw_functions: list[tuple[dict, str, str]],
    *,
    required_names: list[str],
    target_size: int = 32,
) -> list[NormalizedToolSchema]:
    """Pick `target_size` unique, normalizable tools, guaranteeing `required_names` are included.

    `raw_functions` is a list of (raw_function_dict, source, source_revision) triples,
    e.g. gathered while iterating BFCL question records. `required_names` are
    *original* BFCL names (e.g. "country_info.capital") that must end up in
    the catalog -- typically the gold tools of the tasks this catalog will
    serve. Selection beyond the required tools is deterministic: candidates
    are deduped by normalized name and taken in the order first seen.
    """
    required_normalized = {normalize_name(n) for n in required_names}

    seen_names: set[str] = set()
    normalized: list[NormalizedToolSchema] = []
    for raw_function, source, source_revision in raw_functions:
        name = normalize_name(raw_function["name"])
        if name in seen_names:
            continue
        try:
            tool = normalize_function(raw_function, source=source, source_revision=source_revision)
        except UnknownParameterTypeError:
            continue
        seen_names.add(name)
        normalized.append(tool)

    by_name = {t.name: t for t in normalized}
    missing = required_normalized - by_name.keys()
    if missing:
        raise ValueError(f"Required tool(s) not found or not normalizable: {sorted(missing)}")

    required_tools = [by_name[n] for n in sorted(required_normalized)]
    other_tools = [t for t in normalized if t.name not in required_normalized]

    if len(required_tools) + len(other_tools) < target_size:
        raise ValueError(
            f"Only {len(required_tools) + len(other_tools)} normalizable unique tools available, "
            f"need {target_size}"
        )

    n_fillers = target_size - len(required_tools)
    catalog = required_tools + other_tools[:n_fillers]
    assert len(catalog) == target_size
    assert len({t.name for t in catalog}) == target_size
    return catalog


def build_curated_catalog(
    catalog_spec: list[dict],
    raw_functions_by_original_name: dict[str, tuple[dict, str, str]],
) -> list[NormalizedToolSchema]:
    """Build the Phase 3 curated catalog from `configs/catalog_spec.yaml`.

    `catalog_spec` is that file's `tools` list: dicts with `original_name`,
    `domain`, `group`. `raw_functions_by_original_name` maps each original
    BFCL name to (raw_function_dict, source, source_revision) -- typically
    built by scanning the ingested BFCL files once. Raises `ValueError` if
    the spec doesn't actually form 4 domains x 2 groups x 4 tools, or names
    a tool that isn't in the provided raw functions.
    """
    catalog: list[NormalizedToolSchema] = []
    for entry in catalog_spec:
        original_name = entry["original_name"]
        if original_name not in raw_functions_by_original_name:
            raise ValueError(f"catalog_spec names {original_name!r}, which isn't in the provided raw functions")
        raw_function, source, source_revision = raw_functions_by_original_name[original_name]
        catalog.append(
            normalize_function(
                raw_function,
                source=source,
                source_revision=source_revision,
                domain=entry["domain"],
                group=entry["group"],
            )
        )

    validate_curated_catalog(catalog)
    return catalog


def validate_curated_catalog(
    catalog: list[NormalizedToolSchema],
    *,
    n_domains: int = 4,
    n_groups_per_domain: int = 2,
    n_tools_per_group: int = 4,
) -> None:
    """Raise `AssertionError` if `catalog` isn't a well-formed domain/group structure."""
    target_size = n_domains * n_groups_per_domain * n_tools_per_group

    names = [t.name for t in catalog]
    assert len(catalog) == target_size, f"expected {target_size} tools, got {len(catalog)}"
    assert len(set(names)) == target_size, "duplicate tool names in catalog"
    assert all(t.schema_id for t in catalog), "every tool must have a schema_id"

    by_domain: dict[str, list[NormalizedToolSchema]] = {}
    for tool in catalog:
        assert tool.domain is not None, f"{tool.name}: missing domain"
        assert tool.group is not None, f"{tool.name}: missing group"
        by_domain.setdefault(tool.domain, []).append(tool)

    assert len(by_domain) == n_domains, f"expected {n_domains} domains, got {len(by_domain)}: {sorted(by_domain)}"

    for domain, domain_tools in by_domain.items():
        assert len(domain_tools) == n_groups_per_domain * n_tools_per_group, (
            f"domain {domain!r}: expected {n_groups_per_domain * n_tools_per_group} tools, got {len(domain_tools)}"
        )
        by_group: dict[str, list[NormalizedToolSchema]] = {}
        for tool in domain_tools:
            by_group.setdefault(tool.group, []).append(tool)
        assert len(by_group) == n_groups_per_domain, (
            f"domain {domain!r}: expected {n_groups_per_domain} groups, got {len(by_group)}: {sorted(by_group)}"
        )
        for group, group_tools in by_group.items():
            assert len(group_tools) == n_tools_per_group, (
                f"domain {domain!r} group {group!r}: expected {n_tools_per_group} tools, got {len(group_tools)}"
            )

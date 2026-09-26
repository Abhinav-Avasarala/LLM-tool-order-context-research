"""Assemble a 32-tool catalog from normalized BFCL functions.

Phase 2 uses `select_provisional_catalog` to get *a* valid 32-tool catalog
quickly (deduped, normalizable, includes whatever gold tools the vertical
slice needs) so the position/permutation machinery can be end-to-end
tested. It does not attempt the domain/group curation the brief asks for
in Phase 3 (manually checking each tool has a distinct purpose, organizing
into 4 domains x 2 groups x 4 tools) -- `domain`/`group` are left `None`
here and must be filled in when Phase 3 replaces this provisional catalog.
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

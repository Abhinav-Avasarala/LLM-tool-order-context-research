# Decision log

Records deviations from, and open choices within, `docs/research_brief.md`
(brief section 15 asks for finalized choices to be logged here rather than
left implicit in code).

## 2026-09-26 — Phase 1

**BFCL source: HF dataset mirror, not the gorilla GitHub repo, and pinned by
commit sha rather than a tag.**
The brief names `BFCL_v3_multiple.json`, `BFCL_v3_live_multiple.json`, and
their `possible_answer/` counterparts. As of this date, the `gorilla` GitHub
repo (HEAD `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`) has moved on to `v4`
files and no longer carries the `v3` names, and it has no tags to pin an
older `v3` snapshot to. The HF dataset mirror
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) still hosts the exact
`v3` files named in the brief, so ingestion pulls from there, pinned at
revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`. Scientific consequence:
none — this is the same file content the brief specifies, just a different
(still-official, brief-sanctioned) host; recorded here for traceability, not
because it changes anything.

**Skip-and-warn (not fail) BFCL question/answer id mismatches.**
`BFCL_v3_live_multiple.json` has 1 question id (`live_multiple_1052-79-0`)
with no matching id in its own `possible_answer/` file (closest match:
`live_multiple_1052-279-0`, apparently a `79` vs `279` typo upstream in
BFCL itself). `ingest_bfcl.load_questions_with_answers` skips unmatched
questions and prints a warning rather than raising, since this is
development-set scaffolding, not the frozen final set. If similar mismatches
turn up in files used for anything beyond development scaffolding, they
should be handled by the predefined final-set annotation-defect policy
(brief section 3), not silently dropped.

**Type mapping in `normalize_schema.py` raises on unmapped types instead of
guessing.** Surveyed all parameter types across the two ingested question
files: `string, dict, integer, boolean, float, array, any, tuple` (8 `any`
occurrences out of ~4,735 functions). `any` has no unambiguous JSON Schema
mapping, so `normalize_function` raises `UnknownParameterTypeError` for it
rather than guessing `object` or `string`; those 8 functions will need a
manual per-case decision if/when they are considered for the catalog
(Phase 3), consistent with the brief's "manually check" requirement for
catalog tools.

**Python/tooling: dedicated `conda` env `tool_ctx` (Python 3.11, conda-forge
channel only).** The machine's system `python3` (3.10) has no `pip`, and
`conda`'s default `anaconda.org` channels require accepting a ToS
non-interactively; conda-forge sidesteps that. Not a scientific decision,
recorded for reproducibility of the dev setup.

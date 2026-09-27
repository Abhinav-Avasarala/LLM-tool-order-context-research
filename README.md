# Tool Context Study

Does an LLM's tool-selection accuracy depend on (1) the correct tool's
position among the available tool schemas, (2) how much conversation
history precedes the request, and (3) whether that history is semantically
interfering? See `docs/research_brief.md` for the full experimental design;
this README tracks implementation status and setup.

## Status

**Phase 1 (BFCL ingestion) — done.**
**Phase 2 (five-task vertical slice) — done.** Built a provisional 32-tool
catalog + 5 dev tasks from BFCL, generated position-1/position-32 conditions
under `H0`, served `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` on one L40S via
vLLM, ran all 10 conditions, and scored them (10/10 correct on this easy
pilot slice — expected, and not a real result; the point was proving the
pipeline works, not measuring anything). Verified the actual rendered
chat-template prompt (not just our own condition-builder) preserves all 32
tool names, in the declared order, with the gold tool at the declared
position. See `docs/decision_log.md` for choices made along the way
(notably: vLLM/torch/transformers version pinning for this GPU driver).
**Phase 3 (catalog curation + dev tasks + history assembly + dev pilot) —
in progress.** Curated and user-approved the real 32-tool catalog (4
domains x 2 groups x 4 tools, `configs/catalog_spec.yaml` ->
`data/catalog/tools.jsonl`), built the 24-question development set (20
reused from BFCL + 4 freshly authored, see `docs/decision_log.md`),
generated the episode banks (250 neutral + ~431 interference across 8
target groups), and assembled + validated all four history conditions
(H16N, H64N, one H64I per target group). Ran the full 288-condition dev
grid (24 tasks x H0/H16N/H64N/H64I x positions 1/16/32) on
Qwen3-30B-A3B-Instruct-2507-FP8: 90.6% overall, with a pattern
directionally consistent with the brief's hypotheses (accuracy declines as
history grows; position effects sharpen specifically under H64I). This is
a small-n pilot, not a result — see `docs/decision_log.md` and
`results/processed/phase3_dev_grid_qwen3_30b_a3b_fp8_summary.json`.
Smoke-tested both GPT-OSS models (separate `tool_ctx_gptoss` conda env,
`vllm==0.10.1.1` — the earliest release with gpt-oss support that still
matches this host's CUDA/driver pins): 10/10 correct on the same Phase 2
five-task dev slice for both `gpt-oss-20b` and `gpt-oss-120b`. Getting there
required three real fixes, all logged in detail in `docs/decision_log.md`:
(1) GPT-OSS tool-calling over `/v1/chat/completions` crashes deterministically
whenever the model actually emits a tool call (likely root cause: vLLM's
harmony prompt-builder for that endpoint never passes the tool schemas into
the model at all) — switched to `/v1/responses` instead via a new
`parsers/harmony_responses_call.py` adapter and `run_model.py --api-format
responses` flag; (2) `gpt-oss-120b`'s tensor-parallel (2×L40S) startup hangs
indefinitely on this no-NVLink host without `--enforce-eager
--disable-custom-all-reduce`; (3) `/v1/responses`' `usage` field always
reports zero token counts — fixed by reimplementing gpt-oss's harmony
prompt construction client-side (`parsers/harmony_token_count.py`, using
the `openai_harmony` library directly) rather than trusting that field.
All 440 interference episodes have since been manually reviewed and
approved by the user (see the Phase 4 entry below) — **Phase 3 is done.**
Still outstanding, non-blocking: bisecting which of the three gpt-oss-120b
startup flags were actually necessary (all three were changed at once to
get unblocked).

**Phase 4 (final 96-question set, histories, freeze) — done.** 96 fresh
questions (3 per tool) drafted by 8 parallel subagents against the real
32-tool catalog (`.claude/agents/final-question-writer.md`), reviewed and
approved by the user via a published Artifact backed by its `db`
capability, synced to `data/final/tasks.jsonl` with
`review_status: "approved"`. All 440 interference episodes separately
reviewed and approved by the user via a second Artifact
(`src/tool_context_study/check_interference_episodes.py` for the
automated pre-check; 1 of 440 flagged, confirmed a false positive).
Assembled fresh final-set histories with a dedicated `final_seed`
(distinct from the dev pilot's `base_seed`) — this surfaced and fixed a
genuine authoring defect in the neutral episode bank (`neutral_0083`
ended on a `user` turn; the dev pilot's seed had never exposed it). Built
and validated the full 1,152-condition frozen grid (96 tasks × 4 history
conditions × 3 positions; `phase4_build_final_conditions.py`) — exactly
288/384/36-way uniform as required. Re-verified Phase 2's rendered-prompt
rigor at final scale through Qwen's real tokenizer (all 32 tools present
once, order preserved, gold tool at declared position, no truncation;
largest condition renders to 71,900 tokens, comfortably under the
already-proven 80,000 context window). Decided the two remaining open
protocol items from the brief's "still open" list: interference fraction
reaffirmed at 25% (final), and a predefined annotation-defect policy
(exclude-and-log, never edit a frozen task). Froze everything via
`freeze_phase4.py` → `data/manifests/phase4_freeze.json` (hashes +
frozen design parameters). See `docs/decision_log.md` for full detail on
every decision above.

**Forward note for Phase 5:** GPT-OSS has only been smoke-tested at
`--max-model-len 8192` (H0 only) — it needs the same context-window
increase Qwen already got (to ≥80,000) before it can serve H64N/H64I
conditions at final scale.

**Phase 5 (full model runs, analysis) — not yet started.**

## Setup

```bash
conda create -n tool_ctx -c conda-forge --override-channels python=3.11
conda activate tool_ctx
pip install -e ".[dev]"
```

## Usage

Download and validate the BFCL source files this project reuses (pinned
revision in `configs/experiment.yaml`):

```bash
python -m tool_context_study.ingest_bfcl
```

This writes raw copies to `data/raw/bfcl/` and a provenance manifest to
`data/manifests/bfcl_provenance.json`.

Run tests:

```bash
pytest
```

Build the Phase 2 five-task dev slice (provisional 32-tool catalog + 5 dev
tasks + position-1/32 conditions under `H0`):

```bash
python -m tool_context_study.phase2_dev_slice
```

Serve the dev model (see `configs/models/qwen3_30b_a3b_instruct_2507_fp8.yaml`
for the exact launch command and version pins this required on this host),
then run and score the slice against it:

```bash
python -m tool_context_study.run_model \
  --conditions data/conditions/phase2_dev5.jsonl \
  --model qwen3-30b-a3b-instruct-2507-fp8 \
  --output results/raw/phase2_dev5_qwen3_30b_a3b_fp8.jsonl

python -m tool_context_study.score \
  --results results/raw/phase2_dev5_qwen3_30b_a3b_fp8.jsonl \
  --conditions data/conditions/phase2_dev5.jsonl
```

Build the Phase 3 curated 32-tool catalog + 24-question dev set:

```bash
python -m tool_context_study.phase3_catalog_and_dev_tasks
```

Assemble history conditions (dev-pilot set, the default; pass
`--seed-key final_seed --id-prefix history_final` for the frozen final set
instead — already done, output committed under `data/histories/assembled/`):

```bash
python -m tool_context_study.build_histories
```

Build the frozen final condition grid (96 tasks × 4 history × 3 positions;
requires every row in `data/final/tasks.jsonl` to be `review_status:
"approved"`):

```bash
python -m tool_context_study.phase4_build_final_conditions
```

Re-check the interference episode bank for literal tool-name leakage or
answer-shaped replies (writes `data/histories/episodes/interference_flags.json`):

```bash
python -m tool_context_study.check_interference_episodes
```

Re-generate the Phase 4 freeze manifest (only if a frozen input file
legitimately changes — see the annotation-defect policy in
`docs/decision_log.md` first):

```bash
python -m tool_context_study.freeze_phase4
```

Serve and run a model against any condition file (`--api-format responses
--reasoning-effort low` are GPT-OSS-only; omit both for Qwen):

```bash
python -m tool_context_study.run_model \
  --conditions data/conditions/final_grid.jsonl \
  --model <served-model-name> \
  --output results/raw/<output-name>.jsonl

python -m tool_context_study.score \
  --results results/raw/<output-name>.jsonl \
  --conditions data/conditions/final_grid.jsonl
```

## Frozen design

The experimental design (32-tool catalog, positions 1/16/32, history
conditions H0/H16N/H64N/H64I, 96-task final set, models) is specified in the
research brief and must not be silently changed. Any implementation
constraint that forces a deviation should be documented in
`docs/decision_log.md` along with its scientific consequences before being
made.

## Repository layout

```
docs/research_brief.md          the experimental design spec (authoritative)
docs/decision_log.md            every non-obvious choice made, and why

configs/experiment.yaml         seeds, token targets, interference fraction, catalog shape
configs/catalog_spec.yaml       the 32-tool catalog's domain/group assignment (hand-curated)
configs/models/*.yaml           one per model: exact serving command + version pins

data/raw/bfcl/                  BFCL files (gitignored; reproducible via the manifest below)
data/manifests/bfcl_provenance.json   BFCL revision + file hashes
data/manifests/phase4_freeze.json     Phase 4 freeze: hashes of every frozen input + design params
data/catalog/tools.jsonl        the 32-tool catalog, normalized (NormalizedToolSchema)
data/development/tasks.jsonl    24-question dev set (Phase 3; freely revisable, not frozen)
data/final/tasks.jsonl          96 frozen final questions (FinalTaskRecord) -- do not edit
data/histories/episodes/        neutral (250) + interference (440, 8 groups) episode banks
data/histories/assembled/       assembled H64N/H16N/H64I histories --
                                 `history_*` = dev pilot, `history_final_*` = frozen final set
data/conditions/phase3_dev_grid.jsonl   288-condition dev-pilot grid
data/conditions/final_grid.jsonl        1,152-condition frozen final grid -- do not edit

src/tool_context_study/schemas.py             all project-owned record shapes
src/tool_context_study/build_histories.py     assembles H64N/H16N/H64I from episode banks
src/tool_context_study/build_conditions.py    position-controlled ConditionRecord builder
src/tool_context_study/phase4_build_final_conditions.py   builds the frozen 1,152-condition grid
src/tool_context_study/check_interference_episodes.py     automated episode safety checks
src/tool_context_study/freeze_phase4.py       writes the Phase 4 freeze manifest
src/tool_context_study/run_model.py           sends conditions to a running server, saves results
src/tool_context_study/score.py               primary tool-selection scoring
src/tool_context_study/parsers/               per-API-format response parsers + GPT-OSS token counting

.claude/agents/                 persona files used by subagents that generated project content
                                 (episode banks, final questions) -- kept for reproducibility

tests/                           unit tests + tiny fixtures (63 as of Phase 4)
results/raw/                    raw per-condition model responses (gitignored, regenerable)
results/processed/              scored/aggregated summaries (gitignored, regenerable)
analysis/                       notebooks (Phase 5, not yet used)
```

# Tool Context Study

Does an LLM's tool-selection accuracy depend on (1) the correct tool's
position among the available tool schemas, (2) how much conversation
history precedes the request, and (3) whether that history is semantically
interfering — and if so, can low-cost, training-free fixes recover the
lost accuracy? The project has two deliverables, not one: a controlled
characterization of the effect (Phase 5), and an evaluation of three
mitigations against an oracle and a retrieval-pruning reference point
(Phase 6). See `docs/research_brief.md` for the full experimental design
(sections 1-15 characterization, section 16 mitigations); this README
tracks implementation status and setup.

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

**GPT-OSS context-window bump (2026-09-28) — done.** Measured actual
harmony-encoded prompt sizes across the full frozen final grid (largest:
`final_0014__H64I__p1` at 67,319 tokens); bumped both
`configs/models/gpt_oss_{20b,120b}.yaml` to `--max-model-len 80000`
(reusing Qwen's already-proven value) and `--gpu-memory-utilization 0.92`.
Verified by relaunching both models with the new settings and running the
grid's single largest condition plus one H0/H16N/H64N condition each
through `run_model.py` — no inference errors or OOM on either model. See
`docs/decision_log.md`'s Phase 5 prep entry for full detail, including a
silent-failure gotcha worth knowing before manually launching a server on
this host: an env var assignment on its own line in a `bash -c` string
does not attach to the following command, so `CUDA_VISIBLE_DEVICES` can
silently fail to apply and vLLM will load onto the wrong GPU with no
warning.

**Phase 5 (full model runs, analysis) — orchestrator built and smoke-tested,
full run not yet started.** `src/tool_context_study/phase5_run_baseline.py`
runs the frozen 1,152-condition grid against all three models, sharded
across multiple idle GPUs per model (see the concurrency entry above for
why per-GPU concurrency alone isn't enough) — capped at 5 GPUs at once
(6 only for gpt-oss-120b's tensor-parallel pairing, never more; shared
host). Resumable: safe to interrupt and re-run, already-completed
conditions are never resubmitted, and `inference_error`s are retried
automatically (request-level retry with backoff first, then a full
replica-cycle retry for anything that still fails). One model failing
doesn't stop the others -- each phase's results are safely merged to disk
before the next model starts. Readiness is a real `GET /health` poll, not
a log-text guess. Verified end-to-end on tiny hand-built grids (sharding,
concurrent shard execution, merge, resume after a simulated crash,
per-model failure isolation, score) — see `docs/decision_log.md`'s
2026-09-28 entries, including two real bugs this caught before they
touched real data. The actual 9-hour, 3,456-generation run itself hasn't
been started yet. Progress can be checked anytime without a terminal via
`results/logs/phase5_status.txt` (current state) or `.jsonl` (full
timeline):

```bash
nohup python -m tool_context_study.phase5_run_baseline --max-gpus 5 > phase5.log 2>&1 &
disown
```

**Phase 6 (mitigations and reference points) — not yet started.** Newly
scoped in `docs/research_brief.md` section 16 (this section didn't exist
until the project proposal PDF was reviewed against the brief — see
`docs/decision_log.md`). Three mitigations (hierarchical routing,
permutation voting, history pruning) plus two reference points (oracle,
retrieval-based tool pruning), run on Qwen and one GPT-OSS model after
Phase 5's baseline completes — permutation voting specifically reuses
Phase 5's own generations rather than requiring new ones. Mitigation
settings get tuned on the development set and frozen before touching the
final set, same discipline as Phase 4.

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
--reasoning-effort low` are GPT-OSS-only; omit both for Qwen). `--concurrency`
(default 16) dispatches requests concurrently so vLLM can continuously
batch them, but on this single-GPU-per-model hardware it measurably helps
only for short-history conditions — see `docs/decision_log.md`'s
2026-09-28 entry: H64N/H64I are prefill-bound and already keep a single
L40S near-saturated, so concurrency barely moves their wall-clock (0% for
Qwen, -7% for gpt-oss-20b in a 24-condition benchmark):

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
made. Phase 4's freeze (`data/manifests/phase4_freeze.json`) covers the
baseline characterization's inputs only — Phase 6's mitigation settings are
a separate, later tuning pass on the development set, not yet frozen, and
must not be tuned on the final set once they are.

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
data/conditions/                gitignored: materialized ConditionRecord grids, fully
                                 regenerable from the frozen inputs above (too large for
                                 git -- final_grid.jsonl alone is ~230MB). Regenerate with
                                 build_conditions.py / phase4_build_final_conditions.py.

src/tool_context_study/schemas.py             all project-owned record shapes
src/tool_context_study/build_histories.py     assembles H64N/H16N/H64I from episode banks
src/tool_context_study/build_conditions.py    position-controlled ConditionRecord builder
src/tool_context_study/phase4_build_final_conditions.py   builds the frozen 1,152-condition grid
src/tool_context_study/check_interference_episodes.py     automated episode safety checks
src/tool_context_study/freeze_phase4.py       writes the Phase 4 freeze manifest
src/tool_context_study/run_model.py           sends conditions to a running server, saves results
src/tool_context_study/phase5_run_baseline.py Phase 5: shards the full grid across multiple idle
                                               GPUs per model (capped at 5, 6 max), resumable
src/tool_context_study/score.py               primary tool-selection scoring
src/tool_context_study/parsers/               per-API-format response parsers + GPT-OSS token counting
src/tool_context_study/mitigations/           Phase 6, not yet started: hierarchical routing,
                                               permutation voting, history pruning, retrieval-pruning baseline

.claude/agents/                 persona files used by subagents that generated project content
                                 (episode banks, final questions) -- kept for reproducibility

tests/                           unit tests + tiny fixtures (63 as of Phase 4)
results/raw/                    raw per-condition model responses (gitignored, regenerable)
results/processed/              scored/aggregated summaries (gitignored, regenerable)
analysis/                       notebooks (Phase 5/6, not yet used)
```

## Implementation attribution

Per the project proposal's deliverable 4: everything under `src/`,
`configs/`, `tests/`, the catalog curation, the episode banks, the 96 final
questions, and the mitigations (once built) were implemented for this
project. Reused as-is, not implemented here: BFCL's tool schemas and
development-set questions (source material, see `docs/research_brief.md`
section 4), vLLM/SGLang as the serving framework, and the model weights
themselves (Qwen3-30B-A3B-Instruct-2507, gpt-oss-20b, gpt-oss-120b). AI
(Claude Code) was used throughout as the implementation tool, including to
draft the interference/neutral episode banks and the 96 final questions
(both under human review and approval before use — see
`docs/decision_log.md` for the review process and persona files under
`.claude/agents/`); this section will be extended as Phase 5/6 analysis
code is added.

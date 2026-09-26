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
**Phase 3 (catalog curation + dev tasks) — in progress.** Curated and
user-approved the real 32-tool catalog (4 domains x 2 groups x 4 tools,
`configs/catalog_spec.yaml` -> `data/catalog/tools.jsonl`), and built the
24-question development set (`data/development/tasks.jsonl`): 20 questions
reused from BFCL + 4 freshly authored where BFCL had zero ground-truth
coverage for a catalog tool (see `docs/decision_log.md`). Generated the
episode banks: 250 neutral episodes and 440 interference episodes (55 per
target group x 8 groups) in `data/histories/episodes/`, sized against the
frozen H64N/H64I token targets rather than the brief's raw episode-count
guidance (see `docs/decision_log.md` for the sizing math and two corrected
generation-quality issues). **Not yet done for this data:** the brief's
required manual review pass over the interference episodes (only automated
checks have run so far) and the token-count top-up for `personal_finance`,
the thinnest-margin group. Remaining Phase 3 steps: assembling the four
history conditions from these banks, and running Qwen/smoke-testing
GPT-OSS across the dev set. Phases 4-5 (final 96-question set, full model
runs) are not yet implemented.

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

## Frozen design

The experimental design (32-tool catalog, positions 1/16/32, history
conditions H0/H16N/H64N/H64I, 96-task final set, models) is specified in the
research brief and must not be silently changed. Any implementation
constraint that forces a deviation should be documented in
`docs/decision_log.md` along with its scientific consequences before being
made.

## Repository layout

```
configs/            experiment + per-model configuration
data/raw/bfcl/       BFCL files downloaded by ingest_bfcl.py (gitignored, reproducible via manifest)
data/manifests/      provenance manifests (BFCL revision + file hashes)
data/catalog/        the 32-tool catalog (Phase 3)
data/development/    ~24-question development set (Phase 3)
data/final/          96-question frozen final set (Phase 4)
data/histories/       episode banks + assembled histories (Phase 3)
data/conditions/     materialized model-input records (Phase 4)
src/tool_context_study/   project source
tests/               unit tests + tiny fixtures
results/             raw + processed model outputs (Phase 5)
analysis/            notebooks (Phase 5)
```

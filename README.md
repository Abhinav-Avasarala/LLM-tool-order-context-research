# Tool Context Study

Does an LLM's tool-selection accuracy depend on (1) the correct tool's
position among the available tool schemas, (2) how much conversation
history precedes the request, and (3) whether that history is semantically
interfering? See `docs/research_brief.md` for the full experimental design;
this README tracks implementation status and setup.

## Status

**Phase 1 (BFCL ingestion) — done.** See `docs/decision_log.md` for choices
made along the way. Phases 2-5 (tool catalog, history assembly, final task
set, model runs) are not yet implemented.

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

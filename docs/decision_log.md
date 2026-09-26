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

## 2026-09-26 — Phase 2

**vLLM install: pinned to `vllm==0.10.0` against a CUDA 12.6 torch build,
not the plain `pip install vllm` default.** A first `pip install vllm`
resolved to `vllm==0.30.0` with `torch==2.13.0+cu130` by default, but the
host's driver (565.57.01) only supports up to CUDA 12.7 -- `torch.cuda.is_available()`
was `False` under that install. Reinstalled with
`pip install vllm==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu126`,
which resolves `vllm`'s pinned `torch==2.7.1` against the cu126 wheel index
instead of the newer cu130 default; confirmed working (`torch.cuda.is_available() == True`,
device `NVIDIA L40S`). `vllm==0.10.0` supports the `qwen3_moe` architecture
(added well before this release), so this doesn't limit which Qwen3
checkpoints can be served. See `configs/models/qwen3_30b_a3b_instruct_2507_fp8.yaml`.

**Pinned `transformers<5.0.0` alongside vLLM.** With `vllm==0.10.0`'s loose
`transformers>=4.53.2` pin, pip resolved `transformers==5.17.0` (a newer
major release), which removed/renamed `PreTrainedTokenizerBase.all_special_tokens_extended`
-- vLLM 0.10.0's tokenizer-caching code still calls it, so the server
crashed on startup with `AttributeError: Qwen2Tokenizer has no attribute
all_special_tokens_extended`. Constraining to `transformers<5.0.0,>=4.53.2`
(resolved to 4.57.6) fixed it. This is an upstream vLLM/transformers
compatibility gap, not a project decision, but pinning here avoids silently
re-breaking on a future `pip install`.

**Tool-call parser: `hermes`, not a Qwen-specific one.** vLLM 0.10.0's
available tool-call parsers include `qwen3_coder` but nothing named plainly
`qwen3`. Inspecting the downloaded model's own `tokenizer_config.json`
chat template shows it emits `<tool_call>{"name":...,"arguments":...}</tool_call>`
-- the same tag format Hermes-style models use -- so `--tool-call-parser
hermes` is the correct choice for this (non-coder) Qwen3 Instruct model;
`qwen3_coder`'s parser expects a different format specific to Qwen3-Coder
models.

**Served on a single L40S (GPU index 1), FP8 checkpoint
(`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, the official Qwen-published FP8
release, ~31GB), `--max-model-len 8192`.** Matches the brief's "test one
L40S first" guidance for the dev model; 8192 tokens is far more than Phase
2's `H0` (no-history) conditions need (32 short tool schemas + one short
question), revisit when H16N/H64N/H64I histories are implemented.

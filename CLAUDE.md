# tool_context_study

Research project (first academic paper) studying whether an LLM's tool-selection
accuracy depends on (1) the correct tool's position among 32 available tool
schemas, (2) conversation history length (0/16K/64K tokens), and (3) semantic
interference in that history — **and evaluating low-cost mitigations for
whatever effect is found.** The project is not just a characterization study;
"and Practical Fixes" is in the actual project title. Full spec:
`docs/research_brief.md` (sections 1-15 baseline characterization, section 16
mitigations). Choices made along the way, and why: `docs/decision_log.md`.
Current implementation status: `README.md`.

## Ground rules

- **Do not silently change the experimental design** in `docs/research_brief.md`
  (32-tool catalog, positions 1/16/32, history conditions H0/H16N/H64N/H64I,
  96-task final set, models, section 16's mitigations/reference points). If an
  implementation constraint forces a deviation, document the proposed change
  and its scientific consequences in `docs/decision_log.md` before making it —
  don't just make the call silently.
- **Development set vs. final set are not interchangeable.** The ~24-question
  BFCL-derived dev set (and the Phase 2 provisional catalog) can be freely
  rebuilt/iterated on. The 96-question final set, once created, is frozen —
  don't rewrite final questions after seeing model failures; log genuine
  annotation defects and apply a predefined correction policy instead. **This
  applies to mitigation settings too**: hierarchical routing, permutation
  voting, and history pruning must be tuned on the development set only, never
  on the final set, even though they're evaluated later (Phase 6).
- **Baseline (Phase 5) before mitigations (Phase 6).** Mitigation recovery is
  reported against the baseline's own results, and permutation voting reuses
  the baseline's generations directly — it isn't a separate model run.
- Log every finalized non-obvious choice (data source pinning, package version
  pins, tool-call parser choice, etc.) in `docs/decision_log.md`, not just in
  code comments or chat.

## Environment

Two separate conda envs -- do not mix them, and do not `pip install`
gpt-oss's vllm version into `tool_ctx` or vice versa.

```bash
source /home/aavasar/miniconda3/etc/profile.d/conda.sh
conda activate tool_ctx   # Python 3.11, conda-forge channel (avoids anaconda.org ToS prompt)
cd /home/aavasar/projects/tool_context_study
pip install -e ".[dev]"   # if not already installed
pytest                    # 63 tests as of Phase 4, tiny fixtures only
```

`tool_ctx` (above) is the client/dev env: the project package, Qwen serving,
`run_model.py`, `score.py`, everything else. It also has `openai-harmony`
installed (lightweight, no CUDA/torch dependency) purely so
`parsers/harmony_token_count.py` can compute accurate GPT-OSS token counts
client-side without needing vLLM itself.

Qwen serving needs care on this host: the driver caps out at CUDA 12.7, so
a plain `pip install vllm` (which defaults to a CUDA 13 torch build) leaves
`torch.cuda.is_available() == False`. Use:

```bash
pip install "vllm==0.10.0" --extra-index-url https://download.pytorch.org/whl/cu126
pip install "transformers<5.0.0,>=4.53.2"   # vllm 0.10.0's tokenizer cache breaks on transformers 5.x
```

See `configs/models/qwen3_30b_a3b_instruct_2507_fp8.yaml` for the exact
serving command (model, tool-call parser, GPU placement) this produced.

`tool_ctx_gptoss` is a **separate** env, Python 3.12, for serving GPT-OSS
only (`vllm==0.10.0` predates gpt-oss support entirely):

```bash
conda create -n tool_ctx_gptoss -c conda-forge --override-channels python=3.12
conda activate tool_ctx_gptoss
pip install "vllm==0.10.1.1" --extra-index-url https://download.pytorch.org/whl/cu126
pip install "transformers<5.0.0,>=4.55.0"
```

GPT-OSS also needs its own non-obvious serving choices -- see
`configs/models/gpt_oss_20b.yaml` / `gpt_oss_120b.yaml` for exact launch
commands, and `docs/decision_log.md`'s "Phase 3 (GPT-OSS smoke test)" entry
for why: no `--tool-call-parser` flag exists or is needed for it; run
`run_model.py` with `--api-format responses --reasoning-effort low`
(`/v1/chat/completions` crashes on a real tool call under this vLLM
version); `gpt-oss-120b`'s 2-GPU tensor-parallel startup needs
`--enforce-eager --disable-custom-all-reduce` plus `NCCL_P2P_DISABLE=1` or
it hangs indefinitely on this no-NVLink host.

Shared 8x L40S machine, no NVLink — pin `CUDA_VISIBLE_DEVICES` to the idle
GPU(s) rather than defaulting to all of them, and shut servers down when not
actively in use.

## Key fixed parameters (see brief for full detail)

- 32-tool catalog: 4 domains x 8 tools, each domain split into 2 groups of 4
  related tools. Sourced from BFCL, normalized to OpenAI-style function
  schemas (`src/tool_context_study/normalize_schema.py`).
- Gold tool positions tested: 1, 16, 32 — via seeded, recorded permutations
  (`src/tool_context_study/build_conditions.py`), never one fixed ordering
  (that would confound tool identity with position).
- History conditions: H0 (none), H16N (~16K neutral), H64N (~64K neutral,
  H16N is its most-recent suffix), H64I (~64K with 25% of episodes replaced
  by semantic-interference episodes, matched on length/turns).
- Final scale: 96 tasks x 4 history conditions x 3 positions = 1,152
  generations per model x 3 models = 3,456 generations.
- Models: Qwen/Qwen3-30B-A3B-Instruct-2507 (dev + final), openai/gpt-oss-20b,
  openai/gpt-oss-120b (final only, both sizes -- not a choice between them).
  vLLM or SGLang only, no LangGraph.
- Primary metric: exact correct-tool-name selection (exactly one tool call,
  name matches gold), implemented in `score.py`. Secondary (argument
  correctness) is deliberately not implemented -- user's explicit call, not
  a gap.
- Mitigations (Phase 6, brief section 16): hierarchical routing (domain then
  tool), permutation voting (K=3, reuses baseline p1/p16/p32 generations),
  history pruning (embedding retrieval, ~4K token budget, H64N/H64I only).
  Reference points: oracle (= the baseline's own p1 results, no new
  generations) and a retrieval-based tool-pruning baseline (top 8 tools by
  embedding similarity). Runs on Qwen + one GPT-OSS model, not all three --
  which one is still open.

**As of Phase 4, all of this is frozen.** The catalog, the 96 final
questions + gold labels, both interference and neutral episode banks, all
final-set histories, the full 1,152-condition grid, model configs, and
scoring/parsing code are hashed in `data/manifests/phase4_freeze.json`. Do
not edit `data/final/tasks.jsonl`, `data/conditions/final_grid.jsonl`, or
any `data/histories/assembled/history_final_*.json` file -- if a genuine
defect turns up, follow the annotation-defect policy in
`docs/decision_log.md`'s Phase 4 entry (exclude and log, never edit) rather
than changing a frozen file directly.

## Key files (start here in a fresh session)

- `docs/research_brief.md` — the full experimental design spec. Authoritative.
- `docs/decision_log.md` — every non-obvious choice made and why, in
  chronological order. Read this before assuming something is a bug rather
  than a documented decision.
- `README.md` — current phase-by-phase implementation status.
- `data/final/tasks.jsonl` — the 96 frozen final questions (`FinalTaskRecord`).
- `data/conditions/final_grid.jsonl` — the 1,152 frozen model-input records
  (`ConditionRecord`) Phase 5 runs over, built by
  `phase4_build_final_conditions.py`.
- `data/manifests/phase4_freeze.json` — the freeze manifest (hashes +
  frozen design parameters), written by `freeze_phase4.py`.
- `configs/models/*.yaml` — one per model: exact serving command, version
  pins, and any model-specific quirks (see decision log for why each exists).
- `src/tool_context_study/run_model.py` — sends conditions to a running
  server, saves raw + parsed results. `--api-format {chat,responses}` and
  `--reasoning-effort` are GPT-OSS-only knobs.
- `src/tool_context_study/score.py` — primary tool-selection scoring.
- `src/tool_context_study/mitigations/` — Phase 6 (not yet started): hierarchical
  routing, permutation voting, history pruning, retrieval-pruning baseline.

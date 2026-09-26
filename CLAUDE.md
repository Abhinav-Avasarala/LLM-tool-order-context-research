# tool_context_study

Research project (first academic paper) studying whether an LLM's tool-selection
accuracy depends on (1) the correct tool's position among 32 available tool
schemas, (2) conversation history length (0/16K/64K tokens), and (3) semantic
interference in that history. Full spec: `docs/research_brief.md`. Choices
made along the way, and why: `docs/decision_log.md`. Current implementation
status: `README.md`.

## Ground rules

- **Do not silently change the experimental design** in `docs/research_brief.md`
  (32-tool catalog, positions 1/16/32, history conditions H0/H16N/H64N/H64I,
  96-task final set, models). If an implementation constraint forces a
  deviation, document the proposed change and its scientific consequences in
  `docs/decision_log.md` before making it — don't just make the call silently.
- **Development set vs. final set are not interchangeable.** The ~24-question
  BFCL-derived dev set (and the Phase 2 provisional catalog) can be freely
  rebuilt/iterated on. The 96-question final set, once created, is frozen —
  don't rewrite final questions after seeing model failures; log genuine
  annotation defects and apply a predefined correction policy instead.
- Log every finalized non-obvious choice (data source pinning, package version
  pins, tool-call parser choice, etc.) in `docs/decision_log.md`, not just in
  code comments or chat.

## Environment

```bash
source /home/aavasar/miniconda3/etc/profile.d/conda.sh
conda activate tool_ctx   # Python 3.11, conda-forge channel (avoids anaconda.org ToS prompt)
cd /home/aavasar/projects/tool_context_study
pip install -e ".[dev]"   # if not already installed
pytest                    # 40 tests as of Phase 2, tiny fixtures only
```

Model serving (vLLM) needs care on this host: the driver caps out at CUDA
12.7, so a plain `pip install vllm` (which defaults to a CUDA 13 torch build)
leaves `torch.cuda.is_available() == False`. Use:

```bash
pip install "vllm==0.10.0" --extra-index-url https://download.pytorch.org/whl/cu126
pip install "transformers<5.0.0,>=4.53.2"   # vllm 0.10.0's tokenizer cache breaks on transformers 5.x
```

See `configs/models/qwen3_30b_a3b_instruct_2507_fp8.yaml` for the exact
serving command (model, tool-call parser, GPU placement) this produced.
Shared 8x L40S machine, no NVLink — pin `CUDA_VISIBLE_DEVICES` to one idle
GPU rather than defaulting to all of them, and shut servers down when not
actively in use.

## Key fixed parameters (see brief for full detail)

- 32-tool catalog: 4 domains x 8 tools, each domain split into 2 groups of 4
  related tools. Sourced from BFCL, normalized to OpenAI-style function
  schemas (`src/tool_context_study/normalize_schema.py`).
- Gold tool positions tested: 1, 16, 32 — via seeded, recorded permutations
  (`src/tool_context_study/build_conditions.py`), never one fixed ordering
  (that would confound tool identity with position).
- History conditions: H0 (none), H16N (~16K neutral), H64N (~64K neutral,
  H16N is its most-recent suffix), H64I (~64K with ~25% of episodes replaced
  by semantic-interference episodes, matched on length/turns).
- Final scale: 96 tasks x 4 history conditions x 3 positions = 1,152
  generations per model x 3 models = 3,456 generations.
- Models: Qwen/Qwen3-30B-A3B-Instruct-2507 (dev + final), openai/gpt-oss-20b,
  openai/gpt-oss-120b (final only). vLLM or SGLang only, no LangGraph.
- Primary metric: exact correct-tool-name selection (exactly one tool call,
  name matches gold). Secondary: argument correctness, error taxonomy.

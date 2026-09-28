# Research Implementation Brief: Tool Order × Conversation Context

## Purpose of this document

This document is the complete starting context for a coding agent such as Claude Code or Codex. The project studies whether an LLM's ability to select the correct tool changes because of:

1. The correct tool's position in the tool list.
2. The length of the preceding conversation.
3. Semantically interfering material in that conversation.

The immediate goal is to build a small, reproducible research pipeline suitable for a first academic paper. Favor a clean controlled experiment over a complicated agent framework.

Do not silently change the experimental design below. If an implementation constraint requires a change, document the proposed change and its scientific consequences before making it.

---

## 1. Research question

Primary question:

> When the same tools are given to an LLM, does the position of the correct tool affect tool-selection accuracy, and does this effect change as conversation history becomes longer or semantically interfering?

The model is presented with a conversation, one final user request, and 32 available function/tool schemas. It must choose the single correct tool. No tool is actually executed.

### Hypotheses

- **H1 — Position:** Correct-tool selection accuracy differs when the correct tool is at position 1, 16, or 32.
- **H2 — Length:** Accuracy decreases as neutral conversation history grows from 0 to approximately 16K and 64K tokens.
- **H3 — Interference:** At the same 64K history length, semantically interfering history causes lower accuracy than neutral history.
- **H4 — Interaction:** Tool-position effects may become larger under long or interfering history.

This is a first-step paper. Do not add multi-step agents, tool execution, memory systems, or graph orchestration. Retrieval is explicitly requested, but only as scoped in section 16 (two of the three mitigations, and the retrieval-pruning reference point) — not as a general-purpose addition to the baseline characterization in sections 1-15.

The project is not only a characterization study. Per the project proposal ("Tool Order and Conversation Context Effects on LLM Tool Selection, and Practical Fixes"), a second, equally real deliverable is evaluating low-cost, training-free mitigations and reporting how much accuracy they recover — see section 16. The baseline characterization (sections 1-15) must run first: the mitigation study evaluates recovery *against* the baseline's own results, and one mitigation (permutation voting) reuses the baseline's generations directly rather than requiring new ones.

---

## 2. Core experimental design

### Fixed tool catalog

Use the same fixed catalog of 32 tools for every final evaluation task.

The correct tool varies by question, but all 32 schemas are always available. For each question, generate three prompt variants in which the correct tool appears at:

- Position 1
- Position 16
- Position 32

Across these three variants:

- The user request must be identical.
- The conversation history must be identical.
- The set of 32 schemas must be identical.
- Schema names, descriptions, and parameters must be identical.
- Only the ordering of the schemas may change.

Do not keep one universal fixed ordering, because tool identity would then be confounded with position. Use deterministic, recorded permutations that place the task's gold tool at the required position and counterbalance the other tools.

### Recommended tool-catalog structure

Organize the 32 tools hierarchically:

- 4 broad domains
- 8 tools per domain
- Within each domain, 2 related groups of 4 tools

For a typical gold tool, this supplies:

- 3 highly similar tools from its four-tool group
- 4 additional tools from the same broad domain
- 24 tools from other broad domains

This structure supports error analysis without changing the catalog per task.

The final domains and schemas are not selected yet. Select them from BFCL wherever possible, normalize them, and manually check that each tool has a distinct purpose.

### Conversation-history conditions

Create four conditions:

| ID | Condition |
|---|---|
| `H0` | No preceding conversation history |
| `H16N` | Approximately 16K tokens of neutral history |
| `H64N` | Approximately 64K tokens of neutral history |
| `H64I` | Approximately 64K tokens containing semantic interference |

The final user request is appended after the history.

The 16K neutral history should be the most recent approximately 16K-token suffix of the corresponding 64K neutral history. Therefore, `H64N` adds older history while preserving the same recent history found in `H16N`.

Construct `H64I` from a matched neutral history by replacing approximately 25% of its conversation episodes with interfering episodes. Match total token count, number of messages, approximate message-length distribution, and turn structure.

### Experimental size

The final paper set should contain 96 tasks:

```text
32 gold tools × 3 fresh questions per tool = 96 final tasks
```

For each model:

```text
96 tasks × 4 history conditions × 3 gold positions = 1,152 generations
```

For three models:

```text
1,152 × 3 = 3,456 final generations
```

This excludes development runs and small reproducibility reruns.

---

## 3. Development set versus final paper set

These sets must remain separate.

### Development set

Use approximately 24 existing BFCL questions for development and pilot testing.

Development questions may be run repeatedly and inspected manually. Use them to:

- Debug BFCL ingestion and schema normalization.
- Verify that tools reach the model in the requested order.
- Debug model-specific tool-call parsing.
- Test whether 32 tools create reasonable difficulty.
- Test 16K and 64K memory requirements.
- Tune history assembly and validation.
- Finalize scoring and error categories.
- Detect ambiguous tasks or overlapping tool definitions.

Development-set results are not the main results of the paper. They are a practice/pilot benchmark because the experiment is deliberately changed in response to them.

### Final paper set

Create 96 fresh questions, with three questions for each of the 32 tools. These should not merely copy public BFCL questions.

AI may draft the questions, but every final question must be manually checked. Each must:

- Be self-contained.
- Require exactly one of the 32 tools.
- Not be validly answerable by a distractor tool.
- Not mention the literal function name.
- Contain all required arguments or information.
- Expect exactly one tool call.

Before any final model run, freeze:

- The 32 tool schemas.
- The 96 questions and gold labels.
- Gold arguments.
- All histories.
- All tool permutations and random seeds.
- Model revisions and inference settings.
- Parsing and scoring rules.

Generate a manifest containing file hashes. Do not rewrite individual final questions after examining model failures. If a genuine annotation error is discovered, log it transparently and apply a predefined correction policy.

---

## 4. What to reuse from BFCL

BFCL means the Berkeley Function-Calling Leaderboard.

Official sources:

- Repository: <https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard>
- Dataset: <https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard>

Pin the exact repository/data revision used and record its commit hash.

### Start with these files

```text
BFCL_v3_multiple.json
possible_answer/BFCL_v3_multiple.json
BFCL_v3_live_multiple.json
possible_answer/BFCL_v3_live_multiple.json
```

They are JSONL files: one JSON object per line.

### Reuse from BFCL

- Function/tool schemas
- Tool descriptions and parameter schemas
- Existing questions for the development set
- Gold function names for development tasks
- Accepted arguments for secondary argument evaluation
- Optional AST/argument-evaluation utilities

### Build in this project

- The fixed 32-tool catalog
- Fresh final questions
- Neutral conversation histories
- Interfering conversation histories
- Position-controlled permutations
- Custom condition records
- A thin inference runner
- Primary tool-selection scoring
- Error taxonomy and statistical analysis

### Do not initially use

- BFCL's multi-turn agent environment
- BFCL's long-context tasks as this project's conversation histories
- Parallel or multiple-tool-call categories
- Tool execution
- LangGraph or another agent graph
- BFCL's single aggregate score as the paper's main outcome

BFCL is seed material, not the entire experiment.

### Public-benchmark contamination concern

Public BFCL questions may have appeared in model training. Therefore:

- Use original BFCL questions mainly for development.
- Use BFCL schemas as scaffolding for the final benchmark.
- Write fresh final requests and argument values.
- Keep the within-task experimental comparison identical across conditions.

A June 2026 paper already studied a related correct-tool reordering intervention on BFCL with smaller tool lists. This project must emphasize systematic positions under controlled neutral and interfering long histories rather than claiming that correct-tool reordering alone is new:

- <https://arxiv.org/html/2606.16364>

---

## 5. Tool-schema normalization

Normalize every selected function to one canonical OpenAI-style schema:

```json
{
  "type": "function",
  "function": {
    "name": "country_info__capital",
    "description": "Find the capital city of a country.",
    "parameters": {
      "type": "object",
      "properties": {
        "country": {
          "type": "string",
          "description": "Country name."
        }
      },
      "required": ["country"]
    }
  }
}
```

Normalization rules:

- Convert nonstandard parameter root types such as `dict` to `object`.
- Ensure every function and parameter has a clear description.
- Convert function names to a conservative character set such as `[A-Za-z0-9_-]`.
- For example, normalize `country_info.capital` to `country_info__capital`.
- Preserve a mapping from original BFCL names to normalized names.
- Canonicalize JSON before hashing.
- Reject or fix invalid JSON Schema constructs consistently.
- Ensure all 32 normalized names are unique.
- Store provenance for every schema.

Represent each catalog item approximately as:

```json
{
  "schema_id": "sha256:...",
  "name": "country_info__capital",
  "original_name": "country_info.capital",
  "domain": "geography",
  "group": "country_facts",
  "source": "BFCL_v3_multiple",
  "source_revision": "...",
  "schema": {}
}
```

---

## 6. Conversation-history construction

AI can draft histories, but code must assemble and validate them. Do not generate one monolithic 64K conversation.

### Neutral episode bank

Generate approximately 200–300 short, coherent conversation episodes. Each episode should contain about 2–6 messages and roughly 300–800 tokens.

Neutral episodes must:

- Be unrelated to the current final task.
- Avoid literal names of all 32 tools.
- Avoid asking for actions that clearly require the 32 tools.
- Contain no instruction intended to persist into later turns.
- End with the request resolved.
- Vary in topic, style, and length.
- Not be repeated mechanically to reach the token target.

Example representation:

```json
{
  "episode_id": "neutral_0042",
  "kind": "neutral",
  "topic": "biology_explanation",
  "messages": [
    {"role": "user", "content": "Why do leaves change color in autumn?"},
    {"role": "assistant", "content": "Leaves change color because..."}
  ]
}
```

### Interfering episode bank

Generate episodes that semantically activate plausible distractor tools while remaining completed older conversations.

For example, if the gold tool is current weather, interfering episodes could discuss historical weather, forecasts, air quality, or weather alerts. They must not contain literal function names or explicit instructions to select a tool.

Every interfering episode should be manually reviewed because interference is a central treatment variable.

### Assembly

Use deterministic seeds. Build a 64K neutral history from episodes, then derive the recent 16K suffix. Build a matched 64K interfering history by replacing approximately 25% of the neutral episodes with relevant interfering episodes.

Count history tokens using each evaluated model's tokenizer and log:

- `history_tokens`
- `tool_schema_tokens`
- `final_request_tokens`
- `system_tokens`
- `total_input_tokens`

Use the same semantic text across models. Token counts will vary slightly because tokenizers differ. Prefer trimming at episode/message boundaries and record actual counts. Target approximately ±2% when practical.

Automated validation should check:

- Alternating valid roles
- No literal tool names in history
- No accidental occurrence of the final task
- No unresolved old request
- Token target tolerance
- Expected episode IDs and interference tags
- No duplicated blocks caused by assembly bugs

Use a generator model that is not one of the three evaluated models if possible. The generator's identity and prompt must be recorded.

---

## 7. Models and hardware

Available compute:

- 8 × NVIDIA L40S
- Approximately 46 GB VRAM per GPU
- Approximately 368 GB total VRAM
- No NVLink; multi-GPU communication uses PCIe
- Shared machine and storage, so avoid unnecessary model copies and delete unneeded duplicate weights/caches carefully

### Development model

Use:

```text
Qwen/Qwen3-30B-A3B-Instruct-2507
```

Start with an appropriate FP8 deployment on one L40S if it accommodates the chosen context and KV cache. Qwen reports native 262,144-token context and tool-use capability:

- <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>

### Final paper models

Run all frozen final conditions on:

1. `Qwen/Qwen3-30B-A3B-Instruct-2507`
2. `openai/gpt-oss-20b`
3. `openai/gpt-oss-120b`

GPT-OSS models use the Harmony response format and have native function-calling support. Use the official chat template/parser and keep the reasoning-effort setting fixed, initially `low`, for this direct selection task:

- <https://huggingface.co/openai/gpt-oss-20b>
- <https://huggingface.co/openai/gpt-oss-120b>

Hardware starting points:

- Qwen 30B FP8: test one L40S first.
- GPT-OSS 20B MXFP4: one L40S.
- GPT-OSS 120B MXFP4: start with two L40S; move to four only if 64K KV-cache/serving overhead causes OOM.

Because the system lacks NVLink, do not tensor-parallelize across more GPUs than necessary. During development, repeatedly use only Qwen. Smoke-test each GPT-OSS parser on a few development tasks before the final run.

Use vLLM or SGLang directly. Do not use LangGraph. Pin exact package versions after the pilot succeeds.

---

## 8. Inference protocol

The model receives:

1. A minimal fixed system message, if required.
2. The selected history condition.
3. The final user request.
4. The ordered list of 32 tool schemas through the model's native tool interface/chat template.

Use automatic tool selection rather than forcing a call. If the model does not call a tool, record that as `no_tool_call`.

Recommended starting settings:

- Greedy or deterministic decoding
- Temperature `0`
- Maximum output around 128–256 tokens
- Exactly the same settings across all conditions for a model
- GPT-OSS reasoning effort fixed to `low`, unless the development protocol establishes another setting before freezing

Do not execute the tool. Stop after capturing the first model response/tool-call decision.

For every condition, store:

- Model name and exact revision
- Serving framework and version
- Quantization
- Tensor-parallel size
- Tokenizer and chat-template revision
- Tool-call parser
- Decoding parameters
- Condition ID and seed
- Ordered tool names
- Rendered prompt hash
- Input-token counts
- Raw model response
- Parsed calls and arguments
- Latency and error information

Inspect rendered prompts using the official tokenizer/chat template. Verify that all 32 tools appear exactly once, their order is preserved, and history is not silently truncated.

---

## 9. Records and file formats

### Final task record

```json
{
  "task_id": "final_0042",
  "request": "I am preparing a quiz about Peru. Look up its capital city.",
  "gold_tool": "country_info__capital",
  "gold_arguments": {"country": "Peru"},
  "authoring_source": "fresh_from_schema",
  "review_status": "approved"
}
```

### Condition record

```json
{
  "condition_id": "final_0042__h64i__p32",
  "task_id": "final_0042",
  "history_condition": "H64I",
  "history_id": "history_final_0042_i",
  "gold_tool": "country_info__capital",
  "gold_position": 32,
  "permutation_seed": 18472,
  "messages": [],
  "tools": []
}
```

### Result record

```json
{
  "condition_id": "final_0042__h64i__p32",
  "model": "...",
  "model_revision": "...",
  "status": "ok",
  "selected_tool": "country_info__capital",
  "arguments": {"country": "Peru"},
  "raw_response": {},
  "history_tokens": 64211,
  "total_input_tokens": 68102,
  "latency_seconds": 3.14
}
```

---

## 10. Scoring

### Primary metric

The primary outcome is correct tool-selection percentage.

```python
selection_correct = (
    parsed_exactly_one_tool_call
    and selected_tool_name == gold_tool_name
)
```

If the task expects one tool and the model produces multiple calls, score tool selection as incorrect, even if one call is correct.

### Error taxonomy

Assign exactly one primary status:

- `correct`
- `wrong_high_similarity`
- `wrong_same_domain`
- `wrong_unrelated`
- `no_tool_call`
- `malformed_tool_call`
- `multiple_tool_calls`
- `inference_error`

### Secondary metrics

- Argument correctness, evaluated only after tool selection is separated out
- Parseable-call rate
- No-call rate
- Error category distribution
- Latency and input-token counts as descriptive operational statistics

A correct function with incorrect arguments is:

- Correct for primary tool selection
- Incorrect for secondary argument accuracy

---

## 11. Analysis plan

Report tool-selection accuracy for every:

```text
model × history condition × gold position
```

Key paired comparisons:

- Position 1 versus position 32
- `H0` versus `H16N`
- `H16N` versus `H64N`
- `H64N` versus `H64I`
- Whether position differences become larger under `H64I`

Use tasks as the unit of analysis. Report paired differences and bootstrap 95% confidence intervals over tasks.

A later inferential model may use mixed-effects logistic regression:

```text
correct ~ position + history_length + interference
        + position:interference + (1 | task)
```

Do not treat repeated generations of the same task as independent tasks. Initially run once at deterministic decoding, then rerun a random 10% sample to measure serving nondeterminism. Add full repetitions only if outputs materially vary.

---

## 12. Recommended repository structure

```text
tool_context_study/
├── README.md
├── pyproject.toml
├── configs/
│   ├── experiment.yaml
│   └── models/
├── data/
│   ├── raw/bfcl/
│   ├── catalog/tools.jsonl
│   ├── development/tasks.jsonl
│   ├── final/tasks.jsonl
│   ├── histories/episodes/
│   ├── histories/assembled/
│   ├── conditions/
│   └── manifests/
├── src/
│   ├── ingest_bfcl.py
│   ├── normalize_schema.py
│   ├── build_catalog.py
│   ├── validate_catalog.py
│   ├── build_histories.py
│   ├── validate_histories.py
│   ├── build_conditions.py
│   ├── validate_conditions.py
│   ├── run_model.py
│   ├── parsers/
│   ├── score.py
│   └── mitigations/       # Phase 6 (section 16): hierarchical routing,
│                          # permutation voting, history pruning, and the
│                          # retrieval-pruning reference point
├── tests/
├── results/
│   ├── raw/
│   └── processed/
└── analysis/
    └── main_analysis.ipynb
```

Use small unit tests for schema normalization, gold extraction, permutations, token counting, parsing, and scoring.

---

## 13. Implementation phases

### Phase 1 — Environment and BFCL ingestion

1. Initialize the repository and configuration system.
2. Pin BFCL revision and record provenance.
3. Download/read the selected JSONL and possible-answer files.
4. Normalize records into a project-owned schema.
5. Add automated validation and unit tests.

### Phase 2 — Five-task vertical slice

1. Select five unambiguous development questions.
2. Create a provisional 32-tool catalog.
3. Build position-1 and position-32 conditions with `H0`.
4. Serve Qwen and capture raw plus parsed outputs.
5. Inspect rendered prompts and prove tool order is preserved.
6. Score tool-name accuracy.

Do not generate large histories before this end-to-end slice works.

### Phase 3 — Full development pilot

1. Finalize the 32-tool candidate catalog.
2. Prepare 24 development tasks.
3. Generate and validate episode banks.
4. Assemble all four history conditions.
5. Run Qwen across development conditions.
6. Smoke-test both GPT-OSS models.
7. Finalize protocol decisions.

### Phase 4 — Final dataset and freeze

1. Draft 96 fresh questions, three per tool.
2. Manually review every question and gold label.
3. Assemble final histories and permutations.
4. Validate every condition.
5. Freeze configuration, manifests, and hashes.

### Phase 5 — Final runs and analysis

1. Run all 1,152 conditions on each of the three models.
2. Preserve raw outputs before parsing/scoring.
3. Rerun a seeded 10% subset for nondeterminism.
4. Produce processed results, confidence intervals, tables, and figures.
5. Keep a machine-readable record of exclusions and failures.

### Phase 6 — Mitigations and reference points (see section 16)

1. Tune each mitigation's settings on the development set only (never the
   frozen final set) and log the chosen settings before running any
   mitigation on final-set data.
2. Compute the oracle reference point (already implied by the position-1
   baseline results — see section 16 — no new generations needed).
3. Run the retrieval-based tool-pruning reference point.
4. Run all three mitigations (hierarchical routing, permutation voting,
   history pruning) on the frozen final set, on Qwen and one GPT-OSS model.
5. Report each mitigation's recovered accuracy and added compute cost
   against the Phase 5 baseline and the oracle.

---

## 14. First instructions for the coding agent

Begin with Phase 1 and the five-task vertical slice only.

Before writing substantial code:

1. Inspect the workspace and preserve existing user work.
2. Create a concise implementation plan.
3. Confirm current BFCL file structure and pin a revision.
4. Define project-owned Pydantic/dataclass schemas for tools, tasks, conditions, and results.
5. Implement JSONL readers and validators.
6. Add tests using tiny checked-in fixtures rather than full datasets.

The first milestone is complete when one command can:

1. Load five development tasks.
2. Build valid 32-tool position variants.
3. Send them to Qwen with no history.
4. Save raw responses.
5. Parse the selected tool.
6. Produce a small accuracy report.

Do not begin the 96-task final set or large-scale history generation until this milestone works reliably.

---

## 15. Decisions that are fixed versus still open

### Fixed for now

- Study tool order, history length, and semantic interference.
- Use one 32-tool catalog across all final tasks.
- Use positions 1, 16, and 32.
- Use `H0`, `H16N`, `H64N`, and `H64I`.
- Use approximately 24 development questions.
- Use 96 fresh final questions, three per tool.
- Use Qwen for development.
- Evaluate Qwen 30B, GPT-OSS 20B, and GPT-OSS 120B.
- Use correct tool-name percentage as the primary metric.
- Do not execute tools.
- Do not use LangGraph.
- Evaluate three mitigations (section 16): hierarchical routing, permutation
  voting (K=3), history pruning via embedding retrieval (~4K token budget).
- Compare mitigations against two reference points: an oracle (correct tool
  always at position 1 — already available from the baseline's own p1
  results, no separate run needed) and a retrieval-based tool-pruning
  baseline (top 8 tools by embedding similarity).
- Mitigations run on Qwen and *one* GPT-OSS model, not both — which one is
  still open (see below).
- Mitigation settings must be chosen on the development set and frozen
  before any mitigation touches the final set — the same "no tuning on the
  test set" discipline section 3 already requires for the baseline.
- The project README must mark which pipeline components were implemented
  for this project (catalog construction, history assembly/validation,
  condition builder, runner, scoring, mitigations, analysis) as a deliverable,
  distinct from reused external material (BFCL schemas/questions, vLLM,
  model weights).

### Still open and must be decided during development

- The exact 32 BFCL-derived tools and four domains.
- The exact history-generator model and prompting template.
- The final percentage/content density of interference, with 25% as the starting value.
- Exact vLLM or SGLang versions.
- Exact model revisions and quantization artifacts.
- Whether Qwen requires more than one L40S at the chosen 64K serving configuration.
- The predefined policy for rare final-set annotation defects.
- Which single GPT-OSS model (20b or 120b) the mitigation study uses.
- The embedding model for history pruning and the retrieval-pruning
  reference point (must not be one of the three evaluated models).
- The exact subset of history conditions x positions each mitigation is
  evaluated over on the final set (the proposal gives "a few thousand
  generations per model" as a scale target, not an exact cross-product).
- Whether permutation voting's "fresh random orderings to confirm" run
  reuses the frozen final set's existing permutation-seed scheme or needs
  its own seeds.

Record each finalized choice in a decision log rather than leaving it implicit in code.

---

## 16. Mitigations and reference points

Per the project proposal, characterizing the problem is half the
deliverable — the other half is evaluating low-cost, training-free fixes
and reporting how much accuracy they recover, and comparing them against
two reference points. This section is scoped as **Phase 6**, strictly
after the Phase 5 baseline characterization (section 13) — see the note
at the end of section 1 for why the ordering matters.

### Mitigations

All three are chosen and tuned on the **development set only**, then
frozen and logged (matching section 3's dev/final discipline) before
running on the final set.

1. **Hierarchical routing.** The model first picks one of the 4 domains,
   then picks among that domain's 8 tools. This shrinks the candidate set
   instead of reordering it — a different kind of intervention from the
   reordering studied in prior work (Chen 2026) — and uses structure
   already built into the catalog (brief section 2's domain/group
   hierarchy).
2. **Permutation voting.** Present the same 32 tools in K=3 different
   orderings and take the majority tool. This targets position sensitivity
   directly. The baseline's own p1/p16/p32 conditions (same task, same
   history condition) already are three orderings per task, so the
   baseline runs can be reused for the vote directly — no new generations
   required for this part. Also run fresh random orderings to confirm the
   result holds beyond the specific p1/p16/p32 permutations already used.
3. **History pruning.** For `H64N` and `H64I` only, an embedding retriever
   keeps only the episodes most relevant to the final request (about 4K
   tokens) before the model call, replacing the full history. This should
   remove much of `H64I`'s distractor material specifically. Report
   honestly if recovery is only partial — this is a real empirical
   question, not an assumed win.

Because Chen (2026) found that prompt-side repairs recover little of
BFCL's tool-selection failures on smaller tool lists without long
histories, treat that as a hypothesis to test under this project's long
and interfering histories, not as a settled result these mitigations must
match or beat.

### Reference points

- **Oracle.** The correct tool always placed first. This is *already*
  what the baseline's position-1 conditions measure — report it as the
  oracle reference rather than running anything new.
- **Retrieval-based tool-pruning baseline.** Keep only the top 8 tools by
  embedding similarity to the request (not the full 32), then run
  inference over that reduced set. This is a *different* axis from the
  history-pruning mitigation above (that one prunes history episodes; this
  one prunes the tool list itself) and exists to show how much a
  retriever's own errors cost when used as a naive tool-selection shortcut,
  not to be evaluated as one of the three mitigations proper.

### Scope and scale

The mitigation study runs on Qwen and *one* GPT-OSS model (not all three),
adding on the order of a few thousand generations per model on top of the
3,456-generation baseline (permutation voting's primary evaluation reuses
baseline generations and adds comparatively little; hierarchical routing
and history pruning need fresh generations across some subset of the final
set's history conditions x positions).

### Reporting

Report each mitigation's recovered accuracy (relative to the Phase 5
baseline, for the matching model/history condition/position cells) and
its added compute cost, alongside the oracle and retrieval-pruning
reference points. If a mitigation only partially recovers accuracy, or a
history-condition/position combination shows no meaningful baseline effect
at all, report that plainly — per the proposal, a documented null result
for the mitigation comparison is an acceptable, honest outcome, not a
failure to fix before writing up.

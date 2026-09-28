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

## 2026-09-26 — Phase 3 (catalog + dev tasks)

**32-tool catalog (`configs/catalog_spec.yaml`): curated by hand from BFCL's
~891 unique normalizable tools, deliberately dropping literal near-duplicates.**
BFCL contains many tools that do the same thing under different names --
e.g. `weather.get` / `weather.get_weather` / `weather.get_weather_data` are
all "current temperature/wind/precipitation for a place," and `math.gcd` /
`math.hcf` are literally the same operation (GCD = HCF). Including both
members of such a pair in the catalog would violate the brief's "manually
check that each tool has a distinct purpose," so exactly one of each was
kept. Final structure: 4 domains (Physics Calculators, Geography & Weather,
Finance & Economics, Property & Environment) x 2 groups x 4 tools. One
deliberately loose fit: `car_rental` sits in the "Property & Rentals" group
(alongside real-estate tools) for lack of a fourth genuine real-estate tool
in BFCL's `multiple`/`live_multiple` categories -- grouped as "things you
rent or value" rather than strictly real estate. User reviewed and approved
this catalog before it was written to `data/catalog/tools.jsonl`.

**24 dev tasks are a 20+4 mix, not 24 uniform BFCL reuses, because 14 of
the 32 catalog tools never appear as the sole correct answer anywhere in
BFCL_v3_multiple.json / BFCL_v3_live_multiple.json** (verified by direct
grep of the raw files, not just parsing logic) -- they only ever appear as
*distractor* options in questions whose gold is a different tool. Genuine
reuse tops out at 20 questions covering 18 tools (`weather.get` alone
contributes 3 of the 20, since it had 3 usable candidates). Given the
choice to accept 20 or pad to ~24, the user chose to pad: 4 more tasks
(`electric_field.calculate`, `country_info.population`,
`finance.loan_repayment`, `windFarm.potential` -- one per domain, chosen to
maximize domain/group coverage) were freshly authored against those tools'
real parameter schemas. These are tagged `authoring_source: dev_only_authored`
and are explicitly **not** held to the final-96 authoring bar (no independent
review pass) and must never be reused as final-set questions. Net dev-set
coverage: 22 of 32 catalog tools have at least one dev question; the other
10 (`kinematics.calculate_final_speed`, `kinematics.calculate_acceleration`,
`resistance_calculator.calculate`, `country_info.largest_city`,
`weather.get_forecast_by_coordinates`, `corporate_finance.product_price`,
`finance.inflation_adjustment`, `property_valuation.get`, `car_rental`,
`ecological_impact.analyze`) have none in the dev set -- acceptable since
dev-set purpose is pipeline/pilot debugging, not full coverage (full
coverage is the final 96-question set's job, 3 fresh questions per tool).

## 2026-09-26 — Phase 3 (episode banks)

**Generator identity (brief section 6: "the generator's identity and prompt
must be recorded"): Claude Sonnet 5 (`claude-sonnet-5`), via `general-purpose`
subagents with no file-reading or other tool access, given only the
persona/instructions from `.claude/agents/episode-writer-neutral.md` and
`.claude/agents/episode-writer-interference.md` inlined into each prompt,
not via those agent types directly.** Not one of the three evaluated models
(Qwen3-30B-A3B-Instruct-2507, gpt-oss-20b, gpt-oss-120b), satisfying the
brief's "generator model that is not one of the three evaluated models"
preference. This session's harness did not have those project-local custom
subagent types registered as invocable (`Agent` calls to
`episode-writer-neutral` / `episode-writer-interference` errored with
"agent type not found"); functionally equivalent since the agent
definitions are just system-prompt files, but noted here in case a future
session finds those custom types *do* resolve and wants to use them
directly instead. No tool access means each subagent call was a pure
prompt-in/text-out generation with no ability to read other project files
(catalog, dev tasks, history assembly code) or otherwise leak information
beyond what was inlined into its prompt.

**Bank sizes are derived from the frozen H64N/H64I token targets, not
directly from the brief's "~200-300 episodes" neutral-bank guidance.**
Given `history_token_targets.H64N: 64000` and `interference_fraction: 0.25`
(`configs/experiment.yaml`), assembling one non-mechanically-repeated H64N
history requires the neutral pool to hold at least 64K tokens of content,
and assembling H64I for a given target group requires that group's
interference pool to hold at least 16K tokens (25% of 64K). A first
generation pass (200 neutral + 240 interference, 30/group) undershot this
badly: subagents wrote much shorter episodes than instructed (the "220-650
words" instruction was treated as a soft ceiling rather than a target
range -- actual average landed near 195 words instead of ~435), leaving
the entire 200-episode neutral pool at only ~53K tokens (short of the 64K
needed for even one history) and every interference group short of its
16K minimum. A corrective second wave (50 more neutral episodes,
25 more per interference group) with an explicit hard length floor ("every
episode >= 400 words, land in 450-650, use 4-6 messages") brought the pool
to 250 neutral episodes (~90.6K tokens, 1.42x the 64K minimum) and 55
episodes per interference group (~20.5K-34.9K tokens per group, 1.28x-2.18x
the 16K minimum). Written to `data/histories/episodes/neutral.jsonl` (250
lines) and `data/histories/episodes/interference_<group>.jsonl` (one file
per of the 8 groups, 55 lines each, 440 total). Token counts here are a
`words * 1.35` estimate for planning purposes only -- `build_conditions.py`
must still count actual tokens with each evaluated model's own tokenizer
per the brief, and if real ratios come in lower than estimated, top up the
thinnest group (`personal_finance`, currently the smallest margin at
1.28x) before relying on it for assembly.

**Two subagent JSON-formatting defects were repaired programmatically
rather than by discarding/regenerating the batch:** one neutral-episode
batch (`neutral_0201`-`0250`) used unescaped literal `"` characters for
emphasis inside message text, breaking JSON string termination; repaired
with a small custom parser that treats a `"` as content (and escapes it)
unless it's immediately followed by `,`/`}`/`]`/`:` (i.e. looks like a real
string terminator) -- all 50 records recovered and re-validated as
structurally correct JSON afterward. One `corporate_finance` interference
batch had two lines with a raw literal newline inside a string value;
recovered by re-parsing with `json.loads(..., strict=False)`. Both fixes
were mechanical (whitespace/escaping only) and did not alter any actual
episode content. Separately, 4 episodes across 3 groups
(`corporate_finance`, `electromagnetism`, `environmental_impact`) had a
`topic` slug collide with a same-group episode from the other generation
wave (independent agent calls, so cross-wave topic uniqueness wasn't
enforced) -- confirmed the underlying conversations were genuinely
different scenarios, not duplicated content, and just renamed the
second occurrence's slug with a `_v2` suffix.

**Interference-bank content has only been automated-checked, not
manually reviewed, despite the brief's explicit "every interfering episode
should be manually reviewed" requirement.** Ran two automated passes across
all 440 interference episodes: (1) a literal-substring search for all 32
tools' `original_name` fragments from `configs/catalog_spec.yaml` (0 hits),
and (2) a regex heuristic for tool-invocation-shaped assistant answers
(patterns like "the current weather is...", "the distance is... miles",
"the capital of X is...", "your monthly payment would be $...") (0 hits).
Neither pass is a substitute for the brief's required human review pass
-- a human still needs to read through (at least a sample of) the 440
interference episodes before they're treated as final, specifically
checking for subtler cases the regex heuristics can't catch (e.g. an
episode that gets uncomfortably close to actually answering the
tool's question in prose without using a flagged phrase).


## 2026-09-26 — Phase 3 (dev pilot run)

**Ran the full 288-condition dev grid (24 dev tasks x H0/H16N/H64N/H64I x positions 1/16/32) on Qwen3-30B-A3B-Instruct-2507-FP8.**
Required relaunching the vLLM server with a much larger context window than
Phase 2's H0-only setup: measured actual prompt sizes (system + 32-tool
schemas + history + request) via the real tokenizer/chat template and found
H64I conditions reach ~72K tokens. Relaunched with `--max-model-len 80000`
and `--gpu-memory-utilization 0.92` (up from Phase 2's 8192/0.85), giving a
122,576-token KV cache -- comfortably above the largest observed prompt.

All 288 conditions completed with no inference errors. This is a pilot on
the *development* set (not the frozen final 96), with only 24 tasks per
cell -- not a result to draw conclusions from -- but the pattern is worth
noting because it's directionally consistent with the brief's own
hypotheses: overall accuracy declined as history grew (H0 0.958 -> H16N
0.944 -> H64N/H64I 0.861 each), and under H64I specifically, accuracy fell
further as gold-tool position moved later (p1 0.917 -> p16 0.875 -> p32
0.792), a steeper decline than the other three history conditions show.
Full breakdown: `results/processed/phase3_dev_grid_qwen3_30b_a3b_fp8_summary.json`,
raw per-condition responses: `results/raw/phase3_dev_grid_qwen3_30b_a3b_fp8.jsonl`.

**score.py extended with `by_history_condition` and
`by_history_condition_and_position` breakdowns** (previously only reported
by position) -- this is the actual cross-tab the brief's analysis plan
(section 11) needs, not just a Phase 2 vertical-slice convenience.

## 2026-09-27 — Phase 3 (GPT-OSS smoke test)

**Interference-episode manual review remains an open gap, deliberately not
attempted this session.** The brief's requirement that "every interfering
episode should be manually reviewed" (section 6) is still only
automated-checked (see the Phase 3 episode-banks entry above). Given a
choice this session between doing an LLM-assisted close-reading pass,
deferring to the user, or leaving it as an open item, the user chose to
defer it entirely and spend this session's time on the GPT-OSS smoke test
instead. This must be closed out before Phase 3 can be considered done, and
before any Phase 4/5 work that would make interference-history content hard
to revise.

**GPT-OSS served from a separate conda env (`tool_ctx_gptoss`, Python 3.12),
not `tool_ctx`.** `tool_ctx`'s pinned `vllm==0.10.0` predates gpt-oss support
entirely (`GptOssForCausalLM` is absent from its `ModelRegistry`). Rather
than touch `tool_ctx`'s already-validated Qwen pins, a fresh env keeps the
two completely isolated.

**vLLM pinned to `0.10.1.1`, not the PyPI default (`0.30.0`) or the plain
`0.10.1`.** `pip install vllm` (no pin) resolves `0.30.0`, whose prebuilt
CUDA extension requires `libcudart.so.13` (CUDA 13 runtime) regardless of
which torch wheel pip resolves for it -- this host's driver (565.57.01) caps
out at CUDA 12.7 (same constraint recorded in the Phase 2 Qwen entry above),
so `0.30.0` cannot run here at all (`ImportError: libcudart.so.13`).
Checked each vLLM release's declared torch pin directly from wheel metadata
(`pip download --no-deps`) rather than guessing: `0.10.1` is the first
release with gpt-oss support (confirmed via
`ModelRegistry.get_supported_archs()`) that still pins `torch==2.7.1`, the
same version already validated against this driver via the `cu126` wheel
index for Qwen. `0.10.1.1` (a same-torch-pin patch release) was used instead
of bare `0.10.1` on general principle (later patch, same pin), not because
`0.10.1` itself was shown to be worse.

**GPT-OSS models need no `--tool-call-parser` / `--enable-auto-tool-choice`
at all.** vLLM auto-detects Harmony models via
`model_config.hf_config.model_type == "gpt_oss"`
(`vllm/entrypoints/openai/serving_chat.py`) and handles encode/decode
(including tool calls) internally; passing `tool_choice: "auto"` without
those flags, which would error for a non-harmony model, is explicitly
exempted for harmony models in that same file. Confirmed by server log:
`"For gpt-oss, we ignore --enable-auto-tool-choice and always enable tool
use."`

**`/v1/chat/completions` is unusable for GPT-OSS tool-calling on this vLLM
version -- switched to `/v1/responses` instead, which requires its own
request/response adapter.** Every request where gpt-oss-20b actually
attempted a real tool call over `/v1/chat/completions` crashed with an
empty-body HTTP 500 (no traceback in any log, at any logging level tried).
Reproduced deterministically (not the ~7% intermittent rate reported
upstream) across: the full 32-tool catalog and a reduced 3-clean-tool
payload (ruling out a bug in our own catalog schemas); with `reasoning_effort`
unset, `"low"`, `"medium"`, and `"high"`. The crash only occurs when the
model actually emits a tool call -- requests where it answered directly
(e.g. general-knowledge questions the model didn't need a tool for)
returned normally, which is why Phase 2-style smoke testing with "easy"
dev tasks could easily miss this. Matches open, unresolved upstream issues
(vllm-project/vllm#23837, #22578); #22578 specifically confirms
`/v1/responses` as the reliable alternative for gpt-oss tool-calling.
**Likely root cause, found while building the token-counting fix below:**
`vllm/entrypoints/openai/serving_chat.py`'s `_make_request_with_harmony`
(the function that builds the harmony prompt for `/v1/chat/completions`)
calls `get_developer_message()` with **no arguments at all** -- meaning
`request.tools` is never passed into the harmony developer message on this
code path, so the model is never actually told the tool schemas exist.
(Contrast `serving_responses.py`'s `_construct_input_messages_with_harmony`,
which correctly calls `get_developer_message(request.instructions,
request.tools)`.) This is consistent with everything observed: the model
still attempts a tool call (our user messages clearly ask for one), but
without any registered function names/schemas to conform to, produces a
token sequence the harmony streaming parser can't reconcile against
anything -- hence the crash, and hence why it only triggers when the model
actually tries to call a tool. Not independently confirmed by re-patching
vLLM's source (out of scope for a smoke test), but sufficient explanation
to stop treating `/v1/chat/completions` as a lead worth pursuing further
for gpt-oss on this vLLM version.
`/v1/responses` uses a different, non-Chat-Completions tool schema
(`{"type": "function", "name", "description", "parameters"}` flat, not
nested under a `"function"` key) and a different output shape (`output`
list of typed items, e.g. `{"type": "function_call", "name", "arguments",
...}`, rather than `choices[0].message.tool_calls`). Added
`parsers/harmony_responses_call.py` (tool-schema converter +
`parse_responses_output`) and a `--api-format {chat,responses}` flag to
`run_model.py` (default `"chat"`, so Qwen is untouched); GPT-OSS runs pass
`--api-format responses`. Confirmed working: 10/10 parseable tool calls on
the Phase 2 five-task dev slice for both gpt-oss-20b and gpt-oss-120b (see
`results/raw/phase3_gptoss20b_smoke.jsonl` /
`phase3_gptoss120b_smoke.jsonl`), matching Qwen's Phase 2 result on the same
slice. **Resolved same session:** `/v1/responses`' `usage` field reports
all-zero token counts on this vLLM version (confirmed not fixable by any
request parameter -- vLLM's own `serving_responses.py` never populates it
for this path). vLLM's `/tokenize` endpoint was considered as a fix but
rejected: it doesn't special-case Harmony models at all (no `use_harmony`/
tools handling in `serving_tokenization.py`), so it would silently
undercount by ignoring the system/developer/tool-schema overhead entirely.
Fixed properly instead by reimplementing gpt-oss's harmony prompt
construction client-side in `parsers/harmony_token_count.py`
(`count_harmony_input_tokens`), using the `openai_harmony` library directly
(installed into `tool_ctx`, not `tool_ctx_gptoss` -- it's a lightweight
pure-Python package with no CUDA/torch dependency, so the client env
doesn't need the heavy `vllm` package too). Mirrors
`vllm.entrypoints.openai.serving_responses.py`'s
`_construct_input_messages_with_harmony` (new-conversation case) +
`vllm.entrypoints.harmony_utils.render_for_completion` exactly (same
system/developer message construction, same encoding
`HARMONY_GPT_OSS`), read directly from vLLM's own source rather than
guessed. `run_model.py`'s `responses` branch now always computes
`total_input_tokens` this way instead of reading (broken) `usage.input_tokens`.
Sanity-checked: 32-tool + reasoning_effort="low" dev_3 condition -> 2,499
tokens; same condition with 0 tools -> 124 tokens; "low" vs "medium" effort
-> identical count (single-word effort label, same token length) --
directionally consistent with the earlier `/v1/chat/completions` finding
below (141 prompt_tokens for the *same* 32-tool condition when tools
weren't actually wired into the prompt at all). Reran both GPT-OSS smoke
tests with the fix: still 10/10, now with populated (~2,480-2,520 token,
consistent across position since only tool *order* changes) input-token
counts per condition.

**`reasoning_effort` fixed to `"low"` (brief section 7) is passed as
`{"reasoning": {"effort": "low"}}` under `/v1/responses`, not the top-level
`reasoning_effort` field `/v1/chat/completions` uses.** `run_model.py`'s
`--reasoning-effort` flag branches on `--api-format` to build the right
shape.

**gpt-oss-120b's tensor-parallel (2x L40S) startup hangs indefinitely
without `--enforce-eager --disable-custom-all-reduce` (plus
`NCCL_P2P_DISABLE=1`).** First launch attempt (default flags, matching the
brief's "start with two L40S" guidance) never completed: GPU memory stayed
flat at ~670MiB per GPU (no weights ever loaded) for 40+ minutes while both
TP worker processes sat pinned at 100% CPU -- a genuine deadlock, not slow
first-time kernel compilation, confirmed by GPU-memory flatness (a slow
compile would still show weights loading first) and by comparison against
the successful retry, which showed ~36GB/GPU loaded within 30 seconds. Most
likely cause: this host has no NVLink, so multi-GPU communication goes over
PCIe, and vLLM's default CUDA-graph-capture + custom-all-reduce path for
TP=2 apparently doesn't handle that cleanly on this vLLM version. Killed the
hung process and retried with `--enforce-eager` (skips `torch.compile`/
CUDA-graph capture) and `--disable-custom-all-reduce`, plus
`NCCL_P2P_DISABLE=1` as a precaution -- all three were changed at once to
get unblocked, so which one(s) were actually necessary is not isolated;
worth bisecting before the final run, since `--enforce-eager` has a real
per-generation latency cost at 1,152-generation scale that a 10-condition
smoke test doesn't need to absorb. gpt-oss-20b (single GPU, no TP) never hit
this and needs none of these flags.

Both models' working configs are recorded in `configs/models/gpt_oss_20b.yaml`
and `configs/models/gpt_oss_120b.yaml`.

## 2026-09-27 — Phase 4 (96 fresh questions, drafted)

**Drafted by 8 parallel `general-purpose` subagents (one per tool group, 12
questions each), not by Claude directly, per explicit user instruction.**
Persona/instructions committed to
`.claude/agents/final-question-writer.md` (read by each subagent via its
`Read` tool, not invoked as a registered custom agent type -- same
"inlined via file read" pattern as the Phase 3 episode-writer subagents,
since custom subagent types still don't resolve as invocable in this
harness). Each subagent had full tool access (`Read`, `Write`, `Grep`,
`Glob`) and was required to read the entire 32-tool catalog
(`data/catalog/tools.jsonl`) -- not just its own 4 tools -- so it could
check cross-catalog ambiguity (brief section 3's "not validly answerable by
a distractor tool"), and was explicitly told not to read `data/raw/bfcl/`,
to avoid unconsciously reproducing a public BFCL question. Every record was
written with `"review_status": "pending"` -- nothing is auto-approved; the
user reviews and approves/rejects each of the 96 before Phase 4 continues.

**Merged and validated programmatically before handoff** (script not
committed, one-off): parsed all 8 draft files (96 lines total), checked
exactly 3 questions per each of the 32 catalog tools, `gold_tool` values
all match real catalog names, `gold_arguments` keys are a subset of the
tool's real schema properties and a superset of its required properties,
every request checked against literal-name leakage for *all* 32 tools (not
just its own gold tool, in case a question about tool A accidentally names
tool B), exact-duplicate and masked-number near-duplicate request-text
checks. Zero errors on first pass. Re-numbered to `final_0001`-`final_0096`
in deterministic catalog order (domain -> group -> tool -> question index)
and wrote `data/final/tasks.jsonl`. Validated again against the project's
own `FinalTaskRecord` pydantic schema.

**Two judgment calls the writer subagents flagged for the user's review,
not resolved here:**
- `kinematics__calculate_acceleration`: none of its 3 questions exercise
  the optional `distance` parameter (subagent's call -- found no natural
  phrasing that wouldn't either be redundant or create ambiguity with
  `calculate_displacement`).
- `weather__get`: all 3 questions give the country as a natural-language
  name (e.g. "Thailand") rather than a literal ISO alpha-2 code in the
  request text, with the alpha-2 code (`TH`) only appearing in
  `gold_arguments`. Satisfies brief section 3's "unambiguously derivable"
  wording on a literal reading, but is a judgment call about how strictly
  "contains all required arguments" should be read for a well-known,
  deterministic derivation -- worth the user's explicit sign-off rather
  than assuming it's fine.

**All 96 questions approved by the user, including both flagged items,
with no edit requests.** Reviewed via a published Artifact
(claude.ai/artifact/Ab4c1v75Wytd3ZDFc1N9Eb) backed by its `db` capability --
seeded with all 96 draft records, the user worked through them there
(per-question approve/reject + optional note), and every record came back
`review_status: "approved"` with an empty `note`, confirmed by querying
the artifact's database directly (`review_status != "approved"` and
`note != ""` both returned zero documents) rather than taking the user's
word alone. Synced `review_status: "approved"` into
`data/final/tasks.jsonl` for all 96 records; re-validated against
`FinalTaskRecord` and `pytest` still passes.

**Confirmed via score.py: primary scoring is tool-name-only, by explicit
user choice.** `categorize()` in `src/tool_context_study/score.py` compares
only `selected_tool == gold_tool`; `arguments` and `gold_arguments` are
both captured in every result/task record but never compared. This is the
brief's primary metric (section 10); the brief's secondary metric
(argument correctness) is not implemented. User confirmed this is
intentional for now, not a gap to close before continuing.

**Interference fraction reaffirmed at 25% (not just a "starting value" per
brief section 15 anymore) -- final.** The Phase 3 dev pilot's H64I results
are the actual evidence for this: at 25% interference density, accuracy
under H64I showed a clear, monotonic position effect (p1 0.917 -> p16 0.875
-> p32 0.792) that the other three history conditions don't show, meaning
25% is already enough interference to produce the exact signal this study
is designed to detect, without needing more (which would also cost more
episode-bank content and assembly complexity for no clear benefit). No
data suggests 25% is too weak or unnecessarily strong. Decided without
further experimentation, on the strength of the existing dev-pilot
evidence.

**Predefined annotation-defect correction policy (brief sections 3 and 15
-- required before freeze, previously undecided). Exclude, never edit.**
1. **What counts as a genuine defect:** something objectively verifiable
   independent of any model's performance on it -- e.g. `gold_arguments`
   doesn't match the tool's real parameter schema (wrong key/type), the
   request is provably answerable by a second catalog tool that was missed
   during review, a factual error in a lookup-style question (e.g. a wrong
   capital city), a duplicate task, or a technical bug in condition/history
   assembly (encoding, truncation, wrong permutation). **Not a defect:** a
   model getting the task wrong, a question merely being hard, or a
   stylistic second-guess ("this could have been phrased better"). The
   review pass (this session's Artifact-based approval, or any future one)
   is the gate for the second category; this policy only covers the first.
2. **Found before freeze:** just fix it directly in
   `data/final/tasks.jsonl` and log the fix in `docs/decision_log.md` --
   this is still development, not a post-freeze correction.
3. **Found after freeze (during or after Phase 5 runs): never edit or
   replace the frozen task.** Editing a gold label after seeing model
   failures is exactly what brief section 3 forbids, and it would look
   like (and risk actually being) cherry-picking, however unintentional.
   Instead:
   - Log it in a new `data/final/annotation_defects.jsonl` (created only
     if this is ever needed): `task_id`, `defect_description`,
     `discovered_date`, `discovered_by`, and the decision below.
   - Default action: **exclude that task's 12 conditions** (4 history
     conditions x 3 positions) from all aggregate analysis (accuracy
     tables, confidence intervals) -- but never delete or hide the raw
     model outputs already collected for it; they stay in `results/raw/`
     for transparency, just filtered out of `results/processed/` via
     Phase 5's required "machine-readable record of exclusions and
     failures."
   - Report the exclusion count and reason in the paper itself (a
     footnote/appendix line), never silently.
4. **Soft escalation trigger:** if excluded tasks ever exceed roughly 2%
   of the 96 (more than ~2 tasks), treat that as a signal of a possible
   systemic authoring problem worth a fuller audit, not just another
   footnote -- a judgment call for whoever is running Phase 5 at the time,
   not an automatic rule.
5. An `inference_error` / serving-side failure (a condition that never got
   a valid model response due to a serving bug, not a content defect) is
   already handled by the existing `inference_error` status and Phase 5's
   exclusions record -- this policy is specifically for defects in the
   *task itself* (the frozen request/gold label), not serving failures.

**Interference-episode manual review: tooling built, review itself still
in progress.** Corrected the actual episode count: it's 440 (55 per group
x 8 groups), not "~431"/"55-54 mixed" as informally estimated in the Phase
3 episode-banks entry -- confirmed by parsing every line of every
`interference_*.jsonl` file rather than trusting `wc -l` (which undercounts
a file whose last line lacks a trailing newline). Added
`src/tool_context_study/check_interference_episodes.py`, a committed,
reusable version of the automated pass mentioned in that earlier entry
(previously ad hoc, not saved as a script): literal-substring check against
all 32 catalog tool names, plus a broader set of answer-shaped-reply regex
patterns than before. Result: 1 of 440 flagged
(`interference_kinematics_0039`, matched on "...how brief **the distance
is**...", a false positive from a baseball-radar-gun conversation with
no actual tool-answer leakage -- confirmed by reading the episode). Flags
written to `data/histories/episodes/interference_flags.json`.

Published a second review Artifact
(claude.ai/artifact/4UTowDdhbuPxYr2m7a5Dt4) for the human review pass
itself, alongside the same-purpose Phase 4 questions one. Different data
shape from that one: episode content (1.1MB across 440 episodes) is
published as a static supporting file (`episode_data.json`) fetched
client-side, not seeded into the artifact database -- only the mutable
review state (`review_status`, `note`) lives in `db`'s `reviews`
collection, written lazily as the user reviews each episode rather than
pre-seeded, since (unlike the questions review) there's no content to
join against and defaulting an absent doc to `"pending"` client-side is
simpler than batch-seeding 440 near-empty documents. Verified with a
throwaway `set`/`delete` round-trip before handoff. Grouped by domain ->
target group, each episode collapsed to a topic + one-line preview by
default (expand to read the full transcript) given the volume, with the
one flagged episode called out at the top. Review itself not yet done --
this is the tool, not the completed pass.

**All 440 interference episodes manually reviewed and approved by the
user -- brief section 6's central requirement, now satisfied.** Reviewed
using the Artifact above as a *reading* tool only, not its per-episode
approve/reject mechanism (the user's explicit instruction: don't query the
artifact database for approval counts this time, since it wasn't used that
way) -- the approval itself is the user's direct statement in this
conversation, not a database record. Includes the one automated-check
flag (`interference_kinematics_0039`); no episodes excluded or flagged for
rewrite. This closes the last carried-over gap from Phase 3, and clears
the way to freeze the episode banks as part of Phase 4.

## 2026-09-27 — Phase 4 (final histories, conditions, freeze)

**Found and fixed a genuine, previously-undetected authoring defect in the
neutral episode bank: `neutral_0083` ended on a `user` turn.** Discovered
by `validate_history`'s role-alternation check when assembling final
histories with the new `final_seed` -- the Phase 3 dev-pilot's `base_seed`
happened to never place this episode adjacent to another user-starting
episode at a concatenation boundary, so the defect was invisible until a
different seed's episode ordering exposed it (assembly failed with
`AssertionError: history_final_h64n_base: consecutive 'user' turns at
index 625`). Scanned the full 250-episode neutral pool programmatically
for the same class of defect (wrong start/end role, or any internal
non-alternation) -- confirmed exactly one offender. Per the
annotation-defect policy above ("found before freeze: just fix it
directly"), added a short closing assistant turn to `neutral_0083`
(topic: SQL join types) so it ends on `assistant` like every other
episode, re-verified the whole pool is clean, and re-ran assembly
successfully. Also added `assemble_all`'s new `id_prefix` parameter and
`build_histories.py`'s `--seed-key`/`--id-prefix` CLI flags in the same
sitting (needed regardless, to keep dev and final histories as separate
files) -- both changes are backward-compatible (existing dev-pilot call
sites and all prior tests are unaffected by the new optional args/flags).

**Final-set histories assembled with `final_seed` (20260927), distinct
from the dev-pilot's `base_seed` (20260926)** -- deliberately, so the
frozen set is an independent assembly rather than a re-listing of
already-inspected dev-pilot histories, even though both draw from the
same (now fully reviewed) episode banks. Written as
`data/histories/assembled/history_final_{h64n_base,h16n_base,h64i_<group>}.json`,
alongside the untouched dev-pilot `history_*.json` files. All landed
within tolerance: H64N 64,006 tokens (target 64,000, `--tolerance 0.02`),
H16N 16,016 (target 16,000), H64I range 61,137-67,065 across the 8 groups
(target 64,000, `--h64i-tolerance 0.05`) -- consistent with the wider H64I
tolerance's documented rationale (matching-based swaps drift more than
plain trimming).

**Built and validated the full 1,152-condition frozen grid** (96 approved
final tasks x 4 history conditions x 3 positions) via new
`src/tool_context_study/phase4_build_final_conditions.py`, reusing
`build_condition`/`validate_condition` unchanged from the dev pilot.
`load_final_tasks()` refuses to run over any task not
`review_status: "approved"` (a hard check, not just a convention) as a
last-resort guard against building frozen conditions from a partially-
reviewed set. Verified the resulting distribution: exactly 288 per
history condition, 384 per position, 36 per each of the 32 tools --
perfectly uniform, as the design requires.

**Re-verified Phase 2's "inspect rendered prompts" rigor at final scale,
through Qwen's real tokenizer/chat template, not just `validate_condition`'s
structural checks.** All 32 tool names present exactly once; declared
tool order preserved in the actual rendered text (checked via each tool's
exact `"name": "<name>"` JSON marker, not a bare substring search -- a
first naive substring check falsely flagged a reordering because
`weather__get` collided as a substring of
`weather__get_forecast_by_coordinates`, corrected before trusting the
result); gold tool confirmed at its declared position; final user message
present and unmodified in the rendered text (no silent truncation).
Largest condition in the whole grid (`final_0014__H64I__p32`, 32 tools +
H64I history + request) renders to 71,900 tokens -- comfortably inside
the `--max-model-len 80000` Qwen was already proven to handle in the
Phase 3 dev pilot. **Forward note for Phase 5:** GPT-OSS has so far only
been smoke-tested at `--max-model-len 8192` (H0 only, per
`configs/models/gpt_oss_{20b,120b}.yaml`) -- it will need the same kind of
context-window increase Qwen already got before it can serve H64N/H64I
conditions at this size.

## 2026-09-28 — Phase 5 prep: GPT-OSS context-window bump

**Resolved the Phase 4 forward note above.** Measured actual harmony-encoded
prompt sizes across the full frozen 1,152-condition final grid using
`count_harmony_input_tokens()` (the same client-side harmony reimplementation
`run_model.py` already uses to log real token counts, since `/v1/responses`'
`usage` field is always zero -- see the Phase 3 GPT-OSS smoke-test entry
above) rather than assuming Qwen's tokenizer figure carries over. Harmony's
encoding is denser than Qwen's: largest condition is
`final_0014__H64I__p1` at 67,319 input tokens (vs. Qwen's 71,900 for the
same condition under its own chat template) -- by history condition,
maxima are H0 2,720 / H16N 18,136 / H64N 64,372 / H64I 67,319.

**Set `--max-model-len 80000` for both `gpt_oss_20b.yaml` and
`gpt_oss_120b.yaml`** (up from the 8192 H0-only smoke-test value), reusing
Qwen's already-proven value rather than shaving margin to the measured
67,319 + 256-token decode budget -- keeps headroom consistent across all
three models rather than independently optimizing GPT-OSS's. Also bumped
`gpu_memory_utilization` 0.85 -> 0.92 (same value Qwen's own bump used) to
fit the larger KV cache this context length needs.

**Verified, not just configured -- relaunched both models with the new
settings and ran real requests through `run_model.py`, not only a config
edit.** gpt-oss-20b (1x L40S, `CUDA_VISIBLE_DEVICES=2`): server started
cleanly (num_gpu_blocks=72190, no OOM), and a smoke set covering the
grid's single largest condition (`final_0014__H64I__p1`, 67,319 tokens)
plus one condition each from H0/H16N/H64N all returned real responses
(no `inference_error`) -- the H64I request took 29.5s and produced
`status=no_tool_call` (a legitimate scored outcome at `reasoning_effort:
low` on a long, distracting context, not an infra failure; the other
three returned `status=ok` with the correct tool). gpt-oss-120b (2x L40S,
TP=2, same `--enforce-eager --disable-custom-all-reduce` flags Phase 3's
smoke test required): started cleanly (GPU KV cache size 165,216 tokens,
"maximum concurrency for 80,000 tokens per request: 4.02x", no OOM), and
the same 4-condition smoke set all returned real responses -- no
`inference_error` on the 67,319-token H64I request (34.6s) or any other.
3 of the 4 came back `no_tool_call` at `reasoning_effort: low` (vs. 1 of
4 for gpt-oss-20b on the identical set); this is n=4, not a benchmark,
and not evidence about 120b-vs-20b quality -- noted only because the
*pattern* (more non-calls on longer-history conditions) is consistent
with what Phase 5 is designed to measure, not because 4 datapoints
support any claim. Both servers shut down immediately after (`nvidia-smi`
confirmed all GPUs back to idle) per this project's shared-host
convention.

**Process note:** first launch attempt for gpt-oss-20b put
`CUDA_VISIBLE_DEVICES=2` on its own line inside a multi-line `bash -c`
string, separate from the `python -m vllm...` invocation -- since it
wasn't attached to that command (and wasn't `export`ed), the env var
never reached the process, and vLLM silently loaded onto GPU 0 instead
(the shared host's default device, which also already had another user's
small process on it) instead of the intended idle GPU 2. Caught via
`/proc/<pid>/environ` and `nvidia-smi --query-compute-apps`, not by any
vLLM-side warning -- it fails silently. Killed immediately (GPU memory
confirmed freed via `nvidia-smi` before relaunching) and relaunched with
`env VAR=val VAR2=val2 python ...` on one command so the env vars actually
attach to the exec'd process. Worth remembering for any future manual
launch on this host: `CUDA_VISIBLE_DEVICES` not landing on the target GPU
fails silently, not loudly.

**Froze the design via `src/tool_context_study/freeze_phase4.py` ->
`data/manifests/phase4_freeze.json`.** Sha256 + byte count for: the
catalog spec and normalized catalog, the 96 approved final tasks, the full
condition grid, `configs/experiment.yaml`, all 10 `history_final_*.json`
files, all `configs/models/*.yaml`, `score.py`, and every file under
`parsers/`. Also records the frozen design parameters inline (32 tools,
96 questions, positions, history conditions, token targets, interference
fraction, `final_seed`, 1,152 conditions/model) and the current git commit
(noted as having a dirty working tree at freeze time, since this work
hasn't been committed yet -- committing is the user's call, not done
automatically here). Phase 4 is now complete.

## 2026-09-27 — Mitigations/reference-points scope folded into the brief

**`docs/research_brief.md` was missing an entire deliverable.** The user
shared the actual project proposal PDF ("Tool Order and Conversation
Context Effects on LLM Tool Selection, and Practical Fixes") while
checking whether anything needed to happen before Phase 5. Grepped the
existing brief/CLAUDE.md/README.md for "mitigation," "oracle,"
"hierarchical routing," "permutation voting," "history pruning,"
"retrieval," and "implemented by us" -- zero matches anywhere. Every phase
implemented so far (1-4) was built entirely from `docs/research_brief.md`,
which turns out to have only ever covered half the actual proposal (the
characterization study, sections 1-15) -- the mitigation/fixes half
(deliverable 2 of the proposal) was never transcribed into it, and the
brief's own section 1 explicitly said "do not add... retrieval... unless
explicitly requested later," which reads as a placeholder for exactly this
that was never filled in.

**Folded the missing scope into `docs/research_brief.md` as new section
16** (mitigations: hierarchical routing, permutation voting, history
pruning; reference points: oracle, retrieval-based tool-pruning baseline;
scope/scale; reporting), a new "Phase 6" in section 13, and additions to
section 15's fixed/open lists -- transcribed faithfully from the proposal
rather than inventing specifics it doesn't give (e.g. which single
GPT-OSS model the mitigation study uses, the exact embedding model, the
exact history-condition/position subset mitigations run over, are all
logged as "still open" rather than decided here). Also updated
`CLAUDE.md` (ground rules: baseline-before-mitigations ordering,
mitigation-tuning-on-dev-set-only) and `README.md` (project description,
a new Phase 6 status entry, a `mitigations/` repo-layout entry, and a new
"Implementation attribution" section for the proposal's deliverable 4).

**Decided: Phase 5 (baseline characterization) runs before Phase 6
(mitigations), not the reverse or in parallel.** Two concrete dependencies,
not just a sequencing preference: permutation voting's primary evaluation
*is* a majority vote over the baseline's own p1/p16/p32 generations for
each (task, history_condition) pair -- it consumes Phase 5's results
directly rather than requiring its own model calls for that part. And the
oracle reference point is simply the baseline's own position-1 accuracy,
again nothing to compute independently. Mitigation-setting tuning on the
dev set (hierarchical routing needs none; permutation voting's K=3 is
already fixed by the proposal; history pruning needs an embedding model
and to confirm the ~4K token budget) is independent of this ordering and
can happen before, during, or after Phase 5 -- it only has to finish
before any mitigation touches the frozen final set.

## 2026-09-28 — run_model.py concurrency, and why it doesn't fix Phase 5's wall-clock

**Added `--concurrency` to `run_model.py`** (default 16): conditions are now
dispatched via a `ThreadPoolExecutor` against a shared `requests.Session`
(pooled connections sized to `--concurrency`) instead of one at a time,
so vLLM's continuous batching actually has multiple requests to batch.
Results are written as each request completes (not in input order) --
safe because `score.py` already joins results back to conditions by
`condition_id`, never by line position. `call_condition`'s default
`timeout` raised 120s -> 300s since a request can now sit queued behind
others before vLLM admits it. Correctness verified, not assumed: ran the
same 24-condition mixed-history set (6 each of H0/H16N/H64N/H64I) through
both `--concurrency 1` and `--concurrency 8`, on both gpt-oss-20b and
Qwen3-30B-A3B-FP8 -- identical `(status, selected_tool)` per
`condition_id` in every case, and every output file had exactly 24 unique
`condition_id`s (no dropped or duplicated results from the concurrent
write path).

**However: measured wall-clock showed concurrency does *not* meaningfully
cut Phase 5's runtime, contrary to the initial hypothesis.** Both models
served on a single L40S (no TP), `--max-model-len 80000` as bumped above:

| model | concurrency=1 | concurrency=8 |
|---|---|---|
| gpt-oss-20b | 5m07s | 4m47s (-7%) |
| Qwen3-30B-A3B-FP8 | 3m05s | 3m05s (~0%) |

Per-request latency under concurrency=8 didn't drop -- it *rose* sharply
for the long-history conditions (Qwen H64I: 13.5-14.1s isolated -> up to
114s concurrent; individual H0 requests that took <1s isolated took
6-8s concurrent). Total tokens processed stayed constant while wall-clock
barely moved: this is the signature of a **compute-bound, prefill-
dominated** workload, not a GPU-idle-time one. Continuous batching's real
win is overlapping *decode*-phase idle cycles across requests (memory-
bandwidth-bound, one token at a time) -- it has nothing to overlap when a
request's cost is dominated by a single one-shot prefill of 16K-80K
tokens through chunked prefill (`max_num_batched_tokens=2048` by
default), which already keeps the GPU's tensor cores close to saturated
on its own. H64N/H64I conditions -- exactly the ones that dominate this
study's total runtime -- are prefill-dominated by construction (up to
80K input tokens, only 256 output tokens), so this is true of the actual
workload, not just this benchmark's slice of it. The original hypothesis
("GPU likely far from saturated during a single 14s Qwen request") is
falsified by this measurement, not confirmed.

**Implication for the 9-hour problem this was meant to solve: request
concurrency on a single GPU is not the lever.** What should actually cut
wall-clock, not yet built: sharding the condition grid across multiple of
the host's 8 L40S GPUs, each running its own model server instance on an
independent subset of conditions -- true parallelism across compute
budgets, rather than contention for one. This is expected to scale close
to linearly with GPU count (modulo gpt-oss-120b's TP=2 requirement
halving how many replicas fit). `--concurrency` is left in `run_model.py`
regardless (harmless default, real if modest gpt-oss win, and helps more
on hardware/workloads that are actually decode-idle) but is not, by
itself, Phase 5's answer.

## 2026-09-28 — Phase 5 orchestrator: GPU sharding, capped for shared-host use

**Built `src/tool_context_study/phase5_run_baseline.py`** to run the frozen
1,152-condition grid against all three models by sharding across multiple
idle GPUs per model (the actual lever, per the entry above), instead of
one server per model at a time. Explicit user requirement, not a default:
**never uses more than 5 GPUs at once, hard-capped at 6** regardless of
flags (`--max-gpus`, default 5; `parser.error`s above 6) -- this is a
shared 8-GPU host and other users' jobs were observed starting and
stopping mid-session while this work was in progress. The one case
allowed to reach 6: gpt-oss-120b needs `tensor_parallel_size=2`, so an odd
cap like the default 5 would otherwise waste a GPU (2 replicas = 4 GPUs,
1 sitting idle) -- `effective_gpu_cap()` rounds up by exactly one *only*
when `gpus_per_replica == 2` and the cap is odd, never past 6, and never
for the single-GPU models. Idle GPUs are queried fresh (`nvidia-smi`) at
the start of every model phase, not cached from an earlier check, for the
same reason. Corrected a latent bug while wiring this up: Qwen's own
`configs/models/qwen3_30b_a3b_instruct_2507_fp8.yaml` had never been
updated after the Phase 3 dev pilot proved `--max-model-len 80000
--gpu-memory-utilization 0.92` necessary (still said 8192/0.85) -- harmless
as long as a human copy-pasted the pilot's real command by hand, but this
orchestrator reads the field programmatically, so it would have silently
under-provisioned Qwen's context window. Fixed, and added
`tensor_parallel_size` / `conda_env` as explicit structured fields to all
three model configs (previously only in free-text `gpu:`/prose) so the
orchestrator has one source of truth instead of a second, separately
-maintained mapping.

Runs one model at a time (never multiple models' servers simultaneously),
each phase: discover idle GPUs -> launch as many replicas as the cap and
idle count allow -> shard *unfinished* conditions round-robin across them
-> run `run_model.py` per shard concurrently -> merge -> score -> shut
every replica down before the next model. Resumable by design: a
condition already recorded with any status other than `inference_error`
is never resubmitted, and retries automatically (`--max-attempts`,
default 3) within one invocation for genuine failures before raising.

**Caught a real bug in the resume logic via an end-to-end smoke test, not
by inspection.** First implementation deleted shard files at the *start*
of each attempt (to avoid double-counting) and only merged them into the
canonical output *once, at the very end*. Tested by faking a crashed
prior attempt (deleted the canonical file, left one shard file with 4/8
conditions already done) and rerunning on a tiny 8-condition grid: the
already-done conditions got recognized correctly on attempt 1, but their
evidence (the shard file recording them) was then deleted without ever
being written anywhere durable, so attempt 2 "forgot" them and reprocessed
them instead of the real remainder -- oscillating between two overlapping
4-condition subsets for all 3 attempts and raising `RuntimeError:
4/8 conditions still unresolved` despite 8/8 actually having succeeded
across the two attempts, just never in the same file at the same time.
Fixed by making the merge step (`absorb_shards_into_canonical`) run at
the *top* of every attempt, not the end -- it's now the only thing that
touches shard files, always folds them into the canonical file before
anything gets a chance to overwrite them, and leaves the shard directory
clean afterward either way. Re-ran the same fault-injection scenario after
the fix: recovered the 4 pre-existing results, ran only the true
remaining 4, and completed correctly on the first attempt. Added
`tests/test_phase5_run_baseline.py::test_absorb_shards_is_cumulative_across_calls`
as a permanent regression test for this exact failure mode, plus smaller
tests for the round-robin sharding and the GPU-cap rounding rule. Also
verified end-to-end against a real gpt-oss-20b server (not just the pure
functions): 2 replicas on 2 GPUs, correct merge, correct score, and GPUs
confirmed fully freed after shutdown via `nvidia-smi`.

**Not yet run at full scale.** This has only been exercised on tiny
(8-condition) hand-built grids so far, specifically to validate the
sharding/resume/merge machinery in isolation from the ~9-hour real
runtime. The actual 1,152-condition x 3-model Phase 5 run is still
pending.

## 2026-09-28 — Phase 5 orchestrator: robustness pass before the real run

Before committing to an unattended 9-hour run, reviewed the design against
external best-practice suggestions for exactly this scenario (long
unattended job on a shared GPU host). Agreed with all seven and closed
three real gaps:

**Request-level retry with backoff, in `run_model.py`'s `call_condition`
(new `--max-retries`, default 2).** Previously a single connection error
or timeout immediately recorded `inference_error` with zero retry --
correct as a last resort, but `phase5_run_baseline.py`'s own retry
mechanism operates at a much coarser grain (relaunching an entire server
replica and reprocessing everything still unresolved), which is far too
heavyweight to invoke over one flaky request. Retries connection errors,
timeouts, and 5xx with linear backoff; does *not* retry 4xx (a malformed
request retrying can't fix).

**Real `GET /health` polling in `wait_for_server_ready`, replacing a
log-text match.** The previous implementation declared a replica ready by
grepping its log for vLLM's "Application startup complete" string --
which only proves vLLM printed that line, not that the port is actually
accepting connections. Now polls the real endpoint (vLLM's OpenAI server
exposes `/health`, confirmed in its own startup route listing) and
requires an actual 200; the log-based `FAIL_RE` check is kept alongside
for *fast failure* (a traceback or OOM message means "never coming up,"
which a health check alone can't distinguish from "still loading" -- both
look like connection-refused until the process actually dies).

**Per-model isolation in `main()`.** Previously an exception from any one
model's `run_model_phase` (or even just a typo in `--models` resolving to
a missing config file) would crash the entire script, discarding the
opportunity to attempt the remaining models even though each one's
completed results are already safely on disk by the time such an
exception could occur. Now each model runs in its own try/except; a
failure is logged in full (`traceback.print_exc()`, not just `str(exc)`,
since this is the one class of error an unattended run can't ask a human
to reproduce interactively) and the loop moves on. Final summary reports
succeeded/failed model lists and the process exits 1 iff anything failed
-- scriptable, and distinct from a clean success. Caught a related bug
while wiring this up: `load_model_config(model_key)` was called *outside*
the try block in the original draft, so exactly the "bad --models value"
case this was meant to guard against would have bypassed the isolation
entirely -- moved inside.

**A done-signal you can check without a terminal (`write_status`).**
Previously the only signal of progress was stdout, useless once nobody's
watching it. Every meaningful milestone (run/model start, replica ready,
attempt progress, model done/failed, run done) now appends one line to
`results/logs/phase5_status.jsonl` (full timeline) and overwrites
`results/logs/phase5_status.txt` (current state, one `cat` away).

**Re-verified end-to-end after all four changes**, on the same tiny
hand-built grid: `--models does_not_exist,gpt_oss_20b` -- the bad model
key failed cleanly and was logged without aborting the run, gpt-oss-20b
then completed normally via the real `/health`-based readiness check
(confirmed replicas were marked ready only after an actual 200, not a log
line), the status files captured the whole timeline correctly, and the
final summary correctly reported `succeeded=['gpt_oss_20b']`,
`failed=['does_not_exist']`, exit code 1. GPUs confirmed fully freed
afterward via `nvidia-smi`.

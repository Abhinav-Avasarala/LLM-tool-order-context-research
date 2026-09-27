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

**Generated via `general-purpose` subagents with the persona/instructions
from `.claude/agents/episode-writer-neutral.md` and
`.claude/agents/episode-writer-interference.md` inlined into each prompt,
not via those agent types directly.** This session's harness did not have
those project-local custom subagent types registered as invocable (`Agent`
calls to `episode-writer-neutral` / `episode-writer-interference` errored
with "agent type not found"); functionally equivalent since the agent
definitions are just system-prompt files, but noted here in case a future
session finds those custom types *do* resolve and wants to use them
directly instead.

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

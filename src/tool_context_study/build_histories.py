"""Assemble the four conversation-history conditions from the episode banks
(brief section 6): `H0` (no history, trivial), `H16N` (~16K neutral),
`H64N` (~64K neutral), `H64I` (~64K, ~25% swapped for group-matched
interference episodes).

AI drafted the episodes; this module is the "code must assemble and
validate them" half -- it never asks a model to write history text, only
to select, order, and swap whole episodes.

One shared `H64N` is built (not one per final task): every task under a
given history condition and target group draws on the same assembled
history, only the final user request and gold-tool position vary. This
keeps "history content" a controlled variable and matches the brief's
singular phrasing ("Build a 64K neutral history... Build a matched 64K
interfering history"). `H16N` is always derived from that same `H64N`'s
tail, never built independently, so it really is "the same recent history
found in H64N" per the brief.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from tool_context_study.config import DATA_DIR, load_experiment_config
from tool_context_study.schemas import AssembledHistory

EPISODES_DIR = DATA_DIR / "histories" / "episodes"
ASSEMBLED_DIR = DATA_DIR / "histories" / "assembled"

TargetGroup = str
TokenCounter = Callable[[list[dict[str, Any]]], int]

# Groups the brief's 32-tool catalog splits into (configs/catalog_spec.yaml),
# i.e. the possible `target_group` values an interference episode -- and
# therefore an H64I variant -- can be tailored to.
TARGET_GROUPS = [
    "kinematics",
    "electromagnetism",
    "places_and_travel",
    "weather",
    "corporate_finance",
    "personal_finance",
    "property_and_rentals",
    "environmental_impact",
]


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_neutral_pool() -> list[dict[str, Any]]:
    return load_episodes(EPISODES_DIR / "neutral.jsonl")


def load_interference_pool(group: TargetGroup) -> list[dict[str, Any]]:
    path = EPISODES_DIR / f"interference_{group}.jsonl"
    episodes = load_episodes(path)
    for ep in episodes:
        if ep.get("target_group") != group:
            raise ValueError(f"{path}: episode {ep['episode_id']!r} has target_group={ep.get('target_group')!r}")
    return episodes


def make_hf_token_counter(model_name: str) -> TokenCounter:
    """Real token counter: the evaluated model's own tokenizer + chat template.

    Imported lazily so the rest of this module (and anything that only
    needs the pure selection/assembly logic, e.g. tests) doesn't require
    `transformers` to be installed.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _count(messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)

    return _count


def _episode_messages(ep: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": m["role"], "content": m["content"]} for m in ep["messages"]]


def _select_prefix_to_target(
    pool: list[dict[str, Any]],
    target_tokens: int,
    token_counter: TokenCounter,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Shuffle `pool` (seeded) and greedily take a prefix whose token count is
    closest to `target_tokens`, trimming only at episode boundaries.

    Returns (selected_episodes, flattened_messages, token_count).
    """
    order = pool[:]
    rng.shuffle(order)

    selected: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    count = 0
    for ep in order:
        candidate_messages = messages + _episode_messages(ep)
        candidate_count = token_counter(candidate_messages)
        if candidate_count >= target_tokens:
            over = candidate_count - target_tokens
            under = target_tokens - count
            if selected and under <= over:
                return selected, messages, count  # excluding this episode lands closer
            selected.append(ep)
            return selected, candidate_messages, candidate_count
        selected.append(ep)
        messages = candidate_messages
        count = candidate_count

    raise ValueError(
        f"Episode pool exhausted at {count} tokens without reaching target {target_tokens} "
        f"(pool has {len(pool)} episodes) -- the pool needs to be larger."
    )


def _select_suffix_to_target(
    episodes_in_order: list[dict[str, Any]],
    target_tokens: int,
    token_counter: TokenCounter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Take a *tail* (most-recent-first) slice of an already-ordered episode
    sequence whose token count is closest to `target_tokens`, trimming only
    at episode boundaries. No shuffling: this is meant to reproduce a
    specific suffix of an existing H64N, not draw a fresh sample.
    """
    selected: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    count = 0
    for ep in reversed(episodes_in_order):
        candidate_messages = _episode_messages(ep) + messages
        candidate_count = token_counter(candidate_messages)
        if candidate_count >= target_tokens:
            over = candidate_count - target_tokens
            under = target_tokens - count
            if selected and under <= over:
                return selected, messages, count
            selected.insert(0, ep)
            return selected, candidate_messages, candidate_count
        selected.insert(0, ep)
        messages = candidate_messages
        count = candidate_count

    raise ValueError(f"H64N too short ({count} tokens) to carve out a {target_tokens}-token suffix.")


def _select_indices_to_remove(
    per_episode_tokens: list[int],
    target_removed_tokens: int,
    rng: random.Random,
) -> list[int]:
    """Pick a scattered subset of episode indices (out of the whole H64N
    sequence) whose combined token count is closest to `target_removed_tokens`.

    Scanning indices in random (not sequential) order, rather than always
    preferring e.g. the largest episodes first, is what keeps the replaced
    slots spread through the conversation instead of clustered.
    """
    order = list(range(len(per_episode_tokens)))
    rng.shuffle(order)

    chosen: list[int] = []
    total = 0
    for idx in order:
        candidate_total = total + per_episode_tokens[idx]
        if candidate_total >= target_removed_tokens:
            over = candidate_total - target_removed_tokens
            under = target_removed_tokens - total
            if chosen and under <= over:
                return sorted(chosen)
            chosen.append(idx)
            return sorted(chosen)
        chosen.append(idx)
        total = candidate_total

    return sorted(chosen)


def _best_match(
    removed_ep: dict[str, Any],
    removed_tokens: int,
    candidates: list[dict[str, Any]],
    candidate_tokens: dict[str, int],
) -> dict[str, Any]:
    """Pick the interference episode closest to `removed_ep` in token count
    (message-count difference as a tiebreaker only), so the swap keeps the
    history's total token count matched -- the brief's H64I requirement
    that matters most for the ±token-target tolerance. Token closeness is
    the primary key rather than message count because an interference
    bank that skews wordier per message than the neutral episode it's
    replacing (which happens: some target groups' episodes run longer)
    would otherwise systematically inflate the total if we filtered down
    to same-message-count candidates first.
    """
    removed_n_messages = len(removed_ep["messages"])
    return min(
        candidates,
        key=lambda c: (abs(candidate_tokens[c["episode_id"]] - removed_tokens), abs(len(c["messages"]) - removed_n_messages)),
    )


def build_h64n(
    neutral_pool: list[dict[str, Any]],
    *,
    target_tokens: int,
    tolerance: float,
    seed: int,
    token_counter: TokenCounter,
    token_count_model: str,
    history_id: str = "history_h64n_base",
) -> AssembledHistory:
    rng = random.Random(seed)
    selected, messages, count = _select_prefix_to_target(neutral_pool, target_tokens, token_counter, rng)

    low, high = target_tokens * (1 - tolerance), target_tokens * (1 + tolerance)
    if not (low <= count <= high):
        raise ValueError(
            f"H64N landed at {count} tokens, outside the +-{tolerance:.0%} band around {target_tokens} "
            f"({low:.0f}-{high:.0f}). Pool ({len(neutral_pool)} episodes) is too coarse-grained; "
            f"add shorter neutral episodes to allow finer trimming, or widen the tolerance."
        )

    return AssembledHistory(
        history_id=history_id,
        history_condition="H64N",
        seed=seed,
        episode_ids=[ep["episode_id"] for ep in selected],
        messages=messages,
        token_count=count,
        token_count_model=token_count_model,
        n_messages=len(messages),
    )


def build_h16n(
    h64n: AssembledHistory,
    neutral_pool_by_id: dict[str, dict[str, Any]],
    *,
    target_tokens: int,
    tolerance: float,
    token_counter: TokenCounter,
    token_count_model: str,
    history_id: str = "history_h16n_base",
) -> AssembledHistory:
    episodes_in_order = [neutral_pool_by_id[eid] for eid in h64n.episode_ids]
    selected, messages, count = _select_suffix_to_target(episodes_in_order, target_tokens, token_counter)

    low, high = target_tokens * (1 - tolerance), target_tokens * (1 + tolerance)
    if not (low <= count <= high):
        raise ValueError(
            f"H16N (suffix of {h64n.history_id}) landed at {count} tokens, outside the +-{tolerance:.0%} "
            f"band around {target_tokens} ({low:.0f}-{high:.0f})."
        )

    return AssembledHistory(
        history_id=history_id,
        history_condition="H16N",
        derived_from=h64n.history_id,
        seed=h64n.seed,
        episode_ids=[ep["episode_id"] for ep in selected],
        messages=messages,
        token_count=count,
        token_count_model=token_count_model,
        n_messages=len(messages),
    )


def build_h64i(
    h64n: AssembledHistory,
    neutral_pool_by_id: dict[str, dict[str, Any]],
    interference_pool: list[dict[str, Any]],
    *,
    target_group: TargetGroup,
    interference_fraction: float,
    tolerance: float,
    seed: int,
    token_counter: TokenCounter,
    token_count_model: str,
    history_id: str | None = None,
) -> AssembledHistory:
    rng = random.Random(seed)
    base_episodes = [neutral_pool_by_id[eid] for eid in h64n.episode_ids]
    # Marginal (in-context) token contribution of each episode within the
    # full H64N sequence, not an isolated per-episode count: chat-template
    # tokenization isn't perfectly additive across message boundaries, and
    # using the in-context deltas here keeps the removal target (a fraction
    # of h64n.token_count, itself measured on the full sequence) consistent
    # with what we're actually summing over.
    per_episode_tokens: list[int] = []
    running_messages: list[dict[str, Any]] = []
    prev_count = 0
    for ep in base_episodes:
        running_messages += _episode_messages(ep)
        count = token_counter(running_messages)
        per_episode_tokens.append(count - prev_count)
        prev_count = count

    target_removed = h64n.token_count * interference_fraction
    remove_indices = _select_indices_to_remove(per_episode_tokens, target_removed, rng)

    interference_tokens = {ep["episode_id"]: token_counter(_episode_messages(ep)) for ep in interference_pool}
    if len(interference_pool) < len(remove_indices):
        raise ValueError(
            f"target_group={target_group!r}: need to swap {len(remove_indices)} episodes but the "
            f"interference pool only has {len(interference_pool)} -- would require reusing an episode, "
            f"which the brief's 'not repeated mechanically' rule forbids. Add more interference episodes."
        )

    remaining_candidates = interference_pool[:]
    rng.shuffle(remaining_candidates)  # so ties in _best_match don't always favor the same episode_id

    new_episodes = base_episodes[:]
    # One record per swapped slot, kept together so the corrective pass
    # below can mutate a single slot without three parallel lists drifting
    # out of sync with each other.
    swaps = []  # each: {"idx", "removed_ep", "removed_tokens", "match"}
    for idx in remove_indices:
        removed_ep = base_episodes[idx]
        match = _best_match(removed_ep, per_episode_tokens[idx], remaining_candidates, interference_tokens)
        remaining_candidates.remove(match)
        new_episodes[idx] = match
        swaps.append({"idx": idx, "removed_ep": removed_ep, "removed_tokens": per_episode_tokens[idx], "match": match})

    messages = [m for ep in new_episodes for m in _episode_messages(ep)]
    count = token_counter(messages)

    low, high = h64n.token_count * (1 - tolerance), h64n.token_count * (1 + tolerance)

    # Corrective pass: matching-based swaps are only as good as the pool's
    # closest available candidate, so the aggregate can still drift outside
    # tolerance (e.g. a target group whose episodes run systematically
    # wordier than what they're replacing). Repeatedly fix the single
    # worst-offending slot -- try a better unused candidate first, and only
    # drop the swap (revert to the original neutral episode) if no
    # remaining candidate improves it -- rather than failing outright.
    guard = 0
    guard_limit = len(swaps) + len(interference_pool) + 1
    while not (low <= count <= high) and swaps and guard < guard_limit:
        guard += 1
        if count > high:
            worst = max(swaps, key=lambda s: interference_tokens[s["match"]["episode_id"]] - s["removed_tokens"])
        else:
            worst = min(swaps, key=lambda s: interference_tokens[s["match"]["episode_id"]] - s["removed_tokens"])

        current_delta = abs(interference_tokens[worst["match"]["episode_id"]] - worst["removed_tokens"])
        better = None
        if remaining_candidates:
            candidate_best = min(
                remaining_candidates, key=lambda c: abs(interference_tokens[c["episode_id"]] - worst["removed_tokens"])
            )
            if abs(interference_tokens[candidate_best["episode_id"]] - worst["removed_tokens"]) < current_delta:
                better = candidate_best

        if better is not None:
            remaining_candidates.remove(better)
            remaining_candidates.append(worst["match"])
            new_episodes[worst["idx"]] = better
            worst["match"] = better
        else:
            # No unused candidate helps this slot -- drop the swap entirely.
            new_episodes[worst["idx"]] = worst["removed_ep"]
            remaining_candidates.append(worst["match"])
            swaps.remove(worst)

        messages = [m for ep in new_episodes for m in _episode_messages(ep)]
        count = token_counter(messages)

    if not (low <= count <= high):
        raise ValueError(
            f"H64I[{target_group}] landed at {count} tokens, outside the +-{tolerance:.0%} band around "
            f"{h64n.history_id}'s {h64n.token_count} ({low:.0f}-{high:.0f}) even after the corrective pass "
            f"({guard} adjustments). The interference pool's length distribution may be too mismatched "
            f"from the neutral episodes it needs to replace -- consider widening --h64i-tolerance or "
            f"adding shorter/longer interference episodes to that group."
        )

    if not swaps:
        raise ValueError(
            f"H64I[{target_group}]: the corrective pass reverted every swap back to neutral, leaving zero "
            f"interference content -- the interference pool doesn't fit this history at all."
        )

    replaced_neutral_ids = [s["removed_ep"]["episode_id"] for s in swaps]
    used_interference_ids = [s["match"]["episode_id"] for s in swaps]

    return AssembledHistory(
        history_id=history_id or f"history_h64i_{target_group}",
        history_condition="H64I",
        target_group=target_group,
        derived_from=h64n.history_id,
        seed=seed,
        episode_ids=[ep["episode_id"] for ep in new_episodes],
        interference_episode_ids=used_interference_ids,
        replaced_neutral_episode_ids=replaced_neutral_ids,
        messages=messages,
        token_count=count,
        token_count_model=token_count_model,
        n_messages=len(messages),
    )


def validate_history(history: AssembledHistory) -> None:
    """Raise `AssertionError` if `history` doesn't meet the design invariants."""
    roles = [m["role"] for m in history.messages]
    assert roles, f"{history.history_id}: no messages"
    assert roles[0] == "user", f"{history.history_id}: history must start with a user turn"
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"{history.history_id}: consecutive {roles[i]!r} turns at index {i}"
    assert len(set(history.episode_ids)) == len(history.episode_ids), f"{history.history_id}: repeated episode id"
    if history.history_condition == "H64I":
        assert history.target_group is not None, f"{history.history_id}: H64I must set target_group"
        assert history.interference_episode_ids, f"{history.history_id}: H64I with zero interference episodes"
        assert len(history.interference_episode_ids) == len(history.replaced_neutral_episode_ids)


def write_history(history: AssembledHistory) -> Path:
    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSEMBLED_DIR / f"{history.history_id}.json"
    path.write_text(history.model_dump_json(indent=2) + "\n")
    return path


def assemble_all(
    *,
    token_counter: TokenCounter,
    token_count_model: str,
    base_seed: int,
    history_token_targets: dict[str, int],
    interference_fraction: float,
    tolerance: float = 0.02,
    h64i_tolerance: float | None = None,
    id_prefix: str = "history",
) -> list[AssembledHistory]:
    """`h64i_tolerance` defaults to `tolerance` if not given, but H64I
    construction (matching-based swaps) has strictly more sources of drift
    than H64N/H16N (plain trimming), so callers may want to pass a looser
    value -- the brief itself only asks for "approximately +-2% when
    practical", not a hard 2% cutoff.

    `id_prefix` distinguishes independently-assembled history sets that
    share the same episode banks (e.g. Phase 3's dev-pilot histories vs.
    Phase 4's frozen final-set histories) so a later `assemble_all` call
    with a different `base_seed` never overwrites an earlier one's
    `data/histories/assembled/*.json` files -- the default reproduces the
    original Phase 3 dev-pilot ids unchanged.
    """
    h64i_tolerance = tolerance if h64i_tolerance is None else h64i_tolerance
    neutral_pool = load_neutral_pool()
    neutral_pool_by_id = {ep["episode_id"]: ep for ep in neutral_pool}

    h64n = build_h64n(
        neutral_pool,
        target_tokens=history_token_targets["H64N"],
        tolerance=tolerance,
        seed=base_seed,
        token_counter=token_counter,
        token_count_model=token_count_model,
        history_id=f"{id_prefix}_h64n_base",
    )
    validate_history(h64n)

    h16n = build_h16n(
        h64n,
        neutral_pool_by_id,
        target_tokens=history_token_targets["H16N"],
        tolerance=tolerance,
        token_counter=token_counter,
        token_count_model=token_count_model,
        history_id=f"{id_prefix}_h16n_base",
    )
    validate_history(h16n)
    assert h16n.episode_ids == h64n.episode_ids[-len(h16n.episode_ids) :], (
        "H16N must be exactly the tail of H64N's episode sequence"
    )

    histories = [h64n, h16n]
    for i, group in enumerate(TARGET_GROUPS):
        interference_pool = load_interference_pool(group)
        h64i = build_h64i(
            h64n,
            neutral_pool_by_id,
            interference_pool,
            target_group=group,
            interference_fraction=interference_fraction,
            tolerance=h64i_tolerance,
            seed=base_seed + i + 1,
            token_counter=token_counter,
            token_count_model=token_count_model,
            history_id=f"{id_prefix}_h64i_{group}",
        )
        validate_history(h64i)
        histories.append(h64i)

    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="HF repo id whose tokenizer + chat template is used to count history tokens",
    )
    parser.add_argument("--tolerance", type=float, default=0.02, help="Fractional tolerance around each token target")
    parser.add_argument(
        "--h64i-tolerance",
        type=float,
        default=0.05,
        help="Fractional tolerance for H64I vs its base H64N (looser than --tolerance: matching-based "
        "swaps drift more than plain trimming; the brief allows 'approximately +-2%% when practical')",
    )
    parser.add_argument(
        "--seed-key",
        default="base_seed",
        help="Which configs/experiment.yaml seeds.<key> to use, e.g. 'final_seed' for the Phase 4 "
        "frozen final-set histories (default: 'base_seed', the Phase 2/3 dev-pilot histories).",
    )
    parser.add_argument(
        "--id-prefix",
        default="history",
        help="history_id prefix (default 'history', matching the original dev-pilot ids). Pass "
        "'history_final' for Phase 4 so these don't overwrite the dev-pilot files.",
    )
    args = parser.parse_args()

    config = load_experiment_config()
    token_counter = make_hf_token_counter(args.model)

    histories = assemble_all(
        token_counter=token_counter,
        token_count_model=args.model,
        base_seed=config["seeds"][args.seed_key],
        history_token_targets=config["history_token_targets"],
        interference_fraction=config["interference_fraction"],
        tolerance=args.tolerance,
        h64i_tolerance=args.h64i_tolerance,
        id_prefix=args.id_prefix,
    )

    print(f"{'history_id':<28} {'condition':<7} {'group':<22} {'episodes':>8} {'messages':>9} {'tokens':>8}")
    for h in histories:
        path = write_history(h)
        print(
            f"{h.history_id:<28} {h.history_condition:<7} {(h.target_group or '-'):<22} "
            f"{len(h.episode_ids):>8} {h.n_messages:>9} {h.token_count:>8}  -> {path.relative_to(DATA_DIR.parent)}"
        )


if __name__ == "__main__":
    main()

import pytest

from tool_context_study.build_histories import (
    build_h16n,
    build_h64i,
    build_h64n,
    validate_history,
)
from tool_context_study.schemas import AssembledHistory

# A fast, deterministic stand-in for the real HF-tokenizer-based counter
# (word count), so these tests don't need `transformers` installed or a
# model download -- only the selection/assembly *logic* is under test here.
FAKE_TOKENS_PER_WORD = 1


def fake_token_counter(messages):
    return sum(len(m["content"].split()) for m in messages)


def _episode(episode_id: str, n_messages: int, words_per_message: int, **extra) -> dict:
    roles = ["user", "assistant"] * (n_messages // 2)
    messages = [{"role": r, "content": " ".join(f"w{i}" for i in range(words_per_message))} for r in roles]
    return {"episode_id": episode_id, "kind": "neutral", "topic": episode_id, "messages": messages, **extra}


def _neutral_pool(n: int, words_per_message: int = 50) -> list[dict]:
    return [_episode(f"neutral_{i:04d}", 4, words_per_message) for i in range(n)]


def _interference_pool(n: int, group: str, words_per_message: int = 50) -> list[dict]:
    return [
        _episode(
            f"interference_{group}_{i:04d}",
            4,
            words_per_message,
            kind="interference",
            target_domain="fixture_domain",
            target_group=group,
        )
        for i in range(n)
    ]


def test_build_h64n_lands_within_tolerance_and_trims_at_episode_boundaries():
    # 40 episodes x 200 words each (4 msgs x 50 words) = 8000 words total.
    pool = _neutral_pool(40)
    h64n = build_h64n(
        pool,
        target_tokens=1000,
        tolerance=0.05,
        seed=1,
        token_counter=fake_token_counter,
        token_count_model="fake",
    )
    assert 950 <= h64n.token_count <= 1050
    # No partial episode: token_count must equal the sum of whole selected episodes.
    assert h64n.token_count == 200 * len(h64n.episode_ids)
    assert h64n.n_messages == 4 * len(h64n.episode_ids)


def test_build_h64n_raises_when_pool_too_small():
    pool = _neutral_pool(3)  # only 600 words total, can't reach 1000
    with pytest.raises(ValueError):
        build_h64n(
            pool, target_tokens=1000, tolerance=0.02, seed=1, token_counter=fake_token_counter, token_count_model="fake"
        )


def test_build_h64n_is_reproducible_given_same_seed():
    pool = _neutral_pool(40)
    a = build_h64n(pool, target_tokens=1000, tolerance=0.05, seed=7, token_counter=fake_token_counter, token_count_model="fake")
    b = build_h64n(pool, target_tokens=1000, tolerance=0.05, seed=7, token_counter=fake_token_counter, token_count_model="fake")
    assert a.episode_ids == b.episode_ids
    assert a.token_count == b.token_count


def test_h16n_is_exact_tail_of_h64n():
    # Finer-grained episodes (40 words each) so a 250-word suffix target is
    # actually reachable within a tight tolerance.
    pool = _neutral_pool(60, words_per_message=10)  # 4 msgs x 10 words = 40 words/episode
    pool_by_id = {ep["episode_id"]: ep for ep in pool}
    h64n = build_h64n(pool, target_tokens=1000, tolerance=0.05, seed=3, token_counter=fake_token_counter, token_count_model="fake")

    h16n = build_h16n(
        h64n, pool_by_id, target_tokens=250, tolerance=0.1, token_counter=fake_token_counter, token_count_model="fake"
    )

    assert h16n.derived_from == h64n.history_id
    assert h64n.episode_ids[-len(h16n.episode_ids) :] == h16n.episode_ids
    assert 225 <= h16n.token_count <= 275  # 250 +-10%


def test_h64i_only_touches_the_matched_target_group_and_preserves_token_mass():
    pool = _neutral_pool(40)
    pool_by_id = {ep["episode_id"]: ep for ep in pool}
    h64n = build_h64n(pool, target_tokens=1000, tolerance=0.05, seed=5, token_counter=fake_token_counter, token_count_model="fake")

    interference_pool = _interference_pool(20, "weather")
    h64i = build_h64i(
        h64n,
        pool_by_id,
        interference_pool,
        target_group="weather",
        interference_fraction=0.25,
        tolerance=0.1,
        seed=11,
        token_counter=fake_token_counter,
        token_count_model="fake",
    )

    assert h64i.target_group == "weather"
    assert h64i.derived_from == h64n.history_id
    assert all(eid.startswith("interference_weather_") for eid in h64i.interference_episode_ids)
    assert len(h64i.interference_episode_ids) == len(h64i.replaced_neutral_episode_ids)
    # Roughly a quarter of the base history's episodes were swapped.
    frac = len(h64i.interference_episode_ids) / len(h64n.episode_ids)
    assert 0.15 <= frac <= 0.35
    # Token mass close to the original H64N (swaps are token-matched).
    assert abs(h64i.token_count - h64n.token_count) / h64n.token_count <= 0.1
    # The untouched episodes are exactly the H64N ones minus the replaced set.
    untouched = set(h64n.episode_ids) - set(h64i.replaced_neutral_episode_ids)
    assert untouched <= set(h64i.episode_ids)


def test_h64i_raises_when_interference_pool_too_small_to_avoid_reuse():
    # A bigger H64N (20 episodes) so 25% removal is unambiguously more than
    # the 1-episode interference pool we're about to hand it.
    pool = _neutral_pool(20)
    pool_by_id = {ep["episode_id"]: ep for ep in pool}
    h64n = build_h64n(pool, target_tokens=4000, tolerance=0.05, seed=5, token_counter=fake_token_counter, token_count_model="fake")

    tiny_interference_pool = _interference_pool(1, "weather")  # nowhere near 25% of ~20 episodes' worth
    with pytest.raises(ValueError):
        build_h64i(
            h64n,
            pool_by_id,
            tiny_interference_pool,
            target_group="weather",
            interference_fraction=0.25,
            tolerance=0.1,
            seed=11,
            token_counter=fake_token_counter,
            token_count_model="fake",
        )


def _bare_history(messages: list[dict], episode_ids: list[str] | None = None) -> AssembledHistory:
    return AssembledHistory(
        history_id="history_test",
        history_condition="H64N",
        seed=1,
        episode_ids=episode_ids or ["neutral_0000"],
        messages=messages,
        token_count=fake_token_counter(messages),
        token_count_model="fake",
        n_messages=len(messages),
    )


def test_validate_history_rejects_consecutive_same_role_messages():
    bad = _bare_history([{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}])
    with pytest.raises(AssertionError):
        validate_history(bad)


def test_validate_history_rejects_non_user_start():
    bad = _bare_history([{"role": "assistant", "content": "hi"}])
    with pytest.raises(AssertionError):
        validate_history(bad)


def test_validate_history_rejects_repeated_episode_id():
    bad = _bare_history(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        episode_ids=["neutral_0000", "neutral_0000"],
    )
    with pytest.raises(AssertionError):
        validate_history(bad)


def test_validate_history_accepts_well_formed_history():
    good = _bare_history([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    validate_history(good)  # should not raise

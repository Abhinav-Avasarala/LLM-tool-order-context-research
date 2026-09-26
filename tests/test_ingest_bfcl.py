from pathlib import Path

from tool_context_study import ingest_bfcl

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_questions_parses_fixture(monkeypatch):
    monkeypatch.setattr(ingest_bfcl, "RAW_BFCL_DIR", FIXTURES_DIR)

    questions = ingest_bfcl.load_questions("sample_bfcl_multiple.json", revision="test-rev")

    assert len(questions) == 2
    assert questions[0].id == "multiple_0"
    assert questions[0].source_file == "sample_bfcl_multiple.json"
    assert questions[0].source_revision == "test-rev"
    assert questions[0].function[0]["name"] == "triangle_properties.get"


def test_load_answers_parses_fixture(monkeypatch):
    monkeypatch.setattr(ingest_bfcl, "RAW_BFCL_DIR", FIXTURES_DIR)

    answers = ingest_bfcl.load_answers("sample_bfcl_multiple_answers.json", revision="test-rev")

    assert len(answers) == 2
    assert answers[0].id == "multiple_0"
    assert answers[0].ground_truth[0]["triangle_properties.get"]["side1"] == [5]


def test_load_questions_with_answers_joins_by_id(monkeypatch):
    monkeypatch.setattr(ingest_bfcl, "RAW_BFCL_DIR", FIXTURES_DIR)

    pairs = ingest_bfcl.load_questions_with_answers(
        "sample_bfcl_multiple.json", "sample_bfcl_multiple_answers.json", revision="test-rev"
    )

    assert len(pairs) == 2
    ids = {q.id for q, _ in pairs}
    assert ids == {"multiple_0", "multiple_2"}
    for q, a in pairs:
        assert q.id == a.id


def test_load_questions_with_answers_skips_and_warns_on_missing_ground_truth(monkeypatch, capsys):
    monkeypatch.setattr(ingest_bfcl, "RAW_BFCL_DIR", FIXTURES_DIR)

    pairs = ingest_bfcl.load_questions_with_answers(
        "sample_bfcl_mismatched.json", "sample_bfcl_mismatched_answers.json", revision="test-rev"
    )

    # multiple_1 has no matching ground truth in the mismatched-answers fixture.
    assert len(pairs) == 1
    assert pairs[0][0].id == "multiple_0"

    captured = capsys.readouterr()
    assert "multiple_1" in captured.out
    assert "Warning" in captured.out

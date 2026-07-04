import unicodedata

import pytest

from judgekit import scorers
from judgekit.errors import ScoringError, UnknownScorerError
from judgekit.models import Case
from judgekit.scorers import (
    ExactScorer,
    RegexScorer,
    StructuredScorer,
    available_scorers,
    get_scorer,
)


def _case(case_id: str, reference: str | None) -> Case:
    return Case(id=case_id, input="q", reference=reference, dataset_version="sha256:test")


def test_get_scorer_returns_matching_ids() -> None:
    assert get_scorer("exact").scorer_id == "exact"
    assert get_scorer("regex").scorer_id == "regex"
    assert get_scorer("structured").scorer_id == "structured"


def test_get_scorer_unknown_id_lists_available() -> None:
    with pytest.raises(UnknownScorerError) as exc_info:
        get_scorer("nope")
    message = str(exc_info.value)
    assert "exact" in message
    assert "regex" in message
    assert "structured" in message


def test_register_duplicate_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_registry: dict[str, scorers.Scorer] = {"exact": get_scorer("exact")}
    monkeypatch.setattr(scorers, "_REGISTRY", fresh_registry)

    class DummyScorer:
        scorer_id = "exact"
        requires_reference = True

        def score(self, case: Case, output: str) -> float:
            return 0.0

    with pytest.raises(ValueError, match="exact"):
        scorers.register(DummyScorer())


def test_available_scorers_sorted() -> None:
    assert available_scorers() == ("exact", "regex", "structured")


def test_exact_equal_strings() -> None:
    scorer = ExactScorer()
    case = _case("c1", "Paris")
    assert scorer.score(case, "Paris") == 1.0


def test_exact_unequal_strings() -> None:
    scorer = ExactScorer()
    case = _case("c1", "Paris")
    assert scorer.score(case, "London") == 0.0


def test_exact_trailing_space_not_stripped() -> None:
    scorer = ExactScorer()
    case = _case("c1", "Paris")
    assert scorer.score(case, "Paris ") == 0.0


def test_exact_nfc_vs_nfd_not_normalized() -> None:
    scorer = ExactScorer()
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd
    case = _case("c1", nfc)
    assert scorer.score(case, nfd) == 0.0


def test_exact_none_reference_raises() -> None:
    scorer = ExactScorer()
    case = _case("c1", None)
    with pytest.raises(ScoringError, match="c1"):
        scorer.score(case, "anything")


def test_regex_matches_inside_longer_sentence() -> None:
    scorer = RegexScorer()
    case = _case("c1", r"\d{4}-\d{2}-\d{2}")
    assert scorer.score(case, "the event was logged on 2026-07-04 in the ledger") == 1.0


def test_regex_anchored_exact_match() -> None:
    scorer = RegexScorer()
    case = _case("c1", "^Paris$")
    assert scorer.score(case, "Paris") == 1.0


def test_regex_anchored_no_match_with_extra_text() -> None:
    scorer = RegexScorer()
    case = _case("c1", "^Paris$")
    assert scorer.score(case, "Paris, France") == 0.0


def test_regex_non_match() -> None:
    scorer = RegexScorer()
    case = _case("c1", r"\d{4}-\d{2}-\d{2}")
    assert scorer.score(case, "no dates here") == 0.0


def test_regex_invalid_pattern_raises() -> None:
    scorer = RegexScorer()
    case = _case("c1", "(")
    with pytest.raises(ScoringError, match="c1"):
        scorer.score(case, "anything")


def test_regex_none_reference_raises() -> None:
    scorer = RegexScorer()
    case = _case("c1", None)
    with pytest.raises(ScoringError, match="c1"):
        scorer.score(case, "anything")


def test_structured_key_order_and_whitespace_irrelevant() -> None:
    scorer = StructuredScorer()
    case = _case("c1", '{"a": 1, "b": [1, 2]}')
    assert scorer.score(case, '{ "b":[1,2], "a": 1 }') == 1.0


def test_structured_int_vs_float_equal_semantics() -> None:
    scorer = StructuredScorer()
    case = _case("c1", '{"a": 1}')
    assert scorer.score(case, '{"a": 1.0}') == 1.0


def test_structured_unparseable_output_scores_zero() -> None:
    scorer = StructuredScorer()
    case = _case("c1", '{"a": 1}')
    assert scorer.score(case, "not json") == 0.0


def test_structured_unparseable_reference_raises() -> None:
    scorer = StructuredScorer()
    case = _case("c1", "not json")
    with pytest.raises(ScoringError, match="c1"):
        scorer.score(case, '{"a": 1}')


def test_structured_parsed_but_unequal() -> None:
    scorer = StructuredScorer()
    case = _case("c1", '{"a": 1}')
    assert scorer.score(case, '{"a": 2}') == 0.0


def test_structured_none_reference_raises() -> None:
    scorer = StructuredScorer()
    case = _case("c1", None)
    with pytest.raises(ScoringError, match="c1"):
        scorer.score(case, "anything")

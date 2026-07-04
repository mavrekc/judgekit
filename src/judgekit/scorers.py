"""Deterministic scorers: exact match, regex match, and structured (JSON) equality."""

import json
import re
from typing import Protocol

from judgekit.errors import ScoringError, UnknownScorerError
from judgekit.models import Case


class Scorer(Protocol):
    """Structural interface for a deterministic case scorer."""

    scorer_id: str
    requires_reference: bool

    def score(self, case: Case, output: str) -> float: ...


def _require_reference(case: Case) -> str:
    """Return case.reference, or raise ScoringError naming the case id."""
    if case.reference is None:
        raise ScoringError(f"case {case.id!r} has no reference to score against")
    return case.reference


class ExactScorer:
    """Scores 1.0 iff output is exactly equal to the reference string."""

    scorer_id = "exact"
    requires_reference = True

    def score(self, case: Case, output: str) -> float:
        reference = _require_reference(case)
        return 1.0 if output == reference else 0.0


class RegexScorer:
    """Scores 1.0 iff the reference, treated as a regex, matches anywhere in output."""

    scorer_id = "regex"
    requires_reference = True

    def score(self, case: Case, output: str) -> float:
        pattern = _require_reference(case)
        try:
            match = re.search(pattern, output)
        except re.error as exc:
            raise ScoringError(f"case {case.id!r} has an invalid regex pattern") from exc
        return 1.0 if match else 0.0


class StructuredScorer:
    """Scores 1.0 iff output and reference parse as JSON to deep-equal values."""

    scorer_id = "structured"
    requires_reference = True

    def score(self, case: Case, output: str) -> float:
        reference = _require_reference(case)
        try:
            expected = json.loads(reference)
        except json.JSONDecodeError as exc:
            raise ScoringError(f"case {case.id!r} has an unparseable reference") from exc
        try:
            actual = json.loads(output)
        except json.JSONDecodeError:
            return 0.0
        return 1.0 if expected == actual else 0.0


_REGISTRY: dict[str, Scorer] = {}


def register(scorer: Scorer) -> None:
    """Add a scorer to the registry; raise if its id is already taken."""
    if scorer.scorer_id in _REGISTRY:
        raise ValueError(f"scorer id already registered: {scorer.scorer_id!r}")
    _REGISTRY[scorer.scorer_id] = scorer


def get_scorer(scorer_id: str) -> Scorer:
    """Look up a registered scorer by id, raising UnknownScorerError if absent."""
    if scorer_id not in _REGISTRY:
        available = ", ".join(available_scorers())
        raise UnknownScorerError(f"unknown scorer id: {scorer_id!r}; available: {available}")
    return _REGISTRY[scorer_id]


def available_scorers() -> tuple[str, ...]:
    """Return the sorted ids of all registered scorers."""
    return tuple(sorted(_REGISTRY))


register(ExactScorer())
register(RegexScorer())
register(StructuredScorer())

import json
from datetime import timedelta
from pathlib import Path

import pytest

from judgekit.errors import RunError, ScoringError
from judgekit.models import Case, Dataset, RunManifest, RunResult
from judgekit.runner import execute_run, write_artifact
from judgekit.scorers import get_scorer

DATASET_VERSION = "sha256:" + "0" * 64


def _case(case_id: str, reference: str | None = None) -> Case:
    return Case(id=case_id, input="q", reference=reference, dataset_version=DATASET_VERSION)


def _dataset(*cases: Case) -> Dataset:
    return Dataset(dataset_version=DATASET_VERSION, cases=tuple(cases))


class FakeScorer:
    """A minimal test double implementing the Scorer protocol."""

    scorer_id = "fake"

    def __init__(self, requires_reference: bool = False) -> None:
        self.requires_reference = requires_reference
        self.calls: list[str] = []

    def score(self, case: Case, output: str) -> float:
        self.calls.append(case.id)
        return 1.0 if output == "ok" else 0.0


class RaisingScorer:
    """A scorer whose score() always raises ScoringError."""

    scorer_id = "raising"
    requires_reference = False

    def score(self, case: Case, output: str) -> float:
        raise ScoringError(f"boom on {case.id}")


def test_happy_path_exact_scorer() -> None:
    cases = [_case("c1", "yes"), _case("c2", "yes"), _case("c3", "yes"), _case("c4", "yes")]
    dataset = _dataset(*cases)
    outputs = {"c1": "yes", "c2": "yes", "c3": "yes", "c4": "no"}

    artifact = execute_run(
        dataset,
        outputs,
        get_scorer("exact"),
        dataset_path="data/cases.jsonl",
        outputs_path="data/outputs.jsonl",
    )

    assert [r.case_id for r in artifact.results] == ["c1", "c2", "c3", "c4"]
    assert [r.score for r in artifact.results] == [1.0, 1.0, 1.0, 0.0]
    run_ids = {r.run_id for r in artifact.results}
    assert len(run_ids) == 1
    run_id = run_ids.pop()
    assert len(run_id) == 32
    for result in artifact.results:
        assert result.cost == 0.0
        assert result.cached is False
        assert result.judge_config_id is None
        assert result.scorer_id == "exact"
        assert result.dataset_version == dataset.dataset_version
    assert artifact.manifest.summary.n_cases == 4
    assert artifact.manifest.summary.mean_score == 0.75
    assert artifact.manifest.summary.pass_rate == 0.75


def test_missing_outputs_raises_before_scoring() -> None:
    cases = [_case("c1"), _case("c2"), _case("c3")]
    dataset = _dataset(*cases)
    outputs = {"c1": "ok"}
    scorer = FakeScorer()

    with pytest.raises(RunError) as exc_info:
        execute_run(dataset, outputs, scorer, dataset_path="d.jsonl", outputs_path="o.jsonl")

    message = str(exc_info.value)
    assert "c2" in message
    assert "c3" in message
    assert scorer.calls == []


def test_missing_references_for_requires_reference_scorer() -> None:
    cases = [_case("c1", "ref"), _case("c2", None), _case("c3", None)]
    dataset = _dataset(*cases)
    outputs = {"c1": "ref", "c2": "x", "c3": "y"}

    with pytest.raises(RunError) as exc_info:
        execute_run(
            dataset,
            outputs,
            get_scorer("exact"),
            dataset_path="d.jsonl",
            outputs_path="o.jsonl",
        )

    message = str(exc_info.value)
    assert "c2" in message
    assert "c3" in message


def test_scoring_error_propagates() -> None:
    dataset = _dataset(_case("c1"), _case("c2"))
    outputs = {"c1": "ok", "c2": "ok"}

    with pytest.raises(ScoringError):
        execute_run(
            dataset,
            outputs,
            RaisingScorer(),
            dataset_path="d.jsonl",
            outputs_path="o.jsonl",
        )


def test_scorer_not_requiring_reference_skips_reference_check() -> None:
    dataset = _dataset(_case("c1", None), _case("c2", None))
    outputs = {"c1": "ok", "c2": "no"}
    scorer = FakeScorer(requires_reference=False)

    artifact = execute_run(dataset, outputs, scorer, dataset_path="d.jsonl", outputs_path="o.jsonl")

    assert [r.score for r in artifact.results] == [1.0, 0.0]


def test_artifact_round_trip(tmp_path: Path) -> None:
    dataset = _dataset(_case("c1", "yes"), _case("c2", "yes"))
    outputs = {"c1": "yes", "c2": "no"}
    artifact = execute_run(
        dataset,
        outputs,
        get_scorer("exact"),
        dataset_path="data/cases.jsonl",
        outputs_path="data/outputs.jsonl",
    )
    out_path = tmp_path / "run.jsonl"

    write_artifact(artifact, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == artifact.manifest.summary.n_cases + 1
    manifest = RunManifest.model_validate(json.loads(lines[0]))
    assert manifest.kind == "manifest"
    assert manifest.schema_version == 1
    assert manifest.created_at.tzinfo is not None
    assert manifest.created_at.utcoffset() == timedelta(0)
    for line in lines[1:]:
        RunResult.model_validate(json.loads(line))


def test_write_artifact_refuses_existing_path(tmp_path: Path) -> None:
    dataset = _dataset(_case("c1", "yes"))
    artifact = execute_run(
        dataset,
        {"c1": "yes"},
        get_scorer("exact"),
        dataset_path="d.jsonl",
        outputs_path="o.jsonl",
    )
    out_path = tmp_path / "run.jsonl"
    out_path.write_text("existing", encoding="utf-8")

    with pytest.raises(RunError):
        write_artifact(artifact, out_path)


def test_write_artifact_creates_missing_parent_directories(tmp_path: Path) -> None:
    dataset = _dataset(_case("c1", "yes"))
    artifact = execute_run(
        dataset,
        {"c1": "yes"},
        get_scorer("exact"),
        dataset_path="d.jsonl",
        outputs_path="o.jsonl",
    )
    out_path = tmp_path / "a" / "b" / "run.jsonl"

    write_artifact(artifact, out_path)

    assert out_path.exists()


def test_manifest_paths_match_inputs() -> None:
    dataset = _dataset(_case("c1", "yes"))
    artifact = execute_run(
        dataset,
        {"c1": "yes"},
        get_scorer("exact"),
        dataset_path="some/data.jsonl",
        outputs_path="some/outputs.jsonl",
    )

    assert artifact.manifest.dataset_path == "some/data.jsonl"
    assert artifact.manifest.outputs_path == "some/outputs.jsonl"


def test_zero_cases_raises_run_error() -> None:
    with pytest.raises(RunError):
        execute_run(
            _dataset(),
            {},
            FakeScorer(),
            dataset_path="d.jsonl",
            outputs_path="o.jsonl",
        )


def test_extra_output_keys_are_ignored() -> None:
    dataset = _dataset(_case("c1"))
    outputs = {"c1": "ok", "unrelated": "ignored"}

    artifact = execute_run(
        dataset,
        outputs,
        FakeScorer(),
        dataset_path="d.jsonl",
        outputs_path="o.jsonl",
    )

    assert [r.case_id for r in artifact.results] == ["c1"]

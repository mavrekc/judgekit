from datetime import datetime

import pytest
from pydantic import ValidationError

from judgekit.models import (
    Case,
    CaseRecord,
    OutputRecord,
    RunManifest,
    RunResult,
    RunSummary,
)


def test_case_record_all_fields() -> None:
    record = CaseRecord(
        id="c1",
        input="What is 2+2?",
        reference="4",
        human_label=1,
        metadata={"difficulty": "easy"},
    )
    assert record.id == "c1"
    assert record.reference == "4"
    assert record.human_label == 1
    assert record.metadata == {"difficulty": "easy"}


def test_case_record_defaults() -> None:
    record = CaseRecord(id="c1", input="What is 2+2?")
    assert record.reference is None
    assert record.human_label is None
    assert record.metadata == {}


def test_case_record_rejects_unknown_extra_key() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", unknown_field="x")


def test_case_record_rejects_dataset_version_key() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", dataset_version="sha256:abc")


def test_case_record_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="", input="q")


def test_case_requires_dataset_version() -> None:
    with pytest.raises(ValidationError):
        Case(id="c1", input="q")  # type: ignore[call-arg]


def test_case_carries_inherited_fields() -> None:
    case = Case(id="c1", input="q", reference="a", dataset_version="sha256:abc")
    assert case.id == "c1"
    assert case.reference == "a"
    assert case.dataset_version == "sha256:abc"


def test_metadata_rejects_nested_dict() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", metadata={"a": {"b": 1}})


def test_metadata_accepts_scalar_types() -> None:
    record = CaseRecord(
        id="c1",
        input="q",
        metadata={"a": "s", "b": 1, "c": 1.5, "d": True},
    )
    assert record.metadata == {"a": "s", "b": 1, "c": 1.5, "d": True}


def test_output_record_ignores_extra_keys() -> None:
    output = OutputRecord(case_id="c1", output="answer", extra_field="ignored")  # type: ignore[call-arg]
    assert "extra_field" not in output.model_dump()


def _run_result_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": "r1",
        "dataset_version": "sha256:abc",
        "scorer_id": "exact_match",
        "case_id": "c1",
        "score": 0.5,
        "cost": 0.0,
        "cached": False,
    }
    base.update(overrides)
    return base


def test_run_result_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        RunResult(**_run_result_kwargs(score=1.1))  # type: ignore[arg-type]


def test_run_result_rejects_score_below_zero() -> None:
    with pytest.raises(ValidationError):
        RunResult(**_run_result_kwargs(score=-0.1))  # type: ignore[arg-type]


def test_run_result_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        RunResult(**_run_result_kwargs(cost=-1.0))  # type: ignore[arg-type]


def test_run_result_judge_config_id_defaults_to_none() -> None:
    result = RunResult(**_run_result_kwargs())  # type: ignore[arg-type]
    assert result.judge_config_id is None


def test_case_record_is_frozen() -> None:
    record = CaseRecord(id="c1", input="q")
    with pytest.raises(ValidationError):
        record.input = "changed"  # type: ignore[misc]


def test_run_result_is_frozen() -> None:
    result = RunResult(**_run_result_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        result.score = 0.9  # type: ignore[misc]


def test_run_manifest_defaults() -> None:
    manifest = RunManifest(
        run_id="r1",
        created_at=datetime(2026, 7, 4),
        dataset_path="data/cases.jsonl",
        dataset_version="sha256:abc",
        outputs_path="data/outputs.jsonl",
        scorer_id="exact_match",
        summary=RunSummary(n_cases=1, mean_score=0.5, pass_rate=1.0),
    )
    assert manifest.kind == "manifest"
    assert manifest.schema_version == 1

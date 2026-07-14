from datetime import datetime

import pytest
from pydantic import ValidationError

from judgekit.models import (
    AgreementEstimate,
    CalibrationReport,
    Case,
    CaseRecord,
    JudgeConfigRecord,
    JudgeParams,
    JudgeRunManifest,
    JudgeTotals,
    JudgeVerdict,
    KappaEstimate,
    OutputRecord,
    RaterSummary,
    RatingRecord,
    RunManifest,
    RunResult,
    RunSummary,
    SliceAgreement,
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


def test_case_record_rejects_nan_human_label() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", human_label=float("nan"))


def test_case_record_rejects_infinity_human_label() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", human_label=float("inf"))


def test_case_record_rejects_nan_in_metadata() -> None:
    with pytest.raises(ValidationError):
        CaseRecord(id="c1", input="q", metadata={"score": float("nan")})


def test_case_record_still_accepts_valid_metadata_and_label() -> None:
    record = CaseRecord(
        id="c1",
        input="q",
        human_label=1.5,
        metadata={"difficulty": "hard", "weight": 2.0},
    )
    assert record.human_label == 1.5
    assert record.metadata == {"difficulty": "hard", "weight": 2.0}


@pytest.mark.parametrize("label", ["yes", 1, 0.5, True])
def test_rating_record_accepts_each_label_type(label: object) -> None:
    record = RatingRecord(case_id="c1", label=label)  # type: ignore[arg-type]
    assert record.label == label


def test_rating_record_requires_label() -> None:
    with pytest.raises(ValidationError):
        RatingRecord(case_id="c1")  # type: ignore[call-arg]


def test_rating_record_rejects_empty_case_id() -> None:
    with pytest.raises(ValidationError):
        RatingRecord(case_id="", label="yes")


def test_rating_record_ignores_extra_keys() -> None:
    rating = RatingRecord(case_id="c1", label="yes", extra_field="ignored")  # type: ignore[call-arg]
    assert "extra_field" not in rating.model_dump()


def test_rating_record_is_frozen() -> None:
    rating = RatingRecord(case_id="c1", label="yes")
    with pytest.raises(ValidationError):
        rating.label = "no"  # type: ignore[misc]


def test_rating_record_rejects_nan_label() -> None:
    with pytest.raises(ValidationError):
        RatingRecord(case_id="c1", label=float("nan"))


def test_rating_record_rejects_infinity_label() -> None:
    with pytest.raises(ValidationError):
        RatingRecord(case_id="c1", label=float("inf"))


def _calibration_report_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "report_id": "rep1",
        "created_at": datetime(2026, 7, 9),
        "dataset_path": "data/cases.jsonl",
        "dataset_version": "sha256:abc",
        "level": "nominal",
        "raters": (
            RaterSummary(
                rater_id="human",
                ratings_path="data/human.jsonl",
                n_ratings=10,
                n_matched=10,
            ),
        ),
        "n_cases": 10,
        "n_labeled": 10,
        "bootstrap_seed": 0,
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
        "alpha": AgreementEstimate(
            value=0.8, ci_low=0.6, ci_high=0.9, n_used=10, n_resamples_used=1000
        ),
    }
    base.update(overrides)
    return base


def test_calibration_report_with_kappa_roundtrips() -> None:
    report = CalibrationReport(
        **_calibration_report_kwargs(
            kappa=KappaEstimate(
                value=0.7,
                ci_low=0.5,
                ci_high=0.85,
                n_used=10,
                n_resamples_used=1000,
                percent_agreement=0.9,
            ),
            confusion={"true": {"true": 8, "false": 1}, "false": {"true": 1, "false": 0}},
        )
    )  # type: ignore[arg-type]
    restored = CalibrationReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.kappa is not None
    assert restored.kappa.percent_agreement == 0.9


def test_calibration_report_without_kappa_roundtrips() -> None:
    report = CalibrationReport(**_calibration_report_kwargs())  # type: ignore[arg-type]
    restored = CalibrationReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.kappa is None


def test_calibration_report_rejects_extra_key() -> None:
    with pytest.raises(ValidationError):
        CalibrationReport(**_calibration_report_kwargs(unknown_field="x"))  # type: ignore[arg-type]


def test_calibration_report_accepts_negative_alpha_value() -> None:
    report = CalibrationReport(
        **_calibration_report_kwargs(
            alpha=AgreementEstimate(
                value=-0.2, ci_low=-0.5, ci_high=0.1, n_used=10, n_resamples_used=1000
            )
        )
    )  # type: ignore[arg-type]
    assert report.alpha.value == -0.2


@pytest.mark.parametrize("confidence_level", [0.0, 1.0])
def test_calibration_report_rejects_boundary_confidence_level(confidence_level: float) -> None:
    with pytest.raises(ValidationError):
        CalibrationReport(**_calibration_report_kwargs(confidence_level=confidence_level))  # type: ignore[arg-type]


def test_calibration_report_rejects_negative_n_cases() -> None:
    with pytest.raises(ValidationError):
        CalibrationReport(**_calibration_report_kwargs(n_cases=-1))  # type: ignore[arg-type]


def test_slice_agreement_valid() -> None:
    slice_agreement = SliceAgreement(
        n=5,
        percent_agreement=0.8,
        confusion={"true": {"true": 4, "false": 1}},
    )
    assert slice_agreement.n == 5


def test_judge_params_defaults() -> None:
    params = JudgeParams()
    assert params.temperature == 0.0
    assert params.max_tokens == 256
    assert params.top_p is None
    assert params.stop == ()


def test_judge_params_rejects_negative_temperature() -> None:
    with pytest.raises(ValidationError):
        JudgeParams(temperature=-0.1)


def test_judge_params_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValidationError):
        JudgeParams(max_tokens=0)


@pytest.mark.parametrize("top_p", [0.0, 1.1])
def test_judge_params_rejects_top_p_out_of_range(top_p: float) -> None:
    with pytest.raises(ValidationError):
        JudgeParams(top_p=top_p)


def _judge_config_record_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "example-judge",
        "provider": "anthropic",
        "model": "test-model",
        "rubric": "Rate this. $input",
        "labels": ("good", "bad"),
    }
    base.update(overrides)
    return base


def test_judge_config_record_rejects_human_id() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(id="human"))  # type: ignore[arg-type]


def test_judge_config_record_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(id=""))  # type: ignore[arg-type]


def test_judge_config_record_rejects_bad_provider() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(provider="openai"))  # type: ignore[arg-type]


def test_judge_config_record_rejects_single_label() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(labels=("good",)))  # type: ignore[arg-type]


def test_judge_config_record_rejects_nan_label() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(labels=(float("nan"), "bad")))  # type: ignore[arg-type]


def test_judge_config_record_accepts_int_and_bool_labels() -> None:
    record = JudgeConfigRecord(**_judge_config_record_kwargs(labels=(1, True)))  # type: ignore[arg-type]
    assert record.labels == (1, True)


def test_judge_config_record_rejects_rubric_without_input() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(rubric="Rate this reply."))  # type: ignore[arg-type]


def test_judge_config_record_rejects_rubric_with_unknown_placeholder() -> None:
    with pytest.raises(ValidationError, match="output"):
        JudgeConfigRecord(**_judge_config_record_kwargs(rubric="Rate $input against $output."))  # type: ignore[arg-type]


def test_judge_config_record_rejects_invalid_template_syntax() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(rubric="judge ${input"))  # type: ignore[arg-type]


def test_judge_config_record_rejects_base_url_with_anthropic() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(
            **_judge_config_record_kwargs(provider="anthropic", base_url="https://example.com")
        )  # type: ignore[arg-type]


def test_judge_config_record_accepts_base_url_with_openai_compatible() -> None:
    record = JudgeConfigRecord(
        **_judge_config_record_kwargs(provider="openai-compatible", base_url="https://example.com")
    )  # type: ignore[arg-type]
    assert record.base_url == "https://example.com"


def test_judge_config_record_rejects_unknown_extra_key() -> None:
    with pytest.raises(ValidationError):
        JudgeConfigRecord(**_judge_config_record_kwargs(unknown_field="x"))  # type: ignore[arg-type]


def _judge_verdict_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "c1",
        "label": "good",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.01,
        "cached": False,
        "n_attempts": 1,
    }
    base.update(overrides)
    return base


def test_judge_verdict_rejects_zero_attempts() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(**_judge_verdict_kwargs(n_attempts=0))  # type: ignore[arg-type]


def test_judge_verdict_rejects_negative_input_tokens() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(**_judge_verdict_kwargs(input_tokens=-1))  # type: ignore[arg-type]


def test_judge_verdict_rejects_negative_output_tokens() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(**_judge_verdict_kwargs(output_tokens=-1))  # type: ignore[arg-type]


def test_judge_verdict_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(**_judge_verdict_kwargs(cost=-0.1))  # type: ignore[arg-type]


def test_judge_verdict_rejects_nan_label() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(**_judge_verdict_kwargs(label=float("nan")))  # type: ignore[arg-type]


def test_judge_run_manifest_defaults() -> None:
    manifest = JudgeRunManifest(
        run_id="r1",
        created_at=datetime(2026, 7, 14),
        dataset_path="data/cases.jsonl",
        dataset_version="sha256:abc",
        judge_config_path="configs/judge.json",
        judge_config_id="example-judge",
        judge_config_hash="sha256:def",
        provider="anthropic",
        model="test-model",
        cache_dir=".cache/judge",
        totals=JudgeTotals(
            n_cases=1, n_cached=0, n_live=1, input_tokens=10, output_tokens=5, cost=0.01
        ),
    )
    assert manifest.kind == "judge_run"
    assert manifest.schema_version == 1
    assert manifest.max_cost is None

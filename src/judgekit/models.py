"""Core pydantic models for datasets, outputs, and run artifacts."""

import math
from datetime import datetime
from string import Template
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetaValue = str | int | float | bool
Label = str | int | float | bool


def _reject_non_finite(value: str | int | float | bool) -> None:
    """Reject NaN/Infinity, which json.loads accepts but which break label equality."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("must be a finite number, got NaN or Infinity")


class CaseRecord(BaseModel):
    """A single evaluation case, independent of dataset versioning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    input: str
    reference: str | None = None
    human_label: Label | None = None
    metadata: dict[str, MetaValue] = Field(default_factory=dict)

    @field_validator("human_label")
    @classmethod
    def _human_label_finite(cls, value: Label | None) -> Label | None:
        if value is not None:
            _reject_non_finite(value)
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_finite(cls, value: dict[str, MetaValue]) -> dict[str, MetaValue]:
        for item in value.values():
            _reject_non_finite(item)
        return value


class Case(CaseRecord):
    """A CaseRecord bound to the dataset version it was hashed into."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str


class Dataset(BaseModel):
    """A versioned collection of cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str
    cases: tuple[Case, ...]


class OutputRecord(BaseModel):
    """A judged system output, loaded from an external outputs file."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    case_id: str = Field(min_length=1)
    output: str


class RatingRecord(BaseModel):
    """One rater's label for one case, loaded from an external ratings file."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    case_id: str = Field(min_length=1)
    label: Label

    @field_validator("label")
    @classmethod
    def _label_finite(cls, value: Label) -> Label:
        _reject_non_finite(value)
        return value


class RaterSummary(BaseModel):
    """How one rater's file matched up against the dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rater_id: str = Field(min_length=1)
    ratings_path: str
    n_ratings: int = Field(ge=0)
    n_matched: int = Field(ge=0)


class AgreementEstimate(BaseModel):
    """A statistic with its bootstrap confidence interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    ci_low: float
    ci_high: float
    n_used: int = Field(ge=0)
    n_resamples_used: int = Field(ge=0)


class KappaEstimate(AgreementEstimate):
    """Cohen's kappa plus the raw percent agreement behind it."""

    percent_agreement: float


class SliceAgreement(BaseModel):
    """Two-rater agreement counts within one metadata slice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=0)
    percent_agreement: float
    confusion: dict[str, dict[str, int]]


class CalibrationReport(BaseModel):
    """Agreement statistics between human labels and one or more raters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["calibration_report"] = "calibration_report"
    schema_version: int = 1
    report_id: str = Field(min_length=1)
    created_at: datetime
    dataset_path: str
    dataset_version: str = Field(min_length=1)
    judge_config_id: str | None = None
    level: Literal["nominal", "ordinal", "interval"]
    raters: tuple[RaterSummary, ...]
    n_cases: int = Field(ge=0)
    n_labeled: int = Field(ge=0)
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_level: float = Field(gt=0, lt=1)
    kappa: KappaEstimate | None = None
    alpha: AgreementEstimate
    confusion: dict[str, dict[str, int]] | None = None
    confusion_by_slice: dict[str, SliceAgreement] | None = None


class RunResult(BaseModel):
    """The score produced by one scorer for one case in one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    dataset_version: str
    judge_config_id: str | None = None
    scorer_id: str
    case_id: str
    score: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    cached: bool


class RunSummary(BaseModel):
    """Aggregate statistics over a run's results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_cases: int
    mean_score: float
    pass_rate: float


class RunManifest(BaseModel):
    """Metadata describing a run, stored alongside its results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["manifest"] = "manifest"
    schema_version: int = 1
    run_id: str
    created_at: datetime
    dataset_path: str
    dataset_version: str
    outputs_path: str
    scorer_id: str
    summary: RunSummary


class RunArtifact(BaseModel):
    """A run manifest paired with its full set of results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: RunManifest
    results: tuple[RunResult, ...]


class JudgeParams(BaseModel):
    """Sampling parameters sent to the judge model on every call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=256, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: tuple[str, ...] = ()


class JudgePricing(BaseModel):
    """User-declared prices per million tokens; judgekit ships no price table by design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class JudgeConfigRecord(BaseModel):
    """A judge's rubric, model, and labels, independent of its version hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    provider: Literal["anthropic", "openai-compatible"]
    model: str = Field(min_length=1)
    rubric: str
    labels: tuple[Label, ...] = Field(min_length=2)
    params: JudgeParams = Field(default_factory=JudgeParams)
    pricing: JudgePricing | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_label_attempts: int = Field(default=3, ge=1)

    @field_validator("id")
    @classmethod
    def _id_not_human(cls, value: str) -> str:
        if value == "human":
            raise ValueError('"human" is the reserved anchor rater id in calibration reports')
        return value

    @field_validator("rubric")
    @classmethod
    def _rubric_valid_template(cls, value: str) -> str:
        template = Template(value)
        if not template.is_valid():
            raise ValueError("invalid rubric template")
        identifiers = set(template.get_identifiers())
        if "input" not in identifiers:
            raise ValueError("rubric must reference $input")
        unknown = sorted(identifiers - {"input", "reference"})
        if unknown:
            raise ValueError(f"unknown rubric placeholders: {', '.join(unknown)}")
        return value

    @field_validator("labels")
    @classmethod
    def _labels_finite(cls, value: tuple[Label, ...]) -> tuple[Label, ...]:
        for item in value:
            _reject_non_finite(item)
        return value

    @model_validator(mode="after")
    def _base_url_requires_openai_compatible(self) -> "JudgeConfigRecord":
        if self.base_url is not None and self.provider != "openai-compatible":
            raise ValueError("base_url is only supported for the openai-compatible provider")
        return self


class JudgeConfig(JudgeConfigRecord):
    """A JudgeConfigRecord bound to the version hash computed from its behavior fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version_hash: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    """A single judge label for one case, with token and cost accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    label: Label
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)
    cached: bool
    n_attempts: int = Field(ge=1)

    @field_validator("label")
    @classmethod
    def _label_finite(cls, value: Label) -> Label:
        _reject_non_finite(value)
        return value


class JudgeTotals(BaseModel):
    """Aggregate token, cache, and cost accounting over a judge run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_cases: int = Field(ge=0)
    n_cached: int = Field(ge=0)
    n_live: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)


class JudgeRunManifest(BaseModel):
    """Metadata describing a judge run, stored alongside its verdicts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["judge_run"] = "judge_run"
    schema_version: int = 1
    run_id: str
    created_at: datetime
    dataset_path: str
    dataset_version: str
    judge_config_path: str
    judge_config_id: str
    judge_config_hash: str
    provider: str
    model: str
    cache_dir: str
    max_cost: float | None = None
    totals: JudgeTotals


class JudgeRunArtifact(BaseModel):
    """A judge run manifest paired with its full set of verdicts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: JudgeRunManifest
    verdicts: tuple[JudgeVerdict, ...]

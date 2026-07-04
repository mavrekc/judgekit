"""Core pydantic models for datasets, outputs, and run artifacts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MetaValue = str | int | float | bool
Label = str | int | float | bool


class CaseRecord(BaseModel):
    """A single evaluation case, independent of dataset versioning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    input: str
    reference: str | None = None
    human_label: Label | None = None
    metadata: dict[str, MetaValue] = Field(default_factory=dict)


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

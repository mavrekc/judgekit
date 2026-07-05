"""judgekit: evaluate your evaluators, with judge-vs-human validity as the headline metric."""

from judgekit.dataset import load_dataset, load_outputs
from judgekit.errors import (
    DatasetError,
    JudgekitError,
    RunError,
    ScoringError,
    UnknownScorerError,
)
from judgekit.models import (
    Case,
    CaseRecord,
    Dataset,
    OutputRecord,
    RunArtifact,
    RunManifest,
    RunResult,
    RunSummary,
)
from judgekit.runner import execute_run, write_artifact
from judgekit.scorers import Scorer, available_scorers, get_scorer, register

__version__ = "0.1.0"

__all__ = [
    "Case",
    "CaseRecord",
    "Dataset",
    "DatasetError",
    "JudgekitError",
    "OutputRecord",
    "RunArtifact",
    "RunError",
    "RunManifest",
    "RunResult",
    "RunSummary",
    "Scorer",
    "ScoringError",
    "UnknownScorerError",
    "available_scorers",
    "execute_run",
    "get_scorer",
    "load_dataset",
    "load_outputs",
    "register",
    "write_artifact",
]

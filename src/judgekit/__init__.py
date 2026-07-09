"""judgekit: evaluate your evaluators, with judge-vs-human validity as the headline metric."""

from judgekit.calibration import RaterRatings, execute_calibration, write_report
from judgekit.dataset import load_dataset, load_outputs, load_ratings
from judgekit.errors import (
    CalibrationError,
    DatasetError,
    JudgekitError,
    RunError,
    ScoringError,
    StatsError,
    UnknownScorerError,
)
from judgekit.models import (
    AgreementEstimate,
    CalibrationReport,
    Case,
    CaseRecord,
    Dataset,
    KappaEstimate,
    OutputRecord,
    RaterSummary,
    RatingRecord,
    RunArtifact,
    RunManifest,
    RunResult,
    RunSummary,
    SliceAgreement,
)
from judgekit.runner import execute_run, write_artifact
from judgekit.scorers import Scorer, available_scorers, get_scorer, register
from judgekit.stats import (
    BootstrapCI,
    bootstrap_ci,
    cohen_kappa,
    krippendorff_alpha,
    percent_agreement,
)

__version__ = "0.1.0"

__all__ = [
    "AgreementEstimate",
    "BootstrapCI",
    "CalibrationError",
    "CalibrationReport",
    "Case",
    "CaseRecord",
    "Dataset",
    "DatasetError",
    "JudgekitError",
    "KappaEstimate",
    "OutputRecord",
    "RaterRatings",
    "RaterSummary",
    "RatingRecord",
    "RunArtifact",
    "RunError",
    "RunManifest",
    "RunResult",
    "RunSummary",
    "Scorer",
    "ScoringError",
    "SliceAgreement",
    "StatsError",
    "UnknownScorerError",
    "available_scorers",
    "bootstrap_ci",
    "cohen_kappa",
    "execute_calibration",
    "execute_run",
    "get_scorer",
    "krippendorff_alpha",
    "load_dataset",
    "load_outputs",
    "load_ratings",
    "percent_agreement",
    "register",
    "write_artifact",
    "write_report",
]

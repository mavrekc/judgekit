"""Exception hierarchy for judgekit."""


class JudgekitError(Exception):
    """Base class for all judgekit errors."""


class DatasetError(JudgekitError):
    """Raised for dataset loading or validation problems."""


class ScoringError(JudgekitError):
    """Raised when a scorer fails to produce a result."""


class RunError(JudgekitError):
    """Raised for run orchestration failures."""


class StatsError(JudgekitError):
    """Raised when an agreement statistic is undefined or its input is invalid."""


class CalibrationError(JudgekitError):
    """Raised for calibration report alignment or construction problems."""


class UnknownScorerError(JudgekitError):
    """Raised when a requested scorer id is not registered."""

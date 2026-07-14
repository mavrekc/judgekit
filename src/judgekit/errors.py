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


class ProviderError(JudgekitError):
    """Raised when an LLM provider call fails after retries or returns an unusable response."""


class JudgeError(JudgekitError):
    """Raised for judge config or judge run problems."""


class BudgetExceededError(JudgeError):
    """Raised when a judge run reaches its cost ceiling before completing."""


class UnknownScorerError(JudgekitError):
    """Raised when a requested scorer id is not registered."""
